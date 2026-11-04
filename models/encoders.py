import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
from utils import *
from einops import repeat
from equiformer_pytorch import Equiformer
from torch import Tensor

from .faformer import FAFormer
from .ipa import IPABlock
from .flash_ipa.ipa import IPAConfig
from .flash_ipa.edge_embedder import EdgeEmbedderConfig
from .flash_ipa.model import Model, ModelConfig, create_rigid
from .modules import FourierFeats, GaussianSmearing, FiLM, GLU, AdaLN, DiTBlock

class FlashIPAEncoder(nn.Module):
    def __init__(self, hidden_size, heads, depth, dropout):
        super(FlashIPAEncoder, self).__init__()
        max_len = 8192  # max length of the input sequence
        node_embed_size = hidden_size
        edge_embed_size = hidden_size // 2

        # IPA config options
        use_flash_attn = True  # True for flash ipa, False for original ipa
        attn_dtype = "fp16"  # "fp16", "bf16", "fp32". For flash ipa, bf16 or fp16. For original, fp32.
        c_s = node_embed_size  # size of the node embedding
        c_z = edge_embed_size  # size of the edge embedding
        c_hidden = hidden_size  # size of the hidden layer in the ipa
        no_heads = heads  # number of attention heads in the ipa
        z_factor_rank = (
            2  # rank of the z factor rank for the edge embedding factorization
        )
        no_qk_points = 8  # number of query/key points
        no_v_points = 12  # number of value points
        seq_tfmr_num_heads = (
            heads  # number of heads in the node embedding update transformer
        )
        seq_tfmr_num_layers = (
            1  # number of layers in the node embedding update transformer
        )
        num_blocks = depth  # number of blocks in the model

        # Edge embedder config
        mode = "flash_1d_bias"  # "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
        k_nearest_neighbors = 20  # Number of nearest neighbors for the distogram

        ipa_conf = IPAConfig(
            use_flash_attn=use_flash_attn,
            attn_dtype=attn_dtype,
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=no_heads,
            z_factor_rank=z_factor_rank,  # Rank of the factorization of the edge embedding
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
            seq_tfmr_num_heads=seq_tfmr_num_heads,
            seq_tfmr_num_layers=seq_tfmr_num_layers,
            num_blocks=num_blocks,
        )
        edge_features_conf = EdgeEmbedderConfig(
            z_factor_rank=z_factor_rank,  # Rank of the factorization of the edge embedding
            c_s=c_s,  # Size of the node embedding
            c_p=c_z,  # Size of the edge embedding
            mode=mode,  # "orig_no_bias", "orig_2d_bias", "flash_no_bias", "flash_1d_bias", "flash_2d_factorize_bias"
            k=k_nearest_neighbors,  # Number of nearest neighbors for the distogram
            max_len=max_len,  # Maximum length of the input sequence
            num_bins=22,
            feat_dim=c_z,
        )
        model_conf = ModelConfig(
            mode=mode,
            node_embed_size=node_embed_size,
            edge_embed_size=edge_embed_size,
            ipa=ipa_conf,
            edge_features=edge_features_conf,
        )
        self.encoder = Model(model_conf, dropout)
        # self.edge_embedding = GaussianSmearing(0.0, args.threshold, args.hidden_size // 2)
        # self.proj = nn.Linear(args.noise_emb_dim, args.hidden_size)

    def apply_rot_and_tr(self, X: Tensor, R: Tensor, t: Tensor):
        """
        Apply the rotation R and translation t from IPA to the coordinates x.
        """
        rigid = create_rigid(R, t)
        x_refined = rigid.apply(X)

        return x_refined, rigid

    def forward(
        self,
        x: Tensor,
        h: Tensor,
        mask: Tensor,
        sel: int,
        time_steps: Tensor,
        return_coords=False,
    ):
        R = construct_3d_basis(x[:, :, 1], x[:, :, 2], x[:, :, 0])
        t = x[:, :, 1]
        # t = x.view(h.shape[0], -1, 3)

        # if sel > 1:
        #     R = R.repeat_interleave(sel, dim=1)
        #     t = t.repeat_interleave(sel, dim=1)

        # dists = self.edge_embedding((
        #     x.view(1, -1, 3)[:, :, None, :] - x.view(1, -1, 3)[:, None, :, :]
        # ).norm(dim=-1, p=2))
        # fourier_feats = self.fourier_feats(x.view(h.shape[0], -1, 3))
        # h = self.coord_proj(torch.cat((h, fourier_feats), dim=-1))

        batch = {
            "node_embeddings": h,
            "translations": t,
            "rotations": R,
            "node_mask": mask,
            # "edge_embeddings": dists,
        }

        # time_proj = self.proj(time_steps)
        output = self.encoder(batch, time_steps)

        if return_coords:
            coords = x[:, :, 1]
            coords, rigid = self.apply_rot_and_tr(
                coords, output["pred_rotmats"], output["pred_trans"]
            )
            return coords

        return output["embs"]  # [B, N, hidden_size]


class DiTEncoder(nn.Module):
    def __init__(self, args):
        super(DiTEncoder, self).__init__()
        self.model = nn.ModuleList(
            [
                DiTBlock(
                    args=args,
                    dim=args.hidden_size,
                    heads=args.heads,
                    mlp_ratio=args.ff_mult,
                    dropout=args.dropout,
                )
                for _ in range(args.depth)
            ]
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(args.hidden_size * 4, args.hidden_size),
        )

    def forward(
        self, X: Tensor, H: Tensor, mask: Tensor, time_steps: Tensor, dist_emb: Tensor
    ) -> Tensor:
        time_steps = self.time_mlp(time_steps)
        for block in self.model:
            H = block(H, time_steps, dist_emb, (~mask).float())

        return H


class CrossAttentionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.normalizer_layers = nn.ModuleList()
        self.ada_ln_layers = nn.ModuleList()
        self.depth = args.depth

        hidden_size = args.hidden_size
        attn_embed_dim = hidden_size

        for _ in range(self.depth):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=attn_embed_dim,
                    num_heads=8,
                    dropout=args.dropout,
                    batch_first=True,
                )
            )
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 3),
                    # nn.ReLU(),
                    # nn.Dropout(args.dropout),
                    # nn.Linear(hidden_size, hidden_size),
                    GLU(hidden_size * 3, args, act="silu"),
                    nn.Linear(hidden_size * 3, hidden_size),
                )
            )
            self.normalizer_layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(hidden_size),
                        nn.LayerNorm(hidden_size),
                    ]
                )
            )
            self.ada_ln_layers.append(
                nn.ModuleList(
                    [AdaLN(hidden_size, hidden_size), AdaLN(hidden_size, hidden_size)]
                )
            )

        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_time_emb: Tensor,
        kv_time_emb: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = query

        for idx in range(self.depth):
            norm_attn, norm_mlp = self.normalizer_layers[idx]

            query = query + query_time_emb
            key = key + kv_time_emb
            value = value + kv_time_emb

            qn = norm_attn(query)
            kn = norm_attn(key)
            vn = norm_attn(value)

            qn = qn + query_time_emb
            kn = kn + kv_time_emb
            vn = vn + kv_time_emb

            attn_out, _ = self.attn_layers[idx](
                qn, kn, vn, attn_mask=attn_mask, need_weights=False
            )
            attn_out = x + self.dropout(attn_out)

            mlp_in = norm_mlp(attn_out)
            mlp_in = mlp_in + query_time_emb
            mlp_out = self.mlp_layers[idx](mlp_in)
            mlp_out = x + self.dropout(mlp_out)
            mlp_out = mlp_out + query_time_emb

        return mlp_out


class EquiformerEncoder(nn.Module):
    def __init__(self, args):
        super(EquiformerEncoder, self).__init__()
        self.down_proj = nn.Linear(args.hidden_size, args.hidden_size // 2)
        self.up_proj = nn.Linear(748, args.hidden_size)
        self.model = Equiformer(
            num_tokens=None,
            dim=(
                args.hidden_size // 2,
                2,
            ),  # dimensions per type, ascending, length must match number of degrees (num_degrees)
            dim_head=(2, 2),  # dimension per attention head
            heads=(2, 2),  # number of attention heads
            num_linear_attn_heads=args.heads
            // 4,  # number of global linear attention heads, can see all the neighbors
            num_degrees=2,  # number of degrees
            depth=args.depth // 2,  # depth of equivariant transformer
            attend_self=True,  # attending to self or not
            reduce_dim_out=True,  # whether to reduce out to dimension of 1, say for predicting new coordinates for type 1 features
            l2_dist_attention=False,  # set to False to try out MLP attention
            valid_radius=15,
        )

    def forward(self, h, x, mask):
        h = self.down_proj(h)
        embs = self.model(h, x, mask).type0
        embs = self.up_proj(embs)
        return embs


class FAFormerEncoder(nn.Module):
    def __init__(self, args, d_input):
        super(FAFormerEncoder, self).__init__()
        self.faformer = FAFormer(
            # d_input=args.hidden_size + 2 * args.fourier_dim,
            d_input=d_input,
            d_model=args.hidden_size // 2,
            n_layers=args.depth,
            n_neighbors=30,
            n_heads=args.heads,
            d_edge_model=args.hidden_size // 2,
            activation="silu",
            valid_radius=args.threshold,
            embedding_grad_frac=0.5,
            attn_drop=args.dropout,
            proj_drop=args.dropout,
        )
        self.fourier_feats = FourierFeats(args)

    def forward(self, h, x, mask, radius=15.0, pair_h=None):
        # fourier_x = self.fourier_feats(x)
        # h = torch.cat((h, fourier_x), dim=-1)
        embs, coords = self.faformer(h, x, radius, mask)

        return embs


class EGNNEncoder(nn.Module):
    def __init__(self, args):
        super(EGNNEncoder, self).__init__()
        self.egnn = EGNN_Network(
            depth=args.depth,
            dim=args.hidden_size,
            fourier_features=args.fourier_dim,
            m_dim=args.hidden_size // 2,
            soft_edges=True,
            valid_radius=40,
        )

    def forward(self, h, x, valid_radius=15.0):
        # edge_idx, batch = get_graph(
        #     x, valid_radius=valid_radius
        # )

        emb = self.egnn(h, x)

        return emb


normalizers = {
    "layer": nn.LayerNorm,
    "batch": nn.BatchNorm1d,
    "rms": nn.RMSNorm,
    "none": nn.Identity,
}


class SAEncoder(nn.Module):
    def __init__(self, args):
        super(SAEncoder, self).__init__()
        hsize = args.hidden_size
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hsize, args.heads, dropout=args.dropout, batch_first=True
                )
                for _ in range(args.depth)
            ]
        )
        self.normalizers = nn.ModuleList(
            [normalizers[args.normalization](hsize) for _ in range(args.depth)]
        )
        self.lns_b = nn.ModuleList(
            [
                # normalizers[args.normalization](hsize)
                nn.LayerNorm(hsize, bias=False)
                for _ in range(args.depth)
            ]
        )
        self.feed_forwards = nn.ModuleList(
            [
                # nn.Sequential(
                #     nn.Linear(hsize, hsize * 3),
                #     nn.SiLU(),
                #     nn.Linear(hsize * 3, hsize),
                # )
                # SwiGLU(args)
                GLU(hsize, args, act="silu", bias=True)
                for _ in range(args.depth)
            ]
        )
        # initialize all weights with xavier
        for att, ff in zip(self.attention_layers, self.feed_forwards):
            nn.init.kaiming_uniform_(att.in_proj_weight)
            nn.init.kaiming_uniform_(att.out_proj.weight)

            nn.init.kaiming_uniform_(ff.gate[0].weight)
            nn.init.kaiming_uniform_(ff.gate[3].weight)

        self.down_project = nn.Linear(args.hidden_size, hsize)
        self.up_project = nn.Linear(hsize, args.hidden_size)

        self.distance_embedding = GaussianSmearing(0, 180, 2)
        self.distance_proj = nn.Linear(2, 1)
        self.heads = args.heads

    def forward(self, X, h, mask, sel, all_time_steps):
        h = self.down_project(h)  # [B, N, C] -> [B, N, C//4]

        B = X.shape[0]
        X = X.view(B, -1, 3)
        N = X.shape[1]

        distance_feats = torch.cdist(X, X, p=2).unsqueeze(-1)
        distance_embedding = self.distance_embedding(distance_feats)
        distance_h = self.distance_proj(distance_embedding).view(B, N, N)
        distance_h = distance_h.repeat_interleave(self.heads, 0)

        for att, norm_a, norm_b, ff in zip(
            self.attention_layers,
            self.normalizers,
            self.lns_b,
            self.feed_forwards,
        ):
            initial = h
            h = norm_a(h)
            h, _ = att(h, h, h, key_padding_mask=(~mask).float(), attn_mask=distance_h)
            h = h + initial
            pre_res = h
            h = norm_b(h)
            h = ff(h)
            h = h + pre_res
            h = h + all_time_steps

        h = self.up_project(h)

        return h


class BidirectionalLSTMEncoder(nn.Module):
    def __init__(self, args):
        super(BidirectionalLSTMEncoder, self).__init__()
        self.lr_encoder = nn.GRU(
            args.hidden_size,
            args.hidden_size,
            num_layers=args.depth,
            dropout=args.dropout,
            bidirectional=False,
        )
        self.rl_encoder = nn.GRU(
            args.hidden_size,
            args.hidden_size,
            num_layers=args.depth,
            dropout=args.dropout,
            bidirectional=False,
        )

    def forward(self, X):
        h_lr = self.lr_encoder(X)[0]
        X_rl = torch.flip(X, [1])
        h_rl = self.rl_encoder(X_rl)[0]
        h = (h_lr + h_rl) / 2
        return h


class IPAEncoder(nn.Module):
    def __init__(self, args, require_pairwise_repr=False):
        super(IPAEncoder, self).__init__()
        self.net = nn.ModuleList(
            [
                IPABlock(
                    dim=args.hidden_size,
                    ff_mult=args.ff_mult,
                    heads=args.heads,
                    ff_num_layers=args.ff_num_layers,
                    post_norm=False,
                    pairwise_repr_dim=args.hidden_size,
                    require_pairwise_repr=require_pairwise_repr,
                    scalar_key_dim=args.scalar_kv_dim,
                    scalar_value_dim=args.scalar_kv_dim,
                    point_key_dim=args.point_kv_dim,
                    point_value_dim=args.point_kv_dim,
                    use_swiglu=args.use_swiglu,
                    post_attn_dropout=args.dropout,
                    post_ff_dropout=args.dropout,
                )
                for _ in range(args.depth)
            ]
        )
        self.linears = nn.ModuleList(
            [nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.depth)],
        )

        self.require_pairwise_repr = require_pairwise_repr

        self.initialization = args.initialization
        self.initialize_parameters()

    def initialize_parameters(self):
        for net, linear in zip(self.net, self.linears):
            for p in net.parameters():
                if p.dim() > 1:
                    if self.initialization == "xavier_u":
                        nn.init.xavier_uniform_(p)
                    elif self.initialization == "xavier_n":
                        nn.init.xavier_normal_(p)
                    elif self.initialization == "kaiming_u":
                        nn.init.kaiming_uniform_(p)
                    elif self.initialization == "kaiming_n":
                        nn.init.kaiming_normal_(p)
                else:
                    nn.init.zeros_(p)

            if self.initialization == "xavier_u":
                nn.init.xavier_uniform_(linear.weight)
            elif self.initialization == "xavier_n":
                nn.init.xavier_normal_(linear.weight)
            elif self.initialization == "kaiming_u":
                nn.init.kaiming_uniform_(linear.weight)
            elif self.initialization == "kaiming_n":
                nn.init.kaiming_normal_(linear.weight)

            nn.init.zeros_(linear.bias)

    def forward(self, h, x, mask, pair_h=None):
        B, N = h.size(0), h.size(1)
        x = x / 10  # angstrom to nm

        rotations = repeat(torch.eye(3, device=x.device), "... -> b n ...", b=B, n=N)
        translations = x

        prev_h = h

        for i in range(len(self.net)):
            if self.require_pairwise_repr:
                h = self.net[i](
                    h,
                    rotations=rotations,
                    translations=translations,
                    mask=mask,
                    pairwise_repr=pair_h,
                )
            else:
                h = self.net[i](
                    h, rotations=rotations, translations=translations, mask=mask
                )

            # h = self.linears[i](h)

        return h


class PairResidueEncoder(nn.Module):
    def __init__(self, args):
        super(PairResidueEncoder, self).__init__()
        self.W_pair = nn.Embedding(
            args.vocab_size * args.vocab_size, args.hidden_size // 2
        )
        self.dist_coeffs = nn.Embedding(
            args.vocab_size * args.vocab_size, 4 * 4
        )  # backbone atoms only
        nn.init.zeros_(self.dist_coeffs.weight)
        self.dist_embedding = nn.Sequential(
            nn.Linear(4 * 4, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, args.hidden_size // 2),
            nn.ReLU(),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(args.hidden_size // 2, args.hidden_size),
        )

    def forward(self, aa, X_backbone):
        """
        aa: [B, L], B is batch, L is the maximum length of the sequence
        X: [B, L, 4, 3], 4 is the number of atoms in each residue, 3 is the coordinates of each atom
        """
        pair_idx = aa.unsqueeze(-1) * aa.unsqueeze(-2)  # [B, L_max, L_max]
        pair_embedding = self.W_pair(pair_idx)  # [B, L_max, L_max, hidden_size]

        # return pair_embedding * pair_mask # [B, L_max, L_max, hidden_size]

        B, L = aa.size(0), aa.size(1)

        d = (
            torch.linalg.norm(
                X_backbone[:, :, None, :, None] - X_backbone[:, None, :, None, :],
                dim=-1,
                ord=2,
            ).reshape(B, L, L, -1)
            / 10
        )  # [B, L, L, 4]
        c = F.softplus(self.dist_coeffs(pair_idx))  # [B, L, L, 4*4]
        d_gauss = torch.exp(-c * d**2)  # [B, L, L, 4*4]
        d_gauss = self.dist_embedding(d_gauss)  # [B, L, L, hidden_size]

        emb = torch.cat([pair_embedding, d_gauss], dim=-1)  # [B, L, L, hidden_size*2]

        return self.out_mlp(emb)  # [B, L, L, hidden_size]


class DeepSets(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepSets, self).__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.psi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h, num_nonzero):
        """
        h: [B, N, D] tensor of features
        """
        h = self.phi(h)
        # mean pool using mask
        h = h.sum(dim=1) / num_nonzero

        h = self.psi(h)

        return h
