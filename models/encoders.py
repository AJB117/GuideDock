import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from equiformer_pytorch import Equiformer
from torch import Tensor

from utils import construct_3d_basis

from .common import GLU, GaussianSmearing
from .faformer import FAFormer
from .flash_ipa.edge_embedder import EdgeEmbedderConfig
from .flash_ipa.ipa import IPAConfig
from .flash_ipa.model import Model, ModelConfig, create_rigid
from .ipa import IPABlock
from .modules import AdaLN, DiTBlock, FourierFeats


class FlashIPAEncoder(nn.Module):
    def __init__(self, hidden_size, heads, depth, dropout):
        super(FlashIPAEncoder, self).__init__()
        max_len = 8192
        node_embed_size = hidden_size
        edge_embed_size = hidden_size // 2

        use_flash_attn = True
        attn_dtype = "fp16"
        c_s = node_embed_size
        c_z = edge_embed_size
        c_hidden = hidden_size
        no_heads = heads
        z_factor_rank = 2
        no_qk_points = 8
        no_v_points = 12
        seq_tfmr_num_heads = heads
        seq_tfmr_num_layers = 1
        num_blocks = depth

        mode = "flash_1d_bias"
        k_nearest_neighbors = 20

        ipa_conf = IPAConfig(
            use_flash_attn=use_flash_attn,
            attn_dtype=attn_dtype,
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=no_heads,
            z_factor_rank=z_factor_rank,
            no_qk_points=no_qk_points,
            no_v_points=no_v_points,
            seq_tfmr_num_heads=seq_tfmr_num_heads,
            seq_tfmr_num_layers=seq_tfmr_num_layers,
            num_blocks=num_blocks,
        )
        edge_features_conf = EdgeEmbedderConfig(
            z_factor_rank=z_factor_rank,
            c_s=c_s,
            c_p=c_z,
            mode=mode,
            k=k_nearest_neighbors,
            max_len=max_len,
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

        batch = {
            "node_embeddings": h,
            "translations": t,
            "rotations": R,
            "node_mask": mask,
        }

        output = self.encoder(batch, time_steps)

        if return_coords:
            coords = x[:, :, 1]
            coords, rigid = self.apply_rot_and_tr(
                coords, output["pred_rotmats"], output["pred_trans"]
            )
            return coords

        return output["embs"]


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
            ),
            dim_head=(2, 2),
            heads=(2, 2),
            num_linear_attn_heads=args.heads // 4,
            num_degrees=2,
            depth=args.depth // 2,
            attend_self=True,
            reduce_dim_out=True,
            l2_dist_attention=False,
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

        embs, coords = self.faformer(h, x, radius, mask)

        return embs


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
            [nn.LayerNorm(hsize, bias=False) for _ in range(args.depth)]
        )
        self.feed_forwards = nn.ModuleList(
            [GLU(hsize, args, act="silu", bias=True) for _ in range(args.depth)]
        )

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
        h = self.down_project(h)

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
        x = x / 10

        rotations = repeat(torch.eye(3, device=x.device), "... -> b n ...", b=B, n=N)
        translations = x

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

        return h


class PairResidueEncoder(nn.Module):
    def __init__(self, args):
        super(PairResidueEncoder, self).__init__()
        self.W_pair = nn.Embedding(
            args.vocab_size * args.vocab_size, args.hidden_size // 2
        )
        self.dist_coeffs = nn.Embedding(args.vocab_size * args.vocab_size, 4 * 4)
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
        pair_idx = aa.unsqueeze(-1) * aa.unsqueeze(-2)
        pair_embedding = self.W_pair(pair_idx)

        B, L = aa.size(0), aa.size(1)

        d = (
            torch.linalg.norm(
                X_backbone[:, :, None, :, None] - X_backbone[:, None, :, None, :],
                dim=-1,
                ord=2,
            ).reshape(B, L, L, -1)
            / 10
        )
        c = F.softplus(self.dist_coeffs(pair_idx))
        d_gauss = torch.exp(-c * d**2)
        d_gauss = self.dist_embedding(d_gauss)

        emb = torch.cat([pair_embedding, d_gauss], dim=-1)

        return self.out_mlp(emb)


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

        h = h.sum(dim=1) / num_nonzero

        h = self.psi(h)

        return h
