import math
import pdb
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from utils import *

# from .graph import build_cross_graph_batch, build_rigid_graph_batch
# from .mpnn import MPNNEncoder
from .encoders import (
    SAEncoder,
    BidirectionalLSTMEncoder,
    IPAEncoder,
    FAFormerEncoder,
    PairResidueEncoder,
    # EGNNEncoder,
    EquiformerEncoder,
    FlashIPAEncoder,
    CrossAttentionEncoder,
    DiTEncoder,
)
from .modules import GaussianSmearing, FiLM, AdditiveGate, FourierFeats
from .latent_modules import CustomCrossAttentionTransformer, SinusoidalDistanceEmbedding


class FrameAveraging(nn.Module):
    def __init__(self):
        super(FrameAveraging, self).__init__()
        self.ops = torch.tensor(
            [[i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]]
        ).cuda()

    def create_frame(self, X, mask):
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,3]
        C = torch.bmm(X.transpose(1, 2), X)  # [B,3,3] (Cov)
        # _, V = torch.symeig(C.detach(), True)  # [B,3,3]
        _, V = torch.linalg.eigh(C.detach(), UPLO="U")

        F_ops = self.ops.unsqueeze(1).unsqueeze(0) * V.unsqueeze(
            1
        )  # [1,8,1,3] x [B,1,3,3] -> [B,8,3,3]
        h = torch.einsum(
            "boij,bpj->bopi", F_ops.transpose(2, 3), X
        )  # transpose is inverse [B,8,N,3]
        h = h.view(X.size(0) * 8, X.size(1), 3)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum("boij,bopj->bopi", F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        return X * mask.unsqueeze(-1)


class FAEncoder(FrameAveraging):
    def __init__(self, args):
        super(FAEncoder, self).__init__()
        self.bind_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.tgt_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.W_bind = nn.Linear(args.embedding_size, args.hidden_size)
        self.W_tgt = nn.Linear(args.embedding_size, args.hidden_size)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=args.hidden_size,
                nhead=4,
                dim_feedforward=args.hidden_size // 2,
                dropout=0,
                activation="gelu",
            ),
            num_layers=args.depth,
        )
        self.proj = nn.Linear(args.hidden_size + 3, args.hidden_size)
        self.silu = nn.SiLU()

    def forward(self, binder, target):
        bind_X, bind_S, bind_A, _ = binder
        tgt_X, tgt_S, tgt_A, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_S.size(1)
        bind_mask = bind_A[:, :, 1].clamp(max=1).float()
        tgt_mask = tgt_A[:, :, 1].clamp(max=1).float()
        mask = torch.cat([bind_mask, tgt_mask], dim=1)

        if bind_S.dim() == 2:
            bind_S = self.bind_embedding(bind_S)
        else:
            bind_S = self.W_bind(bind_S)

        if tgt_S.dim() == 2:
            tgt_S = self.tgt_embedding(tgt_S)
        else:
            tgt_S = self.W_tgt(tgt_S)

        h_S = torch.cat([bind_S, tgt_S], dim=1)
        X = torch.cat([bind_X, tgt_X], dim=1)
        h_X, _, _ = self.create_frame(X[:, :, 1], mask)
        h_S = h_S.unsqueeze(1).expand(-1, 8, -1, -1).reshape(B * 8, N + M, -1)
        mask = mask.unsqueeze(1).expand(-1, 8, -1).reshape(B * 8, N + M)

        h = torch.cat([h_X, h_S], dim=-1)
        h = self.proj(h)
        h = self.silu(h)
        h = self.encoder(h.transpose(0, 1), src_key_padding_mask=~mask.bool())
        h = h.transpose(0, 1).view(B, 8, N + M, -1)
        return h.mean(dim=1)  # frame averaging


class SwiGLU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.W1 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.W2 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.W3 = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, X):
        return self.W3(self.silu(self.W1(X)) * self.W2(X))


class AllAtomEncoder(FrameAveraging):
    def __init__(self, args):
        super(AllAtomEncoder, self).__init__()

        hidden_size = args.hidden_size
        self.embedding = args.embedding

        if args.embedding:
            self.W_bind = nn.Linear(args.embedding_size, hidden_size)
            self.W_tgt = nn.Linear(args.embedding_size, hidden_size)
        else:
            self.W_bind_ = nn.Embedding(len(ALPHABET), args.embedding_size)
            self.W_tgt_ = nn.Embedding(len(ALPHABET), args.embedding_size)

            torch.nn.init.orthogonal_(self.W_bind_.weight)
            torch.nn.init.orthogonal_(self.W_tgt_.weight)

            self.W_bind = nn.Sequential(
                self.W_bind_,
                nn.Linear(args.embedding_size, hidden_size),
            )
            self.W_tgt = nn.Sequential(
                self.W_tgt_,
                nn.Linear(args.embedding_size, hidden_size),
            )

        self.A_bind = nn.Embedding(args.vocab_size, hidden_size)
        self.A_tgt = nn.Embedding(args.vocab_size, hidden_size)
        self.V_bind = nn.Linear(6, hidden_size)
        self.V_tgt = nn.Linear(6, hidden_size)

        self.encoder_type = args.encoder
        self.resolution = args.resolution
        if self.resolution == "all_atom":
            self.residue_size = 14
        elif self.resolution == "backbone":
            self.residue_size = 4
        elif self.resolution == "CA":
            self.residue_size = 1
        else:
            raise ValueError(f"Unsupported resolution: {self.resolution}")
        self.bb = 0 if self.resolution == "CA" else 1
        self.up_to_bb = 0 if self.resolution == "CA" else 4
        self.encoder_type = args.encoder

        if args.encoder == "sa":
            self.encoder = SAEncoder(args)
            self.encoder_1 = SAEncoder(args)
            self.encoder_2 = SAEncoder(args)
        elif args.encoder == "lstm":
            self.encoder = BidirectionalLSTMEncoder(args)
        elif args.encoder == "ipa":
            if args.require_pairwise_repr:
                self.encoder_pre = IPAEncoder(args, require_pairwise_repr=True)

            self.encoder = IPAEncoder(args)
        elif args.encoder == "faformer":
            self.encoder_1 = FAFormerEncoder(
                args, d_input=hidden_size + 2 * args.fourier_dim
            )
            self.encoder_2 = FAFormerEncoder(
                args, d_input=hidden_size + 2 * args.fourier_dim
            )
            self.encoder_3 = FAFormerEncoder(
                args, d_input=hidden_size // 2 + 2 * args.fourier_dim
            )
        elif args.encoder == "gt":
            self.encoder_1 = GraphTransformer(args.hidden_size, n_layers=args.depth)
            self.encoder_2 = GraphTransformer(args.hidden_size, n_layers=args.depth)
        elif args.encoder == "erwin":
            self.encoder_1
        elif args.encoder == "egnn":
            self.encoder_1 = EGNNEncoder(args)
        elif args.encoder == "mpnn":
            # self.encoder_1 = MPNN(args)
            # self.encoder_2 = MPNN(args)
            # self.encoder_3 = CrossMPNN(args)
            self.encoder = MPNNEncoder(args)
        elif args.encoder == "equiformer":
            self.encoder = EquiformerEncoder(args)
        elif args.encoder == "flash_ipa":
            self.encoder = FlashIPAEncoder(
                hidden_size=hidden_size,
                depth=args.depth,
                heads=args.heads,
                dropout=args.dropout,
            )
            # self.encoder_1 = FlashIPAEncoder(args)
            # self.encoder_2 = FlashIPAEncoder(args)
        elif args.encoder == "cross":
            self.encoder = CrossAttentionEncoder(args)
            self.encoder_1 = CrossAttentionEncoder(args)
            self.encoder_2 = CrossAttentionEncoder(args)
        elif args.encoder == "dit":
            self.encoder = DiTEncoder(args)

        if args.encoder in ("ipa", "faformer", "flash_ipa", "sa", "cross", "dit"):
            if args.require_pairwise_repr:
                self.pair_encoder = PairResidueEncoder(args)
            self.proj = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.proj = nn.Linear(hidden_size * 2 + 3, hidden_size)

        self.heads = args.heads

        self.hidden_size = hidden_size

        self.down_project_1 = nn.Sequential(
            nn.Linear(hidden_size * 3 + 3 + 1, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
        )
        self.down_project_2 = nn.Sequential(
            nn.Linear(hidden_size * 3 + 3 + 1, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
        )
        self.non_conservative = args.non_conservative

        # self.edge_embedding = nn.Sequential(
        #     GaussianSmearing(0.0, 180, args.num_edge_feats),
        #     # GaussianSmearing(0.0, 180, args.num_edge_feats),
        #     # nn.Linear(args.num_edge_feats, args.num_edge_feats),
        #     nn.GELU(),
        #     nn.Linear(args.num_edge_feats, 1, bias=False),
        # )
        # self.edge_embedding = ResidualEdgeEncoder(args.num_edge_feats, end=180)
        self.edge_embedding = nn.Sequential(
            SinusoidalDistanceEmbedding(dim=args.num_edge_feats),
            nn.Linear(args.num_edge_feats // 2, 1),
        )

        self.binder_target_encoding = nn.Embedding(2, hidden_size)

        # self.cross_attention1 = MultiheadAttention(
        #     embed_dim=hidden_size // 2
        #     if args.encoder
        #     not in (
        #         "faformer",
        #         "gt",
        #         "mpnn",
        #         "equiformer",
        #         "flash_ipa",
        #         "sa",
        #         "cross",
        #         "dit",
        #     )
        #     else hidden_size,
        #     num_heads=args.heads,
        #     dropout=args.dropout,
        #     batch_first=True,
        # )

        # self.cross_attention = MultiheadAttention(
        #     embed_dim=hidden_size // 2
        #     if args.encoder
        #     not in (
        #         "faformer",
        #         "gt",
        #         "mpnn",
        #         "equiformer",
        #         "flash_ipa",
        #         "sa",
        #         "cross",
        #         "dit",
        #     )
        #     else hidden_size,
        #     num_heads=args.heads,
        #     dropout=args.dropout,
        #     batch_first=True,
        # )

        self.cross_attention1 = CustomCrossAttentionTransformer(
            dim=hidden_size,
            depth=1,
            mlp_ratio=3,
            dropout=args.dropout,
            heads=args.heads,
        )
        self.cross_attention = CustomCrossAttentionTransformer(
            dim=hidden_size,
            depth=1,
            mlp_ratio=3,
            dropout=args.dropout,
            heads=args.heads,
        )

        final_proj_dim = (
            hidden_size // 2
            if args.encoder
            not in (
                "faformer",
                "gt",
                "mpnn",
                "equiformer",
                "flash_ipa",
                "sa",
                "cross",
                "dit",
            )
            else hidden_size
        )

        # if self.non_conservative:
        #     self.final_proj = DeepSets(final_proj_dim, hidden_size, hidden_size)
        # else:
        self.final_proj1 = nn.Sequential(
            nn.Linear(final_proj_dim, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2),
        )
        self.final_proj2 = nn.Sequential(
            nn.Linear(final_proj_dim, hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size // 2, hidden_size // 2),
        )
        self.proj_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.proj_2 = nn.Linear(hidden_size, hidden_size // 2)

        self.merge_layer_bind = FiLM(hidden_size, args)
        self.merge_layer_tgt = FiLM(hidden_size, args)

        self.n_supernodes = args.n_supernodes
        self.distance_linear = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.distance_proj = nn.Sequential(nn.Linear(1 + hidden_size, hidden_size))
        self.layernorm = nn.LayerNorm(hidden_size)

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dropout=0,
                activation="relu",
            ),
            num_layers=4,
        )

        # self.spline_feature_encoder = nn.Sequential(
        #     FourierFeats(args, 2),
        #     nn.Linear(4 * args.fourier_dim, hidden_size // 2),
        #     nn.Sigmoid()
        # )

    def coarsen_A_types(self, A):
        """
        A: [B, N, hidden_size] tensor of atom embeddings to coarsen N -> n_supernodes
        """
        pass

    def mean(self, X: Tensor, mask: Tensor):
        return (X * mask[..., None]).sum(dim=1) / mask[..., None].sum(dim=1).clamp(
            min=1e-6
        )

    def forward(
        self,
        binder: Features,
        target: Features,
        eps_tr: Tensor,
        Ts=None,
        snr=None,
        coarser_trend=None,
        debug=False,
        time_steps: Tensor = None,
    ):
        sel = self.residue_size
        bb = self.bb

        bind_X, bind_S, bind_A, *_ = binder
        tgt_X, tgt_S, tgt_A, *_ = target
        # bind_X, bind_Si, bind_A = binder.X, binder.Si, binder.A
        # tgt_X, tgt_Si, tgt_A = target.X, target.Si, target.A

        bind_mask_residue = bind_A[:, :, bb].clamp(max=1).float()
        tgt_mask_residue = tgt_A[:, :, bb].clamp(max=1).float()

        B, M, N = bind_S.size(0), bind_S.size(1), tgt_S.size(1)

        bind_A = bind_A.view(B, M * sel)
        tgt_A = tgt_A.view(B, N * sel)

        bind_S = bind_S[:, :, None, :].expand(-1, -1, sel, -1).reshape(B, M * sel, -1)
        tgt_S = tgt_S[:, :, None, :].expand(-1, -1, sel, -1).reshape(B, N * sel, -1)
        # bind_S = self.W_bind(bind_Si)[:, :, None, :].expand(-1, -1, sel, -1).reshape(B, M * sel, -1)
        # tgt_S = self.W_tgt(tgt_Si)[:, :, None, :].expand(-1, -1, sel, -1).reshape(B, N * sel, -1)

        if not self.embedding:
            bind_S = bind_S.squeeze(-1)
            tgt_S = tgt_S.squeeze(-1)

        time_steps_bind = time_steps.view(B, 1, -1).repeat_interleave(M * sel, dim=1)
        time_steps_tgt = time_steps.view(B, 1, -1).repeat_interleave(N * sel, dim=1)

        bind_A_H = self.A_bind(bind_A)
        bind_S_H = self.W_bind(bind_S)
        tgt_A_H = self.A_tgt(tgt_A)
        tgt_S_H = self.W_tgt(tgt_S)

        binder_mask = bind_A.view(B, -1) > 0
        tgt_mask = tgt_A.view(B, -1) > 0

        com_bind = self.mean(bind_X.view(B, -1, 3), binder_mask)[:, None, :].expand(
            -1, M * sel, -1
        )
        com_tgt = self.mean(tgt_X.view(B, -1, 3), tgt_mask)[:, None, :].expand(
            -1, N * sel, -1
        )
        com_bind_mag = torch.norm(com_bind, dim=-1, keepdim=True).log().clamp(min=0.0)
        com_tgt_mag = torch.norm(com_tgt, dim=-1, keepdim=True).log().clamp(min=0.0)
        com_bind_normed = com_bind / (torch.norm(com_bind, dim=-1, keepdim=True) + 1e-6)
        com_tgt_normed = com_tgt / (torch.norm(com_tgt, dim=-1, keepdim=True) + 1e-6)

        # spline_torsion_bind, spline_curvature_bind = binder.T.view(B, M*sel, 1), binder.C.view(B, M * sel, 1).log()
        # spline_torsion_tgt, spline_curvature_tgt = target.T.view(B, N*sel, 1), target.C.view(B, N * sel, 1).log()

        # spline_feats_bind = spline_torsion_bind / 100 + spline_curvature_bind / 100
        # spline_feats_tgt = spline_torsion_tgt / 100 + spline_curvature_tgt / 100
        # spline_feats_bind = self.spline_feature_encoder(torch.cat([spline_torsion_bind, spline_curvature_bind], dim=-1))
        # spline_feats_tgt = self.spline_feature_encoder(torch.cat([spline_torsion_tgt, spline_curvature_tgt], dim=-1))

        bind_H = torch.cat(
            [bind_S_H, bind_A_H, time_steps_bind, com_bind_normed, com_bind_mag], dim=-1
        )
        tgt_H = torch.cat(
            [tgt_S_H, tgt_A_H, time_steps_tgt, com_tgt_normed, com_tgt_mag], dim=-1
        )

        bind_H = self.down_project_1(bind_H)
        tgt_H = self.down_project_2(tgt_H)

        # bind_dist_emb, tgt_dist_emb = self.get_dists(
        #     bind_X, tgt_X, binder_mask, tgt_mask
        # )

        # bind_H_dist, _ = self.cross_attention(
        #     bind_H,
        #     tgt_H,
        #     tgt_H,
        #     # bind_dist_emb
        #     key_padding_mask=(~tgt_mask).float(),
        #     attn_mask=bind_dist_emb,
        #     need_weights=False,
        # )
        # tgt_H_dist, _ = self.cross_attention1(
        #     tgt_H,
        #     bind_H,
        #     bind_H,
        #     # tgt_dist_emb
        #     key_padding_mask=(~binder_mask).float(),
        #     attn_mask=tgt_dist_emb,
        #     need_weights=False,
        # )
        bind_dist_emb = self.get_internal_dists(binder, target)
        bind_H_dist = self.cross_attention(
            bind_H,
            tgt_H,
            attn_mask=bind_dist_emb,
        )
        tgt_H_dist = self.cross_attention1(
            tgt_H, bind_H, attn_mask=bind_dist_emb.transpose(1, 2)
        )

        if self.encoder_type == "dit":
            all_X = torch.cat(
                [bind_X.view(B, M, sel, 3), tgt_X.view(B, N, sel, 3)],
                dim=1,
            )
            all_H = torch.cat([bind_H, tgt_H], dim=1)
            all_mask = torch.cat([binder_mask, tgt_mask], dim=1)
            all_time_steps = time_steps.view(B, 1, -1).repeat_interleave(
                M * sel + N * sel, dim=1
            )
            dist_emb, _ = self.get_dists(all_X, all_X, all_mask, all_mask)

            all_H = self.encoder(all_X, all_H, all_mask, all_time_steps, dist_emb)

            # bind_H = all_H[:, : M * sel, :] + bind_H
            # tgt_H = all_H[:, M * sel :, :] + tgt_H
            bind_H = all_H[:, : M * sel, :]
            tgt_H = all_H[:, M * sel :, :]

        elif self.encoder_type == "cross":
            cross_bind_H = self.encoder_1(
                bind_H, tgt_H, tgt_H, time_steps_bind, time_steps_tgt, bind_dist_emb
            )
            bind_H = cross_bind_H + bind_H

            cross_tgt_H = self.encoder_2(
                tgt_H, bind_H, bind_H, time_steps_tgt, time_steps_bind, tgt_dist_emb
            )
            tgt_H = cross_tgt_H + tgt_H

        elif self.encoder_type in ("flash_ipa", "sa"):
            all_X = torch.cat(
                [bind_X.view(B, M, sel, 3), tgt_X.view(B, N, sel, 3)],
                dim=1,
            )
            bind_H_comb = bind_H.view(B, M, sel, -1).sum(dim=2)  # [B, M, D]
            tgt_H_comb = tgt_H.view(B, N, sel, -1).sum(dim=2)  # [B, N, D]
            # bind_H_comb = bind_H.view(B, M * sel, -1)
            # tgt_H_comb = tgt_H.view(B, N * sel, -1)

            all_H_prev = torch.cat([bind_H_comb, tgt_H_comb], dim=1)
            # all_mask = torch.cat([binder_mask, tgt_mask], dim=1)
            all_mask = torch.cat(
                [bind_mask_residue.bool(), tgt_mask_residue.bool()], dim=1
            )
            all_time_steps = time_steps.view(B, 1, -1).repeat_interleave(
                # (M + N) * sel, dim=1
                (M + N),
                dim=1,
            )

            all_H = self.encoder(
                all_X, all_H_prev, all_mask, sel, all_time_steps
            )  # [B, (M+N)*sel, D]
            all_H = self.layernorm(all_H + all_H_prev)

            # bind_H = all_H[:, : M * sel, :] + bind_H
            # tgt_H = all_H[:, M * sel :, :] + tgt_H
            all_H = self.transformer_encoder(
                all_H.transpose(0, 1), src_key_padding_mask=~all_mask.bool()
            ).transpose(0, 1)

            # bind_H = all_H[:, : M * sel, :] + bind_H_dist.view(B, M * sel, -1)
            # tgt_H = all_H[:, M * sel:, :] + tgt_H_dist.view(B, N * sel, -1)
            bind_H = all_H[:, :M, :] + bind_H_dist.view(B, M, sel, -1).mean(dim=2)
            tgt_H = all_H[:, M:, :] + tgt_H_dist.view(B, N, sel, -1).mean(dim=2)

            # bind_H = self.encoder_1(
            #     bind_X.view(B, M, sel, 3),
            #     bind_H,
            #     binder_mask,
            #     sel,
            #     time_steps_bind,
            # ) + bind_H_dist
            # tgt_H = self.encoder_2(
            #     tgt_X.view(B, N, sel, 3),
            #     tgt_H,
            #     tgt_mask,
            #     sel,
            #     time_steps_tgt,
            # ) + tgt_H_dist

        elif self.encoder_type == "mpnn":
            bind_X_centered = bind_X_centered.view(B, M, sel, -1)
            bind_A = bind_A.view(B, M, sel, -1)
            bind_S = bind_S.view(B, M, sel, -1)

            tgt_X_centered = tgt_X_centered.view(B, N, sel, -1)
            tgt_A = tgt_A.view(B, N, sel, -1)
            tgt_S = tgt_S.view(B, N, sel, -1)

            bind_H, tgt_H = self.encoder(
                bind_X_centered,
                bind_H.view(B, M, sel, -1),
                bind_A,
                bind_S,
                # bind_X, bind_H, bind_A, bind_S,
                tgt_X_centered,
                tgt_H.view(B, N, sel, -1),
                tgt_A,
                tgt_S,
                # tgt_X, tgt_H, tgt_A, tgt_S,
                20,
                eps_tr * 3 + 40,
                time_steps_bind[0][0],
                bind_mask_residue.long(),
                tgt_mask_residue.long(),
            )
        elif self.encoder_type == "equiformer":
            all_H = torch.cat([bind_H, tgt_H], dim=1)
            all_X = torch.cat([bind_X_centered, tgt_X_centered], dim=1)
            all_mask = torch.cat([binder_mask, tgt_mask], dim=1)

            H = self.encoder(all_H, all_X, ~all_mask)
        elif self.encoder_type == "gt":
            bind_H = self.encoder_1(
                bind_X.view(B, -1, 3), bind_H, time_steps, binder_mask
            )
            tgt_H = self.encoder_2(tgt_X.view(B, -1, 3), tgt_H, time_steps, tgt_mask)
        else:
            #! separate encoders
            bind_H = self.encoder_1(bind_H, bind_X, ~binder_mask, 30)
            tgt_H = self.encoder_2(tgt_H, tgt_X, ~tgt_mask, 30)

            all_H = torch.cat([bind_H, tgt_H], dim=1)
            all_X = torch.cat([bind_X, tgt_X], dim=1)
            all_mask = torch.cat([binder_mask, tgt_mask], dim=1)
            H = self.encoder_3(all_H, all_X, ~all_mask, 40 * eps_tr + 30)

            bind_H = torch.cat((bind_H, H[:, : M * sel, :]), dim=-1)
            tgt_H = torch.cat((tgt_H, H[:, M * sel :, :]), dim=-1)

            bind_H = self.proj_1(bind_H)
            tgt_H = self.proj_2(tgt_H)

            #! 1 encoder, separate embeddings
            # all_H = torch.cat([bind_H, tgt_H], dim=1)
            # all_X = torch.cat([bind_X_centered, tgt_X_centered], dim=1)
            # all_mask = torch.cat([binder_mask, tgt_mask], dim=1)
            # if self.encoder_type == "egnn":
            #     all_H = all_H.view(B, -1)
            #     H = self.encoder_1(all_H, all_X, 130)
            # else:
            #     H = self.encoder_1(all_H, all_X, ~all_mask, 130)
            # bind_H = H[:, :M * sel, :]
            # tgt_H = H[:, M * sel :, :]

        # if self.non_conservative:
        #     h = self.cross_attention(bind_H, tgt_H, tgt_H)[0]
        #     num_nonzero = binder_mask.sum() // 3
        #     h = self.final_proj(h, num_nonzero)
        #     return h.view(B, -1)

        # h1 = self.final_proj1(h1)
        # h1, _ = self.cross_attention(
        #     bind_H,
        #     tgt_H,
        #     tgt_H,
        #     key_padding_mask=(~tgt_mask).float(),
        #     attn_mask=bind_dist_emb,
        #     need_weights=False,
        # )
        # h1 = self.final_proj1(h1)
        # h2, _ = self.cross_attention(
        #     tgt_H,
        #     bind_H,
        #     bind_H,
        #     key_padding_mask=(~binder_mask).float(),
        #     attn_mask=tgt_dist_emb,
        #     need_weights=False,
        # )
        # h2 = self.final_proj2(h2)
        h1, h2 = bind_H, tgt_H

        return h1.view(B, M, sel, -1), h2.view(B, N, sel, -1)

    def get_internal_dists(self, binder: Features, target: Features):
        B = binder.X.size(0)
        # cplx = torch.cat([binder.X.view(B, -1, 3), target.X.view(B, -1, 3)], dim=1)
        # cplx_internal_dist = torch.cdist(cplx, cplx)  # (B, M+N, M+N)

        # binder_mask = binder.A.view(B, -1).clamp(max=1.0).bool()
        # target_mask = target.A.view(B, -1).clamp(max=1.0).bool()
        # cplx_mask = torch.cat([binder_mask, target_mask], dim=1)  # (B, M+N)
        # pair_mask = cplx_mask.unsqueeze(2) & cplx_mask.unsqueeze(1)  # (B, M+N, M+N)

        # cplx_dist_H = self.edge_embedding(cplx_internal_dist).squeeze(-1)
        # dtype = cplx_dist_H.dtype
        # cplx_dist_H = cplx_dist_H.masked_fill((~pair_mask).unsqueeze(1), torch.finfo(dtype).min)
        dists = torch.cdist(
            binder.X.view(B, -1, 3), target.X.view(B, -1, 3), p=2
        )  # (B, M, N)
        binder_mask = binder.A.view(B, -1).clamp(max=1.0).bool()
        target_mask = target.A.view(B, -1).clamp(max=1.0).bool()
        pair_mask = binder_mask.unsqueeze(2) & target_mask.unsqueeze(1)
        # cplx_attn_bias = self.edge_embedding(dists.unsqueeze(-1)).squeeze(-1)
        cplx_attn_bias = self.edge_embedding(dists.unsqueeze(-1))
        dtype = cplx_attn_bias.dtype
        cplx_attn_bias = cplx_attn_bias.masked_fill(
            ~pair_mask.unsqueeze(-1), torch.finfo(dtype).min
        ).squeeze(-1)

        return cplx_attn_bias

    def get_dists(
        self, bind_X: Tensor, tgt_X: Tensor, binder_mask: Tensor, tgt_mask: Tensor
    ):
        sel = self.residue_size
        B, M, N = bind_X.shape[0], bind_X.shape[1], tgt_X.shape[1]

        bind_tgt_dists = torch.cdist(bind_X.view(B, -1, 3), tgt_X.view(B, -1, 3), p=2)
        bind_dist_emb = self.edge_embedding(bind_tgt_dists.unsqueeze(-1)).view(
            B, M * sel, N * sel
        )

        tgt_dist_emb = bind_dist_emb.transpose(1, 2)

        bind_dist_emb = bind_dist_emb.masked_fill((~tgt_mask).unsqueeze(1), -1e9)
        tgt_dist_emb = tgt_dist_emb.masked_fill((~binder_mask).unsqueeze(1), -1e9)
        # bind_dist_emb = bind_dist_emb.masked_fill((~tgt_mask).unsqueeze(1), 0)
        # tgt_dist_emb = tgt_dist_emb.masked_fill((~binder_mask).unsqueeze(1), 0)

        bind_dist_emb = bind_dist_emb.repeat_interleave(self.heads, 0)
        tgt_dist_emb = tgt_dist_emb.repeat_interleave(self.heads, 0)

        return bind_dist_emb, tgt_dist_emb
