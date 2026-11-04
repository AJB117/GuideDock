import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from torch.nn import MultiheadAttention


class AttentionPooling(nn.Module):
    """
    Attention Pooling
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn = MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x:    (B, N, C)
        mask: (B, N) with 1 for valid tokens and 0 for padding
        """
        B, N, C = x.shape
        x = self.ln(x)
        query = self.query.repeat(B, 1, 1)  # (B, 1, C)
        attn_out, _ = self.attn(
            query,
            x,
            x,
            need_weights=False,
        )  # (B, 1, C)
        attn_out = attn_out.squeeze(1)
        return attn_out  # (B, C)


class StatPooling(nn.Module):
    """
    Statistics Pooling: mean, std, max, min
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Sequential(
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, mask):
        """
        x:    (B, N, C)
        mask: (B, N) with 1 for valid tokens and 0 for padding
        """
        B, N, C = x.shape
        w = mask.float()  # (B, N)
        w3 = w.unsqueeze(-1)  # (B, N, 1)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)

        # Masked mean over N
        mean = (x * w3).sum(dim=1) / denom  # (B, C)

        # Masked std over N
        mean2 = (x.pow(2) * w3).sum(dim=1) / denom  # (B, C)
        std = torch.sqrt((mean2 - mean.pow(2)).clamp(min=1e-8))

        # Masked max over N
        x_masked = x * w3 + (1 - w3) * (-1e9)  # (B, N, C)
        max_pool, _ = x_masked.max(dim=1)  # (B, C)
        min_pool, _ = x_masked.min(dim=1)  # (B, C)
        pooled = torch.cat([mean, std, max_pool, min_pool], dim=-1)  # (B, 4*C)

        return self.linear(pooled)  # (B, C)


class MomentPooling(nn.Module):
    """
    Moment Pooling: mean, std, skewness, kurtosis, ...
    """

    def __init__(self, dim, num_moments=3, eps=1e-8):
        super().__init__()
        self.num_moments = num_moments
        self.dim = dim
        self.eps = eps
        self.linear = nn.Sequential(
            nn.Linear(num_moments * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, mask):
        """
        x:    (B, N, C)
        mask: (B, N) with 1 for valid tokens and 0 for padding
        """
        B, N, C = x.shape
        w = mask.float()  # (B, N)
        w3 = w.unsqueeze(-1)  # (B, N, 1)
        denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)

        # Masked mean over N
        mean = (x * w3).sum(dim=1) / denom  # (B, C)
        moments = [mean]

        if self.num_moments > 1:
            centered = x - mean.unsqueeze(1)  # (B, N, C)
            for i in range(2, self.num_moments + 1):
                # Masked i-th central moment (raw), then take sign-preserving root
                raw_moment = (centered.pow(i) * w3).sum(dim=1) / denom  # (B, C)
                rooted = torch.sign(raw_moment) * torch.pow(
                    raw_moment.abs() + self.eps, 1.0 / i
                )
                moments.append(rooted)

        pooled = torch.cat(moments, dim=-1)  # (B, num_moments*C)
        return self.linear(pooled)  # (B, C)


class DiTBlock(nn.Module):
    def __init__(self, args, dim, heads, mlp_ratio=4, dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            GLU(mlp_ratio * dim, args, act="silu", bias=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
        )

        self.adaln_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.adaln_mod[1].weight)
        nn.init.zeros_(self.adaln_mod[1].bias)

    def forward(self, x, cond, dist_emb, mask):
        B, N, C = x.shape

        # AdaLN modulation
        mod = self.adaln_mod(cond).view(B, N, 6, C)
        scale1, shift1, scale2, shift2, scale3, shift3 = (
            mod[:, :, 0],
            mod[:, :, 1],
            mod[:, :, 2],
            mod[:, :, 3],
            mod[:, :, 4],
            mod[:, :, 5],
        )  # each is (B, N, C)

        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h, _ = self.attn(
            h,
            h,
            h,
            need_weights=False,
            key_padding_mask=mask.float(),
            attn_mask=dist_emb,
        )
        x = x + h

        h = self.norm2(x)
        h = h * (1 + scale2) + shift2
        h = self.ffn(h)
        h = h * (1 + scale3) + shift3
        x = x + h

        return x


class AttentionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.normalizer_layers = nn.ModuleList()
        self.depth = 1

        hidden_size = args.hidden_size
        attn_embed_dim = (
            hidden_size // 2
            if args.encoder not in ("gt", "mpnn", "equiformer", "flash_ipa", "sa")
            else hidden_size
        )

        for _ in range(self.depth):
            self.attn_layers.append(
                MultiheadAttention(
                    embed_dim=attn_embed_dim,
                    num_heads=8,
                    dropout=args.dropout,
                    batch_first=True,
                )
            )
            self.mlp_layers.append(
                nn.Sequential(
                    # nn.Linear(hidden_size, hidden_size),
                    # nn.ReLU(),
                    # nn.Dropout(args.dropout),
                    # nn.Linear(hidden_size, hidden_size),
                    GLU(
                        hidden_size,
                        args,
                        act="silu",
                    ),
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

        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None
    ) -> Tensor:
        x = query
        for idx in range(self.depth):
            norm_attn, norm_mlp = self.normalizer_layers[idx]

            # qn = norm_attn(x)
            # kn = norm_attn(key)
            # vn = norm_attn(value)

            x, _ = self.attn_layers[idx](
                x, key, value, attn_mask=attn_mask, need_weights=False
            )
            # x = x + self.dropout(attn_out)

            # mlp_in = norm_mlp(x)
            # mlp_out = self.mlp_layers[idx](x)
            # x = x + self.dropout(mlp_out)

        return x


class GatedResidualModule(nn.Module):
    def __init__(self, module, dim):
        super().__init__()
        self.module = module
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, 1, bias=False),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.gate[0].weight)

    def forward(self, x, *args, **kwargs):
        out = self.module(x, *args, **kwargs)
        gate = self.gate(torch.cat((x, out, out - x), dim=-1))
        return (1 - gate) * x + gate * out


class ResidualModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return x + self.module(x, *args, **kwargs)


class ForceMOE(nn.Module):
    def __init__(self, args, depth=2):
        super().__init__()
        in_dim, hdim = args.hidden_size, args.hidden_size
        self.proj = nn.Linear(in_dim * 2, hdim)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        GatedResidualModule(
                            GLU(hdim, args, act="silu", bias=True), hdim
                        )
                        for _ in range(depth)
                    ]
                )
                for _ in range(3)
            ]
        )
        self.reducer = nn.Linear(hdim, 3)

        self.expert_intervals = [
            (0, 0.1),
            (0.1, 0.6),
            (0.6, 1.0),
        ]

    def forward(self, force: Tensor, noise_emb: Tensor, time_step: Tensor):
        x = torch.cat((force, noise_emb), dim=-1)
        x = self.proj(x)

        # assign a separate expert based on time_step
        expert_outputs = []
        for idx, (start, end) in enumerate(self.expert_intervals):
            # mask = ((time_step >= start) & (time_step < end)).squeeze(-1)
            mask = ((start <= time_step) & (time_step < end)).squeeze(-1)
            if mask.sum() > 0:
                expert_input = x[mask]
                expert_output = self.experts[idx](expert_input) + expert_input
                expert_outputs.append((mask, expert_output))

        # combine expert outputs
        out = torch.zeros_like(x)
        for mask, expert_output in expert_outputs:
            out[mask] = expert_output

        x = self.reducer(out)
        return x


class ScalarMOE(nn.Module):
    def __init__(self, args, depth=2):
        super().__init__()
        in_dim, hdim = args.hidden_size, args.hidden_size
        self.proj = nn.Linear(1 + in_dim, hdim)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        GatedResidualModule(
                            GLU(hdim, args, act="silu", bias=False), hdim
                        )
                        for _ in range(depth)
                    ]
                )
                for _ in range(3)
            ]
        )
        self.reducer = nn.Linear(hdim, 3)

        self.expert_intervals = [
            (0, 0.1),
            (0.1, 0.6),
            (0.6, 1.0),
        ]

    def forward(self, force: Tensor, noise_emb: Tensor, time_step: Tensor):
        x = torch.cat((force, noise_emb), dim=-1)
        x = self.proj(x)

        # assign a separate expert based on time_step
        expert_outputs = []
        for idx, (start, end) in enumerate(self.expert_intervals):
            mask = ((start <= time_step) & (time_step < end)).squeeze(-1)
            if mask.sum() > 0:
                expert_input = x[mask]
                expert_output = self.experts[idx](expert_input) + expert_input
                expert_outputs.append((mask, expert_output))

        # combine expert outputs
        out = torch.zeros_like(x)
        for mask, expert_output in expert_outputs:
            out[mask] = expert_output

        x = self.reducer(out)
        return x


class ForceHead(nn.Module):
    # Force / Score, tomato / tomawto
    def __init__(self, args, depth=2):
        super().__init__()
        in_dim, hdim = args.hidden_size, args.hidden_size
        self.proj = nn.Linear(in_dim * 2, hdim)
        self.head = nn.Sequential(
            # nn.ReLU(),
            # nn.Dropout(args.dropout),
            # *[GLU(hdim, args, act="relu") for _ in range(depth)]
            *[
                GatedResidualModule(GLU(hdim, args, act="silu", bias=True), hdim)
                for _ in range(depth)
            ]
            # nn.Linear(hdim, hdim),
            # nn.ReLU(),
            # nn.Dropout(args.dropout),
            # nn.Dropout(args.dropout),
            # GLU(args.hidden_size, args, act="relu"),
            # nn.Dropout(args.dropout),
        )
        self.reducer = nn.Linear(hdim, 3)

    def forward(self, force: Tensor, noise_emb: Tensor):
        x = torch.cat((force, noise_emb), dim=-1)
        x = self.proj(x)
        x = self.head(x) + x
        x = self.reducer(x)

        return x


class ScalarHead(nn.Module):
    def __init__(self, args, depth=2):
        super().__init__()
        in_dim, hdim = args.hidden_size, args.hidden_size
        self.proj = nn.Linear(1 + in_dim, hdim)
        self.head = nn.Sequential(
            # nn.ReLU(),
            # nn.Dropout(args.dropout),
            # nn.Linear(hdim, hdim),
            # nn.SiLU(),
            # nn.Dropout(args.dropout),
            # *[GLU(hdim, args, act="relu") for _ in range(depth)]
            *[
                GatedResidualModule(GLU(hdim, args, act="silu", bias=False), hdim)
                for _ in range(depth)
            ]
            # GLU(args.hidden_size // 2, args, act="relu"),
        )
        self.reducer = nn.Linear(hdim, 1)

    def forward(self, norm: Tensor, time_steps: Tensor):
        x = torch.cat((norm, time_steps), dim=-1)
        x = self.proj(x)
        x = self.head(x) + x
        x = self.reducer(x)

        return x


class AdaLN(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.a_norm = nn.LayerNorm(dim_a, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(dim_b, bias=False)
        self.s_scale = nn.Linear(dim_b, dim_a)
        self.s_bias = nn.Linear(dim_b, dim_a, bias=False)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = F.sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, sigma_embed_dim=32, embedding_scale=10000):
        super().__init__()
        self.embed_dim = sigma_embed_dim
        self.scale = embedding_scale
        self.W = nn.Parameter(
            torch.randn(self.embed_dim // 2) * self.scale, requires_grad=False
        )

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


@torch.compile
def get_sinusoidal_feats(x, emb):
    emb = (x.float()[:, :, None] * emb[:, None]).squeeze(-1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class SinusoidalEmbedding(nn.Module):
    def __init__(self, sigma_embed_dim=32, embedding_scale=10000):
        super().__init__()
        self.embed_dim = sigma_embed_dim
        self.scale = embedding_scale
        self.max_positions = 1e4

    def forward(self, x):
        """
        From https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
        """
        if self.embed_dim == 1:
            return x

        x = self.scale * x

        half_dim = self.embed_dim // 2
        emb = math.log(self.max_positions) / (half_dim)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=x.device) * -emb
        )
        emb = get_sinusoidal_feats(x, emb)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode="constant")
        assert emb.shape == (x.shape[0], self.embed_dim)
        return emb


class GLU(nn.Module):
    def __init__(self, dim, args, act="relu", bias=False):
        super(GLU, self).__init__()

        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()
        elif act == "gelu":
            self.act = nn.GELU()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            raise NotImplementedError

        self.gate = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=bias),
            self.act,
            nn.Dropout(args.dropout),
            nn.Linear(dim * 2, dim * 2, bias=bias),
        )

    def forward(self, x):
        gate = self.gate(x)
        x, gate = gate.chunk(2, dim=-1)

        return x * self.act(gate)


class AdditiveGate(nn.Module):
    def __init__(self, dim, dropout):
        super(AdditiveGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim, bias=False), nn.Sigmoid(), nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        hidden = torch.cat((x1, x2, x1 - x2), dim=-1)
        gate = self.gate(hidden)

        return (x1 * gate) + (1 - gate) * x2


class FiLM(nn.Module):
    def __init__(self, dim, args):
        super(FiLM, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=True),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(dim * 2, dim * 2, bias=True),
        )

    def forward(self, x1, x2):
        gate = self.gate(x2)
        mult_gate, add_gate = gate.chunk(2, dim=-1)

        return (x1 * mult_gate) + add_gate


@torch.compile
def get_sine_cosine_feats(X: Tensor, weights: Tensor):
    inner = 2 * torch.pi * torch.bmm(X, weights.transpose(1, 2))
    sin_emb = torch.sin(inner)
    cos_emb = torch.cos(inner)
    return sin_emb, cos_emb


class FourierFeats(nn.Module):
    def __init__(self, args, data_dim=3):
        super(FourierFeats, self).__init__()
        self.dim = args.fourier_dim
        self.B = torch.normal(mean=0, std=1, size=(self.dim, data_dim))

    def forward(self, X):
        """
        X: [B, N, data_dim] tensor of coordinates, angles, whatever
        """
        gaussian_weights = (
            self.B.unsqueeze(0).repeat_interleave(X.size(0), dim=0).to(X.device)
        )  # [B, dim, data_dim]
        sin_component, cos_component = get_sine_cosine_feats(X, gaussian_weights)
        fourier_feats = torch.cat(
            [cos_component, sin_component], dim=-1
        )  # [B, N, 2*dim]

        return fourier_feats  # [B, N, 2*dim]


@torch.compile
def get_dist_feats(dist, offset, coeff):
    dist = dist.unsqueeze(-1) - offset.view(1, 1, 1, -1)
    return torch.exp(coeff * torch.pow(dist, 2))


class GaussianSmearing(nn.Module):
    # used to embed the edge dists
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        """
        dist: (B, N, N, 1)
        """
        return get_dist_feats(dist, self.offset, self.coeff)


class GraphGaussianSmearing(nn.Module):
    # used to embed the edge dists
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))  # (E, num_gaussians)
