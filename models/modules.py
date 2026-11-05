import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention

from .common import GLU


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

        mod = self.adaln_mod(cond).view(B, N, 6, C)
        scale1, shift1, scale2, shift2, scale3, shift3 = (
            mod[:, :, 0],
            mod[:, :, 1],
            mod[:, :, 2],
            mod[:, :, 3],
            mod[:, :, 4],
            mod[:, :, 5],
        )

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


class ForceHead(nn.Module):
    def __init__(self, args, depth=2):
        super().__init__()
        in_dim, hdim = args.hidden_size, args.hidden_size
        self.proj = nn.Linear(in_dim * 2, hdim)
        self.head = nn.Sequential(
            *[
                GatedResidualModule(GLU(hdim, args, act="silu", bias=True), hdim)
                for _ in range(depth)
            ]
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
            *[
                GatedResidualModule(GLU(hdim, args, act="silu", bias=False), hdim)
                for _ in range(depth)
            ]
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
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")
        assert emb.shape == (x.shape[0], self.embed_dim)
        return emb


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
        )
        sin_component, cos_component = get_sine_cosine_feats(X, gaussian_weights)
        fourier_feats = torch.cat([cos_component, sin_component], dim=-1)

        return fourier_feats
