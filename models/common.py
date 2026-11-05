import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalDistanceEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, 2).float() * -(math.log(max_period) / half_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, dist):
        """
        dist: [B, N, M, 1] or [B, N, 1] - Raw distances (in Angstroms)
        """

        num_dims = dist.dim()
        x = dist * self.div_term.view(*([1] * (num_dims - 1)), -1)

        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        return emb


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.num_gaussians = num_gaussians
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        """
        dist: (B, M, N)
        returns: (B, M, N, num_gaussians)
        """
        dist = dist.unsqueeze(-1) - self.offset.view(1, 1, 1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class BlockAdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_params = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.to_params.weight)
        nn.init.zeros_(self.to_params.bias)

    def forward(self, cond):
        return self.to_params(cond).chunk(6, dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: Tensor, mask=None):
        out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: Tensor, context: Tensor, mask=None):
        out, _ = self.attn(x, context, context, attn_mask=mask, need_weights=False)
        return out


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


class GLU(nn.Module):
    def __init__(self, dim, args=None, act="relu", bias=False):
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

        layers = [nn.Linear(dim, dim * 2, bias=bias), self.act]
        if args is not None:
            layers.append(nn.Dropout(args.dropout))
        layers.append(nn.Linear(dim * 2, dim * 2, bias=bias))
        self.gate = nn.Sequential(*layers)

    def forward(self, x):
        gate = self.gate(x)
        x, gate = gate.chunk(2, dim=-1)

        return x * self.act(gate)


class SinusoidalNoiseEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        t = t * 1000.0
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device))
            * torch.arange(half, device=t.device)
            / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return F.pad(emb, (0, self.dim - emb.shape[-1]))
