import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

from .encoders import FlashIPAEncoder
from utils import ALPHABET, Features, randomly_rotate_and_translate


class SinusoidalDistanceEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # Precompute the frequency divisors
        # Log-space distribution ensures we capture multiple scales of distance equally
        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, 2).float() * -(math.log(max_period) / half_dim)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, dist):
        """
        dist: [B, N, M, 1] or [B, N, 1] - Raw distances (in Angstroms)
        """
        # Reshape for broadcasting: [..., 1] * [dim/2]
        num_dims = dist.dim()
        x = dist * self.div_term.view(*([1] * (num_dims - 1)), -1)

        # Concatenate sin and cos
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        return emb  # [B, N, M, dim]


class ResidualEdgeEncoder(nn.Module):
    def __init__(self, edge_dim: int, end: int = 180):
        super(ResidualEdgeEncoder, self).__init__()
        self.end = end
        self.edge_dim = edge_dim

        self.gaussian_embedding = GaussianSmearing(0.0, end, edge_dim)
        self.lin1 = nn.Linear(edge_dim, edge_dim)
        self.lin2 = nn.Linear(edge_dim, 1)

    def forward(self, edge_dist: Tensor) -> Tensor:
        """
        edge_dist: (B, M, N) in [0, end]
        returns: (B, M, N)
        """
        edge_emb_1 = self.gaussian_embedding[0](edge_dist)  # (B, M, N, edge_dim)
        edge_emb = F.gelu(self.lin1(edge_emb_1))  # (B, M, N, edge_dim)
        edge_emb = edge_emb + edge_emb_1  # residual connection
        edge_emb = self.lin2(edge_emb).squeeze(-1)  # (B, M, N)
        return edge_emb


class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super(RelativePositionalEncoding, self).__init__()
        self.max_len = max_len
        self.dim = dim
        self.embedding = nn.Embedding(2 * max_len + 1, dim)
        self.weights = nn.Parameter(torch.randn(dim, 1))

    def forward(self, length: int) -> Tensor:
        """
        Return relative positional encodings for sequences of given length
        Output shape: (length, length, dim)
        """
        range_vec = torch.arange(length, device=self.embedding.weight.device)
        distance_mat = range_vec[None, :] - range_vec[:, None]  # (length, length)
        distance_mat_clipped = (
            torch.clamp(distance_mat, -self.max_len, self.max_len) + self.max_len
        )  # shift to [0, 2*max_len]
        embeddings = self.embedding(distance_mat_clipped)  # (length, length, dim)
        # return self.lin(embeddings).unsqueeze(0).squeeze(-1) # (1, length, length)
        return (
            embeddings.matmul(self.weights).squeeze(-1).unsqueeze(0)
        )  # (1, length, length)

    def pair_relative_embeddings(self, len_q: int, len_k: int) -> Tensor:
        """
        Return relative positional encodings for pairs of sequences of given lengths
        Output shape: (len_q + len_k, len_q + len_k, dim)
        """
        total_len = len_q + len_k
        range_vec = torch.arange(total_len, device=self.embedding.weight.device)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = (
            torch.clamp(distance_mat, -self.max_len, self.max_len) + self.max_len
        )
        embeddings = self.embedding(distance_mat_clipped)
        # return self.lin(embeddings).unsqueeze(0).squeeze(-1)  # (1, total_len, total_len)
        return (
            embeddings.matmul(self.weights).squeeze(-1).unsqueeze(0)
        )  # (1, total_len, total_len)


class ColumnPooling(nn.Module):
    def __init__(self, dim):
        super(ColumnPooling, self).__init__()
        self.dim = dim
        self.attention = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x: Tensor, attn_mask: Tensor, direction="columns"):
        """
        Return weighted sum over the columns/rows
        x: (B, N, M, D)
        attn_mask: (B, N, M) or None
        """
        if direction == "rows":
            x = x.transpose(1, 2)
            if attn_mask is not None:
                attn_mask = attn_mask.transpose(1, 2)

        weights = self.attention(x)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(-1)
            weights = weights.masked_fill(~attn_mask, -1e9)

        weights = F.softmax(weights, dim=2)  # (B, N, M, 1)
        out = (x * weights).sum(dim=2)  # (B, N, D)

        return out  # (B, N, D) if columns else (B, M, D)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=2,
        bias=True,
        normalizer=None,
        activation=nn.SiLU(),
    ):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            output_dim = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
            if i < num_layers - 1:
                layers.append(activation)
                if normalizer is not None:
                    layers.append(normalizer(hidden_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ResidualBlock(nn.Module):
    def __init__(self, module):
        super(ResidualBlock, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class GaussianSmearing(nn.Module):
    # used to embed the edge dists
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


class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm (timestep-conditioned) with zero init
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor, t_embed: Tensor):
        # t_embed: (B, dim)
        g = self.gamma * t_embed
        b = self.beta * t_embed
        return self.norm(x) * (1 + g) + b


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


class DiTMLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.block_adaln = BlockAdaLN(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # Self-attn part
        self.sa = SelfAttention(dim, heads, dropout)
        # self.sa = TransformerBlock(dim, heads, mlp_ratio, dropout)

        # Cross-attn part
        self.cross = CrossAttention(dim, heads, dropout)
        # self.cross = CrossAttentionBlock(dim, heads, mlp_ratio, dropout)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond, t_embed, mask=None):
        """
        x:      (B, N, D)
        cond:   (B, N, D)
        t_emb:  (B, D)
        mask:   (B, N) or (B, N, 1), True = valid, False = pad, or None
        """
        x = x + cond  # token-level bias

        # Self-attention
        s1, sc1, g1, s2, sc2, g2 = self.block_adaln(t_embed)

        h = self.norm1(x)
        h = h * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1)
        h = self.sa(h, mask=mask)
        # h = self.sa(h, attn_mask=mask)
        x = x + g1.unsqueeze(1) * h

        h = self.norm2(x)
        h = h * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1)
        h = self.cross(h, cond, mask=mask)
        # h = self.cross(h, cond, attn_mask=mask)
        x = x + g2.unsqueeze(1) * h

        h = self.norm3(x)
        h = self.mlp(h)
        x = x + h

        return x


class DiTBackbone(nn.Module):
    def __init__(self, depth, dim, heads):
        super().__init__()
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])

    def forward(self, x, cond, t_emb, mask=None):
        for block in self.blocks:
            x = block(x, cond, t_emb, mask=mask)
        return x


class DiTOutput(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaln = BlockAdaLN(dim)
        self.proj = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        s, sc, _, *_ = self.adaln(cond)
        x = self.norm(x)
        x = x * (1 + sc) + s
        return self.proj(x)


class DiTConditioning(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t_emb):
        return self.net(t_emb)  # (B, D)


class DiT(nn.Module):
    def __init__(self, dim, depth, heads, out_dim):
        super().__init__()
        self.cond1 = DiTConditioning(dim)
        self.cond2 = DiTConditioning(dim)
        self.backbone = DiTBackbone(depth, dim, heads)
        self.out = DiTOutput(dim, out_dim)

    def forward(self, x, context, t, mask=None):
        t = self.cond1(t)
        cond = self.cond2(context)
        x = self.backbone(x, cond, t, mask=mask)
        return self.out(x, cond)


class DiTBlock_(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, cross=True, dropout=0.0):
        super().__init__()
        self.use_cross = cross

        # Self-attn part
        self.adaln1 = AdaLNZero(dim)
        self.sa = SelfAttention(dim, heads, dropout)

        # Cross-attn part (optional)
        if cross:
            self.adaln2 = AdaLNZero(dim)
            self.ca = CrossAttention(dim, heads, dropout)

        # MLP
        self.adaln3 = AdaLNZero(dim)
        self.ff = DiTMLP(dim, mlp_ratio)

    def forward(self, x, t_embed, context=None, mask=None):
        # Self-attention
        x = x + self.sa(self.adaln1(x, t_embed), mask)

        # Cross-attention
        if self.use_cross and context is not None:
            x = x + self.ca(self.adaln2(x, t_embed), context, mask)

        # MLP
        x = x + self.ff(self.adaln3(x, t_embed))
        return x


class DiTTransformer(nn.Module):
    """
    Diffusion model operating over latent tokens (B, L, D)

    - z_t: noisy latents
    - context: binder & target embeddings, shape (B, L_ctx, D)
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        predict: str = "noise",  # {noise | x0}
    ):
        super().__init__()
        self.predict = predict

        # stack of blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim,
                    heads,
                    mlp_ratio,
                    cross=True,  # want cross-attn in every block
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # final LN & linear
        self.final_ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)  # predict ε or x0

    def forward(self, zt, context, t_embed, mask=None):
        """
        zt:      (B, L, D)
        t_embed: (B, D)
        context: (B, L_ctx, D)
        """
        x = zt
        for block in self.blocks:
            x = block(x, t_embed, context, mask)

        x = self.final_ln(x)
        out = self.head(x)

        return out


class FusionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = TransformerBlock(dim, heads, mlp_ratio=4, dropout=dropout)
        self.cross = CrossAttentionBlock(dim, heads, mlp_ratio=4, dropout=dropout)
        # self.film = FiLM(dim, dim)
        self.fuser = nn.Sequential(
            nn.Linear(dim * 2, dim),
            # nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.adaln1 = AdaLNZero(dim)
        self.adaln2 = AdaLNZero(dim)

    def forward(self, x, context, noise_emb, res=None, attn_mask=None):
        x = self.adaln1(x, noise_emb)
        x1 = self.attn(x, attn_mask=attn_mask)
        x2 = self.cross(x, context, attn_mask=attn_mask)

        x = self.fuser(torch.cat([x1, x2], dim=-1))
        # x = self.film(x, noise_emb)
        x = self.adaln2(x, noise_emb)

        if res is not None:
            x = x + res

        return x


class FusionAttentionModule(nn.Module):
    def __init__(self, latent_dim, heads, depth, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [FusionBlock(latent_dim, heads, dropout) for _ in range(depth)]
        )

    def forward(self, x, context, t, attn_mask=None):
        res = x

        for layer in self.layers:
            x = layer(x, context, t, res=res, attn_mask=attn_mask)
            res = x

        return x


class CustomCrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        mlp_ratio,
        depth,
        dropout=0.0,
        normalizer=nn.RMSNorm,
    ):
        super(CustomCrossAttentionTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim,
                    heads,
                    mlp_ratio,
                    dropout,
                    normalizer,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, context, t=None, attn_mask=None):
        for layer in self.layers:
            x = layer(x, context, t=t, attn_mask=attn_mask)

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        mlp_ratio,
        dropout=0.0,
        normalizer=nn.RMSNorm,
    ):
        super(CrossAttentionBlock, self).__init__()
        self.heads = heads
        self.dim = dim
        self.dim_head = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)

        self.norm1 = normalizer(dim)
        self.norm2 = normalizer(dim)

        self.proj_out = nn.Linear(dim, dim, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=False),
            nn.SiLU(),
            nn.Linear(dim * mlp_ratio, dim, bias=False),
            nn.Dropout(dropout),
        )

    def reshape_heads(self, x: Tensor) -> Tensor:
        # (B, T, C) -> (B, heads, T, dim_head)
        B = x.shape[0]
        return x.view(B, -1, self.heads, self.dim_head).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        t: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        x:       (B, N, C)
        context: (B, M, C)
        t:       (B, N, C) or None, time conditioning if diffusion
        attn_mask: (B, 1, N, M) or (B, N, M) or None
        """
        B, N, C = x.shape
        _, M, _ = context.shape

        if t is not None:
            x = x + t

        x_norm = self.norm1(x)

        q = self.to_q(x_norm)  # (B, N, C)
        kv = self.to_kv(context)  # (B, M, 2C)
        k, v = kv.chunk(2, dim=-1)

        q = self.reshape_heads(q)  # (B, H, N, Dh)
        k = self.reshape_heads(k)  # (B, H, M, Dh)
        v = self.reshape_heads(v)  # (B, H, M, Dh)

        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, H, N, M)
        attn = attn / (self.dim_head**0.5)

        # ---- Apply mask ----
        if attn_mask is not None:
            # ensure broadcastable: (B, 1, N, M)
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, N, Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # merge heads
        out = self.proj_out(out)

        x = x + out

        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


class CustomTransformer(nn.Module):
    """
    A stack of Transformer Blocks
    """

    def __init__(
        self,
        dim,
        heads,
        mlp_ratio,
        depth,
        dropout=0.0,
        normalizer=nn.RMSNorm,
    ):
        super(CustomTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    mlp_ratio,
                    dropout,
                    normalizer,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: Tensor, t: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None
    ):
        for layer in self.layers:
            x = layer(x, t=t, attn_mask=attn_mask)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer Block
    """

    def __init__(
        self,
        dim,
        heads,
        mlp_ratio,
        dropout=0.0,
        normalizer=nn.RMSNorm,
    ):
        super(TransformerBlock, self).__init__()
        self.heads = heads
        self.dim_head = dim // heads

        self.qkv = nn.Linear(dim, heads * self.dim_head * 3, bias=False)

        self.norm1 = normalizer(dim)
        self.norm2 = normalizer(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio, bias=False),
            nn.SiLU(),
            nn.Linear(dim * mlp_ratio, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self, x, t: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None
    ):
        """
        x:    (B, N, dim)
        t:    (B, N, dim) or None, time conditioning if diffusion
        mask: (B, N) or (B, N, 1), True = valid, False = pad, or None
        """
        B, N, _ = x.size()

        if t is not None:
            x = x + t

        x_norm = self.norm1(x)

        # qkv: (B, N, 3 * heads * dim_head)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, N, self.heads, 3 * self.dim_head)
        qkv = qkv.permute(0, 2, 1, 3)  # (B, heads, N, 3 * dim_head)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, heads, N, dim_head)

        qk_prod = torch.matmul(q, k.transpose(-2, -1))  # (B, heads, N, N)
        qk_prod = qk_prod / (self.dim_head**0.5)

        if attn_mask is not None:
            qk_prod = qk_prod + attn_mask

        attn_weights = F.softmax(qk_prod, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (B, heads, N, dim_head)

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(
            B, N, self.heads * self.dim_head
        )  # (B, N, dim)
        x = x + attn_out

        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


class GLU(nn.Module):
    def __init__(self, dim, act="relu", bias=False):
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
            nn.Linear(dim * 2, dim * 2, bias=bias),
        )

    def forward(self, x):
        gate = self.gate(x)
        x, gate = gate.chunk(2, dim=-1)

        return x * self.act(gate)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: h -> gamma(z) * h + beta(z)
    """

    def __init__(self, latent_dim: int, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, feature_dim * 2)

        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: Tensor, t: Tensor) -> Tensor:
        """
        h: (B, N, feature_dim)
        z: (B, 1, latent_dim)
        """
        gamma, beta = self.fc(t).chunk(
            2, dim=-1
        )  # (B,1,feature_dim), (B,1,feature_dim)
        return h * gamma + beta


class ConvFiLM(nn.Module):
    """
    Feature-wise Linear Modulation: h -> gamma(z) * h + beta(z)
    """

    def __init__(self, latent_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, channels * 2)

        # nn.init.zeros_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """
        h: (B, C, H, W)
        z: (B, latent_dim)
        """
        gamma, beta = self.fc(z).chunk(2, dim=-1)  # (B,C), (B,C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        return h * (1 + gamma) + beta


# For diffusion conditioning
class ProteinEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        emb_size: int,
        heads: int,
        depth: int,
        dropout: float = 0.0,
        resolution: str = "backbone",
        vocab_size: int = len(ALPHABET) * 3,
    ):
        super().__init__()
        self.hdim = hidden_size
        self.heads = heads
        self.encoder = FlashIPAEncoder(
            hidden_size=self.hdim, heads=heads, depth=depth, dropout=dropout
        )
        self.W_bind = nn.Linear(emb_size, hidden_size)
        self.W_tgt = nn.Linear(emb_size, hidden_size)
        self.A_bind = nn.Embedding(vocab_size, hidden_size)
        self.A_tgt = nn.Embedding(vocab_size, hidden_size)

        self.down_project_1 = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size, bias=False)
        )
        self.down_project_2 = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size, bias=False)
        )
        self.resolution = resolution

        if self.resolution == "all_atom":
            self.sel = 14
        elif self.resolution == "backbone":
            self.sel = 4
        elif self.resolution == "CA":
            self.sel = 1
        else:
            raise ValueError(f"Unsupported resolution: {self.resolution}")
        self.bb = 0 if self.resolution == "CA" else 1

    def get_sequence_feats(
        self, binder: Features, target: Features
    ) -> Tuple[Tensor, Tensor]:
        bind_X, bind_S, bind_A = binder.X, binder.S, binder.A
        tgt_X, tgt_S, tgt_A = target.X, target.S, target.A

        B, M, N = bind_X.size(0), bind_X.size(1), tgt_X.size(1)

        bind_S = (
            bind_S[:, :, None, :]
            .expand(-1, -1, self.sel, -1)
            .reshape(B, M * self.sel, -1)
        )
        tgt_S = (
            tgt_S[:, :, None, :]
            .expand(-1, -1, self.sel, -1)
            .reshape(B, N * self.sel, -1)
        )

        bind_A = bind_A.view(B, M * self.sel)
        tgt_A = tgt_A.view(B, N * self.sel)

        bind_A_H = self.A_bind(bind_A)
        bind_S_H = self.W_bind(bind_S)
        tgt_A_H = self.A_tgt(tgt_A)
        tgt_S_H = self.W_tgt(tgt_S)

        bind_H = torch.cat([bind_S_H, bind_A_H], dim=-1)
        tgt_H = torch.cat([tgt_S_H, tgt_A_H], dim=-1)

        return bind_H, tgt_H

    def get_geom_feats(
        self, binder: Features, target: Features
    ) -> Tuple[Tensor, Tensor]:
        bind_torsion, tgt_torsion = binder.T, target.T
        bind_curvature, tgt_curvature = binder.C, target.C

        B, M, N = bind_torsion.size(0), bind_torsion.size(1), tgt_torsion.size(1)

        bind_torsion = binder.T.view(B, M * self.sel, 1) / 100
        tgt_torsion = target.T.view(B, N * self.sel, 1) / 100

        return bind_torsion, tgt_torsion

    def embed(self, binder: Features, target: Features) -> Tuple[Tensor, Tensor]:
        seq_bind_H, seq_tgt_H = self.get_sequence_feats(binder, target)
        geom_bind_H, geom_tgt_H = self.get_geom_feats(binder, target)

        bind_H = torch.cat([seq_bind_H, geom_bind_H], dim=-1)
        tgt_H = torch.cat([seq_tgt_H, geom_tgt_H], dim=-1)

        bind_H = self.down_project_1(bind_H)
        tgt_H = self.down_project_2(tgt_H)

        return bind_H, tgt_H

    def pad_to_max_length(
        self,
        bind_H: Tensor,
        tgt_H: Tensor,
        bind_X: Tensor,
        tgt_X: Tensor,
        bind_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        max_length = max(bind_H.size(1), tgt_H.size(1))
        # pad to max length for efficient batching
        pad_bind_H = F.pad(bind_H, (0, 0, 0, max_length - bind_H.size(1)), value=0.0)
        pad_tgt_H = F.pad(tgt_H, (0, 0, 0, max_length - tgt_H.size(1)), value=0.0)
        pad_bind_X = F.pad(
            bind_X, (0, 0, 0, 0, 0, max_length - bind_X.size(1)), value=0.0
        )
        pad_tgt_X = F.pad(tgt_X, (0, 0, 0, 0, 0, max_length - tgt_X.size(1)), value=0.0)

        pad_bind_mask = F.pad(
            bind_mask,
            (0, max_length - bind_mask.size(1)),
            value=False,
        )
        pad_tgt_mask = F.pad(
            tgt_mask,
            (0, max_length - tgt_mask.size(1)),
            value=False,
        )

        return pad_bind_H, pad_tgt_H, pad_bind_X, pad_tgt_X, pad_bind_mask, pad_tgt_mask

    def forward(
        self,
        binder: Features,
        target: Features,
        bind_H: Optional[Tensor] = None,
        tgt_H: Optional[Tensor] = None,
        return_coords=False,
    ):
        bind_X, bind_S, bind_A, *_ = binder
        tgt_X, tgt_S, tgt_A, *_ = target

        bind_mask_ca = bind_A[:, :, self.bb].clamp(max=1).bool()  # [B, M]
        tgt_mask_ca = tgt_A[:, :, self.bb].clamp(max=1).bool()  # [B, N]

        B, M, N = bind_S.size(0), bind_S.size(1), tgt_S.size(1)

        bind_A = bind_A.view(B, M * self.sel) > 0
        tgt_A = tgt_A.view(B, N * self.sel) > 0

        bind_X = bind_X.view(B, M * self.sel, 3)
        tgt_X = tgt_X.view(B, N * self.sel, 3)

        bind_X = randomly_rotate_and_translate(bind_X, tr_max=20, mask=bind_A)
        tgt_X = randomly_rotate_and_translate(tgt_X, tr_max=20, mask=tgt_A)

        bind_X = bind_X.view(B, M, self.sel, 3)
        tgt_X = tgt_X.view(B, N, self.sel, 3)

        if bind_H is None and tgt_H is None:
            bind_H, tgt_H = self.embed(binder, target)

            bind_H = bind_H.view(B, M, self.sel, -1).sum(dim=2)  # pool to [B, M, D]
            tgt_H = tgt_H.view(B, N, self.sel, -1).sum(dim=2)  # pool to [B, N, D]

        bind_H, tgt_H, bind_X, tgt_X, bind_mask_ca, tgt_mask_ca = (
            self.pad_to_max_length(
                bind_H,
                tgt_H,
                bind_X,
                tgt_X,
                bind_mask_ca,
                tgt_mask_ca,
            )
        )

        combined_H = torch.cat([bind_H, tgt_H], dim=0)  # (2B, L, D)
        combined_X = torch.cat([bind_X, tgt_X], dim=0)
        combined_mask = torch.cat([bind_mask_ca, tgt_mask_ca], dim=0)

        output = self.encoder(
            x=combined_X,
            h=combined_H,
            mask=combined_mask,
            sel=self.sel,
            time_steps=None,
            return_coords=return_coords,
        )

        encoded_H = output
        bind_out, tgt_out = torch.chunk(encoded_H, 2, dim=0)

        # undo the padding
        bind_out = bind_out[:, :M, :].contiguous()
        tgt_out = tgt_out[:, :N, :].contiguous()

        return bind_out, tgt_out
