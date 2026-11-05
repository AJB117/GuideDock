from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import ALPHABET, Features, randomly_rotate_and_translate

from .common import BlockAdaLN, CrossAttention, SelfAttention
from .encoders import FlashIPAEncoder


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.block_adaln = BlockAdaLN(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.sa = SelfAttention(dim, heads, dropout)

        self.cross = CrossAttention(dim, heads, dropout)

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
        x = x + cond

        s1, sc1, g1, s2, sc2, g2 = self.block_adaln(t_embed)

        h = self.norm1(x)
        h = h * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1)
        h = self.sa(h, mask=mask)

        x = x + g1.unsqueeze(1) * h

        h = self.norm2(x)
        h = h * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1)
        h = self.cross(h, cond, mask=mask)

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
        return self.net(t_emb)


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

        q = self.to_q(x_norm)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)

        q = self.reshape_heads(q)
        k = self.reshape_heads(k)
        v = self.reshape_heads(v)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (self.dim_head**0.5)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
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

        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, N, self.heads, 3 * self.dim_head)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        qk_prod = torch.matmul(q, k.transpose(-2, -1))
        qk_prod = qk_prod / (self.dim_head**0.5)

        if attn_mask is not None:
            qk_prod = qk_prod + attn_mask

        attn_weights = F.softmax(qk_prod, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(
            B, N, self.heads * self.dim_head
        )
        x = x + attn_out

        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


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
        _bind_curvature, _tgt_curvature = binder.C, target.C

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

        bind_mask_ca = bind_A[:, :, self.bb].clamp(max=1).bool()
        tgt_mask_ca = tgt_A[:, :, self.bb].clamp(max=1).bool()

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

            bind_H = bind_H.view(B, M, self.sel, -1).sum(dim=2)
            tgt_H = tgt_H.view(B, N, self.sel, -1).sum(dim=2)

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

        combined_H = torch.cat([bind_H, tgt_H], dim=0)
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

        bind_out = bind_out[:, :M, :].contiguous()
        tgt_out = tgt_out[:, :N, :].contiguous()

        return bind_out, tgt_out
