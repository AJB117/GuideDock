import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cuequivariance_torch import (
    triangle_attention,
    triangle_multiplicative_update,
)
from triangle_multiplicative_module import TriangleMultiplicativeModule
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Triangle Multiplication
# -------------------------


class TriangleMultiplicationOutgoing(nn.Module):
    """
    pair[i, j] += sum_k W_left(pair[i, k]) * W_right(pair[k, j])
    """

    def __init__(self, c_pair, c_hidden):
        super().__init__()
        self.ln = nn.LayerNorm(c_pair)

        self.left = nn.Linear(c_pair, c_hidden, bias=False)
        self.right = nn.Linear(c_pair, c_hidden, bias=False)

        self.gate = nn.Linear(c_pair, c_hidden)
        self.out = nn.Linear(c_hidden, c_pair)

    def forward(self, pair):
        # pair: [B, L, L, C]
        x = self.ln(pair)

        a = self.left(x)  # [B, L, L, H]
        b = self.right(x)  # [B, L, L, H]
        g = torch.sigmoid(self.gate(x))  # [B, L, L, H]

        # sum over k: (i,k) * (k,j)
        out = torch.einsum("bikh,bkjh->bijh", a, b)
        out = out * g

        return self.out(out)


class TriangleMultiplicationIncoming(nn.Module):
    """
    pair[i, j] += sum_k W_left(pair[k, i]) * W_right(pair[j, k])
    """

    def __init__(self, c_pair, c_hidden):
        super().__init__()
        self.ln = nn.LayerNorm(c_pair)

        self.left = nn.Linear(c_pair, c_hidden, bias=False)
        self.right = nn.Linear(c_pair, c_hidden, bias=False)

        self.gate = nn.Linear(c_pair, c_hidden)
        self.out = nn.Linear(c_hidden, c_pair)

    def forward(self, pair):
        # pair: [B, L, L, C]
        x = self.ln(pair)

        a = self.left(x)
        b = self.right(x)
        g = torch.sigmoid(self.gate(x))

        # swap indices for incoming triangles
        out = torch.einsum("bkih,bjkh->bijh", a, b)
        out = out * g

        return self.out(out)


# -------------------------
# Pair Transition (FFN)
# -------------------------


class PairTransition(nn.Module):
    def __init__(self, c_pair, expansion=4):
        super().__init__()
        self.ln = nn.LayerNorm(c_pair)
        self.fc1 = nn.Linear(c_pair, c_pair * expansion)
        self.fc2 = nn.Linear(c_pair * expansion, c_pair)

    def forward(self, pair):
        # pair: [B, L, L, C]
        x = self.ln(pair)
        x = F.silu(self.fc1(x))
        return self.fc2(x)


# -------------------------
# Pair Embedding Initializer
# -------------------------


class PairEmbeddingInit(nn.Module):
    def __init__(
        self,
        c_pair,
        num_aa=21,
        max_relpos=32,
        symmetric=True,
    ):
        super().__init__()
        self.c_pair = c_pair
        self.num_aa = num_aa
        self.max_relpos = max_relpos
        self.symmetric = symmetric

        self.relpos = nn.Embedding(2 * max_relpos + 1, c_pair)
        self.aa_pair = nn.Embedding(num_aa * num_aa, c_pair)
        self.ln = nn.LayerNorm(c_pair)

        nn.init.normal_(self.relpos.weight, std=0.02)
        nn.init.normal_(self.aa_pair.weight, std=0.02)

    def forward(self, aatype):
        # aatype: [B, L]
        B, L = aatype.shape
        device = aatype.device

        # relative positions
        idx = torch.arange(L, device=device)
        rel = idx[:, None] - idx[None, :]
        rel = rel.clamp(-self.max_relpos, self.max_relpos) + self.max_relpos
        rel = self.relpos(rel)  # [L, L, C]
        rel = rel.unsqueeze(0).expand(B, -1, -1, -1)

        # amino-acid pairs
        aa_i = aatype[:, :, None]
        aa_j = aatype[:, None, :]
        aa_idx = aa_i * self.num_aa + aa_j
        aa = self.aa_pair(aa_idx)

        pair = rel + aa

        if self.symmetric:
            pair = 0.5 * (pair + pair.transpose(1, 2))

        return self.ln(pair)


# -------------------------
# Full Pair Stack
# -------------------------


class PairStack(nn.Module):
    """
    AF2-style pair stack using ONLY triangle multiplicative updates.
    """

    def __init__(
        self,
        c_pair,
        c_hidden,
        n_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "tri_out": TriangleMultiplicationOutgoing(c_pair, c_hidden),
                        "tri_in": TriangleMultiplicationIncoming(c_pair, c_hidden),
                        "transition": PairTransition(c_pair),
                    }
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, pair):
        # pair: [B, L, L, C]
        for blk in self.blocks:
            pair = pair + blk["tri_out"](pair)
            pair = pair + blk["tri_in"](pair)
            pair = pair + blk["transition"](pair)
        return pair


# -------------------------
# End-to-end module
# -------------------------


class TriangleModule(nn.Module):
    def __init__(self, c_pair=128, c_hidden=64, n_blocks=4):
        super().__init__()
        self.stack = PairStack(
            c_pair=c_pair,
            c_hidden=c_hidden,
            n_blocks=n_blocks,
        )

    def forward(self, pair):
        # aatype: [B, L]
        pair = self.stack(pair)
        return pair


# -------------------------
# Smoke test
# -------------------------

if __name__ == "__main__":
    B, L = 2, 128
    aatype = torch.randint(0, 21, (B, L))
    model = TriangleModule(c_pair=128, c_hidden=64, n_blocks=3)
    pair = model(aatype)
    print(pair.shape)  # [B, L, L, 128]


class TriangleBlock(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.in_triangle_updates = nn.ModuleList(
            [
                TriangleMultiplicativeModule(dim=dim, hidden_dim=dim, mix="ingoing")
                for _ in range(depth)
            ]
        )
        self.out_triangle_updates = nn.ModuleList(
            [
                TriangleMultiplicativeModule(dim=dim, hidden_dim=dim, mix="outgoing")
                for _ in range(depth)
            ]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.SiLU(),
                    nn.Linear(dim, dim),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, mask=None):
        for in_tri, out_tri, ffn in zip(
            self.in_triangle_updates, self.out_triangle_updates, self.ffns
        ):
            x = out_tri(x, mask=mask) + x
            x = in_tri(x, mask=mask) + x
            x = x + ffn(x)

        return x


class TriangleMultiplicativeUpdate(nn.Module):
    """
    Wrapper around cuequivariance_torch.triangle_multiplicative_update.

    x: (B, N, N, D)
    mask: (B, N, N) or None
    direction: "outgoing" or "incoming"
    """

    def __init__(self, d_hidden: int, direction: str = "outgoing", eps: float = 1e-5):
        super().__init__()
        assert direction in ("outgoing", "incoming")
        self.direction = direction
        self.d_hidden = d_hidden
        self.eps = eps

        D = d_hidden

        # Input norm
        self.norm_in_weight = nn.Parameter(torch.ones(D))
        self.norm_in_bias = nn.Parameter(torch.zeros(D))

        # Input projections (2D -> D) and gating
        self.p_in_weight = nn.Parameter(torch.empty(2 * D, D))
        self.p_in_bias = nn.Parameter(torch.zeros(2 * D))
        self.g_in_weight = nn.Parameter(torch.empty(2 * D, D))
        self.g_in_bias = nn.Parameter(torch.zeros(2 * D))

        # Output norm
        self.norm_out_weight = nn.Parameter(torch.ones(D))
        self.norm_out_bias = nn.Parameter(torch.zeros(D))

        # Output projections and gating (D -> D)
        self.p_out_weight = nn.Parameter(torch.empty(D, D))
        self.p_out_bias = nn.Parameter(torch.zeros(D))
        self.g_out_weight = nn.Parameter(torch.empty(D, D))
        self.g_out_bias = nn.Parameter(torch.zeros(D))

        self.reset_parameters()

    def reset_parameters(self):
        D = self.d_hidden
        # Roughly AF2-style initializations
        nn.init.ones_(self.norm_in_weight)
        nn.init.zeros_(self.norm_in_bias)
        nn.init.ones_(self.norm_out_weight)
        nn.init.zeros_(self.norm_out_bias)

        nn.init.xavier_uniform_(self.p_in_weight)
        nn.init.zeros_(self.p_in_bias)
        nn.init.xavier_uniform_(self.g_in_weight)
        nn.init.zeros_(self.g_in_bias)

        nn.init.xavier_uniform_(self.p_out_weight)
        nn.init.zeros_(self.p_out_bias)
        nn.init.xavier_uniform_(self.g_out_weight)
        nn.init.zeros_(self.g_out_bias)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (B, N, N, D)
        mask: (B, N, N) with 1/True for valid, or None
        returns: (B, N, N, D)
        """
        out = triangle_multiplicative_update(
            x=x,
            direction=self.direction,
            mask=mask,
            norm_in_weight=self.norm_in_weight,
            norm_in_bias=self.norm_in_bias,
            p_in_weight=self.p_in_weight,
            p_in_bias=self.p_in_bias,
            g_in_weight=self.g_in_weight,
            g_in_bias=self.g_in_bias,
            norm_out_weight=self.norm_out_weight,
            norm_out_bias=self.norm_out_bias,
            p_out_weight=self.p_out_weight,
            p_out_bias=self.p_out_bias,
            g_out_weight=self.g_out_weight,
            g_out_bias=self.g_out_bias,
            eps=self.eps,
            precision=None,  # let cuEquivariance decide
        )
        return out


class _TriangleAttentionBase(nn.Module):
    """
    Base class for triangle attention.

    Operates on pair representation z: (B, N, N, C)

    Child classes define how to permute axes for
    'starting node' vs 'ending node' semantics.
    """

    def __init__(
        self, c_z: int, n_heads: int, head_dim: int | None = None, bias: bool = False
    ):
        super().__init__()
        assert c_z % n_heads == 0 or head_dim is not None, (
            "Either c_z divisible by n_heads or provide head_dim explicitly."
        )

        self.c_z = c_z
        self.n_heads = n_heads
        self.head_dim = head_dim or (c_z // n_heads)
        assert self.n_heads * self.head_dim == c_z, "c_z must match n_heads * head_dim."

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(c_z, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(c_z, n_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(c_z, n_heads * self.head_dim, bias=bias)

    def _shape_qkv(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Default: treat z as (B, N, N, C), with
        - triangle dimension index = dim 1 (e.g. i)
        - query/key dimension index = dim 2 (e.g. j/k)

        Produces q,k,v for triangle_attention:

            q, k, v: (B, N, H, Q/K, D)
        """
        B, N, M, C = z.shape
        assert N == M, "Triangle attention assumes square pair matrix (N x N)."

        # Project
        q = self.q_proj(z)  # (B, N, N, H*D)
        k = self.k_proj(z)
        v = self.v_proj(z)

        # Reshape to (B, N, N, H, D) then permute -> (B, N, H, N, D)
        def reshape_projection(x):
            x = x.view(B, N, N, self.n_heads, self.head_dim)
            x = x.permute(0, 1, 3, 2, 4).contiguous()  # (B, N_tri, H, N_q_or_k, D)
            return x

        q = reshape_projection(q)
        k = reshape_projection(k)
        v = reshape_projection(v)
        return q, k, v

    def _apply_triangle_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q,k,v: (B, N, H, Q/K, D)
        pair_bias: (B, 1, H, Q, K) or None
        mask: (B, N, 1, 1, K) or None

        Returns: (B, N, H, Q, D)
        """
        B, N, H, Q, D = q.shape
        _, _, _, K, _ = k.shape
        assert Q == K, "This wrapper assumes Q == K == N."

        if pair_bias is None:
            # Bias must be float32 according to docs
            pair_bias = q.new_zeros(B, 1, H, Q, K, dtype=torch.float32)
        else:
            # Ensure correct shape & dtype
            if pair_bias.dim() == 4:  # (B, H, Q, K)
                pair_bias = pair_bias.unsqueeze(1)  # (B, 1, H, Q, K)
            assert pair_bias.shape == (B, 1, H, Q, K), (
                f"Expected pair_bias of shape (B, 1, H, Q, K), got {pair_bias.shape}"
            )
            pair_bias = pair_bias.to(dtype=torch.float32)

        scale = 1.0 / math.sqrt(D)

        out = triangle_attention(
            q=q,
            k=k,
            v=v,
            bias=pair_bias,
            mask=mask,
            scale=scale,
            return_aux=False,
        )  # (B, N, H, Q, D)
        return out

    def forward(
        self,
        z: torch.Tensor,
        pair_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Use TriangleAttentionStartingNode or TriangleAttentionEndingNode."
        )


class TriangleAttentionStartingNode(_TriangleAttentionBase):
    """
    Triangle attention "starting at node":

    For each pair (i, j), attends over all (i, k).

    Input:  z (B, N, N, C)
    Output: same shape (B, N, N, C)
    """

    def forward(
        self,
        z: torch.Tensor,
        pair_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z: (B, N, N, C)
        pair_bias: (B, 1, H, N, N) or (B, H, N, N) or None
        mask: (B, N, 1, 1, N) or None
            - mask[b, i, 0, 0, k] corresponds to M_k for a given i.
        """
        B, N, _, C = z.shape
        # Q/K/V from z (B, N, N, C)
        q, k, v = self._shape_qkv(z)  # (B, N, H, N, D)

        out = self._apply_triangle_attention(
            q=q,
            k=k,
            v=v,
            pair_bias=pair_bias,
            mask=mask,
        )  # (B, N, H, N, D)

        # Bring back to (B, N, N, C)
        out = out.permute(0, 1, 3, 2, 4).contiguous()  # (B, N, N, H, D)
        out = out.view(B, N, N, self.n_heads * self.head_dim)
        return out


class TriangleAttentionEndingNode(_TriangleAttentionBase):
    """
    Triangle attention "ending at node":

    For each pair (i, j), attends over all (k, j).

    Implemented by transposing i/j axes before and after
    using the same internal q/k/v shaping as StartingNode.

    Input:  z (B, N, N, C)
    Output: (B, N, N, C)
    """

    def forward(
        self,
        z: torch.Tensor,
        pair_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z: (B, N, N, C)
        pair_bias: (B, 1, H, N, N) or (B, H, N, N) or None
            - Interpret pair_bias[b, :, :, i, j] as bias for edge (i, j)
              regardless of direction; we'll transpose consistently.
        mask: (B, N, 1, 1, N) or None
            - mask[b, j, 0, 0, k] corresponds to M_k for fixed j.
        """
        B, N, _, C = z.shape

        # Swap i and j axes: (B, j, i, C)
        z_t = z.transpose(1, 2).contiguous()

        # Adjust bias & mask if provided:
        if pair_bias is not None:
            # pair_bias: (B, 1, H, N, N) or (B, H, N, N)
            if pair_bias.dim() == 4:
                pair_bias_t = pair_bias.unsqueeze(1)  # (B, 1, H, N, N)
            else:
                pair_bias_t = pair_bias
            # transpose along the last two dims (i/j)
            pair_bias_t = pair_bias_t.transpose(-1, -2).contiguous()
        else:
            pair_bias_t = None

        if mask is not None:
            # mask: (B, N, 1, 1, N); swap N dims
            mask_t = mask.transpose(1, 4).contiguous()  # (B, N, 1, 1, N)
        else:
            mask_t = None

        # q,k,v from transposed z_t
        q, k, v = self._shape_qkv(z_t)  # (B, N(j), H, N(i), D)

        out_t = self._apply_triangle_attention(
            q=q,
            k=k,
            v=v,
            pair_bias=pair_bias_t,
            mask=mask_t,
        )  # (B, N(j), H, N(i), D)

        # back to (B, j, i, C)
        out_t = out_t.permute(0, 1, 3, 2, 4).contiguous()  # (B, N(j), N(i), H, D)
        out_t = out_t.view(B, N, N, self.n_heads * self.head_dim)

        # transpose back to (B, i, j, C)
        out = out_t.transpose(1, 2).contiguous()
        return out


class TriangleBlock_(nn.Module):
    """
    A cohesive triangle block for pair representations, roughly AF2-style:

        z <- z + TriMul(outgoing)
        z <- z + TriMul(incoming)
        z <- z + TriAttn(starting node)
        z <- z + TriAttn(ending node)

    You can of course re-order or drop pieces as desired.

    Args:
        c_z:      pair channel dimension
        n_heads:  number of attention heads
        dropout:  dropout on each residual update
    """

    def __init__(self, c_z: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.c_z = c_z
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Triangle multiplicative updates
        self.tri_mul_out = TriangleMultiplicativeUpdate(c_z, direction="outgoing")
        self.tri_mul_in = TriangleMultiplicativeUpdate(c_z, direction="incoming")

        # Triangle attentions
        self.tri_attn_start = TriangleAttentionStartingNode(c_z, n_heads)
        self.tri_attn_end = TriangleAttentionEndingNode(c_z, n_heads)

        # Optional layer norms or gates around residuals (simple version)
        self.norm_z1 = nn.LayerNorm(c_z)
        self.norm_z2 = nn.LayerNorm(c_z)
        self.norm_z3 = nn.LayerNorm(c_z)
        self.norm_z4 = nn.LayerNorm(c_z)

    def forward(
        self,
        z: torch.Tensor,
        pair_bias: torch.Tensor | None = None,
        tri_mul_mask: torch.Tensor | None = None,
        tri_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z:            (B, N, N, C)
        pair_bias:    (B, 1, H, N, N) or (B, H, N, N) or None
        tri_mul_mask: (B, N, N) or None
        tri_attn_mask:(B, N, 1, 1, N) or None

        returns: updated z (B, N, N, C)
        """

        # 1) Triangle multiplicative update (outgoing)
        z_res = z
        z = self.norm_z1(z)
        z = self.tri_mul_out(z, mask=tri_mul_mask)
        z = z_res + self.dropout(z)

        # 2) Triangle multiplicative update (incoming)
        z_res = z
        z = self.norm_z2(z)
        z = self.tri_mul_in(z, mask=tri_mul_mask)
        z = z_res + self.dropout(z)

        # 3) Triangle attention starting node
        z_res = z
        z = self.norm_z3(z)
        z = self.tri_attn_start(z, pair_bias=pair_bias, mask=tri_attn_mask)
        z = z_res + self.dropout(z)

        # 4) Triangle attention ending node
        z_res = z
        z = self.norm_z4(z)
        z = self.tri_attn_end(z, pair_bias=pair_bias, mask=tri_attn_mask)
        z = z_res + self.dropout(z)

        return z  # (B, N, N, C)


# if __name__ == "__main__":

#     def run_test(device="cuda" if torch.cuda.is_available() else "cpu"):
#         print(f"Running on: {device}")

#         # Very small sizes for speed
#         B = 1
#         N = 800
#         C = 32
#         H = 4

#         # Create tiny input pair representation
#         z = torch.randn(B, N, N, C, device=device, requires_grad=True)

#         # Masks
#         res_mask = torch.ones(B, N, device=device, dtype=torch.bool)
#         tri_mul_mask = (
#             res_mask[:, :, None] & res_mask[:, None, :]
#         ).float()  # (B, N, N)
#         tri_attn_mask = res_mask[:, :, None, None, None].expand(B, N, 1, 1, N)

#         # Create block
#         block = TriangleBlock(c_z=C, n_heads=H, dropout=0.0).to(device)

#         # Run forward
#         print("Forward pass...")
#         context = torch.cuda.amp.autocast(dtype=torch.bfloat16)
#         with context:
#             out = block(
#                 z,
#                 pair_bias=None,
#                 tri_mul_mask=tri_mul_mask,
#                 tri_attn_mask=tri_attn_mask,
#             )
#         # Shape check
#         assert out.shape == (B, N, N, C), f"Output shape mismatch: {out.shape}"

#         # Make sure values changed
#         assert not torch.allclose(out, z), (
#             "Output identical to input—block may be inactive."
#         )

#         # Backward pass
#         print("Backward pass...")
#         loss = out.sum()
#         loss.backward()

#         # Gradient check
#         assert z.grad is not None, "No gradient propagated to input."
#         assert torch.isfinite(z.grad).all(), "Non-finite gradients detected."

#         print(
#             "✓ TriangleBlock test passed.\n"
#             "  Forward OK, Backward OK, Shapes OK, Gradients OK."
#         )

#     if __name__ == "__main__":
#         run_test()
