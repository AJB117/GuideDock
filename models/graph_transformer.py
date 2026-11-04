"""
This code has been imported from:
(lucidrains - graph-transformer-pytorch)
https://github.com/lucidrains/graph-transformer-pytorch
"""
import pdb
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Optional
import os, sys
# from torch.nn.attention 

from contextlib import nullcontext
from torch import nn, einsum
from einops import rearrange, repeat
# from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention import SDPBackend, sdpa_kernel


class DistanceEmbedding(nn.Module):
    def __init__(self, dim):
        super(DistanceEmbedding, self).__init__()

        self.dim = dim    
        self.mus = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.sigmas = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )
        self.scale = torch.tensor(2 * torch.pi)

    def forward(self, dists):
        """
        Args:
            dists: [bs, n_nodes, n_nodes, 1]
        """
        dist_param = (dists - self.mus)**2 # [bs, n_nodes, n_nodes, dim]
        dist_param = dist_param / (2 * self.sigmas**2)

        rbf = 1/(self.sigmas * torch.sqrt(self.scale)) * torch.exp(-dist_param)
        rbf = self.mlp(rbf)

        return rbf



class GraphTransformer(nn.Module):
    """
    Graph neural network module.
    """

    def __init__(
        self,
        hidden_nf,
        n_layers=8,
        use_intrinsic_coords: bool = True,
        use_abs_coords: bool = True,
        use_distances: bool = False,
        conservative: bool = False,
        use_flash_attention: bool = True,
        use_swiglu: bool = False,
        use_positional_embedding: bool = False,
        pos_emb_dim: int = 32,
        ff_mult: int = 3,
    ):
        """_summary_

        Args:
            num_beads (_type_): _description_
            hidden_nf (_type_): _description_
            device (str, optional): _description_. Defaults to "cpu".
            n_layers (int, optional): _description_. Defaults to 4.
            use_intrinsic_coords (bool, optional): _description_. Defaults to False.
            use_abs_coords (bool, optional): _description_. Defaults to True.
            use_distances (bool, optional): _description_. Defaults to True.
            conservative (bool, optional): _description_. Defaults to True.
        """
        super(GraphTransformer, self).__init__()
        self.use_intrinsic_coords = use_intrinsic_coords
        self.use_distances = use_distances
        self.use_abs_coords = use_abs_coords
        self.conservative = conservative

        in_node_nf = hidden_nf + use_abs_coords * 3 + 1
        in_edge_nf = (
            3 * use_intrinsic_coords
            + use_distances
            + 1 * (not use_intrinsic_coords) * (not use_distances)
        )
        self.node_embedding = nn.Sequential(nn.Linear(in_node_nf, hidden_nf))
        self.edge_embedding = nn.Linear(in_edge_nf, hidden_nf)

        if self.conservative:
            self.node_decoder = nn.Linear(hidden_nf, 1)
        else:
            self.node_decoder = nn.Linear(hidden_nf, 3)

        self.graphtransformer = GraphTransformerLucid(
            dim=hidden_nf,
            depth=n_layers,
            edge_dim=hidden_nf,  # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
            with_feedforwards=True,  # whether to add a feedforward after each attention layer, suggested by literature to be needed
            gated_residual=True,  # to use the gated residual to prevent over-smoothing
            use_flash_attention=use_flash_attention,  # whether to use the flash attention mechanism
            use_swiglu=use_swiglu,
            ff_mult=ff_mult,  # multiplier for the feedforward layer
        )
        self.distance_embedding = DistanceEmbedding(64)

        if use_positional_embedding:
            self.pos_emb = SinusoidalPosEmb(dim=pos_emb_dim)

        self.use_positional_embedding = use_positional_embedding

        # self.to(self.device)
        # self.reset_parameters()

    def reset_parameters(self):
        # recursively iterate over all modules and initialize any linear layers 
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)


    def forward(
        self,
        x,
        h,
        t,
        mask,
        return_energy=False,
        alphas: Optional[torch.Tensor] = None,
    ):
        # alphas: Optional tensor of shape [Batch_size,]
        # Center at 0 to be translation invariant
        x = x.requires_grad_(requires_grad=True)
        with torch.enable_grad() if self.conservative else nullcontext():
            bs, n_nodes, _ = x.shape
            t = t.reshape(-1, 1, 1).repeat(1, x.shape[1], 1)
            # h = h.unsqueeze(0).repeat(bs, 1, 1).to(x.device)

            # Compute edge attributes if necessary
            edge_attr = self.get_edge_attr(x)
            edge_attr = self.edge_embedding(edge_attr)

            # Concatenate node inputs
            if self.use_abs_coords:
                if self.use_positional_embedding:
                    x_pos_emb = self.pos_emb(x)
                    x = x+x_pos_emb
                nodes = torch.cat((h, x, t), dim=2)
            else:
                nodes = torch.cat((h, t), dim=2)

            nodes = self.node_embedding(nodes)

            nodes, _ = self.graphtransformer(nodes, edge_attr, mask=mask)

            return nodes

            output = self.node_decoder(nodes)
            if self.conservative:
                energy = output
                if return_energy:
                    return energy
                forces = compute_forces(energy, x, self.training)
            else:
                forces = output
            return forces


    def get_edge_attr(self, x):
        if self.use_distances and not self.use_intrinsic_coords:
            xa = x.unsqueeze(1)
            xb = x.unsqueeze(2)
            diff = xa - xb
            dist = torch.sum(
                diff**2, dim=3, keepdim=True
            )  # torch.norm(diff, dim=3).unsqueeze(3)
            dist = self.distance_embedding(dist)
            return dist
        elif self.use_intrinsic_coords and not self.use_distances:
            xa = x.unsqueeze(1)
            xb = x.unsqueeze(2)
            diff = xa - xb
            return diff
        elif self.use_intrinsic_coords and self.use_distances:
            xa = x.unsqueeze(1)
            xb = x.unsqueeze(2)
            diff = xa - xb
            dist = torch.sum(
                diff**2, dim=3, keepdim=True
            )  # torch.norm(diff, dim=3).unsqueeze(3)
            return torch.cat([diff, dist], dim=3)
        else:
            bs, n_nodes, _ = x.size()
            return torch.zeros(bs, n_nodes, n_nodes, 1).to(x.device).detach()


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training=True
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        outputs=energy,  # [n_graphs, ]
        inputs=positions,  # [n_nodes, 3]
        grad_outputs=torch.ones_like(energy),
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        only_inputs=True,  # Diff only w.r.t. inputs
        allow_unused=True,
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        raise Exception("Gradient after computing forces is None.")
    return -1 * gradient


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


List = nn.ModuleList


# sinusoidal encoding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.dim = dim
        j = torch.arange(1, dim+1, dtype=torch.float32)
        freq_init = (2.0 ** j) * torch.pi             # [2^1 π, 2^2 π, ..., 2^d π]
        w_init = freq_init.unsqueeze(1).repeat(1, 3)  # shape (d,3)
        self.emb_sine = nn.Parameter(w_init)  # (d, 3)
        self.emb_cosine = nn.Parameter(w_init)  # (d, 3)
        self.linear = nn.Linear(dim * 3 * 2, 3, bias=False)

    def forward(self, x):
        # X: (B, N, 3)
        x = x.unsqueeze(2)  # (B, N, 1, 3)
        x_sine = x * self.emb_sine  # (B, N, d, 3)
        x_cos = x * self.emb_cosine  # (B, N, d, 3)

        all_feats = torch.cat(
            [
                torch.sin(x_sine).reshape(x.shape[0], x.shape[1], -1),
                torch.cos(x_cos).reshape(x.shape[0], x.shape[1], -1),
            ],
            dim=-1,
        ) # (B, N, d*3)

        all_feats = self.linear(all_feats)  # (B, N, 3)

        return all_feats

# swiglu
# class GLU(nn.Module):
class GLU(torch.jit.ScriptModule):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=True)
        self.V = nn.Linear(dim, dim, bias=True)

    @torch.jit.script_method
    def forward(self, x):
        lhs = self.W(x)
        rhs = self.V(x)

        return F.relu(rhs) * lhs

# normalizations


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        # self.norm = nn.RMSNorm(dim)
        # self.norm = nn.BatchNorm1d(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


# gated residual


class Residual(nn.Module):
    def forward(self, x, res):
        return x + res


class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim * 3, 1, bias=False), nn.Sigmoid())

    def forward(self, x, res):
        gate_input = torch.cat((x, res, x - res), dim=-1)
        gate = self.proj(gate_input)
        return x * gate + res * (1 - gate)


# attention


class Attention(nn.Module):
    # NOTE: we don't use positional embeddings
    def __init__(self, dim, pos_emb=None, dim_head=64, heads=8, edge_dim=None):
        super().__init__()
        edge_dim = default(edge_dim, dim)

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, inner_dim * 2)
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges, mask=None):
        h = self.heads

        q = self.to_q(nodes)
        k, v = self.to_kv(nodes).chunk(2, dim=-1)

        e_kv = self.edges_to_kv(edges)

        q, k, v, e_kv = map(
            lambda t: rearrange(t, "b ... (h d) -> (b h) ... d", h=h), (q, k, v, e_kv)
        )

        ek, ev = e_kv, e_kv

        k, v = map(lambda t: rearrange(t, "b j d -> b () j d "), (k, v))
        k = k + ek
        v = v + ev

        sim = einsum("b i d, b i j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b i -> b i ()") & rearrange(mask, "b j -> b () j")
            mask = repeat(mask, "b i j -> (b h) i j", h=h)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b i j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


def create_score_mod(edge_attr: torch.Tensor):
    def score_mod(
            score: torch.Tensor, # all scalars
            batch: torch.Tensor,
            head: torch.Tensor,
            q_idx: torch.Tensor,
            k_idx: torch.Tensor,
    ) -> torch.Tensor:
        edge_feat = edge_attr[batch, head, q_idx, k_idx]
        return score * edge_feat

    return score_mod

class EdgeNodeFlashAttention(nn.Module):
    def __init__(self, dim, pos_emb=None, dim_head=64, heads=8, edge_dim=None):
        super().__init__()
        edge_dim = edge_dim or dim
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.edge_to_bias = nn.Linear(edge_dim, heads)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, nodes, edges, mask=None):
        B, N, _ = nodes.shape
        h, d = self.heads, self.dim_head
        
        # Update nodes with aggregated edge information
        # Sum edge features along one dimension (e.g., incoming edges)
        in_edge_updates = edges.mean(dim=2)  # (B, N, dim)
        out_edge_updates = edges.mean(dim=1)

        edge_gate = self.edge_gate(in_edge_updates+out_edge_updates)  # (B, N, dim)
        updated_nodes = nodes + edge_gate * in_edge_updates + (1-edge_gate) * out_edge_updates  # (B, N, dim)
        # updated_nodes = nodes + in_edge_updates + out_edge_updates
        
        edge_bias = self.edge_to_bias(edges)
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # (B, heads, N, N)

        # Standard attention with updated nodes
        q = self.to_q(updated_nodes)
        kv = self.to_kv(updated_nodes)
        k, v = kv.chunk(2, dim=-1)
        
        q = q.view(B, N, h, d).transpose(1, 2)
        k = k.view(B, N, h, d).transpose(1, 2)
        v = v.view(B, N, h, d).transpose(1, 2)
        
        # Optional mask handling
        # attn_mask = None
        # if mask is not None:
        #     attn_mask = ~mask.unsqueeze(1).unsqueeze(2)
        attn_mask = edge_bias
        # attn_mask = torch.zeros((B, h, N, N)).float().to(nodes.device)
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]): 
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                 scale=self.scale, dropout_p=0.0)
        
        attn_out = attn_out.transpose(1, 2).reshape(B, N, h * d)
        return self.to_out(attn_out)

class FlashAttention(nn.Module):
    # NOTE: we don't use positional embeddings
    def __init__(self, dim, pos_emb=None, dim_head=64, heads=8, edge_dim=None):
        super().__init__()
        edge_dim = default(edge_dim, dim)
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.pos_emb = pos_emb

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        # This layer is still used if you wish to add edge features into keys/values,
        # but here we use a separate projection for attention bias.
        self.edges_to_kv = nn.Linear(edge_dim, inner_dim)
        # New: project edge features to a scalar bias per head.
        self.edges_to_bias = nn.Linear(edge_dim, heads)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, nodes, edges, mask=None):
        """
        nodes: (B, N, dim)
        edges: (B, N, N, edge_dim)  -- one edge feature for each query-key pair
        mask: Optional[Tensor] of shape (B, N), where True indicates a valid token.
        """
        B, N, _ = nodes.shape
        h, d = self.heads, self.dim_head

        # Compute queries, keys, and values from nodes
        q = self.to_q(nodes)          # (B, N, inner_dim)
        kv = self.to_kv(nodes)         # (B, N, inner_dim * 2)
        k, v = kv.chunk(2, dim=-1)     # each: (B, N, inner_dim)

        # Reshape and transpose to shape: (B, heads, N, d)
        q = q.view(B, N, h, d).transpose(1, 2)
        k = k.view(B, N, h, d).transpose(1, 2)
        v = v.view(B, N, h, d).transpose(1, 2)

        # Compute an edge bias for each query-key pair and head.
        # edges: (B, N, N, edge_dim) --> bias: (B, N, N, heads)
        edge_bias = self.edges_to_bias(edges)
        # Rearrange to shape: (B, heads, N, N)
        edge_bias = edge_bias.permute(0, 3, 1, 2)

        # Optionally incorporate an additional mask for invalid tokens.
        # if mask is not None:
        #     # Assume mask is of shape (B, N) with True for valid tokens.
        #     # We create a bias for keys: invalid keys get a huge negative value.
        #     mask_bias = (1.0 - mask.float()).unsqueeze(1).unsqueeze(2) * -torch.finfo(q.dtype).max
        #     # Broadcast and add to the edge bias.
        #     attn_mask = edge_bias + mask_bias  # (B, heads, N, N)
        # else:
        attn_mask = edge_bias
        
        # Use PyTorch's flash attention function.
        # Note: attn_mask is a floating-point tensor added to the raw scores.
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]):
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale,
                                                            dropout_p=0.0, is_causal=False)
        # attn_out: (B, heads, N, d)

        # Merge heads back into the feature dimension: (B, N, inner_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, h * d)
        return self.to_out(attn_out)

# optional feedforward


def FeedForward(dim, ff_mult=4, use_swiglu=False):
    if use_swiglu:
        return nn.Sequential(
            nn.Linear(dim, dim * ff_mult), GLU(dim * ff_mult), nn.Linear(dim * ff_mult, dim)
        )
    else:
        return nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.SiLU(), nn.Linear(dim * ff_mult, dim)
        )


# classes


class GraphTransformerLucid(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head=64,
        edge_dim=None,
        heads=8,
        gated_residual=True,
        with_feedforwards=False,
        norm_edges=False,
        use_flash_attention=False, # WARNING: flash attention uses only edge features as bias, not augmenting the node kvs
        use_swiglu=False,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = List([])
        edge_dim = default(edge_dim, dim)
        self.norm_edges = nn.LayerNorm(edge_dim) if norm_edges else nn.Identity()

        pos_emb = None

        for _ in range(depth):
            self.layers.append(
                List(
                    [
                        List(
                            [
                                PreNorm(
                                    dim,
                                    Attention(
                                        dim,
                                        pos_emb=pos_emb,
                                        edge_dim=edge_dim,
                                        dim_head=dim_head,
                                        heads=heads,
                                    ) if not use_flash_attention else FlashAttention(
                                        dim,
                                        pos_emb=pos_emb,
                                        dim_head=dim_head,
                                        heads=heads,
                                        edge_dim=edge_dim,
                                    ),
                                ),
                                GatedResidual(dim),
                            ]
                        ),
                        List([PreNorm(dim, FeedForward(dim, ff_mult=ff_mult, use_swiglu=use_swiglu)), 
                              GatedResidual(dim),
                              ])
                        if with_feedforwards
                        else None,
                    ]
                )
            )

    def forward(self, nodes, edges, mask=None):
        edges = self.norm_edges(edges)

        for attn_block, ff_block in self.layers:
            attn, attn_residual = attn_block
            nodes = attn_residual(attn(nodes, edges, mask=mask), nodes)

            if exists(ff_block):
                ff, ff_residual = ff_block
                nodes = ff_residual(ff(nodes), nodes)

        return nodes, edges


if __name__ == "__main__":
    import time

    # Init variables
    n_nodes = 10
    hidden_nf = 256
    bs = 128

    # Init model
    model = GraphTransformer(
        n_nodes, hidden_nf=hidden_nf, n_layers=5, conservative=False
    ).cuda()

    # Init parameters
    x = torch.ones(bs, n_nodes, 3).cuda()
    h = torch.ones(n_nodes, n_nodes).cuda()
    t = torch.ones(bs).cuda()

    # Run model
    t1 = time.time()
    n_iterations = 10
    for i in range(n_iterations):
        forces = model(x, h, t)
    t2 = time.time()

    # Print output shape
    print(forces.shape)
    print(f"Elapsed time {(t2 - t1)/n_iterations}")
