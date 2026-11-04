import math
import pdb
import torch
import torch.nn as nn
from torch.nn import HuberLoss
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union
from models.latent_modules import (
    ProteinEncoder,
    GaussianSmearing,
    CustomTransformer,
    ColumnPooling,
    RelativePositionalEncoding,
)
from models.triangle import TriangleBlock, TriangleModule, PairEmbeddingInit
from utils import Features, ALPHABET
from latent_utils import (
    get_pairwise_cplx_attn_mask,
    get_single_attn_bias,
    get_cplx_attn_bias,
)


def kl_sigmoid_anneal(step, k=0.0008, x0=2000):
    """
    Sigmoid KL annealing.
    k controls slope; x0 is midpoint (≈ where KL weight = 0.5)
    """
    return float(1 / (1 + torch.exp(-k * torch.tensor(step - x0))))


def sharpness_loss(pred: Tensor, target: Tensor, mask: Tensor, eps=1e-6):
    """
    Structural edge-consistency loss.
    Encourages pred to have the same *edge map* as target.
    """
    # Sobel kernels (better than Laplacian for stable gradients)
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=pred.device, dtype=torch.float32
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=pred.device, dtype=torch.float32
    ).view(1, 1, 3, 3)

    # Compute edge maps
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)

    gx_p = F.conv2d(pred, sobel_x, padding=1)
    gy_p = F.conv2d(pred, sobel_y, padding=1)
    gx_t = F.conv2d(target, sobel_x, padding=1)
    gy_t = F.conv2d(target, sobel_y, padding=1)

    mag_p = torch.sqrt(gx_p**2 + gy_p**2 + eps)
    mag_t = torch.sqrt(gx_t**2 + gy_t**2 + eps)

    # Contact regions have higher structural priority
    contact_mask = (target.squeeze(1) < 8.0).float() * mask
    contact_mask = contact_mask.unsqueeze(1)

    # Robust difference in edge magnitudes
    diff = torch.sqrt((mag_p - mag_t) ** 2 + eps)

    loss = (diff * contact_mask).sum() / contact_mask.sum().clamp(min=1)
    return loss


def triangle_loss_efficient(
    binder: object,
    target: object,
    pred: Tensor,
    mask: Tensor,
    num_anchors: int = 48,
) -> Tensor:
    """
    Highly optimized Triangle Inequality Loss.

    Features:
    - Zero full distance matrix materialization (saves O((M+N)^2) memory).
    - Stochastic anchoring (reduces complexity to O(MN * num_anchors)).
    - Fully vectorized (no python loops).

    Args:
        binder/target: Objects with .X (coords) and .A (features/masks).
        pred: (B, M, N) predicted interface distance matrix.
        mask: (B, M, N) mask for pred, often ignored in favor of node masks.
        num_anchors: Number of residues to sample from each side to check triangles against.
    """
    B, M, N = pred.shape
    device = pred.device

    # Extract coordinates (B, L, 3) and node masks (B, L)
    b_coords = binder.X[:, :, 1]
    t_coords = target.X[:, :, 1]
    b_mask = binder.A[:, :, 1].clamp(max=1.0)
    t_mask = target.A[:, :, 1].clamp(max=1.0)

    # We process two types of triangles based on where the anchor 'k' falls.
    # We define the triangle as (i, j, k) where i=Binder, j=Target.
    # pred[i, j] provides d_ij.

    total_loss = 0.0
    total_count = 0.0

    # =========================================================================
    # TYPE A: Anchor 'k' is in the BINDER
    # Triangle: (Binder_i, Target_j, Binder_k)
    # Edges:
    #   d_ij: Pred (Binder_i -> Target_j)
    #   d_ik: Euclidean (Binder_i -> Binder_k)
    #   d_kj: Pred (Binder_k -> Target_j)
    # =========================================================================
    S_B = min(M, num_anchors)
    if S_B > 0:
        # Sample random anchors from Binder
        k_idx = torch.randperm(M, device=device)[:S_B]

        # 1. d_ik (Euclidean): Distances from all Binder_i to Anchors_k
        # Shape: (B, M, S_B)
        k_coords_b = b_coords[:, k_idx, :]
        d_ik = torch.cdist(b_coords, k_coords_b).unsqueeze(2)  # (B, M, 1, S_B)

        # 2. d_kj (Pred): Predicted distances from Anchors_k to all Target_j
        # Pull rows k from pred. Shape: (B, S_B, N)
        # Transpose/Reshape to (B, 1, N, S_B) for broadcasting
        d_kj = pred[:, k_idx, :].transpose(1, 2).unsqueeze(1)

        # 3. d_ij (Pred): (B, M, N) -> (B, M, N, 1)
        d_ij = pred.unsqueeze(-1)

        # 4. Masks
        # Combine: mask_i (M,1,1) * mask_j (1,N,1) * mask_k (1,1,S)
        m_k = b_mask[:, k_idx]  # (B, S_B)
        mask_A = (
            b_mask.view(B, M, 1, 1) * t_mask.view(B, 1, N, 1) * m_k.view(B, 1, 1, S_B)
        )

        # 5. Compute Violations (d_ij, d_ik, d_kj)
        if mask_A.sum() > 0:
            # Check all 3 permutations
            v1 = F.relu(d_ij - (d_ik + d_kj))  # Direct path too long
            v2 = F.relu(d_ik - (d_ij + d_kj))  # Euclidean shortcut impossible
            v3 = F.relu(d_kj - (d_ij + d_ik))  # Pred shortcut impossible

            total_loss = total_loss + ((v1 + v2 + v3) * mask_A).sum()
            total_count = total_count + mask_A.sum()

    # =========================================================================
    # TYPE B: Anchor 'k' is in the TARGET
    # Triangle: (Binder_i, Target_j, Target_k)
    # Edges:
    #   d_ij: Pred (Binder_i -> Target_j)
    #   d_ik: Pred (Binder_i -> Target_k)
    #   d_kj: Euclidean (Target_j -> Target_k)
    # =========================================================================
    S_T = min(N, num_anchors)
    if S_T > 0:
        # Sample random anchors from Target
        k_idx = torch.randperm(N, device=device)[:S_T]

        # 1. d_kj (Euclidean): Distances from all Target_j to Anchors_k
        # Shape: (B, N, S_T) -> (B, 1, N, S_T)
        k_coords_t = t_coords[:, k_idx, :]
        d_kj = torch.cdist(t_coords, k_coords_t).unsqueeze(1)

        # 2. d_ik (Pred): Predicted distances from all Binder_i to Anchors_k
        # Pull cols k from pred. Shape: (B, M, S_T)
        # Reshape to (B, M, 1, S_T)
        d_ik = pred[:, :, k_idx].unsqueeze(2)

        # 3. d_ij (Pred): (B, M, N, 1)
        d_ij = pred.unsqueeze(-1)

        # 4. Masks
        m_k = t_mask[:, k_idx]  # (B, S_T)
        mask_B = (
            b_mask.view(B, M, 1, 1) * t_mask.view(B, 1, N, 1) * m_k.view(B, 1, 1, S_T)
        )

        # 5. Compute Violations
        if mask_B.sum() > 0:
            v1 = F.relu(d_ij - (d_ik + d_kj))
            v2 = F.relu(d_ik - (d_ij + d_kj))
            v3 = F.relu(d_kj - (d_ij + d_ik))

            total_loss = total_loss + ((v1 + v2 + v3) * mask_B).sum()
            total_count = total_count + mask_B.sum()

    return total_loss / total_count.clamp(min=1.0)


def triangle_loss(
    binder: object,
    target: object,
    pred: Tensor,
    mask: Tensor,
    num_anchors: int = 32,
) -> Tensor:
    """
    Vectorized, Loop-Free, Memory-Efficient Triangle Inequality Loss.

    Uses stochastic subsampling (anchors) to reduce memory from O(L^3) to O(L^2 * num_anchors).
    Penalizes all 3 permutations of the inequality for robustness.

    num_anchors: Number of random 'k' nodes to test per batch.
                 32-64 is usually sufficient for training convergence.
    """
    B, M, N = pred.shape
    L = M + N
    device = pred.device

    # 1. Coordinate extraction
    binder_ca = binder.X[:, :, 1]  # (B, M, 3)
    target_ca = target.X[:, :, 1]  # (B, N, 3)

    # 2. Build Full Distance Matrix D (B, L, L)
    # We keep this part as it's relatively cheap O(L^2) compared to O(L^3)
    D = torch.zeros(B, L, L, device=device, dtype=pred.dtype)

    with torch.no_grad():
        # Internal distances are fixed rigid bodies
        internal_bind = torch.cdist(binder_ca, binder_ca)
        internal_tgt = torch.cdist(target_ca, target_ca)

    D[:, :M, :M] = internal_bind
    D[:, M:, M:] = internal_tgt
    D[:, :M, M:] = pred
    D[:, M:, :M] = pred.transpose(1, 2)

    # 3. Subsampling
    # Instead of all k, pick 'num_anchors' random indices from [0, L)
    # This keeps the tensor size small: (B, M, N, num_anchors)

    # Ensure we don't sample more anchors than available residues
    S = min(num_anchors, L)
    k_indices = torch.randperm(L, device=device)[:S]  # (S,)

    # Extract columns for the random k anchors
    # D_anchors shape: (B, L, S)
    D_anchors = D[:, :, k_indices]

    # 4. Prepare Triplet Distances (Broadcasting)
    # We focus on the Interaction Block (Binder x Target) for (i, j)
    # i in [0, M), j in [M, M+N), k in Random S

    # d_ij: The predicted interface distances
    d_ij = pred.unsqueeze(-1)  # (B, M, N, 1)

    # d_ik: Distance from Binder(i) to Anchors(k)
    # Take first M rows of D_anchors
    d_ik = D_anchors[:, :M, :].unsqueeze(2)  # (B, M, 1, S)

    # d_jk: Distance from Target(j) to Anchors(k)
    # Take last N rows of D_anchors
    d_jk = D_anchors[:, M:, :].unsqueeze(1)  # (B, 1, N, S)

    # 5. Compute Violations (All 3 Permutations)
    # Check 1: Direct edge too long? d(i,j) > d(i,k) + d(k,j)
    v1 = F.relu(d_ij - (d_ik + d_jk))

    # Check 2: Shortcut via k? d(i,k) > d(i,j) + d(j,k)
    v2 = F.relu(d_ik - (d_ij + d_jk))

    # Check 3: Shortcut via k? d(j,k) > d(j,i) + d(i,k)
    v3 = F.relu(d_jk - (d_ij + d_ik))

    violations = v1 + v2 + v3  # (B, M, N, S)

    # 6. Apply Masks
    # Binder mask (i)
    m_bind = binder.A[:, :, 1].clamp(max=1.0)  # (B, M)
    # Target mask (j)
    m_tgt = target.A[:, :, 1].clamp(max=1.0)  # (B, N)
    # Full mask for k
    m_full = torch.cat((m_bind, m_tgt), dim=1)  # (B, L)

    # Slice mask for k anchors
    m_k_anchors = m_full[:, k_indices]  # (B, S)

    # Combine masks: (B, M, 1, 1) * (B, 1, N, 1) * (B, 1, 1, S)
    joint_mask = (
        m_bind.view(B, M, 1, 1) * m_tgt.view(B, 1, N, 1) * m_k_anchors.view(B, 1, 1, S)
    )

    # 7. Aggregate
    loss_sum = (violations * joint_mask).sum()
    valid_count = joint_mask.sum().clamp(min=1.0)

    return loss_sum / valid_count


class NonConvProteinContactMapVAE(nn.Module):
    """
    Simpler VAE over *continuous* distance matrices:
      x: (B, M, N) of real-valued distances.

    pair_mask: (B, M, N) boolean mask of valid entries.

    This version includes:
      - NO convolutions
      - Symmetry enforcement of the reconstructed distances
      - Charbonnier reconstruction loss
      - Triangle inequality & sharpness regularizers
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        kl_weight: float,
        mean: float,
        std: float,
        emb_size: int,
        protein_encoder_depth=5,
        hidden_size=128,
        latent_dim=32,
        lambda_tri=0.0001,
        lambda_sharp=0.0,
        use_symmetry=False,
        pred_type="direct",
    ):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.kl_weight = kl_weight
        self.mean = float(mean)
        self.std = float(std)
        self.use_symmetry = use_symmetry
        self.pred_type = pred_type

        self.lambda_tri = lambda_tri
        self.lambda_sharp = lambda_sharp
        self.warmup_start = 1
        self.warmup_steps = 8000
        self.protein_encoder = ProteinEncoder(
            hidden_size=hidden_size,
            emb_size=emb_size,
            depth=protein_encoder_depth,
            heads=8,
        )
        self.pre_mixer = CustomTransformer(
            dim=hidden_size,
            heads=8,
            mlp_ratio=4,
            depth=6,
        )
        self.post_mixer = CustomTransformer(
            dim=hidden_size,
            heads=8,
            mlp_ratio=4,
            depth=6,
        )
        self.bind_norm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.tgt_norm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.gaussian_encoder = GaussianSmearing(start=0.0, stop=mean, num_gaussians=64)
        num_centers = self.gaussian_encoder.num_gaussians

        # self.distance_mlp = nn.Sequential(
        #     nn.Linear(num_centers, 1),
        #     # nn.GELU(),
        #     # nn.Linear(num_centers // 2, 1),
        # )
        self.distance_mlp = nn.Sequential(
            # nn.Linear(num_centers, 1),
            nn.Linear(num_centers, num_centers // 2),
            nn.SiLU(),
            nn.Linear(num_centers // 2, 1),
        )
        self.encoder_stack = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.SiLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.SiLU(),
            nn.Linear(hidden_size, latent_dim * 2),
        )

        self.up_project = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
        )

        # ---- Decoder stack ----
        self.bind_z_final = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.tgt_z_final = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.z_mixer = CustomTransformer(
            dim=hidden_size,
            heads=8,
            mlp_ratio=4,
            depth=4,
        )

        self.huber_loss = HuberLoss(delta=0.2)

        if self.pred_type == "direct":
            self.dist_predictor = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.dist_predictor = ProteinEncoder(
                hidden_size=hidden_size,
                emb_size=hidden_size,
                depth=7,
                heads=8,
            )

    def grid_pad(self, binder_H: Tensor, target_H: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Pad binder_H and target_H to grid size (self.rows, self.cols)
        """
        binder_H = binder_H[:, : self.rows, :]  # crop to (B, rows, H)
        target_H = target_H[:, : self.cols, :]  # crop to (B, cols, H)

        # pad bind_H and target_H to match x_img if needed
        pad_bind_H = F.pad(
            binder_H, (0, 0, 0, self.rows - binder_H.size(1))
        )  # (B, rows, H)
        pad_target_H = F.pad(
            target_H, (0, 0, 0, self.cols - target_H.size(1))
        )  # (B, , H)

        return pad_bind_H, pad_target_H

    def encode_protein(
        self,
        binder: Features,
        target: Features,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        returns:
            pair_H: (B, H, M, N) pairwise features
            bind_H: (B, M, H) binder features
            target_H: (B, N, H) target features
        """
        bind_H, target_H = self.protein_encoder(binder, target)  # [B, M, H], [B, N, H]

        cplx_H = torch.cat((bind_H, target_H), dim=1)  # (B, M+N, H)
        cplx_H = self.pre_mixer(cplx_H)  # (B, M+N, H)
        bind_H, target_H = cplx_H.split([bind_H.size(1), target_H.size(1)], dim=1)

        bind_H, target_H = self.bind_norm(bind_H), self.tgt_norm(target_H)

        return bind_H, target_H  # [B, rows, H], [B, cols, H]

    def bin_distances(self, distances: Tensor) -> Tensor:
        """Discretize continuous distances into contact bins.

        distances: (B, M, N) in Å
        returns: (B, M, N) long tensor of bin indices.

        Bins are defined by ``self.contact_bin_edges``:
          bin 0: [0, edges[0])
          bin 1: [edges[0], edges[1]), ...
          bin K: [edges[K-1], +inf)
        """
        if self.contact_bin_edges.numel() == 0:
            raise RuntimeError(
                "bin_distances() called but contact_bin_edges is empty. "
                "Construct the VAE with target_mode='contact' to enable binning."
            )

        # torch.bucketize gives indices in [0, len(edges)]
        labels = torch.bucketize(distances, self.contact_bin_edges)
        return labels.long()

    def encode(
        self,
        binder: Features,
        target: Features,
        x_dists: Tensor,
        binder_H: Tensor,
        target_H: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        x_dists: (B, 1, M, N) normalized distances
        binder_H: (B, M, H) binder features
        target_H: (B, N, H) target features

        returns mu, logvar, and skip features
        """
        encoded_dists = self.gaussian_encoder(
            x_dists.squeeze(1)  # (B, M, N, num_centers)
        )

        dist_H = self.distance_mlp(encoded_dists).squeeze(-1)  # (B, M, N)
        cplx_H = torch.cat(
            [binder_H, target_H],
            dim=1,  # (B, M+N, H)
        )

        # B, L = cplx_H.size(0), cplx_H.size(1)
        M = binder_H.size(1)

        # attn_bias = cplx_H.new_zeros(B, L, L)  # (B, M+N, M+N)
        attn_bias, _ = get_pairwise_cplx_attn_mask(binder, target)  # (B, M+N, M+N)

        # fill the attention bias with pairwise distances but only for binder-target pairs
        attn_bias[:, :M, M:] = dist_H  # (B, M, N)
        attn_bias[:, M:, :M] = dist_H.transpose(1, 2)  # (B, N, M)

        # expand for multi-head attention
        attn_bias = attn_bias.unsqueeze(1)  # (B, 1, M+N, M+N)
        # attn_bias = attn_bias.repeat(1, 8, 1, 1)  # (B, num_heads, M+N, M+N)

        cplx_H = self.post_mixer(cplx_H, attn_mask=attn_bias)  # (B, M+N, H)

        pre_latent = self.encoder_stack(cplx_H)  # (B, M+N, latent_dim*2)
        mu, logvar = pre_latent.chunk(2, dim=-1)  # each (B, M+N, latent_dim)

        return mu, logvar  # each (B, M+N, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(
        self, z: Tensor, binder: Features, target: Features, upscale=False
    ) -> Tensor:
        """
        z: (B, M+N, latent_dim) latent code for each residue
        upscale: whether to upscale the reconstructed distances according to mean/std
        returns:
            x_recon: (B, M, N) reconstructed normalized distances
        """
        z = self.up_project(z)  # (B, M+N, H)
        M, N = binder.X.size(1), target.X.size(1)

        # bind_z = self.bind_z_final(bind_z)  # (B, M, H)
        # target_z = self.tgt_z_final(target_z)  # (B, N, H)

        cplx_z = z  # (B, M+N, H)
        attn_mask, _ = get_pairwise_cplx_attn_mask(binder, target)  # (B, M+N, M+N)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, M+N, M+N)
        cplx_z = self.z_mixer(cplx_z, attn_mask=attn_mask)  # (B, M+N, H)
        bind_z, target_z = cplx_z.split([M, N], dim=1)  # (B, M, H), (B, N, H)

        # dist_H = bind_z.unsqueeze(2) + target_z.unsqueeze(1)  # (B, M, N, H)
        dist_H = torch.cat(
            [
                bind_z.unsqueeze(2).expand(-1, -1, N, -1),
                target_z.unsqueeze(1).expand(-1, M, -1, -1),
            ],
            dim=-1,
        )

        if self.pred_type == "direct":
            x_recon = self.dist_predictor(dist_H).squeeze(-1)  # (B, M, N)
            # x_recon = 0.5 * (x_recon + x_recon.transpose(1, 2))  # enforce symmetry
        elif self.pred_type == "ipa":
            bind_recon, tgt_recon = self.dist_predictor(
                binder, target, bind_H=bind_z, tgt_H=target_z, return_coords=True
            )
            x_recon = torch.cdist(
                bind_recon, tgt_recon
            )  # (B, M, N) reconstructed distances
            return x_recon, bind_recon, tgt_recon
        else:
            raise ValueError(f"Unknown pred_type: {self.pred_type}")

        # x_recon = 0.5 * (x_recon + x_recon.transpose(1, 2))  # enforce symmetry
        # # output gram matrix X^T X
        # x_recon = torch.matmul(x_recon.transpose(1, 2), x_recon)  # (B, M, N)

        if upscale:
            x_recon = (x_recon * self.std) + self.mean
            # x_recon = x_recon.exp()  # back to physical distances

        return x_recon

    def get_latents(self, x: Tensor, binder: Features, target: Features) -> Tensor:
        """
        returns:
            z: (B, M+N, latent_dim) latent codes for each residue
            binder_H: (B, M, H) binder features
            target_H: (B, N, H) target features
        """

        # Keep original physical distances
        x_phys = x

        # Normalize distances (for reconstruction)
        x_norm = (x_phys - self.mean) / self.std  # (B,M,N)
        x_img = x_norm.unsqueeze(1)  # (B,1,M,N)

        # Get protein features
        binder_H, target_H = self.encode_protein(binder, target)

        # Encode
        mu, logvar = self.encode(binder, target, x_img, binder_H, target_H)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        return z, binder_H, target_H

    def forward(
        self,
        x: Tensor,  # (B, M, N) raw distances (Å)
        binder: Features,
        target: Features,
        pair_mask: Tensor,  # (B, M, N) bool or float
        step=0,
        return_latent=False,
        return_recon=False,
    ):
        """
        returns:
            loss, recon_loss, kl_loss [, x_recon_physical]
        """
        assert x.dim() == 3, f"Expected x of shape (B,M,N), got {x.shape}"

        if pair_mask.dtype != torch.float32:
            mask = pair_mask.float()
        else:
            mask = pair_mask

        # Keep original physical distances
        x_phys = x

        # Normalize distances (for reconstruction)
        # x_norm = ((x_phys + 1e-8).log() - self.mean) / self.std  # (B,M,N)
        x_norm = (x_phys - self.mean) / self.std  # (B,M,N)
        x_img = x_norm.unsqueeze(1)  # (B,1,M,N)

        # Create masks and pad if necessary
        bind_mask_padded = binder.A[:, :, 1].clamp(max=1.0).bool()  # (B, M, 1)
        tgt_mask_padded = target.A[:, :, 1].clamp(max=1.0).bool()  # (B, N, 1)
        pair_mask = bind_mask_padded.unsqueeze(2) & tgt_mask_padded.unsqueeze(
            1
        )  # (B, M, N)
        # bind_mask_padded, tgt_mask_padded = self.pad_to_grid(
        #     bind_mask, tgt_mask
        # )  # (B, self.rows), (B, self.cols)

        # Get protein features
        bind_H, target_H = self.encode_protein(binder, target)

        # Encode
        mu, logvar = self.encode(binder, target, x_img, bind_H, target_H)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        if return_latent:
            # return mu, logvar
            return z

        # Decode → normalized reconstruction
        if self.pred_type == "ipa":
            x_recon_norm, bind_recon, tgt_recon = self.decode(z, binder, target)
        else:
            x_recon_norm = self.decode(z, binder, target)  # (B,M,N)

        # ---------------- Coordinate loss ----------------
        if self.pred_type == "ipa":
            min_bind_size = min(bind_recon.size(1), binder.X.size(1))
            bind_coord_loss = (
                bind_recon[:, :min_bind_size] - binder.X[:, :min_bind_size, 1]
            ) ** 2  # (B, M, 3)

            min_tgt_size = min(tgt_recon.size(1), target.X.size(1))
            tgt_coord_loss = (
                tgt_recon[:, :min_tgt_size] - target.X[:, :min_tgt_size, 1]
            ) ** 2  # (B, N, 3)

            num_bind = bind_mask.sum(dim=1)  # (B,)
            num_tgt = tgt_mask.sum(dim=1)  # (B,)

            bind_coord_loss = bind_coord_loss * bind_mask.unsqueeze(-1)
            tgt_coord_loss = tgt_coord_loss * tgt_mask.unsqueeze(-1)

            bind_coord_loss = bind_coord_loss.sum(dim=-1).sum(dim=-1) / num_bind.clamp(
                min=1.0
            )  # (B, )
            tgt_coord_loss = tgt_coord_loss.sum(dim=-1).sum(dim=-1) / num_tgt.clamp(
                min=1.0
            )

            coord_loss = bind_coord_loss.mean() + tgt_coord_loss.mean()
        else:
            coord_loss = torch.tensor(0.0, device=x.device)

        # ---------------- Reconstruction loss (Charbonnier) ----------------
        eps = 1e-6
        # diff = x_recon_norm - x_norm
        # charb = torch.sqrt(diff * diff + eps)  # robust L2
        # recon_loss = (
        #     (charb * pair_mask).sum(dim=(1, 2))
        #     / pair_mask.float().clamp(min=1.0).sum(dim=(1, 2))
        # ).mean()

        recon_loss = self.huber_loss(x_recon_norm * pair_mask, x_norm * pair_mask)

        # recon_loss = (diff**2 * pair_mask).sum(dim=(1,2)) / pair_mask.float().sum(dim=(1,2))
        # recon_loss = recon_loss.mean()

        # ---------------- Geometry / sharpness regularizers ----------------
        # Convert reconstruction back to physical distances for these terms
        # x_recon_phys = (x_recon_norm * self.std + self.mean).exp() + 1e-8
        x_recon_phys = x_recon_norm * self.std + self.mean

        triangle_ineq_loss = triangle_loss_efficient(
            binder, target, x_recon_phys.clamp(max=30), mask
        )
        # triangle_ineq_loss = torch.tensor(0.0, device=x.device)
        # sharp_loss = sharpness_loss(x_recon_phys, x_phys, mask)
        sharp_loss = torch.tensor(0.0, device=x.device)

        # ---------------- Per-Channel KL term ----------------
        # kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # mask invalid residues before KL computation
        valid_mask = torch.cat((bind_mask_padded, tgt_mask_padded), dim=1)  # (B, M+N)
        mu = mu * valid_mask.unsqueeze(-1)
        logvar = logvar * valid_mask.unsqueeze(-1)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = (kl_loss * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum().clamp(
            min=1.0
        )
        kl_loss = self.kl_weight * kl_loss

        # kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1).mean(dim=0)
        # kl_loss = self.kl_weight * kl_loss.mean()

        # free_bits = 0.3
        # kl_loss = torch.clamp(kl_loss, min=free_bits)

        # lambda_scale = (
        #     min(1.0, step / self.warmup_steps) if step >= self.warmup_start else 0.0
        # )
        lambda_scale = kl_sigmoid_anneal(step) if step >= self.warmup_start else 0.0

        # ---------------- Warmup for lambda_tri and lambda_sharp ----------------
        triangle_ineq_loss = lambda_scale * triangle_ineq_loss
        sharp_loss = 0.0
        kl_loss = lambda_scale * kl_loss

        # ---------------- Total loss ----------------
        loss = (
            coord_loss
            + recon_loss
            + kl_loss
            + self.lambda_tri * triangle_ineq_loss
            + self.lambda_sharp * sharp_loss
        )

        losses = {
            "total_loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "triangle_ineq_loss": triangle_ineq_loss,
            "sharp_loss": sharp_loss,
            "coord_loss": coord_loss,
        }
        if return_recon:
            return losses, x_recon_phys
            # Return physical distances
            # return (
            #     loss,
            #     recon_loss,
            #     kl_loss,
            #     triangle_ineq_loss,
            #     sharp_loss,
            #     x_recon_phys,
            # )

        # return loss, recon_loss, kl_loss, triangle_ineq_loss, sharp_loss
        return losses


class TriangularDistanceMapVAE(nn.Module):
    """
    Simpler VAE over *continuous* distance matrices:
      x: (B, M, N) of real-valued distances.

    pair_mask: (B, M, N) boolean mask of valid entries.

    This version includes:
      - NO convolutions
      - Charbonnier reconstruction loss
      - Triangle inequality & sharpness regularizers
    """

    def __init__(
        self,
        kl_weight: float,
        mean: float,
        std: float,
        hidden_size=128,
        latent_dim=8,
        lambda_tri=0.0001,
        lambda_sharp=0.0,
        use_symmetry=False,
        pred_type="direct",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.kl_weight = kl_weight
        self.mean = float(mean)
        self.std = float(std)
        self.use_symmetry = use_symmetry
        self.pred_type = pred_type

        self.lambda_tri = lambda_tri
        self.lambda_sharp = lambda_sharp
        self.warmup_start = 1
        self.warmup_steps = 8000

        self.W_bind = nn.Linear(hidden_size, hidden_size)
        self.W_tgt = nn.Linear(hidden_size, hidden_size)

        self.pair_embedding = PairEmbeddingInit(
            c_pair=hidden_size // 2, max_relpos=3000
        )
        self.triangular_module = TriangleModule(
            c_pair=hidden_size // 2,
            c_hidden=hidden_size,
            num_blocks=8,
        )

        self.pre_mixer = CustomTransformer(
            dim=hidden_size,
            heads=8,
            mlp_ratio=4,
            depth=6,
        )
        self.post_mixer = CustomTransformer(
            dim=hidden_size,
            heads=8,
            mlp_ratio=4,
            depth=6,
        )
        num_gaussians = hidden_size
        self.gaussian_encoder = GaussianSmearing(
            start=0.0, stop=mean, num_gaussians=num_gaussians
        )
        num_centers = num_gaussians

        self.distance_mlp = nn.Sequential(
            nn.Linear(num_centers, num_centers),
            nn.SiLU(),
            nn.Linear(num_centers, num_centers),
        )
        self.proj_to_latent = nn.Linear(num_gaussians, latent_dim * 2)

        self.up_project = nn.Sequential(
            nn.Linear(latent_dim, num_gaussians),
        )
        self.triangle_encoder = TriangleBlock(num_gaussians, depth=8)

        # ---- Decoder stack ----
        self.bind_z_final = nn.Sequential(
            nn.Linear(num_gaussians, num_gaussians),
            nn.SiLU(),
            nn.Linear(num_gaussians, num_gaussians),
        )
        self.tgt_z_final = nn.Sequential(
            nn.Linear(num_gaussians, num_gaussians),
            nn.SiLU(),
            nn.Linear(num_gaussians, num_gaussians),
        )

        self.triangle_decoder = TriangleBlock(num_gaussians, depth=8)

        self.dist_predictor = nn.Linear(num_gaussians * 2, 1)

    def get_internal_dists(self, binder: Features, target: Features):
        bind_internal_dist = torch.cdist(binder.X[:, :, 1], binder.X[:, :, 1])
        target_internal_dist = torch.cdist(target.X[:, :, 1], target.X[:, :, 1])

        # bind_internal_dist = (bind_internal_dist - self.mean) / self.std
        # target_internal_dist = (target_internal_dist - self.mean) / self.std

        bind_dist_H = self.distance_mlp(
            self.gaussian_encoder(bind_internal_dist)
        ).squeeze(-1)
        tgt_dist_H = self.distance_mlp(
            self.gaussian_encoder(target_internal_dist)
        ).squeeze(-1)

        return bind_dist_H, tgt_dist_H  # each (B, L, L)

    def get_cplx_pair_mask(self, binder: Features, target: Features) -> Tensor:
        bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()

        cplx_mask = torch.cat([bind_mask, tgt_mask], dim=1)  # (B, M+N)
        cplx_pair_mask = cplx_mask.unsqueeze(2) & cplx_mask.unsqueeze(
            1
        )  # (B, M+N, M+N)

        return cplx_pair_mask

    def get_cplx_dists(
        self, x_dists: Tensor, binder: Features, target: Features
    ) -> Tensor:
        """
        x_dists: (B, 1, M, N) normalized distances
        returns:
            cplx_dists: (B, M+N, M+N) full complex distance matrix
        """
        bind_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        B, M, N = x_dists.size(0), binder.X.size(1), target.X.size(1)
        L = M + N

        device = x_dists.device
        cplx_dists = torch.zeros(B, L, L, device=device, dtype=x_dists.dtype)

        with torch.no_grad():
            internal_bind = torch.cdist(bind_ca, bind_ca)
            internal_tgt = torch.cdist(target_ca, target_ca)

        cplx_dists[:, :M, :M] = internal_bind
        cplx_dists[:, M:, M:] = internal_tgt
        cplx_dists[:, :M, M:] = x_dists.squeeze(1)
        cplx_dists[:, M:, :M] = x_dists.squeeze(1).transpose(1, 2)

        return cplx_dists  # (B, M+N, M+N)

    def encode(
        self,
        binder: Features,
        target: Features,
        x_img: Tensor,
        binder_H: Tensor,
        target_H: Tensor,
        pair_H: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        x_dists: (B, 1, M+N, M+N) normalized distances
        returns mu, logvar
        """
        M = binder.X.size(1)

        cplx_dist_H = self.distance_mlp(encoded_dists).squeeze(
            -1
        )  # (B, M+N, M+N, n_centers)
        cplx_H = self.triangle_encoder(cplx_dist_H, cplx_pair_mask)  # (B, M+N, M+N, H)

        cplx_H = torch.cat((bind_H, target_H), dim=1)  # (B, M+N, H)

        pre_latent = self.proj_to_latent(cplx_H)  # (B, M+N, latent_dim*2)
        mu, logvar = pre_latent.chunk(2, dim=-1)  # each (B, M+N, latent_dim)

        return (
            mu,
            logvar,
            encoded_dists,
        )  # each (B, M+N, latent_dim), (B, M+N, M+N, num_centers)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(
        self,
        encoded_dists: Tensor,
        z: Tensor,
        binder: Features,
        target: Features,
        cplx_pair_mask: Tensor,
        upscale=False,
    ) -> Tensor:
        """
        encoded_dists: (B, M+N, M+N, num_centers) encoded distance features from encoder
        z: (B, M+N, latent_dim) latent code for each residue
        upscale: whether to upscale the reconstructed distances according to mean/std
        returns:
            x_recon: (B, M, N) reconstructed normalized distances
        """
        z = self.up_project(z)  # (B, M+N, H)
        M, N = binder.X.size(1), target.X.size(1)

        bind_z = self.bind_z_final(z[:, :M, :])  # (B, M, H)
        target_z = self.tgt_z_final(z[:, M:, :])  # (B, N, H)

        cplx_z = bind_z.unsqueeze(2) * target_z.unsqueeze(1)  # (B, M, N, H)

        # put cplx_z back into full complex distance matrix
        dist_z = torch.zeros_like(encoded_dists)  # (B, M+N, M+N, H)
        dist_z[:, :M, M:, :] = cplx_z  # (B, M, N, H)
        dist_z[:, M:, :M, :] = cplx_z.transpose(1, 2)
        dist_z[:, :M, :M, :] = encoded_dists[:, :M, :M, :]  # (B, M, M, H)
        dist_z[:, M:, M:, :] = encoded_dists[:, M:, M:, :]  # (B, N, N, H)

        dist_z = self.triangle_decoder(dist_z, cplx_pair_mask)  # (B, M+N, M+N, H)

        # extract binder-target distances
        bind_z = dist_z[:, :M, M:, :].mean(dim=2)  # (B, M, H)
        target_z = dist_z[:, M:, :M, :].mean(dim=1)  # (B, N, H)

        dist_H = torch.cat(
            [
                bind_z.unsqueeze(2).expand(-1, -1, N, -1),
                target_z.unsqueeze(1).expand(-1, M, -1, -1),
            ],
            dim=-1,
        )

        x_recon = self.dist_predictor(dist_H).squeeze(-1)  # (B, M, N)

        if upscale:
            x_recon = (x_recon * self.std) + self.mean
            # x_recon = x_recon.exp()  # back to physical distances

        return x_recon

    def get_latents(
        self, x: Tensor, binder: Features, target: Features, pair_mask: Tensor
    ) -> Tensor:
        """
        returns:
            z: (B, M+N, latent_dim) latent codes for each residue
            binder_H: (B, M, H) binder features
            target_H: (B, N, H) target features
        """
        x_phys = x

        # Normalize distances (for reconstruction)
        x_norm = (x_phys - self.mean) / self.std  # (B,M,N)
        x_img = x_norm.unsqueeze(1)  # (B,1,M,N)

        # Encode
        mu, logvar = self.encode(binder, x_img, pair_mask)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        return z, binder_H, target_H

    def forward(
        self,
        x: Tensor,  # (B, M, N) raw distances (Å)
        binder: Features,
        target: Features,
        pair_mask: Tensor,  # (B, M, N) bool or float
        step=0,
        return_latent=False,
        return_recon=False,
    ):
        """
        returns:
            loss, recon_loss, kl_loss [, x_recon_physical]
        """
        assert x.dim() == 3, f"Expected x of shape (B,M,N), got {x.shape}"

        # Keep original physical distances
        x_phys = x

        cplx_dists = self.get_cplx_dists(x.unsqueeze(1), binder, target)
        cplx_pair_mask = self.get_cplx_pair_mask(binder, target)

        # Normalize distances (for reconstruction)
        x_norm = (cplx_dists - self.mean) / self.std  # (B,M,N)
        x_img = x_phys.unsqueeze(1)  # (B,1,M,N)

        # Embed the sequence embeddings
        binder_H = self.W_bind(binder.S)  # (B, M, H)
        target_H = self.W_tgt(target.S)  # (B, N, H)

        whole_seq = torch.cat((binder.Si, target.Si), dim=1)  # (B, M+N)
        pair_H = self.pair_embedding(whole_seq)  # (B, M+N, M+N, H/2)

        # Create masks and pad if necessary
        bind_mask_padded = binder.A[:, :, 1].clamp(max=1.0).bool()  # (B, M, 1)
        tgt_mask_padded = target.A[:, :, 1].clamp(max=1.0).bool()  # (B, N, 1)
        pair_mask = bind_mask_padded.unsqueeze(2) & tgt_mask_padded.unsqueeze(
            1
        )  # (B, M, N)

        # Encode
        mu, logvar, encoded_dists = self.encode(
            binder, target, x_img, binder_H, target_H, pair_H
        )

        # Latent sample
        z = self.reparameterize(mu, logvar)

        if return_latent:
            # return mu, logvar
            return z

        # Decode → normalized reconstruction
        x_recon_norm = self.decode(
            encoded_dists.detach(), z, binder, target, cplx_pair_mask
        )  # (B,M,N)

        # ---------------- Coordinate loss ----------------
        coord_loss = torch.tensor(0.0, device=x.device)

        # ---------------- Reconstruction loss (Charbonnier) ----------------
        eps = 1e-6

        diff = x_recon_norm - x_norm
        charb = torch.sqrt(diff * diff + eps)  # robust L2
        recon_loss = (
            (charb * pair_mask).sum(dim=(1, 2))
            / pair_mask.float().clamp(min=1.0).sum(dim=(1, 2))
        ).mean()

        # recon_loss = (diff**2 * pair_mask).sum(dim=(1,2)) / pair_mask.float().sum(dim=(1,2))
        # recon_loss = recon_loss.mean()

        # ---------------- Geometry / sharpness regularizers ----------------
        # Convert reconstruction back to physical distances for these terms
        # x_recon_phys = (x_recon_norm * self.std + self.mean).exp() + 1e-8
        x_recon_phys = x_recon_norm * self.std + self.mean

        # triangle_ineq_loss = triangle_loss_efficient(
        #     binder, target, x_recon_phys.clamp(max=30), mask
        # )

        triangle_ineq_loss = torch.tensor(0.0, device=x.device)
        # sharp_loss = sharpness_loss(x_recon_phys, x_phys, mask)
        sharp_loss = torch.tensor(0.0, device=x.device)

        # ---------------- Per-Channel KL term ----------------
        # kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # mask invalid residues before KL computation
        valid_mask = torch.cat((bind_mask_padded, tgt_mask_padded), dim=1)  # (B, M+N)
        mu = mu * valid_mask.unsqueeze(-1)
        logvar = logvar * valid_mask.unsqueeze(-1)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = (kl_loss * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum().clamp(
            min=1.0
        )
        kl_loss = self.kl_weight * kl_loss

        # kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1).mean(dim=0)
        # kl_loss = self.kl_weight * kl_loss.mean()

        # free_bits = 0.3
        # kl_loss = torch.clamp(kl_loss, min=free_bits)

        # lambda_scale = (
        #     min(1.0, step / self.warmup_steps) if step >= self.warmup_start else 0.0
        # )
        lambda_scale = kl_sigmoid_anneal(step) if step >= self.warmup_start else 0.0

        # ---------------- Warmup for lambda_tri and lambda_sharp ----------------
        triangle_ineq_loss = lambda_scale * triangle_ineq_loss
        sharp_loss = 0.0
        kl_loss = lambda_scale * kl_loss

        # ---------------- Total loss ----------------
        loss = (
            coord_loss
            + recon_loss
            + kl_loss
            + self.lambda_tri * triangle_ineq_loss
            + self.lambda_sharp * sharp_loss
        )

        if return_recon:
            # Return physical distances
            return (
                loss,
                recon_loss,
                kl_loss,
                triangle_ineq_loss,
                sharp_loss,
                x_recon_phys,
            )

        return loss, recon_loss, kl_loss, triangle_ineq_loss, sharp_loss


class DistanceMapVAE(nn.Module):
    """
    Simpler VAE over *continuous* distance matrices:
      x: (B, M, N) of real-valued distances.

    pair_mask: (B, M, N) boolean mask of valid entries.

    This version includes:
      - NO convolutions
      - Charbonnier reconstruction loss
      - Triangle inequality & sharpness regularizers
    """

    def __init__(
        self,
        kl_weight: float,
        mean: float,
        std: float,
        hidden_size=128,
        latent_dim=8,
        lambda_tri=0.0001,
        lambda_sharp=0.0,
        use_symmetry=False,
        pred_type="direct",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.kl_weight = kl_weight
        self.mean = float(mean)
        self.std = float(std)
        self.use_symmetry = use_symmetry
        self.pred_type = pred_type

        self.lambda_tri = lambda_tri
        self.lambda_sharp = lambda_sharp
        self.warmup_start = 1
        self.warmup_steps = 8000
        self.pre_mixer = CustomTransformer(
            dim=hidden_size, heads=8, mlp_ratio=4, depth=6
        )
        self.post_mixer = CustomTransformer(
            dim=hidden_size,
            heads=8,
            mlp_ratio=4,
            depth=6,
        )
        num_gaussians = hidden_size
        self.gaussian_encoder = GaussianSmearing(
            start=0.0, stop=mean, num_gaussians=num_gaussians
        )
        num_centers = num_gaussians

        self.distance_mlp = nn.Sequential(
            nn.Linear(num_centers, num_centers),
            nn.RMSNorm(num_centers),
            nn.SiLU(),
            nn.Linear(num_centers, num_centers),
        )
        self.bind_enc = nn.Sequential(
            nn.RMSNorm(num_gaussians),
            nn.Linear(num_gaussians, num_gaussians),
        )
        self.tgt_enc = nn.Sequential(
            nn.RMSNorm(num_gaussians),
            nn.Linear(num_gaussians, num_gaussians),
        )
        self.proj_to_latent = nn.Sequential(
            nn.RMSNorm(num_gaussians),
            nn.Linear(num_gaussians, latent_dim * 2),
        )

        self.up_project = nn.Linear(latent_dim, num_gaussians)

        self.cplx_column_pooling = ColumnPooling(num_gaussians)
        self.cplx_row_pooling = ColumnPooling(num_gaussians)

        # ---- Decoder stack ----
        self.bind_z_final = nn.Sequential(
            nn.RMSNorm(num_gaussians),
            nn.Linear(num_gaussians, num_gaussians),
        )
        self.tgt_z_final = nn.Sequential(
            nn.RMSNorm(num_gaussians),
            nn.Linear(num_gaussians, num_gaussians),
        )

        self.bind_inter_final = nn.Sequential(
            nn.RMSNorm(num_gaussians),
            nn.Linear(num_gaussians, num_gaussians),
        )
        self.tgt_inter_final = nn.Sequential(
            nn.RMSNorm(num_gaussians),
            nn.Linear(num_gaussians, num_gaussians),
        )

        self.inter_bind_pooling = ColumnPooling(num_gaussians)
        self.inter_tgt_pooling = ColumnPooling(num_gaussians)

        self.dist_predictor = nn.Sequential(
            nn.Linear(num_gaussians, num_gaussians),
            nn.RMSNorm(num_gaussians),
            nn.SiLU(),
            nn.Linear(num_gaussians, 1),
        )

    def encode(
        self, x_dists: Tensor, cplx_mask: Tensor, bind_mask: Tensor, target_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        x_dists: (B, 1, M, N) normalized distances
        cplx_mask: (B, M+N, M+N) float attention mask
        bind_mask: (B, M, 1) bool mask for binder residues
        target_mask: (B, N, 1) bool mask for target residues
        returns mu, logvar
        """
        M = x_dists.size(2)
        encoded_dists = self.gaussian_encoder(
            x_dists.squeeze(1)  # (B, M+N, M+N, num_centers)
        )

        cplx_dist_H = self.distance_mlp(encoded_dists)  # (B, M, N, n_centers)

        num_bind = bind_mask.sum(dim=-1)  # (B,)
        num_tgt = target_mask.sum(dim=-1)
        pair_mask = (bind_mask.unsqueeze(2) & target_mask.unsqueeze(1)).unsqueeze(
            -1
        )  # (B, M, N, 1)

        cplx_dist_H = cplx_dist_H * pair_mask.float()

        # extract distance embeddings between binder and target
        # bind_H = cplx_dist_H.sum(dim=2)  # (B, M, H)
        # target_H = cplx_dist_H.sum(dim=1)  # (B, N, H)

        # bind_H = bind_H / num_bind.view(-1, 1, 1)  # (B, M, H)
        # target_H = target_H / num_tgt.view(-1, 1, 1)  # (B, N, H)
        bind_H = self.cplx_row_pooling(cplx_dist_H, attn_mask=pair_mask.squeeze(-1))
        target_H = self.cplx_column_pooling(
            cplx_dist_H, direction="rows", attn_mask=pair_mask.squeeze(-1)
        )

        bind_H = self.bind_enc(bind_H)  # (B, M, H)
        target_H = self.tgt_enc(target_H)  # (B, N, H)

        cplx_H = torch.cat((bind_H, target_H), dim=1)  # (B, M+N, H)
        cplx_H = self.pre_mixer(cplx_H, attn_mask=cplx_mask)  # (B, M+N, H)

        pre_latent = self.proj_to_latent(cplx_H)  # (B, M+N, latent_dim*2)
        mu, logvar = pre_latent.chunk(2, dim=-1)  # each (B, M+N, latent_dim)

        return mu, logvar  # each (B, M+N, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(
        self,
        z: Tensor,
        binder: Features,
        target: Features,
        cplx_mask: Tensor,
        binder_mask: Tensor,
        target_mask: Tensor,
        upscale=False,
    ) -> Tensor:
        """
        z: (B, M+N, latent_dim) latent code for each residue
        cplx_mask: (B, M+N, M+N) float attention mask
        upscale: whether to upscale the reconstructed distances according to mean/std
        returns:
            x_recon: (B, M, N) reconstructed normalized distances
        """
        z = self.up_project(z)  # (B, M+N, H)

        num_bind = binder_mask.sum(dim=-1)  # (B,)
        num_tgt = target_mask.sum(dim=-1)
        M, N = binder.X.size(1), target.X.size(1)

        z = self.post_mixer(z, attn_mask=cplx_mask)  # (B, M+N, H)

        bind_internal_dist = torch.cdist(
            binder.X[:, :, 1], binder.X[:, :, 1]
        )  # (B, M, M)
        target_internal_dist = torch.cdist(
            target.X[:, :, 1], target.X[:, :, 1]
        )  # (B, N, N)

        bind_internal_dist = (bind_internal_dist - self.mean) / self.std
        target_internal_dist = (target_internal_dist - self.mean) / self.std

        bind_inter_H = self.distance_mlp(
            self.gaussian_encoder(bind_internal_dist)
        )  # (B, M, M, H)
        target_inter_H = self.distance_mlp(
            self.gaussian_encoder(target_internal_dist)
        )  # (B, N, N, H)

        bind_z = self.bind_z_final(z[:, :M, :])  # (B, M, H)
        target_z = self.tgt_z_final(z[:, M:, :])  # (B, N, H)

        bind_pair_mask = (
            binder_mask.unsqueeze(2) & binder_mask.unsqueeze(1)
        ).unsqueeze(-1)  # (B, M, M, 1)
        bind_inter_H = bind_inter_H * bind_pair_mask.float()

        target_pair_mask = (
            target_mask.unsqueeze(2) & target_mask.unsqueeze(1)
        ).unsqueeze(-1)  # (B, N, N, 1)
        target_inter_H = target_inter_H * target_pair_mask.float()

        # bind_inter_H = bind_inter_H.sum(dim=2) / num_bind.view(-1, 1, 1)
        bind_inter_H = self.inter_bind_pooling(
            bind_inter_H, direction="rows", attn_mask=bind_pair_mask.squeeze(-1)
        )
        bind_inter_H = self.bind_inter_final(bind_inter_H)  # (B, M, H)

        # target_inter_H = target_inter_H.sum(dim=2) / num_tgt.view(-1, 1, 1)
        target_inter_H = self.inter_tgt_pooling(
            target_inter_H, target_pair_mask.squeeze(-1)
        )
        target_inter_H = self.tgt_inter_final(target_inter_H)  # (B, N, H)

        bind_z = bind_z + bind_inter_H  # (B, M, H)
        target_z = target_z + target_inter_H

        cplx_z = bind_z.unsqueeze(2) * target_z.unsqueeze(1)  # (B, M, N, H)

        x_recon = self.dist_predictor(cplx_z).squeeze(-1)  # (B, M, N)

        if upscale:
            x_recon = (x_recon * self.std) + self.mean
            # x_recon = x_recon.exp()  # back to physical distances

        return x_recon

    def get_latents(
        self, x: Tensor, binder: Features, target: Features, pair_mask: Tensor
    ) -> Tensor:
        """
        returns:
            z: (B, M+N, latent_dim) latent codes for each residue
            binder_H: (B, M, H) binder features
            target_H: (B, N, H) target features
        """
        x_phys = x

        # Normalize distances (for reconstruction)
        x_norm = (x_phys - self.mean) / self.std  # (B,M,N)
        x_img = x_norm.unsqueeze(1)  # (B,1,M,N)

        # Encode
        mu, logvar = self.encode(binder, x_img, pair_mask)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        return z, binder_H, target_H

    def forward(
        self,
        x: Tensor,  # (B, M, N) raw distances (Å)
        binder: Features,
        target: Features,
        pair_mask: Tensor,  # (B, M, N) bool or float
        step=0,
        return_latent=False,
        return_recon=False,
    ):
        """
        returns:
            loss, recon_loss, kl_loss [, x_recon_physical]
        """
        assert x.dim() == 3, f"Expected x of shape (B,M,N), got {x.shape}"

        if pair_mask.dtype != torch.float32:
            mask = pair_mask.float()
        else:
            mask = pair_mask

        # Keep original physical distances
        x_phys = x

        # Normalize distances (for reconstruction)
        x_norm = (x_phys - self.mean) / self.std  # (B,M,N)
        x_img = x_norm.unsqueeze(1)  # (B,1,M,N)

        # Create masks and pad if necessary
        bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()  # (B, M, 1)
        tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()  # (B, N, 1)
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)  # (B, M, N)

        cplx_mask, _ = get_pairwise_cplx_attn_mask(binder, target)
        cplx_mask = cplx_mask.unsqueeze(1)

        # Encode
        mu, logvar = self.encode(x_img, cplx_mask, bind_mask, tgt_mask)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        if return_latent:
            # return mu, logvar
            return z

        # Decode → normalized reconstruction
        x_recon_norm = self.decode(
            z, binder, target, cplx_mask, bind_mask, tgt_mask, upscale=True
        )  # (B,M,N)

        # ---------------- Coordinate loss ----------------
        coord_loss = torch.tensor(0.0, device=x.device)

        # ---------------- Reconstruction loss (Charbonnier) ----------------
        eps = 1e-6

        diff = x_recon_norm - x_norm
        charb = torch.sqrt(diff * diff + eps)  # robust L2
        recon_loss = (
            (charb * pair_mask).sum(dim=(1, 2))
            / pair_mask.float().clamp(min=1.0).sum(dim=(1, 2))
        ).mean()

        # recon_loss = (diff**2 * pair_mask).sum(dim=(1,2)) / pair_mask.float().sum(dim=(1,2))
        # recon_loss = recon_loss.mean()

        # ---------------- Geometry / sharpness regularizers ----------------
        # Convert reconstruction back to physical distances for these terms
        # x_recon_phys = (x_recon_norm * self.std + self.mean).exp() + 1e-8
        x_recon_phys = x_recon_norm * self.std + self.mean

        triangle_ineq_loss = triangle_loss_efficient(
            binder, target, x_recon_phys.clamp(max=30), mask
        )

        # triangle_ineq_loss = torch.tensor(0.0, device=x.device)
        # sharp_loss = sharpness_loss(x_recon_phys, x_phys, mask)
        sharp_loss = torch.tensor(0.0, device=x.device)

        # ---------------- Per-Channel KL term ----------------
        # kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # mask invalid residues before KL computation
        valid_mask = torch.cat((bind_mask, tgt_mask), dim=1)  # (B, M+N)
        mu = mu * valid_mask.unsqueeze(-1)
        logvar = logvar * valid_mask.unsqueeze(-1)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = (kl_loss * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum().clamp(
            min=1.0
        )
        kl_loss = self.kl_weight * kl_loss

        # kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1).mean(dim=0)
        # kl_loss = self.kl_weight * kl_loss.mean()

        # free_bits = 0.3
        # kl_loss = torch.clamp(kl_loss, min=free_bits)

        # lambda_scale = (
        #     min(1.0, step / self.warmup_steps) if step >= self.warmup_start else 0.0
        # )
        lambda_scale = kl_sigmoid_anneal(step) if step >= self.warmup_start else 0.0

        # ---------------- Warmup for lambda_tri and lambda_sharp ----------------
        triangle_ineq_loss = lambda_scale * triangle_ineq_loss
        sharp_loss = 0.0
        kl_loss = lambda_scale * kl_loss

        # ---------------- Total loss ----------------
        loss = (
            coord_loss
            + recon_loss
            + kl_loss
            + self.lambda_tri * triangle_ineq_loss
            + self.lambda_sharp * sharp_loss
        )

        if return_recon:
            # Return physical distances
            return (
                loss,
                recon_loss,
                kl_loss,
                triangle_ineq_loss,
                sharp_loss,
                x_recon_phys,
            )

        return loss, recon_loss, kl_loss, triangle_ineq_loss, sharp_loss


class TokenDistanceVAE(nn.Module):
    """
    Simpler VAE over *continuous* distance matrices:
      x: (B, M, N) of real-valued distances.

    pair_mask: (B, M, N) boolean mask of valid entries.

    This version includes:
      - NO convolutions
      - Charbonnier reconstruction loss
      - Triangle inequality & sharpness regularizers
    """

    def __init__(
        self,
        kl_weight: float,
        mean: float,
        std: float,
        hidden_size=128,
        latent_dim=8,
        heads=8,
        lambda_tri=0.0001,
        lambda_sharp=0.0,
        use_symmetry=False,
        emb_size=768,
        pred_type="direct",
        use_nll=False,
        dropout=0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.kl_weight = kl_weight
        self.mean = float(mean)
        self.std = float(std)
        self.use_symmetry = use_symmetry
        self.pred_type = pred_type
        self.heads = heads
        self.use_nll = use_nll

        self.lambda_tri = lambda_tri
        self.lambda_sharp = lambda_sharp
        self.warmup_start = 8000
        self.warmup_steps = 8000

        #! Embedding layers
        # self.embedding_layer = nn.Embedding(21, emb_size)
        # self.W_bind = nn.Sequential(
        #     nn.Linear(emb_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        # )
        # self.W_tgt = nn.Sequential(
        #     nn.Linear(emb_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        # )

        #! Distance encoding
        num_gaussians = hidden_size
        self.gaussian_encoder = GaussianSmearing(
            start=0.0, stop=mean, num_gaussians=num_gaussians
        )
        num_centers = num_gaussians

        self.distance_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_centers, num_centers),
            nn.LayerNorm(num_centers),
            nn.SiLU(),
            nn.Linear(num_centers, 1),
        )

        #! Residue encoder
        self.enc_ipa_mixer = ProteinEncoder(
            hidden_size=hidden_size,
            emb_size=emb_size,
            heads=heads,
            dropout=dropout,
            depth=4,
        )
        self.enc_cplx_mixer = CustomTransformer(
            dim=hidden_size, heads=heads, mlp_ratio=4, depth=6, dropout=dropout
        )

        self.bind_enc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.tgt_enc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proj_to_latent = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, latent_dim * 2),
        )

        #! Decoder stack
        self.up_project = nn.Linear(latent_dim, hidden_size)
        self.dec_ipa_mixer = ProteinEncoder(
            hidden_size=hidden_size,
            emb_size=emb_size,
            heads=heads,
            dropout=dropout,
            depth=4,
        )
        self.post_mixer = CustomTransformer(
            dim=hidden_size, heads=heads, mlp_ratio=4, depth=6, dropout=dropout
        )

        self.bind_z_final = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.tgt_z_final = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        #! Distance predictor
        self.inter_query = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            # nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

        end_dim = 2 if use_nll else 1

        self.dist_predictor = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, end_dim),
        )

        self.aa_predictor = nn.Linear(hidden_size, 21)

    def encode(
        self, x_dists: Tensor, binder: Features, target: Features
    ) -> Tuple[Tensor, Tensor]:
        """
        x_dists: (B, 1, M, N) normalized distances
        binder: Features for binder
        target: Features for target
        binder_H: (B, M, H) binder features
        target_H: (B, N, H) target features
        returns mu, logvar
        """
        M = x_dists.size(1)

        binder_H, target_H = self.enc_ipa_mixer(binder, target)

        cplx_dists = self.gaussian_encoder(
            x_dists.squeeze(1)  # (B, M+N, M+N, num_centers)
        )

        # Distance embeddings
        bind_dist_H, tgt_dist_H = self.get_internal_dists(binder, target)
        cplx_dist_H = self.distance_mlp(cplx_dists).squeeze(-1)  # (B, M, N)
        # cplx_dist_H = torch.zeros(
        #     (binder.X.size(0), M, target.X.size(1)), device=x_dists.device
        # )

        # bind_attn_bias, tgt_attn_bias = get_single_attn_bias(
        #     binder,
        #     target,
        #     bind_dists=bind_dist_H,
        #     target_dists=tgt_dist_H,
        # )
        cplx_attn_bias = get_cplx_attn_bias(
            binder,
            target,
            bind_dists=bind_dist_H,
            target_dists=tgt_dist_H,
            inter_dists=cplx_dist_H,
        )

        # extract distance embeddings between binder and target
        # binder_H_pre = self.enc_bind_mixer(binder_H, attn_mask=bind_attn_bias)
        # target_H_pre = self.enc_tgt_mixer(target_H, attn_mask=tgt_attn_bias)
        binder_H_pre = binder_H
        target_H_pre = target_H
        cplx_H = self.enc_cplx_mixer(
            torch.cat((binder_H, target_H), dim=1), attn_mask=cplx_attn_bias
        )
        binder_H, target_H = cplx_H[:, :M, :], cplx_H[:, M:, :]

        # Residual connection
        binder_H = binder_H + binder_H_pre
        target_H = target_H + target_H_pre

        binder_H = self.bind_enc(binder_H)  # (B, M, H)
        target_H = self.tgt_enc(target_H)  # (B, N, H)

        # Final projection to latent
        cplx_H = torch.cat((binder_H, target_H), dim=1)  # (B, M+N, H)
        pre_latent = self.proj_to_latent(cplx_H)  # (B, M+N, latent_dim*2)

        mu, logvar = pre_latent.chunk(2, dim=-1)  # each (B, M+N, latent_dim)

        return mu, logvar  # each (B, M+N, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)  # [B, M+N, latent_dim]

    def get_internal_dists(self, binder: Features, target: Features):
        bind_internal_dist = torch.cdist(binder.X[:, :, 1], binder.X[:, :, 1])
        target_internal_dist = torch.cdist(target.X[:, :, 1], target.X[:, :, 1])

        # bind_internal_dist = (bind_internal_dist - self.mean) / self.std
        # target_internal_dist = (target_internal_dist - self.mean) / self.std

        bind_dist_H = self.distance_mlp(
            self.gaussian_encoder(bind_internal_dist)
        ).squeeze(-1)
        tgt_dist_H = self.distance_mlp(
            self.gaussian_encoder(target_internal_dist)
        ).squeeze(-1)

        return bind_dist_H, tgt_dist_H  # each (B, L, L)

    def decode(
        self,
        z: Tensor,
        binder: Features,
        target: Features,
        upscale=False,
        return_aa_logits=False,
    ) -> Tensor:
        """
        If self.use_nll:
            returns (mu, log_sigma) in either normalized space (upscale=False)
            or physical space (upscale=True), and optionally aa_logits.
        Else:
            returns x_recon (same as before), and optionally aa_logits.
        """
        M, N = binder.X.size(1), target.X.size(1)

        # Project up to model dim
        z_pre = self.up_project(z)  # (B, M+N, H)

        binder_z, target_z = z_pre[:, :M, :], z_pre[:, M:, :]  # (B, M, H), (B, N, H)
        binder_z, target_z = self.dec_ipa_mixer(
            binder, target, bind_H=binder_z, tgt_H=target_z
        )  # each (B, L, H)
        z_pre = torch.cat([binder_z, target_z], dim=1)  # (B, M+N, H)

        # Get inter distance score
        inter_score = self.inter_query(z_pre)  # (B, M+N, 1)
        bind_inter_score, tgt_inter_score = inter_score[:, :M], inter_score[:, M:]
        inter_score = (
            bind_inter_score.unsqueeze(2) * tgt_inter_score.unsqueeze(1)
        ).squeeze(-1)  # (B, M, N)

        # Get internal distance embeddings
        bind_dist_H, target_dist_H = self.get_internal_dists(binder, target)

        cplx_attn_bias = get_cplx_attn_bias(
            binder,
            target,
            bind_dists=bind_dist_H,
            target_dists=target_dist_H,
            inter_dists=inter_score,
        )

        z = self.post_mixer(z_pre, attn_mask=cplx_attn_bias)  # (B, M+N, H)
        z = z + z_pre  # residual

        binder_z, target_z = z[:, :M, :], z[:, M:, :]  # (B, M, H), (B, N, H)

        binder_z = self.bind_z_final(binder_z)  # (B, M, H)
        target_z = self.tgt_z_final(target_z)  # (B, N, H)

        cplx_z = binder_z.unsqueeze(2) * target_z.unsqueeze(1)  # (B, M, N, H)

        dist_z = cplx_z * inter_score.unsqueeze(-1)
        dist_z = torch.cat(
            [
                dist_z,
                binder_z.unsqueeze(2).expand(-1, -1, N, -1),
                target_z.unsqueeze(1).expand(-1, M, -1, -1),
            ],
            dim=-1,
        )  # (B, M, N, 3H)

        dist_out = self.dist_predictor(dist_z)  # (B, M, N, 1) or (B, M, N, 2)

        aa_logits = self.aa_predictor(z)  # (B, M+N, 21)

        # ---------- Heteroscedastic head ----------
        if self.use_nll:
            mu = dist_out[..., 0]  # (B, M, N)
            log_sigma = dist_out[..., 1]  # (B, M, N)

            if upscale:
                # mu_phys = mu_norm * std + mean
                mu = (mu * self.std) + self.mean
                # sigma_phys = sigma_norm * std  => log_sigma_phys = log_sigma_norm + log(std)
                log_sigma = log_sigma + math.log(self.std + 1e-8)

            if return_aa_logits:
                return (mu, log_sigma), aa_logits

            return mu, log_sigma

        # ---------- Deterministic head (original behavior) ----------
        x_recon = dist_out.squeeze(-1)  # (B, M, N)

        if upscale:
            x_recon = (x_recon * self.std) + self.mean

        if return_aa_logits:
            return x_recon, aa_logits

        return x_recon

    def get_latents(
        self, x: Tensor, binder: Features, target: Features, pair_mask: Tensor
    ) -> Tensor:
        """
        returns:
            z: (B, M+N, latent_dim) latent codes for each residue
            binder_H: (B, M, H) binder features
            target_H: (B, N, H) target features
        """
        x_phys = x

        # Normalize distances (for reconstruction)
        x_norm = (x_phys - self.mean) / self.std  # (B,M,N)
        x_img = x_norm.unsqueeze(1)  # (B,1,M,N)

        # Encode
        mu, logvar = self.encode(binder, x_img, pair_mask)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        return z, binder_H, target_H

    def get_f1(self, logits, labels, mask):
        """
        logits: (B, L, 21)
        labels: (B, L)
        mask: (B, L) bool
        returns:
            f1_score: float
        """
        preds = logits.argmax(dim=-1)
        recall = (
            ((preds == labels) & mask).sum(dim=1).float()
            / mask.float().sum(dim=1).clamp(min=1.0)
        ).mean()
        precision = (
            ((preds == labels) & mask).sum(dim=1).float()
            / (preds != 20).sum(dim=1).clamp(min=1.0).float()
        ).mean()
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        return f1_score

    def get_aa_f1_score(
        self, aa_logits: Tensor, binder: Features, target: Features
    ) -> float:
        """
        aa_logits: (B, M+N, 21) predicted AA logits
        binder: Features for binder
        target: Features for target
        returns:
            f1_score: float
        """
        M = binder.X.size(1)
        aa_bind_logits = aa_logits[:, :M, :]
        aa_tgt_logits = aa_logits[:, M:, :]

        bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()  # (B, M, 1)
        tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()

        aa_bind_labels = binder.Si.squeeze(-1)  # (B, M)
        aa_tgt_labels = target.Si.squeeze(-1)  # (B, N)

        bind_f1 = self.get_f1(aa_bind_logits, aa_bind_labels, bind_mask.squeeze(-1))
        tgt_f1 = self.get_f1(aa_tgt_logits, aa_tgt_labels, tgt_mask.squeeze(-1))

        f1_score = (bind_f1 + tgt_f1) / 2.0

        return f1_score

    def forward(
        self,
        x: Tensor,  # (B, M, N) raw distances (Å)
        binder: Features,
        target: Features,
        pair_mask: Tensor,  # (B, M, N) bool or float
        step=0,
        return_latent=False,
        return_recon=False,
    ):
        assert x.dim() == 3, f"Expected x of shape (B,M,N), got {x.shape}"

        if pair_mask.dtype != torch.float32:
            mask = pair_mask.float()
        else:
            mask = pair_mask

        # # Get AA embeddings
        # binder_H = self.W_bind(binder.S)  # (B, M, H)
        # target_H = self.W_tgt(target.S)  # (B, N, H)

        x_phys = x
        x_norm = (x_phys - self.mean) / self.std  # (B, M, N)

        # Residue-valid masks
        bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()  # (B, M, 1)
        tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()  # (B, N, 1)
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)  # (B, M, N)
        pair_mask_f = pair_mask.float()

        # Encode (you currently pass x_img as x_phys; keep as-is)
        x_img = x_phys
        mu, logvar = self.encode(x_img, binder, target)

        z = self.reparameterize(mu, logvar)

        if return_latent:
            return z

        # ---- Decode in NORMALIZED space for losses ----
        if self.use_nll:
            (x_mu, x_log_sigma), aa_recon = self.decode(
                z, binder, target, upscale=False, return_aa_logits=True
            )  # each (B, M, N)

            # masked heteroscedastic Gaussian NLL in normalized space
            x_log_sigma = x_log_sigma.clamp(min=-7.0, max=7.0)
            inv_var = torch.exp(-2.0 * x_log_sigma)  # 1/sigma^2

            # 0.5 * [ (err^2)/sigma^2 + 2log(sigma) + log(2pi) ]
            err2 = (x_norm - x_mu) ** 2
            nll = 0.5 * (err2 * inv_var + 2.0 * x_log_sigma + math.log(2.0 * math.pi))

            recon_loss = (nll * pair_mask_f).sum(dim=(1, 2)) / pair_mask_f.sum(
                dim=(1, 2)
            ).clamp(min=1.0)
            recon_loss = recon_loss.mean()

            # For geometry / return: convert mean to physical once
            x_recon_phys = (x_mu * self.std) + self.mean
            uncertainty = torch.exp(x_log_sigma) * self.std  # physical space

        else:
            x_recon_norm, aa_recon = self.decode(
                z, binder, target, upscale=False, return_aa_logits=True
            )  # (B, M, N) normalized

            # Charbonnier recon in normalized space
            eps = 1e-6
            diff = x_recon_norm - x_norm
            charb = torch.sqrt(diff * diff + eps)
            recon_loss = (charb * pair_mask_f).sum(dim=(1, 2)) / pair_mask_f.sum(
                dim=(1, 2)
            ).clamp(min=1.0)
            recon_loss = recon_loss.mean()

            # convert once for geometry / return
            x_recon_phys = (x_recon_norm * self.std) + self.mean

        # ---------------- AA reconstruction loss (unchanged) ----------------
        aa_bind_logits = aa_recon[:, : binder.X.size(1), :]  # (B, M, 21)
        aa_tgt_logits = aa_recon[:, binder.X.size(1) :, :]  # (B, N, 21)
        aa_bind_labels = binder.Si.squeeze(-1)  # (B, M)
        aa_tgt_labels = target.Si.squeeze(-1)  # (B, N)

        aa_bind_loss = F.cross_entropy(
            aa_bind_logits.permute(0, 2, 1), aa_bind_labels, reduction="none"
        )
        aa_tgt_loss = F.cross_entropy(
            aa_tgt_logits.permute(0, 2, 1), aa_tgt_labels, reduction="none"
        )

        aa_bind_loss = (aa_bind_loss * bind_mask.squeeze(-1).float()).sum(
            dim=1
        ) / bind_mask.float().sum(dim=1).clamp(min=1.0)
        aa_tgt_loss = (aa_tgt_loss * tgt_mask.squeeze(-1).float()).sum(
            dim=1
        ) / tgt_mask.float().sum(dim=1).clamp(min=1.0)

        aa_recon_loss = aa_bind_loss.mean() + aa_tgt_loss.mean()

        # ---------------- Geometry / sharpness regularizers ----------------
        triangle_ineq_loss = triangle_loss_efficient(
            binder, target, x_recon_phys.clamp(max=30), mask
        )
        sharp_loss = torch.tensor(0.0, device=x.device)

        # ---------------- Per-Channel KL term (unchanged) ----------------
        valid_mask = torch.cat((bind_mask, tgt_mask), dim=1)  # (B, M+N)
        mu = mu * valid_mask.unsqueeze(-1)
        logvar = logvar * valid_mask.unsqueeze(-1)

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = (kl_loss * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum().clamp(
            min=1.0
        )
        kl_loss = self.kl_weight * kl_loss

        lambda_scale = kl_sigmoid_anneal(step) if step >= self.warmup_start else 0.0
        triangle_ineq_loss = lambda_scale * triangle_ineq_loss
        kl_loss = lambda_scale * kl_loss

        loss = (
            recon_loss
            + 0.001 * aa_recon_loss
            + kl_loss
            + self.lambda_tri * triangle_ineq_loss
            + self.lambda_sharp * sharp_loss
        )

        aa_f1 = self.get_aa_f1_score(aa_recon, binder, target)

        losses = {
            "loss": loss,
            "recon_loss": recon_loss,
            "aa_recon_loss": aa_recon_loss,
            "aa_f1": aa_f1,
            "kl_loss": kl_loss,
            "triangle_ineq_loss": triangle_ineq_loss,
        }

        if return_recon:
            return losses, x_recon_phys, uncertainty

        return losses


class CVAE(nn.Module):
    """
    Conditional VAE over *continuous* distance matrices:
      x: (B, M, N) of real-valued distances.

    pair_mask: (B, M, N) boolean mask of valid entries.

    This version includes:
      - NO convolutions
      - Charbonnier reconstruction loss
      - Triangle inequality & sharpness regularizers
    """

    def __init__(
        self,
        kl_weight: float,
        mean: float,
        std: float,
        hidden_size=128,
        latent_dim=8,
        heads=8,
        lambda_tri=0.0001,
        lambda_sharp=0.0,
        use_symmetry=False,
        emb_size=768,
        pred_type="direct",
        use_nll=False,
        dropout=0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.kl_weight = kl_weight
        self.mean = float(mean)
        self.std = float(std)
        self.use_symmetry = use_symmetry
        self.pred_type = pred_type
        self.heads = heads
        self.use_nll = use_nll

        self.lambda_tri = lambda_tri
        self.lambda_sharp = lambda_sharp
        self.warmup_start = 8000
        self.warmup_steps = 8000

        #! Embedding layers
        # self.embedding_layer = nn.Embedding(21, emb_size)
        # self.W_bind = nn.Sequential(
        #     nn.Linear(emb_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        # )
        # self.W_tgt = nn.Sequential(
        #     nn.Linear(emb_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        # )

        #! Distance encoding
        num_gaussians = hidden_size
        self.gaussian_encoder = GaussianSmearing(
            start=0.0, stop=mean, num_gaussians=num_gaussians
        )
        num_centers = num_gaussians

        self.distance_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_centers, num_centers),
            nn.LayerNorm(num_centers),
            nn.SiLU(),
            nn.Linear(num_centers, 1),
        )

        #! Residue encoder
        self.enc_ipa_mixer = ProteinEncoder(
            hidden_size=hidden_size,
            emb_size=emb_size,
            heads=heads,
            dropout=dropout,
            depth=6,
        )
        self.enc_cplx_mixer = CustomTransformer(
            dim=hidden_size, heads=heads, mlp_ratio=4, depth=8, dropout=dropout
        )

        self.bind_enc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.tgt_enc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proj_to_latent = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, latent_dim * 2),
        )

        #! Decoder stack
        self.up_project = nn.Linear(latent_dim, hidden_size)
        self.bind_emb_proj = nn.Linear(emb_size, hidden_size)
        self.tgt_emb_proj = nn.Linear(emb_size, hidden_size)
        self.dec_ipa_mixer = ProteinEncoder(
            hidden_size=hidden_size,
            emb_size=emb_size,
            heads=heads,
            dropout=dropout,
            depth=6,
        )
        self.post_mixer = CustomTransformer(
            dim=hidden_size, heads=heads, mlp_ratio=4, depth=8, dropout=dropout
        )

        self.bind_z_final = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.tgt_z_final = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )

        #! Distance predictor
        self.inter_query = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            # nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

        end_dim = 2 if use_nll else 1

        self.dist_predictor = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, end_dim),
        )

        self.aa_predictor = nn.Linear(hidden_size, 21)

    def encode(
        self, x_dists: Tensor, binder: Features, target: Features, return_Hs=False
    ) -> Tuple[Tensor, Tensor]:
        """
        x_dists: (B, 1, M, N) normalized distances
        binder: Features for binder
        target: Features for target
        binder_H: (B, M, H) binder features
        target_H: (B, N, H) target features
        returns mu, logvar
        """
        M = x_dists.size(1)

        binder_H_first, target_H_first = self.enc_ipa_mixer(binder, target)

        # cplx_dists = self.gaussian_encoder(
        #     x_dists.squeeze(1)  # (B, M+N, M+N, num_centers)
        # )

        # Distance embeddings
        bind_dist_H, tgt_dist_H = self.get_internal_dists(binder, target)
        # cplx_dist_H = self.distance_mlp(cplx_dists).squeeze(-1)  # (B, M, N)
        cplx_dist_H = binder_H_first.unsqueeze(2) * target_H_first.unsqueeze(1)
        cplx_dist_H = self.distance_mlp(cplx_dist_H).squeeze(-1)

        # cplx_dist_H = torch.zeros(
        #     (binder.X.size(0), M, target.X.size(1)), device=x_dists.device
        # )

        # bind_attn_bias, tgt_attn_bias = get_single_attn_bias(
        #     binder,
        #     target,
        #     bind_dists=bind_dist_H,
        #     target_dists=tgt_dist_H,
        # )
        cplx_attn_bias = get_cplx_attn_bias(
            binder,
            target,
            bind_dists=bind_dist_H,
            target_dists=tgt_dist_H,
            inter_dists=cplx_dist_H,
        )

        # extract distance embeddings between binder and target
        # binder_H_pre = self.enc_bind_mixer(binder_H, attn_mask=bind_attn_bias)
        # target_H_pre = self.enc_tgt_mixer(target_H, attn_mask=tgt_attn_bias)
        cplx_H = self.enc_cplx_mixer(
            torch.cat((binder_H_first, target_H_first), dim=1), attn_mask=cplx_attn_bias
        )
        binder_H, target_H = cplx_H[:, :M, :], cplx_H[:, M:, :]

        # Residual connection
        binder_H = binder_H + binder_H_first
        target_H = target_H + target_H_first

        binder_H = self.bind_enc(binder_H)  # (B, M, H)
        target_H = self.tgt_enc(target_H)  # (B, N, H)

        # Final projection to latent
        cplx_H = torch.cat((binder_H, target_H), dim=1)  # (B, M+N, H)
        pre_latent = self.proj_to_latent(cplx_H)  # (B, M+N, latent_dim*2)

        mu, logvar = pre_latent.chunk(2, dim=-1)  # each (B, M+N, latent_dim)

        if return_Hs:
            return mu, logvar, binder_H, target_H

        return mu, logvar  # each (B, M+N, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)  # [B, M+N, latent_dim]

    def get_internal_dists(self, binder: Features, target: Features):
        bind_internal_dist = torch.cdist(binder.X[:, :, 1], binder.X[:, :, 1])
        target_internal_dist = torch.cdist(target.X[:, :, 1], target.X[:, :, 1])

        # bind_internal_dist = (bind_internal_dist - self.mean) / self.std
        # target_internal_dist = (target_internal_dist - self.mean) / self.std

        bind_dist_H = self.distance_mlp(
            self.gaussian_encoder(bind_internal_dist)
        ).squeeze(-1)
        tgt_dist_H = self.distance_mlp(
            self.gaussian_encoder(target_internal_dist)
        ).squeeze(-1)

        return bind_dist_H, tgt_dist_H  # each (B, L, L)

    def decode(
        self,
        z: Tensor,
        binder: Features,
        target: Features,
        upscale=False,
        return_aa_logits=False,
    ) -> Tensor:
        """
        If self.use_nll:
            returns (mu, log_sigma) in either normalized space (upscale=False)
            or physical space (upscale=True), and optionally aa_logits.
        Else:
            returns x_recon (same as before), and optionally aa_logits.
        """
        M, N = binder.X.size(1), target.X.size(1)

        # Project up to model dim
        z_pre = self.up_project(z)  # (B, M+N, H)

        binder_S, target_S = binder.S, target.S  # [B, M, emb_size], [B, N, emb_size]
        binder_emb = self.bind_emb_proj(binder_S)  # (B, M, H)
        target_emb = self.tgt_emb_proj(target_S)  # (B, N, H)
        binder_z, target_z = z_pre[:, :M, :], z_pre[:, M:, :]  # (B, M, H), (B, N, H)

        # Mix in E1 sequence embeddings
        binder_z = binder_z + binder_emb
        target_z = target_z + target_emb

        binder_z, target_z = self.dec_ipa_mixer(
            binder, target, bind_H=binder_z, tgt_H=target_z
        )  # each (B, L, H)
        z_pre = torch.cat([binder_z, target_z], dim=1)  # (B, M+N, H)

        # Get inter distance score
        inter_score = self.inter_query(z_pre)  # (B, M+N, 1)
        bind_inter_score, tgt_inter_score = inter_score[:, :M], inter_score[:, M:]
        inter_score = (
            bind_inter_score.unsqueeze(2) * tgt_inter_score.unsqueeze(1)
        ).squeeze(-1)  # (B, M, N)

        # Get internal distance embeddings
        bind_dist_H, target_dist_H = self.get_internal_dists(binder, target)

        cplx_attn_bias = get_cplx_attn_bias(
            binder,
            target,
            bind_dists=bind_dist_H,
            target_dists=target_dist_H,
            inter_dists=inter_score,
        )

        z = self.post_mixer(z_pre, attn_mask=cplx_attn_bias)  # (B, M+N, H)
        z = z + z_pre  # residual

        binder_z, target_z = z[:, :M, :], z[:, M:, :]  # (B, M, H), (B, N, H)

        binder_z = self.bind_z_final(binder_z)  # (B, M, H)
        target_z = self.tgt_z_final(target_z)  # (B, N, H)

        cplx_z = binder_z.unsqueeze(2) * target_z.unsqueeze(1)  # (B, M, N, H)

        dist_z = cplx_z * inter_score.unsqueeze(-1)
        dist_z = torch.cat(
            [
                dist_z,
                binder_z.unsqueeze(2).expand(-1, -1, N, -1),
                target_z.unsqueeze(1).expand(-1, M, -1, -1),
            ],
            dim=-1,
        )  # (B, M, N, 3H)

        dist_out = self.dist_predictor(dist_z)  # (B, M, N, 1) or (B, M, N, 2)

        aa_logits = self.aa_predictor(z)  # (B, M+N, 21)

        # ---------- Heteroscedastic head ----------
        if self.use_nll:
            mu = dist_out[..., 0]  # (B, M, N)
            log_sigma = dist_out[..., 1]  # (B, M, N)

            if upscale:
                # mu_phys = mu_norm * std + mean
                mu = (mu * self.std) + self.mean
                # sigma_phys = sigma_norm * std  => log_sigma_phys = log_sigma_norm + log(std)
                log_sigma = log_sigma + math.log(self.std + 1e-8)

            if return_aa_logits:
                return (mu, log_sigma), aa_logits

            return mu, log_sigma

        # ---------- Deterministic head (original behavior) ----------
        x_recon = dist_out.squeeze(-1)  # (B, M, N)

        if upscale:
            x_recon = (x_recon * self.std) + self.mean

        if return_aa_logits:
            return x_recon, aa_logits

        return x_recon

    def get_diff_conditioning(self, binder: Features, target: Features):
        """
        Same as encoder, but does NOT include ground-truth distance info.
        returns:
            binder_H: (B, M, H) binder features
            target_H: (B, N, H) target features
        """
        M = binder.X.size(1)

        binder_H_first, target_H_first = self.enc_ipa_mixer(binder, target)
        bind_dist_H, tgt_dist_H = self.get_internal_dists(binder, target)
        cplx_attn_bias = get_cplx_attn_bias(
            binder,
            target,
            bind_dists=bind_dist_H,
            target_dists=tgt_dist_H,
            inter_dists=None,
        )
        cplx_H = self.enc_cplx_mixer(
            torch.cat((binder_H_first, target_H_first), dim=1), attn_mask=cplx_attn_bias
        )
        binder_H, target_H = cplx_H[:, :M, :], cplx_H[:, M:, :]

        # Residual connection
        binder_H = binder_H + binder_H_first
        target_H = target_H + target_H_first

        binder_H = self.bind_enc(binder_H)  # (B, M, H)
        target_H = self.tgt_enc(target_H)  # (B, N, H)

        return binder_H, target_H

    def get_latents(self, x: Tensor, binder: Features, target: Features) -> Tensor:
        """
        returns:
            z: (B, M+N, latent_dim) latent codes for each residue
            binder_H: (B, M, H) binder features
            target_H: (B, N, H) target features
        """

        # Normalize distances (for reconstruction)
        # x_norm = (x_phys - self.mean) / self.std  # (B,M,N)
        # x_img = x_norm.unsqueeze(1)  # (B,1,M,N)
        # x_img = x_phys.unsqueeze(1)  # (B,1,M,N)

        # Encode
        mu, logvar = self.encode(x, binder, target)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        return z

    def get_f1(self, logits, labels, mask):
        """
        logits: (B, L, 21)
        labels: (B, L)
        mask: (B, L) bool
        returns:
            f1_score: float
        """
        preds = logits.argmax(dim=-1)
        recall = (
            ((preds == labels) & mask).sum(dim=1).float()
            / mask.float().sum(dim=1).clamp(min=1.0)
        ).mean()
        precision = (
            ((preds == labels) & mask).sum(dim=1).float()
            / (preds != 20).sum(dim=1).clamp(min=1.0).float()
        ).mean()
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        return f1_score

    def get_aa_f1_score(
        self, aa_logits: Tensor, binder: Features, target: Features
    ) -> float:
        """
        aa_logits: (B, M+N, 21) predicted AA logits
        binder: Features for binder
        target: Features for target
        returns:
            f1_score: float
        """
        M = binder.X.size(1)
        aa_bind_logits = aa_logits[:, :M, :]
        aa_tgt_logits = aa_logits[:, M:, :]

        bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()  # (B, M, 1)
        tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()

        aa_bind_labels = binder.Si.squeeze(-1)  # (B, M)
        aa_tgt_labels = target.Si.squeeze(-1)  # (B, N)

        bind_f1 = self.get_f1(aa_bind_logits, aa_bind_labels, bind_mask.squeeze(-1))
        tgt_f1 = self.get_f1(aa_tgt_logits, aa_tgt_labels, tgt_mask.squeeze(-1))

        f1_score = (bind_f1 + tgt_f1) / 2.0

        return f1_score

    def forward(
        self,
        x: Tensor,  # (B, M, N) raw distances (Å)
        binder: Features,
        target: Features,
        pair_mask: Tensor,  # (B, M, N) bool or float
        step=0,
        return_latent=False,
        return_recon=False,
    ):
        assert x.dim() == 3, f"Expected x of shape (B,M,N), got {x.shape}"

        if pair_mask.dtype != torch.float32:
            mask = pair_mask.float()
        else:
            mask = pair_mask

        # # Get AA embeddings
        # binder_H = self.W_bind(binder.S)  # (B, M, H)
        # target_H = self.W_tgt(target.S)  # (B, N, H)

        x_phys = x
        x_norm = (x_phys - self.mean) / self.std  # (B, M, N)

        # Residue-valid masks
        bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()  # (B, M, 1)
        tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()  # (B, N, 1)
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)  # (B, M, N)
        pair_mask_f = pair_mask.float()

        # Encode (you currently pass x_img as x_phys; keep as-is)
        x_img = x_phys
        mu, logvar = self.encode(x_img, binder, target)

        z = self.reparameterize(mu, logvar)

        if return_latent:
            return z

        # ---- Decode in NORMALIZED space for losses ----
        if self.use_nll:
            (x_mu, x_log_sigma), aa_recon = self.decode(
                z, binder, target, upscale=False, return_aa_logits=True
            )  # each (B, M, N)

            # masked heteroscedastic Gaussian NLL in normalized space
            x_log_sigma = x_log_sigma.clamp(min=-7.0, max=7.0)
            inv_var = torch.exp(-2.0 * x_log_sigma)  # 1/sigma^2

            # 0.5 * [ (err^2)/sigma^2 + 2log(sigma) + log(2pi) ]
            err2 = (x_norm - x_mu) ** 2
            nll = 0.5 * (err2 * inv_var + 2.0 * x_log_sigma + math.log(2.0 * math.pi))

            recon_loss = (nll * pair_mask_f).sum(dim=(1, 2)) / pair_mask_f.sum(
                dim=(1, 2)
            ).clamp(min=1.0)
            recon_loss = recon_loss.mean()

            # For geometry / return: convert mean to physical once
            x_recon_phys = (x_mu * self.std) + self.mean
            uncertainty = torch.exp(x_log_sigma) * self.std  # physical space

        else:
            x_recon_norm, aa_recon = self.decode(
                z, binder, target, upscale=False, return_aa_logits=True
            )  # (B, M, N) normalized

            # Charbonnier recon in normalized space
            eps = 1e-6
            diff = x_recon_norm - x_norm
            charb = torch.sqrt(diff * diff + eps)
            recon_loss = (charb * pair_mask_f).sum(dim=(1, 2)) / pair_mask_f.sum(
                dim=(1, 2)
            ).clamp(min=1.0)
            recon_loss = recon_loss.mean()

            # convert once for geometry / return
            x_recon_phys = (x_recon_norm * self.std) + self.mean

            uncertainty = None

        # ---------------- AA reconstruction loss (unchanged) ----------------
        aa_bind_logits = aa_recon[:, : binder.X.size(1), :]  # (B, M, 21)
        aa_tgt_logits = aa_recon[:, binder.X.size(1) :, :]  # (B, N, 21)
        aa_bind_labels = binder.Si.squeeze(-1)  # (B, M)
        aa_tgt_labels = target.Si.squeeze(-1)  # (B, N)

        aa_bind_loss = F.cross_entropy(
            aa_bind_logits.permute(0, 2, 1), aa_bind_labels, reduction="none"
        )
        aa_tgt_loss = F.cross_entropy(
            aa_tgt_logits.permute(0, 2, 1), aa_tgt_labels, reduction="none"
        )

        aa_bind_loss = (aa_bind_loss * bind_mask.squeeze(-1).float()).sum(
            dim=1
        ) / bind_mask.float().sum(dim=1).clamp(min=1.0)
        aa_tgt_loss = (aa_tgt_loss * tgt_mask.squeeze(-1).float()).sum(
            dim=1
        ) / tgt_mask.float().sum(dim=1).clamp(min=1.0)

        aa_recon_loss = aa_bind_loss.mean() + aa_tgt_loss.mean()

        # ---------------- Geometry / sharpness regularizers ----------------
        triangle_ineq_loss = triangle_loss_efficient(
            binder, target, x_recon_phys.clamp(max=30), mask
        )
        sharp_loss = torch.tensor(0.0, device=x.device)

        # ---------------- Per-Channel KL term (unchanged) ----------------
        valid_mask = torch.cat((bind_mask, tgt_mask), dim=1)  # (B, M+N)
        mu = mu * valid_mask.unsqueeze(-1)
        logvar = logvar * valid_mask.unsqueeze(-1)

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = (kl_loss * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum().clamp(
            min=1.0
        )
        kl_loss = self.kl_weight * kl_loss

        lambda_scale = kl_sigmoid_anneal(step) if step >= self.warmup_start else 0.0
        triangle_ineq_loss = lambda_scale * triangle_ineq_loss
        kl_loss = lambda_scale * kl_loss

        loss = (
            recon_loss
            + 0.005 * aa_recon_loss
            + kl_loss
            + self.lambda_tri * triangle_ineq_loss
            + self.lambda_sharp * sharp_loss
        )

        aa_f1 = self.get_aa_f1_score(aa_recon, binder, target)

        losses = {
            "loss": loss,
            "recon_loss": recon_loss,
            "aa_recon_loss": aa_recon_loss,
            "aa_f1": aa_f1,
            "kl_loss": kl_loss,
            "triangle_ineq_loss": triangle_ineq_loss,
        }

        if return_recon:
            return losses, x_recon_phys, uncertainty

        return losses
