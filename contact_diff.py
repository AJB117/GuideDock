import os
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform.rotation import Rotation as Rot
import matplotlib.pyplot as plt

# ============================================================================
# PDB Parsing
# ============================================================================


def parse_pdb(
    pdb_file: str, backbone_atoms: List[str] = ["N", "CA", "C", "O"]
) -> np.ndarray:
    """
    Parse PDB file and extract backbone atom coordinates.

    Args:
        pdb_file: Path to PDB file
        backbone_atoms: List of atom names to extract (default: N, CA, C)

    Returns:
        coords: [M, K, 3] where M is number of residues, K is number of backbone atoms
    """
    residues = {}

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name not in backbone_atoms:
                    continue

                try:
                    # Extract residue info
                    chain = line[21].strip()
                    res_num = int(line[22:26].strip())
                    res_key = (chain, res_num)

                    # Extract coordinates
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    # Store
                    if res_key not in residues:
                        residues[res_key] = {}
                    residues[res_key][atom_name] = np.array([x, y, z])
                except (ValueError, IndexError) as e:
                    continue

    # Convert to array format
    sorted_residues = sorted(residues.keys())
    coords_list = []

    for res_key in sorted_residues:
        res_atoms = []
        for atom_name in backbone_atoms:
            if atom_name in residues[res_key]:
                res_atoms.append(residues[res_key][atom_name])
            else:
                # If atom is missing, use NaN (will be filtered later)
                res_atoms.append(np.array([np.nan, np.nan, np.nan]))
        coords_list.append(res_atoms)

    coords = np.array(coords_list)  # [M, K, 3]

    # Filter out residues with missing atoms
    valid_residues = ~np.isnan(coords).any(axis=(1, 2))
    coords = coords[valid_residues]

    print(
        f"Parsed {pdb_file}: {coords.shape[0]} residues with {coords.shape[1]} backbone atoms"
    )

    return coords


# ============================================================================
# DDPM Model and Training
# ============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    # ensure groups divides num_channels
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding -> (B, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) or (B, 1) scalar timesteps (float or int)
        returns: (B, dim)
        """
        if t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        t = t.float()

        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -torch.arange(half, device=device) * (math.log(10000.0) / (half - 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb  # (B, dim)


class ResBlock(nn.Module):
    """ResNet block with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = _make_gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = _make_gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        t_emb: (B, time_dim)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # time embedding
        temb = self.time_proj(t_emb)  # (B, out_ch)
        h = h + temb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention over spatial dimensions."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _make_gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        h_ = self.norm(x)
        qkv = self.qkv(h_)
        q, k, v = qkv.reshape(b, 3, c, h * w).unbind(1)  # each: (B, C, HW)

        # scaled dot-product attention
        scale = 1.0 / math.sqrt(c)
        attn = torch.softmax((q.transpose(1, 2) @ k) * scale, dim=-1)  # (B, HW, HW)
        out = attn @ v.transpose(1, 2)  # (B, HW, C)
        out = out.transpose(1, 2).reshape(b, c, h, w)

        out = self.proj_out(out)
        return x + out


class SimpleUNet(nn.Module):
    """Stronger U-Net for denoising protein distance matrices (non-square OK)."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, time_dim: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        # Time embedding (sinusoidal + MLP)
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)

        self.enc1 = ResBlock(hidden_dim, hidden_dim, time_dim)
        self.down1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)

        self.enc2 = ResBlock(hidden_dim, hidden_dim * 2, time_dim)
        self.down2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=4, stride=2, padding=1)

        self.enc3 = ResBlock(hidden_dim * 2, hidden_dim * 4, time_dim)

        # Bottleneck
        self.bot1 = ResBlock(hidden_dim * 4, hidden_dim * 4, time_dim)
        self.attn = AttentionBlock(hidden_dim * 4)
        self.bot2 = ResBlock(hidden_dim * 4, hidden_dim * 4, time_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(hidden_dim * 4, hidden_dim * 2, time_dim)

        self.up2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(hidden_dim * 2, hidden_dim, time_dim)

        self.dec1 = ResBlock(hidden_dim, hidden_dim, time_dim)
        self.conv_out = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=in_channels, H, W)
        t: (B,) or (B, 1) diffusion timesteps / noise levels
        """
        orig_size = x.shape[-2:]

        # time embedding
        t_emb = self.time_embed(t)      # (B, time_dim)
        t_emb = self.time_mlp(t_emb)    # (B, time_dim)

        # Encoder
        x0 = self.conv_in(x)            # (B, hidden, H, W)

        e1 = self.enc1(x0, t_emb)       # (B, hidden, H, W)
        d1 = self.down1(e1)             # (B, hidden, H/2, W/2)

        e2 = self.enc2(d1, t_emb)       # (B, 2h, H/2, W/2)
        d2 = self.down2(e2)             # (B, 2h, H/4, W/4)

        e3 = self.enc3(d2, t_emb)       # (B, 4h, H/4, W/4)

        # Bottleneck
        b = self.bot1(e3, t_emb)
        b = self.attn(b)
        b = self.bot2(b, t_emb)

        # Decoder
        u3 = self.up3(b)                # (B, 2h, ~H/2, ~W/2)
        if u3.shape[-2:] != e2.shape[-2:]:
            u3 = F.interpolate(u3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u3 = torch.cat([u3, e2], dim=1)  # (B, 4h, H/2, W/2)
        u3 = self.dec3(u3, t_emb)        # (B, 2h, H/2, W/2)

        u2 = self.up2(u3)                # (B, h, ~H, ~W)
        if u2.shape[-2:] != e1.shape[-2:]:
            u2 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, e1], dim=1)  # (B, 2h, H, W)
        u2 = self.dec2(u2, t_emb)        # (B, h, H, W)

        u1 = self.dec1(u2, t_emb)        # (B, h, H, W)
        out = self.conv_out(u1)          # (B, in_channels, H, W)

        # Make sure size matches exactly (in case of odd sizes)
        if out.shape[-2:] != orig_size:
            out = F.interpolate(out, size=orig_size, mode="bilinear", align_corners=False)

        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class DDPM:
    """Denoising Diffusion Probabilistic Model for distance matrices."""

    def __init__(self, model, timesteps=1000, device="cpu"):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # Define beta schedule (linear)
        # self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.betas = torch.linspace(1e-4, 0.002, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process: add noise to x_0."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (
            (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        )

        return (
            sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_sample(self, x_t, t):
        """Reverse diffusion: denoise one step."""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (
            (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        )
        sqrt_recip_alphas_t = (1 / self.alphas[t]).sqrt().view(-1, 1, 1, 1)

        # Predict noise
        t_input = t.float() / self.timesteps
        predicted_noise = self.model(x_t, t_input)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            posterior_variance_t = betas_t
            return model_mean + posterior_variance_t.sqrt() * noise

    def sample(self, shape, num_samples=1):
        """Generate samples from noise."""
        self.model.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(num_samples, *shape).to(self.device)

            # Iteratively denoise
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((num_samples,), t, dtype=torch.long).to(
                    self.device
                )
                x = self.p_sample(x, t_batch)

            # Apply ReLU to ensure non-negative distances
            x = F.relu(x)
            return x


# ============================================================================
# Kabsch Algorithm
# ============================================================================


def kabsch_alignment(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal rotation and translation to align P onto Q using Kabsch algorithm.

    Args:
        P: Source coordinates [N, 3]
        Q: Target coordinates [N, 3]

    Returns:
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
    """
    # Center the point clouds
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_Q - R @ centroid_P

    return R, t

def se3_optimize_from_distances(
    binder_coords_init,     # [M,3] initial binder CA coordinates
    receptor_coords,        # [N,3] receptor CA coordinates
    distance_matrix,        # [M,N] predicted distances
    weights=None,           # [M,N] optional
    iters=1,
    lr=1e-4,
    device="cpu"
):
    """
    Optimize (R,t) such that || R b_i + t - r_j || ≈ D_ij.

    Returns binder_coords_opt, R(3,3), t(3).
    """

    # Move everything to torch
    binder = torch.tensor(binder_coords_init, device=device, dtype=torch.float32)
    receptor = torch.tensor(receptor_coords, device=device, dtype=torch.float32)
    D = torch.tensor(distance_matrix, device=device, dtype=torch.float32)

    M, N = D.shape

    if weights is None:
        # Default weighting: emphasize "close" distances, ignore far ones
        weights = torch.exp(-D / (D.mean() + 1e-6))
        weights = weights / (weights.sum() + 1e-6)

    weights = weights.to(device)

    # Optimization variables
    # Rotation represented as a quaternion
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, requires_grad=True)
    t = torch.zeros(3, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([q, t], lr=lr)

    def quat_to_rot(q):
        """Normalize and convert quaternion to rotation matrix."""
        q = q / q.norm()
        w, x, y, z = q
        R = torch.stack([
            torch.tensor([1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)]),
            torch.tensor([2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)]),
            torch.tensor([2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)])
        ])
        return R.to(device)

    for _ in range(iters):
        optimizer.zero_grad()

        R = quat_to_rot(q)                   # (3,3)
        B = (binder @ R.T) + t               # (M,3)

        # Compute predicted distances
        diff = B[:, None, :] - receptor[None, :, :]   # (M,N,3)
        dist_pred = torch.norm(diff, dim=-1)          # (M,N)

        # Loss: weighted L2 distance matching
        loss = (weights * (dist_pred - D)**2).sum()

        loss.backward()
        optimizer.step()

    # Return optimized binder coordinates
    R_final = quat_to_rot(q)
    t_final = t.detach()
    binder_opt = ((binder @ torch.tensor(R_final).T) + torch.tensor(t_final))

    return binder_opt, R_final, t_final

# ============================================================================
# Training and Sampling Pipeline
# ============================================================================


def compute_distance_matrix(
    binder_coords: np.ndarray, receptor_coords: np.ndarray
) -> np.ndarray:
    """
    Compute distance matrix between binder and receptor.

    Args:
        binder_coords: [M, K, 3]
        receptor_coords: [N, K, 3]

    Returns:
        distance_matrix: [M, N] pairwise distances in Angstroms
    """
    M, K, _ = binder_coords.shape
    N, _, _ = receptor_coords.shape

    # Use CA atoms (index 1) or first atom if K=1
    ca_idx = 1 if K > 1 else 0

    binder_ca = binder_coords[:, ca_idx, :]  # [M, 3]
    receptor_ca = receptor_coords[:, ca_idx, :]  # [N, 3]

    # Compute pairwise distances
    distances = np.linalg.norm(binder_ca[:, None, :] - receptor_ca[None, :, :], axis=2)

    return distances.astype(np.float32)


def train_ddpm(
    distance_matrices: List[np.ndarray], epochs=1000, batch_size=4, device="cpu"
):
    """Train DDPM on distance matrices."""
    # Normalize distance matrices to reasonable range
    all_distances = np.concatenate([dm.flatten() for dm in distance_matrices])
    # max_distance = np.percentile(all_distances, 95)  # Use 95th percentile to avoid outliers
    max_distance = all_distances.max()  # Use max distance for normalization

    # Normalize to [0, 1] range for training
    normalized_matrices = [dm / max_distance for dm in distance_matrices]
    distance_matrices_tensor = (
        torch.FloatTensor(np.array(normalized_matrices)).unsqueeze(1).to(device)
    )
    # repeat for batch size
    distance_matrices_tensor = distance_matrices_tensor.repeat(
        batch_size * 100, 1, 1, 1
    )

    print(
        f"Distance matrix range: {all_distances.min():.2f} - {all_distances.max():.2f} Å"
    )
    print(f"Normalizing by max distance: {max_distance:.2f} Å")

    # Initialize model
    model = SimpleUNet().to(device)
    ddpm = DDPM(model, timesteps=2000, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = len(distance_matrices_tensor) // batch_size
        for i in tqdm(range(num_batches)):
            batch = distance_matrices_tensor[i * batch_size : (i + 1) * batch_size]

            # Sample random timesteps
            t = torch.randint(0, ddpm.timesteps, (batch_size,)).to(device)

            # Generate noise
            noise = torch.randn_like(batch)

            # Forward diffusion
            x_t = ddpm.q_sample(batch, t, noise)

            # Predict noise
            t_input = t.float() / ddpm.timesteps
            predicted_noise = model(x_t, t_input)

            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # if (epoch + 1) % 0 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches:.4f}")

    # Store normalization factor
    ddpm.max_distance = max_distance
    torch.save(model.state_dict(), "ddpm_distance_model.pth")

    return ddpm


# ============================================================================
# Main Pipeline
# ============================================================================


def main_pipeline(
    binder_coords: np.ndarray,
    receptor_coords: np.ndarray,
    contact_threshold: float = 8.0,   # unused but kept for compatibility
    epochs: int = 100,
    num_samples: int = 5,
):
    """
    Complete pipeline:
      1. Compute true binder–receptor distances
      2. Train DDPM to model distance matrices
      3. Sample new predicted distance matrices
      4. Convert each predicted distance matrix -> SE(3) pose
         using gradient-based optimization on rigid-body transforms.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------------
    # 1. Compute distance matrix (CA)
    # -------------------------------
    print("\n1. Computing distance matrix from input coordinates...")
    dist0 = compute_distance_matrix(binder_coords, receptor_coords)
    print(f"Distance matrix shape: {dist0.shape}")
    print(f"Distance range: {dist0.min():.2f}–{dist0.max():.2f} Å")

    # -------------------------------
    # 2. Prepare dataset (single example)
    # -------------------------------
    print("\n2. Preparing training dataset...")
    training_matrices = [dist0]

    # -------------------------------
    # 3. Train DDPM
    # -------------------------------
    print(f"\n3. Training DDPM for {epochs} epochs...")
    # if not os.path.exists("ddpm_distance_model.pth"):
    ddpm = train_ddpm(training_matrices, epochs=epochs, batch_size=32, device=device)
    # else:
    #     ckpt = torch.load("ddpm_distance_model.pth", map_location=device)
    #     model = SimpleUNet().to(device)
    #     model.load_state_dict(ckpt)
    #     ddpm = DDPM(model, timesteps=1000, device=device)
    #     ddpm.max_distance = np.max(dist0)
    #     print("✓ DDPM model loaded.")

    # -------------------------------
    # 4. Sample distance matrices
    # -------------------------------
    print(f"\n4. Sampling {num_samples} distance matrices...")
    M, N = dist0.shape
    sampled = ddpm.sample(shape=(1, M, N), num_samples=num_samples)  # (S,1,M,N)
    sampled = sampled[:, 0, :, :].cpu().numpy()                       # (S,M,N)


    # Denormalize
    sampled = sampled * ddpm.max_distance
    MSE = np.mean((sampled - dist0[None, :, :])**2)
    # visualize one example
    print(f"Sampled distance matrices MSE vs true: {MSE:.4f} (normalized units)")
    print(f"Sampled distance matrices range: {sampled.min():.2f}–{sampled.max():.2f} Å")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("True Distance Matrix")
    plt.imshow(dist0, cmap="viridis", vmin=0, vmax=ddpm.max_distance)
    plt.colorbar(label="Distance (Å)")
    plt.subplot(1, 3, 2)
    plt.title("Sampled Distance Matrix")
    plt.imshow(sampled[0], cmap="viridis", vmin=0, vmax=ddpm.max_distance)
    plt.colorbar(label="Distance (Å)")
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(sampled[0] - dist0, cmap="bwr", vmin=-ddpm.max_distance/2, vmax=ddpm.max_distance/2)
    plt.colorbar(label="Distance Difference (Å)")
    plt.tight_layout()
    plt.savefig("distance_matrix_comparison.png")

    

    # -----------------------------------------
    # 5. Convert distance matrices -> 3D poses
    #    using rigid-body SE3 optimization
    # -----------------------------------------
    print("\n5. Reconstructing binder poses via SE(3) optimization...")
    binder_poses = []

    # Extract CA atoms for SE(3) optimization
    K = binder_coords.shape[1]
    ca_idx = 1 if K > 1 else 0

    # binder_CA_init = binder_coords[:, ca_idx, :]    # (M,3)

    global_offset = torch.rand(3) * 10
    random_rotation = Rot.random().as_matrix()
    binder_coords = (binder_coords @ random_rotation.T ) + np.array(global_offset.detach().cpu())

    binder_CA_init = binder_coords[:, ca_idx, :]
    receptor_CA    = receptor_coords[:, ca_idx, :]  # (N,3)


    for i, D_pred in enumerate(sampled):
        print(f"   Optimizing pose {i+1}/{num_samples}...")

        # Run the SE(3) distance-fitting solver
        
        binder_CA_opt, R, t = se3_optimize_from_distances(
            binder_CA_init,
            receptor_CA,
            D_pred,
            iters=500,
            lr=1e-4,
            device=device,
        )

        # Apply the SE(3) transform to the FULL backbone coordinates
        binder_full = torch.tensor(binder_coords).cuda().reshape(-1, 3).float()            # (M*K,3)
        binder_full = (binder_full @ R.T) + t                 # transform
        binder_full = binder_full.reshape(binder_coords.shape)

        binder_poses.append(binder_full)

    print(f"\n✓ Done! Generated {num_samples} rigid-body poses.")

    return sampled, binder_poses

def save_pdb(
    coords: np.ndarray, output_file: str, atom_names: List[str] = ["N", "CA", "C"]
):
    """
    Save coordinates to PDB file.

    Args:
        coords: [M, K, 3] coordinates
        output_file: Output PDB file path
        atom_names: List of atom names
    """
    with open(output_file, "w") as f:
        atom_num = 1
        for res_num, residue in enumerate(coords, start=1):
            for atom_idx, atom_name in enumerate(atom_names):
                if atom_idx < len(residue):
                    x, y, z = residue[atom_idx]
                    # PDB format
                    line = f"ATOM  {atom_num:5d}  {atom_name:<3s} ALA A{res_num:4d}    "
                    line += f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
                    f.write(line)
                    atom_num += 1
        f.write("END\n")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DDPM on protein contact maps and generate 3D poses"
    )
    parser.add_argument(
        "--binder", type=str, required=True, help="Path to binder PDB file"
    )
    parser.add_argument(
        "--receptor", type=str, required=True, help="Path to receptor PDB file"
    )
    parser.add_argument(
        "--backbone-atoms",
        type=str,
        nargs="+",
        default=["N", "CA", "C"],
        help="Backbone atom names to extract (default: N CA C)",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=8.0,
        help="Distance threshold for contacts in Angstroms (default: 8.0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of poses to generate (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for generated poses (default: output)",
    )

    args = parser.parse_args()

    # Parse PDB files
    print("=" * 70)
    print("Loading PDB files...")
    print("=" * 70)
    binder_coords = parse_pdb(args.binder, args.backbone_atoms)
    receptor_coords = parse_pdb(args.receptor, args.backbone_atoms)

    # Run pipeline
    print("\n" + "=" * 70)
    print("Running DDPM training and sampling pipeline...")
    print("=" * 70)
    sampled_contact_maps, sampled_poses = main_pipeline(
        binder_coords,
        receptor_coords,
        contact_threshold=args.contact_threshold,
        epochs=args.epochs,
        num_samples=args.num_samples,
    )
    sampled_poses = [np.array(pose.cpu()) for pose in sampled_poses]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # Save distance matrices
    for i, dmat in enumerate(sampled_contact_maps):
        np.save(output_dir / f"distance_matrix_{i + 1}.npy", dmat)
        print(f"Saved: {output_dir / f'distance_matrix_{i + 1}.npy'}")

    # Save poses as PDB files
    for i, pose in enumerate(sampled_poses):
        save_pdb(pose, output_dir / f"binder_pose_{i + 1}.pdb", args.backbone_atoms)
        print(f"Saved: {output_dir / f'binder_pose_{i + 1}.pdb'}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print("\nSampled distance matrix statistics:")
    for i, dmat in enumerate(sampled_contact_maps):
        close_contacts = (dmat < 10.0).sum()
        print(
            f"  Matrix {i + 1}: min={dmat.min():.2f}Å, max={dmat.max():.2f}Å, mean={dmat.mean():.2f}Å, contacts<10Å={close_contacts}"
        )

    print("\nSampled pose statistics:")
    for i, pose in enumerate(sampled_poses):
        centroid = pose.mean(axis=(0, 1))
        print(
            f"  Pose {i + 1}: {pose.shape[0]} residues, centroid=({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
        )

    print(f"\n✓ All outputs saved to: {output_dir.absolute()}")
