from contact_diff import parse_pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

# ============================================================
# 1. Utils
# ============================================================


def compute_contact_recall_precision(
    pred_contacts: np.ndarray, true_contacts: np.ndarray, threshold: float = 0.5
):
    pred_bin = (pred_contacts >= threshold).astype(np.float32)
    true_bin = (true_contacts >= threshold).astype(np.float32)

    true_positives = np.sum(pred_bin * true_bin)
    predicted_positives = np.sum(pred_bin)
    actual_positives = np.sum(true_bin)

    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)

    return precision, recall


def compute_contact_map(dist_matrix: np.ndarray, threshold: float = 10.0):
    return (dist_matrix < threshold).astype(np.float32)
    # return dist_matrix.astype(np.float32)


def visualize_contact_maps(gt, recon, sample, save_path="contact_maps.png"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.imshow(gt, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Reconstruction")
    plt.imshow(recon, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Sample")
    plt.imshow(sample, cmap="viridis")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved contact map comparison → {save_path}")


# ============================================================
# 2. VAE
# ============================================================


class ContactMapVAE(nn.Module):
    """
    VAE for contact maps: contact map → z → reconstruction.
    Latent z then used in diffusion.
    """

    def __init__(self, M, N, latent_dim=64):
        super().__init__()
        self.M, self.N = M, N
        self.in_dim = M * N
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(self.in_dim, 256), nn.ReLU())
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.in_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z):
        out = self.decoder(z)
        return out.view(-1, self.M, self.N)

    def forward(self, x):
        """
        x: (B, M, N)
        Returns:
            recon, mu, logvar, z
        """
        B = x.size(0)
        x_flat = x.view(B, -1)

        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class ContactMapEncoder(nn.Module):
    """
    Flattened contact map -> latent vector.
    """

    def __init__(self, M, N, latent_dim=64):
        super().__init__()
        self.in_dim = M * N
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        # x: (B, M, N)
        return self.net(x.view(x.size(0), -1))


class ContactMapDecoder(nn.Module):
    """
    latent vector -> reconstructed contact map
    """

    def __init__(self, M, N, latent_dim=64):
        super().__init__()
        self.M, self.N = M, N
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, M * N),
            # nn.Sigmoid(),  # produce soft contact probabilities
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(z.size(0), self.M, self.N)


# ============================================================
# 3. Latent DDPM (R^d diffusion)
# ============================================================


class LatentDDPM(nn.Module):
    """
    Tiny DDPM in latent space.
    Predicts noise εθ(z_t, t).
    """

    def __init__(self, latent_dim=64, timesteps=500, device="cpu"):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.device = device

        # Simple β schedule
        self.betas = torch.linspace(1e-4, 2e-3, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.ac = torch.cumprod(self.alphas, dim=0)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        a = self.ac[t].view(-1, 1)
        return torch.sqrt(a) * z0 + torch.sqrt(1 - a) * noise

    def predict_noise(self, zt, t):
        t_embed = t.float().unsqueeze(1) / self.timesteps
        return self.net(torch.cat([zt, t_embed], dim=1))

    def p_sample(self, zt, t):
        beta = self.betas[t].view(-1, 1)
        a = self.alphas[t].view(-1, 1)
        ac = self.ac[t].view(-1, 1)

        eps = self.predict_noise(zt, t)
        mean = (1 / torch.sqrt(a)) * (zt - beta * eps / torch.sqrt(1 - ac))

        if t[0] == 0:
            return mean

        noise = torch.randn_like(zt)
        return mean + torch.sqrt(beta) * noise

    @torch.no_grad()
    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((num_samples,), t, dtype=torch.long).to(self.device)
            z = self.p_sample(z, t_batch)
        return z

class LatentFlowMatching(nn.Module):
    """
    Placeholder for latent flow matching model.
    """

    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )

    def predict_velocity(self, z, t):
        t_embed = t.float().unsqueeze(1) / 1000  # assuming 1000 timesteps
        return self.net(torch.cat([z, t_embed], dim=1))

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        a = torch.exp(-0.5 * t.float().unsqueeze(1) / 1000)
        return a * z0 + torch.sqrt(1 - a ** 2) * noise

    def p_sample(self, zt, t, dt=1):
        v = self.predict_velocity(zt, t)
        return zt - v * dt

    @torch.no_grad()
    def sample(self, num_samples=1):
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        for t in reversed(range(1000)):
            t_batch = torch.full((num_samples,), t, dtype=torch.long).to(
                next(self.parameters()).device
            )
            z = self.p_sample(z, t_batch)
        return z




# ============================================================
# 4. Training loop
# ============================================================


def train_vae_latent_diffusion(
    contact_map: np.ndarray, latent_dim=64, epochs=1000, device="cpu"
):
    M, N = contact_map.shape
    x = torch.tensor(contact_map, device=device).unsqueeze(0)

    vae = ContactMapVAE(M, N, latent_dim).to(device)
    ddpm = LatentDDPM(latent_dim, timesteps=500, device=device).to(device)

    opt = torch.optim.Adam(list(vae.parameters()) + list(ddpm.parameters()), lr=1e-3)

    for epoch in range(epochs):
        opt.zero_grad()

        # -----------------
        # 1. Forward VAE
        # -----------------
        logits, mu, logvar, z0 = vae(x)

        # VAE losses
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = F.binary_cross_entropy_with_logits(logits, x)

        # -----------------
        # 2. Latent DDPM
        # -----------------
        t = torch.randint(0, ddpm.timesteps, (1,), device=device)
        noise = torch.randn_like(z0)
        zt = ddpm.q_sample(z0, t, noise)
        eps_pred = ddpm.predict_noise(zt, t)
        diff_loss = F.mse_loss(eps_pred, noise)

        loss = recon_loss + 0.1 * kl + diff_loss
        loss.backward()
        opt.step()

        if (epoch + 1) % 250 == 0:
            print(
                f"Epoch {epoch + 1:4d} | recon={recon_loss.item():.4f} KL={kl.item():.4f} diff={diff_loss.item():.4f}"
            )

    return vae, ddpm


# ============================================================
# 5. Wrapper Pipeline
# ============================================================


def run_latent_diffusion_pipeline_VAE(binder_coords, receptor_coords):
    """
    Full interface-latent VAE + DDPM pipeline.
    """

    # Compute CA–CA distance matrix
    binder_ca = binder_coords[:, 1]
    receptor_ca = receptor_coords[:, 1]
    dmat = np.linalg.norm(binder_ca[:, None, :] - receptor_ca[None, :, :], axis=2)
    dmat = dmat

    cmap = compute_contact_map(dmat)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    vae, ddpm = train_vae_latent_diffusion(
        cmap, latent_dim=64, epochs=700, device=device
    )

    # Clean reconstruction
    with torch.no_grad():
        x = torch.tensor(cmap, device=device).unsqueeze(0)
        recon, mu, logvar, z0 = vae(x)
        recon = recon[0]
        recon = recon.clamp(0, 1).cpu().numpy()

    # Sample from DDPM latent prior
    with torch.no_grad():
        z_sample = ddpm.sample(num_samples=1)
        sample = vae.decode(z_sample)[0]
        sample = sample.clamp(0, 1).cpu().numpy()


    visualize_contact_maps(
        gt=cmap, recon=recon, sample=sample, save_path="vae_latent_contact_maps.png"
    )
    recall, precision = compute_contact_recall_precision(
        pred_contacts=sample, true_contacts=cmap, threshold=0.5
    )

    return cmap, recon, sample, recall, precision


def train_latent_diffusion(
    contact_map: np.ndarray, latent_dim=64, epochs=2000, device="cpu"
):
    """
    Train encoder + decoder + latent DDPM to model the GT contact map.
    """
    M, N = contact_map.shape
    contact_map_t = torch.tensor(
        contact_map, dtype=torch.float32, device=device
    ).unsqueeze(0)

    encoder = ContactMapEncoder(M, N, latent_dim).to(device)
    decoder = ContactMapDecoder(M, N, latent_dim).to(device)
    ddpm = LatentDDPM(latent_dim, timesteps=500, device=device).to(device)

    params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(ddpm.parameters())
    )
    opt = torch.optim.AdamW(params, lr=1e-3, fused=True)

    for epoch in range(epochs):
        opt.zero_grad()

        # Encode
        z0 = encoder(contact_map_t)  # (1,latent_dim)

        # Sample a time
        t = torch.randint(0, ddpm.timesteps, (1,), device=device)

        # Noise
        noise = torch.randn_like(z0)

        # Forward noise
        z_t = ddpm.q_sample(z0, t, noise)

        # Predict noise
        eps_pred = ddpm.predict_noise(z_t, t)

        # DDPM noise loss in latent space
        loss_diff = F.mse_loss(eps_pred, noise)

        # Reconstruction loss (only on clean latent)
        recon = decoder(z0)
        loss_recon = F.mse_loss(recon, contact_map_t)

        loss = loss_diff + loss_recon
        loss.backward()
        opt.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1:4d}, Loss = {loss.item():.5f}")

    return encoder, decoder, ddpm


def run_latent_diffusion_pipeline(binder_coords, receptor_coords):
    """
    - Compute CA–CA distance matrix
    - Convert to contact map
    - Train latent diffusion to overfit on that one map
    - Sample new contact maps
    - Visualize
    """

    # Compute distance matrix using CA
    M = binder_coords.shape[0]
    N = receptor_coords.shape[0]
    binder_ca = binder_coords[:, 1]  # CA index
    receptor_ca = receptor_coords[:, 1]

    dmat = np.linalg.norm(binder_ca[:, None, :] - receptor_ca[None, :, :], axis=2)
    cmap = compute_contact_map(dmat)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    encoder, decoder, ddpm = train_latent_diffusion(
        cmap, latent_dim=64, epochs=8000, device=device
    )

    # Reconstruct (clean)
    with torch.no_grad():
        gt = torch.tensor(cmap, device=device).unsqueeze(0)
        z0 = encoder(gt)
        recon = decoder(z0)[0].cpu().numpy()

    # Sample from diffusion
    with torch.no_grad():
        z_sample = ddpm.sample(1)
        sample = decoder(z_sample)[0].cpu().numpy()

    visualize_contact_maps(cmap, recon, sample, save_path="latent_contact_maps.png")

    return cmap, recon, sample


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
    args = parser.parse_args()
    binder = parse_pdb(args.binder)
    receptor = parse_pdb(args.receptor)
    # cmap_gt, cmap_recon, cmap_sample = run_latent_diffusion_pipeline(
    #     binder_coords=binder, receptor_coords=receptor
    # )
    cmap_gt, cmap_recon, cmap_sample, recall, precision = run_latent_diffusion_pipeline_VAE(
        binder_coords=binder, receptor_coords=receptor
    )
    print(f"Sampled contact map precision: {precision:.4f}, recall: {recall:.4f}")
