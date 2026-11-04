import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .latent_modules import ProteinEncoder, ConvFiLM
from utils import Features, ALPHABET
from typing import Tuple


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


def triangle_loss(pred: Tensor, target, mask: Tensor):
    """
    Vectorized triangle inequality loss:
        d(i,j) + d(j,k) >= d(i,k)
    """
    B, M, N = pred.shape

    # Expand dims: (B, M, N) → (B, M, N, 1) + (B, M, 1, N)
    dij = pred.unsqueeze(3)  # (B, M, N, 1)
    djk = pred.unsqueeze(1)  # (B, 1, M, N)
    dik = pred.unsqueeze(2)  # (B, M, 1, N)

    # Compute violations: dij + djk < dik
    tri_ineq = dij + djk - dik  # (B, M, N, N)
    tri_ineq = F.relu(tri_ineq)

    # Build mask (B, M, N, N)
    m_i = mask.unsqueeze(3)  # (B, M, N, 1)
    m_j = mask.unsqueeze(1)  # (B, 1, M, N)
    m_k = mask.unsqueeze(2)  # (B, M, 1, N)
    m = m_i * m_j * m_k

    loss = (tri_ineq * m).sum() / m.sum().clamp(min=1.0)
    return loss


class ContactMapVAE(nn.Module):
    """
    VAE over *continuous* distance matrices:
      x: (B, M, N) of real-valued distances.

    pair_mask: (B, M, N) boolean mask of valid entries.

    This version includes:
      - UNet-style encoder-decoder with skip connections
      - FiLM conditioning in the decoder
      - Symmetry enforcement of the reconstructed distances
      - Charbonnier reconstruction loss
      - Triangle inequality & sharpness regularizers
    """

    def __init__(
        self,
        M: int,
        N: int,
        kl_weight: float,
        mean: float,
        std: float,
        hidden_size: int = 128,  # latent_dim
        lambda_tri: float = 0.001,
        lambda_sharp: float = 0.0,
    ):
        super().__init__()

        self.M = M
        self.N = N
        self.latent_dim = hidden_size
        self.kl_weight = kl_weight
        self.mean = float(mean)
        self.std = float(std)

        self.lambda_tri = lambda_tri
        self.lambda_sharp = lambda_sharp
        self.warmup_steps = 10000

        # ------------- Encoder (UNet-style) -------------
        # Resolution levels:
        #  level1: (B,  64,  M,   N)
        #  level2: (B, 128, M/2, N/2)
        #  level3: (B, 256, M/4, N/4)
        #  bottleneck: (B, 256, M/4, N/4)

        self.enc_block1 = ResBlock(1, 64, stride=1)
        self.enc_block2 = ResBlock(64, 128, stride=2)
        self.enc_block3 = ResBlock(128, 256, stride=2)
        self.enc_block4 = ResBlock(256, 256, stride=1)

        # Dynamically infer encoder bottleneck spatial shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1, M, N)
            h = self.enc_block1(dummy)
            h = self.enc_block2(h)
            h = self.enc_block3(h)
            h = self.enc_block4(h)
            _, C, H, W = h.shape
        self.enc_C = C
        self.enc_H = H
        self.enc_W = W
        self.enc_feat_size = C * H * W

        # ------------- Latent projections -------------

        self.fc_mu = nn.Sequential(
            nn.Linear(self.enc_feat_size, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.enc_feat_size, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # ------------- Decoder (UNet + FiLM) -------------

        self.fc_dec = nn.Linear(self.latent_dim, self.enc_feat_size)

        # Bottleneck decoder block
        self.dec_block1 = ResBlock(self.enc_C, 256, stride=1)
        self.film1 = ConvFiLM(self.latent_dim, 256)

        # Up 1: (B,256,M/4,N/4) → upsample → concat skip3 (256) → 512 → 256
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_block2 = ResBlock(256 + 256, 256)
        self.film2 = ConvFiLM(self.latent_dim, 256)

        # Up 2: (B,256,M/2,N/2) → upsample → concat skip2 (128) → 384 → 128
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_block3 = ResBlock(256 + 128, 128)
        self.film3 = ConvFiLM(self.latent_dim, 128)

        # Up 3: (B,128,M,N) → upsample → concat skip1 (64) → 192 → 64
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_block4 = ResBlock(128 + 64, 64)
        self.film4 = ConvFiLM(self.latent_dim, 64)

        # Final conv → 1 channel of distances (normalized)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    # ------------- core VAE bits -------------

    def encode(self, x_img: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tuple]:
        """
        x_img: (B, 1, M, N) normalized distances
        returns mu, logvar, and skip features
        """
        h1 = self.enc_block1(x_img)  # (B, 64, M,   N)
        h2 = self.enc_block2(h1)  # (B,128, M/2, N/2)
        h3 = self.enc_block3(h2)  # (B,256, M/4, N/4)
        h4 = self.enc_block4(h3)  # (B,256, M/4, N/4)

        h_flat = h4.reshape(h4.size(0), -1)  # (B, C*H*W)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        skips = (h1, h2, h3)
        return mu, logvar, h4, skips

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z: Tensor, bottleneck: Tensor, skips) -> Tensor:
        """
        z:         (B, latent_dim)
        bottleneck:(B, C, H, W)
        skips:     tuple of encoder feature maps
        returns normalized reconstruction: (B, M, N)
        """
        h1, h2, h3 = skips  # skip1 (64), skip2(128), skip3(256)

        # Bottleneck (already in correct shape)
        h = bottleneck  # (B,256,H,W)
        h = self.dec_block1(h)
        h = self.film1(h, z)

        # Up 1: -> (B,256,M/2,N/2) + skip3(256) -> (B,512,M/2,N/2) -> 256
        h = self.up1(h)
        # Crop if needed to match skip size (in case of odd dims)
        if h.shape[-2:] != h3.shape[-2:]:
            h = h[:, :, : h3.size(2), : h3.size(3)]
        h = torch.cat([h, h3], dim=1)
        h = self.dec_block2(h)
        h = self.film2(h, z)

        # Up 2: -> (B,256,M,N) + skip2(128) -> (B,384,M,N) -> 128
        h = self.up2(h)
        if h.shape[-2:] != h2.shape[-2:]:
            h = h[:, :, : h2.size(2), : h2.size(3)]
        h = torch.cat([h, h2], dim=1)
        h = self.dec_block3(h)
        h = self.film3(h, z)

        # Up 3: -> (B,128,2M,2N) but we only need (M,N); then concat skip1(64)
        h = self.up3(h)
        if h.shape[-2:] != h1.shape[-2:]:
            h = h[:, :, : h1.size(2), : h1.size(3)]
        h = torch.cat([h, h1], dim=1)
        h = self.dec_block4(h)
        h = self.film4(h, z)

        out = self.out_conv(h)  # (B,1,M,N)
        out = out[:, :, : self.M, : self.N]  # crop if needed
        x_recon = out.squeeze(1)  # (B,M,N)

        # Enforce symmetry of the distance matrix
        x_recon = 0.5 * (x_recon + x_recon.transpose(1, 2))

        return x_recon

    def forward(
        self,
        x: Tensor,  # (B, M, N) raw distances (Å)
        pair_mask: Tensor,  # (B, M, N) bool or float
        weights=None,
        step=0,
        return_logits=False,
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

        # Encode
        mu, logvar, bottleneck, skips = self.encode(x_img)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        # Decode → normalized reconstruction
        x_recon_norm = self.decode(z, bottleneck, skips)  # (B,M,N)

        # ---------------- Reconstruction loss (Charbonnier) ----------------
        eps = 1e-6
        diff = x_recon_norm - x_norm
        charb = torch.sqrt(diff * diff + eps)  # robust L2
        recon_loss = (charb * mask).sum() / mask.sum().clamp(min=1.0)

        # ---------------- Geometry / sharpness regularizers ----------------
        # Convert reconstruction back to physical distances for these terms
        x_recon_phys = x_recon_norm * self.std + self.mean

        triangle_ineq_loss = triangle_loss(x_recon_phys, x_phys, mask)
        sharp_loss = sharpness_loss(x_recon_phys, x_phys, mask)

        # ---------------- KL term ----------------
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = self.kl_weight * kl_loss

        # ---------------- Warmup for lambda_tri and lambda_sharp ----------------
        lambda_scale = min(1.0, step / self.warmup_steps)
        triangle_ineq_loss = lambda_scale * triangle_ineq_loss
        sharp_loss = lambda_scale * sharp_loss

        # ---------------- Total loss ----------------
        loss = (
            recon_loss
            + kl_loss
            + self.lambda_tri * triangle_ineq_loss
            + self.lambda_sharp * sharp_loss
        )

        if return_logits:
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


class ProteinContactMapVAE(nn.Module):
    """
    VAE over *continuous* distance matrices:
      x: (B, M, N) of real-valued distances.

    pair_mask: (B, M, N) boolean mask of valid entries.

    This version includes:
      - UNet-style encoder-decoder with skip connections
      - FiLM conditioning in the decoder
      - Symmetry enforcement of the reconstructed distances
      - Charbonnier reconstruction loss
      - Triangle inequality & sharpness regularizers
    """

    def __init__(
        self,
        M: int,
        N: int,
        kl_weight: float,
        mean: float,
        std: float,
        emb_size: int,
        protein_encoder_depth: int = 5,
        hidden_size: int = 64,  # latent_dim
        lambda_tri: float = 0.001,
        lambda_sharp: float = 0.0,
        channel_depths: list = [64, 128, 256],
        use_symmetry: bool = False,
        target_mode: str = "distances",
        contact_bin_edges: Tensor =None,
        class_weights: Tensor = None,
    ):
        super().__init__()

        self.M = M
        self.N = N
        self.latent_dim = hidden_size
        self.kl_weight = kl_weight
        self.mean = float(mean)
        self.std = float(std)
        self.use_symmetry = use_symmetry
        self.target_mode = target_mode

        if self.target_mode == "contact":
            if contact_bin_edges is None:
                # Example bins in Å:
                #   [0–2), [2–4), [4–8), [8–16), [16, +inf)
                contact_bin_edges = [2.0, 4.0, 8.0, 16.0]
            self.register_buffer(
                "contact_bin_edges",
                torch.tensor(contact_bin_edges, dtype=torch.float32),
                persistent=True,
            )
            # Number of bins = number of edges + 1
            self.num_bins = len(contact_bin_edges) + 1
            self.class_weights = class_weights  # for weighted cross-entropy loss
        else:
            # Empty buffer so binning isn't accidentally used in distance mode
            self.register_buffer(
                "contact_bin_edges",
                torch.tensor([], dtype=torch.float32),
                persistent=False,
            )
            self.num_bins = 1  # Single distance output
            self.class_weights = None

        self.lambda_tri = lambda_tri
        self.lambda_sharp = lambda_sharp
        self.warmup_steps = 10000
        self.protein_encoder = ProteinEncoder(
            hidden_size=hidden_size,
            emb_size=emb_size,
            depth=protein_encoder_depth,
            heads=8,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
        )
        tfmr_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size,
            batch_first=True,
        )
        self.cplx_mixer = nn.TransformerEncoder(
            tfmr_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )

        # self.pair_embedding = nn.Embedding(
        #     len(ALPHABET) * len(ALPHABET), hidden_size // 2
        # )

        self.bind_norm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.tgt_norm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        l1, l2, l3 = channel_depths

        # ------------- Encoder (UNet-style) -------------
        # Resolution levels:
        #  level1: (B,  64,  M,   N)
        #  level2: (B, 128, M/2, N/2)
        #  level3: (B, 256, M/4, N/4)
        #  bottleneck: (B, 256, M/4, N/4)

        self.enc_block1 = ResBlock(1 + hidden_size, l1, stride=1)
        self.enc_block2 = ResBlock(l1, l2, stride=2)
        self.enc_block3 = ResBlock(l2, l3, stride=2)
        self.enc_block4 = ResBlock(l3, l3, stride=1)

        # Dynamically infer encoder bottleneck spatial shape
        with torch.no_grad():
            dummy = torch.zeros(1, 1 + hidden_size, M, N)
            h = self.enc_block1(dummy)
            h = self.enc_block2(h)
            h = self.enc_block3(h)
            h = self.enc_block4(h)
            _, C, H, W = h.shape

        self.enc_C = C
        self.enc_H = H
        self.enc_W = W
        self.enc_feat_size = C * H * W

        # ------------- Latent projections -------------

        self.fc_mu = nn.Sequential(
            nn.Linear(self.enc_feat_size, self.latent_dim),
            nn.SiLU(),
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.enc_feat_size, self.latent_dim),
            nn.SiLU(),
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # ------------- Decoder (UNet + FiLM) -------------

        self.fc_dec = nn.Linear(self.latent_dim, self.enc_feat_size)

        # Bottleneck decoder block
        self.dec_block1 = ResBlock(self.enc_C, l3, stride=1)
        self.film1 = ConvFiLM(self.latent_dim, l3)

        # Up 1: (B,256,M/4,N/4) → upsample → concat skip3 (256) → 512 → 256
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_block2 = ResBlock(l3 + l3, l3)
        self.film2 = ConvFiLM(self.latent_dim, l3)

        # Up 2: (B,256,M/2,N/2) → upsample → concat skip2 (128) → 384 → 128
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_block3 = ResBlock(l3 + l2, l2)
        self.film3 = ConvFiLM(self.latent_dim, l2)

        # Up 3: (B,128,M,N) → upsample → concat skip1 (64) → 192 → 64
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_block4 = ResBlock(l2 + l1, l1)
        self.film4 = ConvFiLM(self.latent_dim, l1)

        # Final conv → 1 channel of distances (normalized)
        # self.out_conv = nn.Conv2d(l1, 1, kernel_size=1)
        if self.target_mode == "contact":
            self.out_conv = nn.Conv2d(l1, self.num_bins, kernel_size=1)
        else:
            self.out_conv = nn.Conv2d(l1, 1, kernel_size=1)

        # ------------- Neural Metric parameters -----------------
        self.metric_dim = l1  # final embedding channel dim
        self.num_components = 8  # number of Mahalanobis bases
        self.alpha = 0.5  # maxmean blending

        # Wide Norm parameters W_i
        self.W_metric = nn.Parameter(
            torch.randn(self.num_components, self.metric_dim, self.metric_dim)
        )
        self.bind_final = nn.Linear(l1, self.metric_dim)
        self.tgt_final = nn.Linear(l1, self.metric_dim)

        # Concave metric shaping: min_j ( w_ij * x + b_ij )
        self.num_segments = 4
        self.w_raw = nn.Parameter(torch.randn(self.num_components, self.num_segments))
        self.b_raw = nn.Parameter(torch.randn(self.num_components, self.num_segments))

    def encode_protein(
        self, binder: Features, target: Features
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        returns:
            pair_H: (B, H, M, N) pairwise features
            bind_H: (B, M, H) binder features
            target_H: (B, N, H) target features
        """
        bind_H, target_H = self.protein_encoder(
            binder, target
        )  # [B, M_full, H], [B, N_full, H]
        bind_H, target_H = self.bind_norm(bind_H), self.tgt_norm(target_H)
        bind_H = bind_H[:, : self.M, :]  # crop to (B, M_, H)
        target_H = target_H[:, : self.N, :]  # crop to (B, N_, H)

        # pad bind_H and target_H to match x_img if needed
        bind_H = F.pad(bind_H, (0, 0, 0, self.M - bind_H.size(1)))  # (B, M, H)
        target_H = F.pad(target_H, (0, 0, 0, self.N - target_H.size(1)))  # (B, N, H)

        pair_H = bind_H.unsqueeze(2) + target_H.unsqueeze(1)  # (B, M, N, H)
        pair_H = pair_H.permute(0, 3, 1, 2)  # (B, H, M, N)

        return pair_H, bind_H, target_H

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

    def compute_neural_metric(self, bind_H, target_H):
        """
        bind_H:   (B, M, C)
        target_H: (B, N, C)
        returns distance matrix (B, M, N)
        """

        # pairwise embedding differences
        diff = bind_H[:, :, None, :] - target_H[:, None, :, :]  # (B, M, N, C)

        # Wide Norm components: Mahalanobis norms per component
        # → compute || W_i diff ||_2

        # (K,B,M,N,C) = W @ diff
        Wx = torch.einsum("kcd,bmnd->kbmnc", self.W_metric, diff)

        # compute L2 norms along channel
        d_base = Wx.norm(dim=-1)  # (K, B, M, N)

        # transpose to (B,M,N,K)
        d_base = d_base.permute(1, 2, 3, 0)

        # concave shaping: min_j ( w_ij * x + b_ij )
        w = torch.nn.functional.softplus(self.w_raw)  # (K,S)
        b = torch.nn.functional.softplus(self.b_raw)  # (K,S)
        b[:, 0] = 0.0  # enforce f(0)=0

        # shape for broadcasting
        x_exp = d_base.unsqueeze(-1)  # (B,M,N,K,1)
        lines = w.view(
            1, 1, 1, self.num_components, self.num_segments
        ) * x_exp + b.view(1, 1, 1, self.num_components, self.num_segments)

        f_d = lines.min(dim=-1).values  # (B,M,N,K)

        # final metric via max-mean
        mx = f_d.max(dim=-1).values
        mn = f_d.mean(dim=-1)

        return self.alpha * mx + (1 - self.alpha) * mn

    # ------------- core VAE bits -------------

    def encode(
        self, x_img: Tensor, pair_H: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple]:
        """
        x_img: (B, 1, M, N) normalized distances
        returns mu, logvar, and skip features
        """

        x = torch.cat([x_img, pair_H], dim=1)  # (B, 1+H, M, N)

        h1 = self.enc_block1(x)  # (B, 64, M, N)
        h2 = self.enc_block2(h1)  # (B,128, M/2, N/2)
        h3 = self.enc_block3(h2)  # (B,256, M/4, N/4)
        h4 = self.enc_block4(h3)  # (B,256, M/4, N/4)

        h_flat = h4.reshape(h4.size(0), -1)  # (B, C*H*W)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)

        skips = (h1, h2, h3)
        return mu, logvar, h4, skips

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(
        self,
        z: Tensor,
        bottleneck: Tensor,
        skips: tuple,
        binder_H: Tensor,
        target_H: Tensor,
    ) -> Tensor:
        """
        z:         (B, latent_dim)
        bottleneck:(B, C, H, W)
        skips:     tuple of encoder feature maps
        binder_H: (B, M, H)
        target_H: (B, N, H)
        returns normalized reconstruction: (B, M, N)
        """
        cplx_H = torch.cat((binder_H, target_H), dim=1)  # (B, M+N, H)
        cplx_H = self.cplx_mixer(cplx_H)  # (B, M+N, H)

        z = self.cross_attention(
            query=z.unsqueeze(1),
            key=cplx_H,
            value=cplx_H,
        )[0].squeeze(1)

        h1, h2, h3 = skips  # skip1 (64), skip2(128), skip3(256)

        # Bottleneck (already in correct shape)
        h = bottleneck  # (B,256,H,W)
        h = self.dec_block1(h)
        h = self.film1(h, z)

        # Up 1: -> (B,256,M/2,N/2) + skip3(256) -> (B,512,M/2,N/2) -> 256
        h = self.up1(h)
        # Crop if needed to match skip size (in case of odd dims)
        if h.shape[-2:] != h3.shape[-2:]:
            h = h[:, :, : h3.size(2), : h3.size(3)]
        h = torch.cat([h, h3], dim=1)
        h = self.dec_block2(h)
        h = self.film2(h, z)

        # Up 2: -> (B,256,M,N) + skip2(128) -> (B,384,M,N) -> 128
        h = self.up2(h)
        if h.shape[-2:] != h2.shape[-2:]:
            h = h[:, :, : h2.size(2), : h2.size(3)]
        h = torch.cat([h, h2], dim=1)
        h = self.dec_block3(h)
        h = self.film3(h, z)

        # Up 3: -> (B,128,2M,2N) but we only need (M,N); then concat skip1(64)
        h = self.up3(h)
        if h.shape[-2:] != h1.shape[-2:]:
            h = h[:, :, : h1.size(2), : h1.size(3)]
        h = torch.cat([h, h1], dim=1)
        h = self.dec_block4(h)
        h = self.film4(h, z)  # [B, l1, M, N]

        if not self.use_symmetry:
            out = self.out_conv(h)  # (B,1,M,N)
        else:
            # extract residue embeddings for binder and target
            B = h.size(0)
            out = h.reshape(B, self.M, self.N, -1)  # (B, M, N, C)
            bind_H_refined = out.mean(dim=2)  # (B, M, C)
            target_H_refined = out.mean(dim=1)  # (B, N, C)
            # x_recon_phys = self.compute_neural_metric(bind_H_refined, target_H_refined)
            bind_H_refined = self.bind_final(bind_H_refined)
            target_H_refined = self.tgt_final(target_H_refined)
            x_recon_phys = (
                bind_H_refined.unsqueeze(1) - target_H_refined.unsqueeze(2)
            ).norm(dim=-1, p=2)

            return x_recon_phys

        if self.target_mode == "contact":
            # Return raw logits for contact classification
            # Shape: (B, num_bins, M, N)
            return 0.5 * (out + out.transpose(2, 3))  # enforce symmetry

        # out = out[:, :, : self.M, : self.N]  # crop if needed
        x_recon = out.squeeze(1)  # (B,M,N)

        # Enforce symmetry of the distance matrix
        x_recon = 0.5 * (x_recon + x_recon.transpose(1, 2))

        return x_recon

    def forward(
        self,
        x: Tensor,  # (B, M, N) raw distances (Å)
        binder: Features,
        target: Features,
        pair_mask: Tensor,  # (B, M, N) bool or float
        step=0,
        return_recon=False,
        return_latent=False,
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

        if self.target_mode == "contact":
            true_bins = self.bin_distances(x_phys).cuda()  # (B, M, N) long tensor

        # Normalize distances (for reconstruction)
        x_norm = ((x_phys + 1e-8).log() - self.mean) / self.std  # (B,M,N)
        x_img = x_norm.unsqueeze(1)  # (B,1,M,N)

        # Get protein features
        pair_H, binder_H, target_H = self.encode_protein(binder, target)

        # Encode
        mu, logvar, bottleneck, skips = self.encode(x_img, pair_H)

        # Latent sample
        z = self.reparameterize(mu, logvar)

        if return_latent:
            return z

        # Decode → normalized reconstruction
        x_recon_norm = self.decode(z, bottleneck, skips, binder_H, target_H)  # (B,M,N)

        if self.target_mode == "contact":
            logits_flat = x_recon_norm.permute(0, 2, 3, 1).reshape(
                -1, self.num_bins
            )  # (B*M*N, num_bins)
            true_bins_flat = true_bins.reshape(-1)  # (B*M*N,)
            mask_flat = mask.reshape(-1)  # (B*M*N,)
            ce_loss = F.cross_entropy(
                logits_flat, true_bins_flat, reduction="none", weight=self.class_weights
            )  # (B*M*N,)

            recon_loss = (ce_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
            x_recon_phys = x_recon_norm.argmax(
                dim=1
            )
            triangle_ineq_loss = torch.tensor(0.0, device=x.device)
            sharp_loss = torch.tensor(0.0, device=x.device)
        else:
            # ---------------- Reconstruction loss (Charbonnier) ----------------
            eps = 1e-6
            diff = x_recon_norm - x_norm
            charb = torch.sqrt(diff * diff + eps)  # robust L2
            recon_loss = (charb * mask).sum() / mask.sum().clamp(min=1.0)

            # ---------------- Geometry / sharpness regularizers ----------------
            # Convert reconstruction back to physical distances for these terms
            x_recon_phys = (x_recon_norm * self.std + self.mean).exp() + 1e-8

            triangle_ineq_loss = triangle_loss(x_recon_phys, x_phys, mask)
            sharp_loss = sharpness_loss(x_recon_phys, x_phys, mask)

        # ---------------- KL term ----------------
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = self.kl_weight * kl_loss
        lambda_scale = min(1.0, step / self.warmup_steps)


        # ---------------- Warmup for lambda_tri and lambda_sharp ----------------
        triangle_ineq_loss = lambda_scale * triangle_ineq_loss
        sharp_loss = lambda_scale * sharp_loss
        kl_loss = lambda_scale * kl_loss

        # ---------------- Total loss ----------------
        loss = (
            recon_loss
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


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.silu(out + residual)
        return out
