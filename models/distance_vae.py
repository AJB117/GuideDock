import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.common import GaussianSmearing
from models.latent_modules import CustomTransformer, ProteinEncoder
from utils import (
    Features,
    get_cplx_attn_bias,
)


def kl_sigmoid_anneal(step, k=0.0008, x0=2000):
    """
    Sigmoid KL annealing.
    k controls slope; x0 is midpoint (≈ where KL weight = 0.5)
    """
    return float(1 / (1 + torch.exp(-k * torch.tensor(step - x0))))


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

    b_coords = binder.X[:, :, 1]
    t_coords = target.X[:, :, 1]
    b_mask = binder.A[:, :, 1].clamp(max=1.0)
    t_mask = target.A[:, :, 1].clamp(max=1.0)

    total_loss = 0.0
    total_count = 0.0

    S_B = min(M, num_anchors)
    if S_B > 0:
        k_idx = torch.randperm(M, device=device)[:S_B]

        k_coords_b = b_coords[:, k_idx, :]
        d_ik = torch.cdist(b_coords, k_coords_b).unsqueeze(2)

        d_kj = pred[:, k_idx, :].transpose(1, 2).unsqueeze(1)

        d_ij = pred.unsqueeze(-1)

        m_k = b_mask[:, k_idx]
        mask_A = (
            b_mask.view(B, M, 1, 1) * t_mask.view(B, 1, N, 1) * m_k.view(B, 1, 1, S_B)
        )

        if mask_A.sum() > 0:
            v1 = F.relu(d_ij - (d_ik + d_kj))
            v2 = F.relu(d_ik - (d_ij + d_kj))
            v3 = F.relu(d_kj - (d_ij + d_ik))

            total_loss = total_loss + ((v1 + v2 + v3) * mask_A).sum()
            total_count = total_count + mask_A.sum()

    S_T = min(N, num_anchors)
    if S_T > 0:
        k_idx = torch.randperm(N, device=device)[:S_T]

        k_coords_t = t_coords[:, k_idx, :]
        d_kj = torch.cdist(t_coords, k_coords_t).unsqueeze(1)

        d_ik = pred[:, :, k_idx].unsqueeze(2)

        d_ij = pred.unsqueeze(-1)

        m_k = t_mask[:, k_idx]
        mask_B = (
            b_mask.view(B, M, 1, 1) * t_mask.view(B, 1, N, 1) * m_k.view(B, 1, 1, S_T)
        )

        if mask_B.sum() > 0:
            v1 = F.relu(d_ij - (d_ik + d_kj))
            v2 = F.relu(d_ik - (d_ij + d_kj))
            v3 = F.relu(d_kj - (d_ij + d_ik))

            total_loss = total_loss + ((v1 + v2 + v3) * mask_B).sum()
            total_count = total_count + mask_B.sum()

    return total_loss / total_count.clamp(min=1.0)


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

        self.inter_query = nn.Sequential(
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

        bind_dist_H, tgt_dist_H = self.get_internal_dists(binder, target)

        cplx_dist_H = binder_H_first.unsqueeze(2) * target_H_first.unsqueeze(1)
        cplx_dist_H = self.distance_mlp(cplx_dist_H).squeeze(-1)

        cplx_attn_bias = get_cplx_attn_bias(
            binder,
            target,
            bind_dists=bind_dist_H,
            target_dists=tgt_dist_H,
            inter_dists=cplx_dist_H,
        )

        cplx_H = self.enc_cplx_mixer(
            torch.cat((binder_H_first, target_H_first), dim=1), attn_mask=cplx_attn_bias
        )
        binder_H, target_H = cplx_H[:, :M, :], cplx_H[:, M:, :]

        binder_H = binder_H + binder_H_first
        target_H = target_H + target_H_first

        binder_H = self.bind_enc(binder_H)
        target_H = self.tgt_enc(target_H)

        cplx_H = torch.cat((binder_H, target_H), dim=1)
        pre_latent = self.proj_to_latent(cplx_H)

        mu, logvar = pre_latent.chunk(2, dim=-1)

        if return_Hs:
            return mu, logvar, binder_H, target_H

        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def get_internal_dists(self, binder: Features, target: Features):
        bind_internal_dist = torch.cdist(binder.X[:, :, 1], binder.X[:, :, 1])
        target_internal_dist = torch.cdist(target.X[:, :, 1], target.X[:, :, 1])

        bind_dist_H = self.distance_mlp(
            self.gaussian_encoder(bind_internal_dist)
        ).squeeze(-1)
        tgt_dist_H = self.distance_mlp(
            self.gaussian_encoder(target_internal_dist)
        ).squeeze(-1)

        return bind_dist_H, tgt_dist_H

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

        z_pre = self.up_project(z)

        binder_S, target_S = binder.S, target.S
        binder_emb = self.bind_emb_proj(binder_S)
        target_emb = self.tgt_emb_proj(target_S)
        binder_z, target_z = z_pre[:, :M, :], z_pre[:, M:, :]

        binder_z = binder_z + binder_emb
        target_z = target_z + target_emb

        binder_z, target_z = self.dec_ipa_mixer(
            binder, target, bind_H=binder_z, tgt_H=target_z
        )
        z_pre = torch.cat([binder_z, target_z], dim=1)

        inter_score = self.inter_query(z_pre)
        bind_inter_score, tgt_inter_score = inter_score[:, :M], inter_score[:, M:]
        inter_score = (
            bind_inter_score.unsqueeze(2) * tgt_inter_score.unsqueeze(1)
        ).squeeze(-1)

        bind_dist_H, target_dist_H = self.get_internal_dists(binder, target)

        cplx_attn_bias = get_cplx_attn_bias(
            binder,
            target,
            bind_dists=bind_dist_H,
            target_dists=target_dist_H,
            inter_dists=inter_score,
        )

        z = self.post_mixer(z_pre, attn_mask=cplx_attn_bias)
        z = z + z_pre

        binder_z, target_z = z[:, :M, :], z[:, M:, :]

        binder_z = self.bind_z_final(binder_z)
        target_z = self.tgt_z_final(target_z)

        cplx_z = binder_z.unsqueeze(2) * target_z.unsqueeze(1)

        dist_z = cplx_z * inter_score.unsqueeze(-1)
        dist_z = torch.cat(
            [
                dist_z,
                binder_z.unsqueeze(2).expand(-1, -1, N, -1),
                target_z.unsqueeze(1).expand(-1, M, -1, -1),
            ],
            dim=-1,
        )

        dist_out = self.dist_predictor(dist_z)

        aa_logits = self.aa_predictor(z)

        if self.use_nll:
            mu = dist_out[..., 0]
            log_sigma = dist_out[..., 1]

            if upscale:
                mu = (mu * self.std) + self.mean

                log_sigma = log_sigma + math.log(self.std + 1e-8)

            if return_aa_logits:
                return (mu, log_sigma), aa_logits

            return mu, log_sigma

        x_recon = dist_out.squeeze(-1)

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

        binder_H = binder_H + binder_H_first
        target_H = target_H + target_H_first

        binder_H = self.bind_enc(binder_H)
        target_H = self.tgt_enc(target_H)

        return binder_H, target_H

    def get_latents(self, x: Tensor, binder: Features, target: Features) -> Tensor:
        """
        returns:
            z: (B, M+N, latent_dim) latent codes for each residue
            binder_H: (B, M, H) binder features
            target_H: (B, N, H) target features
        """

        mu, logvar = self.encode(x, binder, target)

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

        bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()

        aa_bind_labels = binder.Si.squeeze(-1)
        aa_tgt_labels = target.Si.squeeze(-1)

        bind_f1 = self.get_f1(aa_bind_logits, aa_bind_labels, bind_mask.squeeze(-1))
        tgt_f1 = self.get_f1(aa_tgt_logits, aa_tgt_labels, tgt_mask.squeeze(-1))

        f1_score = (bind_f1 + tgt_f1) / 2.0

        return f1_score

    def forward(
        self,
        x: Tensor,
        binder: Features,
        target: Features,
        pair_mask: Tensor,
        step=0,
        return_latent=False,
        return_recon=False,
    ):
        assert x.dim() == 3, f"Expected x of shape (B,M,N), got {x.shape}"

        if pair_mask.dtype != torch.float32:
            mask = pair_mask.float()
        else:
            mask = pair_mask

        x_phys = x
        x_norm = (x_phys - self.mean) / self.std

        bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)
        pair_mask_f = pair_mask.float()

        x_img = x_phys
        mu, logvar = self.encode(x_img, binder, target)

        z = self.reparameterize(mu, logvar)

        if return_latent:
            return z

        if self.use_nll:
            (x_mu, x_log_sigma), aa_recon = self.decode(
                z, binder, target, upscale=False, return_aa_logits=True
            )

            x_log_sigma = x_log_sigma.clamp(min=-7.0, max=7.0)
            inv_var = torch.exp(-2.0 * x_log_sigma)

            err2 = (x_norm - x_mu) ** 2
            nll = 0.5 * (err2 * inv_var + 2.0 * x_log_sigma + math.log(2.0 * math.pi))

            recon_loss = (nll * pair_mask_f).sum(dim=(1, 2)) / pair_mask_f.sum(
                dim=(1, 2)
            ).clamp(min=1.0)
            recon_loss = recon_loss.mean()

            x_recon_phys = (x_mu * self.std) + self.mean
            uncertainty = torch.exp(x_log_sigma) * self.std

        else:
            x_recon_norm, aa_recon = self.decode(
                z, binder, target, upscale=False, return_aa_logits=True
            )

            eps = 1e-6
            diff = x_recon_norm - x_norm
            charb = torch.sqrt(diff * diff + eps)
            recon_loss = (charb * pair_mask_f).sum(dim=(1, 2)) / pair_mask_f.sum(
                dim=(1, 2)
            ).clamp(min=1.0)
            recon_loss = recon_loss.mean()

            x_recon_phys = (x_recon_norm * self.std) + self.mean

            uncertainty = None

        aa_bind_logits = aa_recon[:, : binder.X.size(1), :]
        aa_tgt_logits = aa_recon[:, binder.X.size(1) :, :]
        aa_bind_labels = binder.Si.squeeze(-1)
        aa_tgt_labels = target.Si.squeeze(-1)

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

        triangle_ineq_loss = triangle_loss_efficient(
            binder, target, x_recon_phys.clamp(max=30), mask
        )
        sharp_loss = torch.tensor(0.0, device=x.device)

        valid_mask = torch.cat((bind_mask, tgt_mask), dim=1)
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
