import random

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from models.common import GaussianSmearing
from models.latent_modules import CustomTransformer, DiT
from models.triangle import PairEmbeddingInit, TriangleModule
from utils import Features, get_cplx_attn_bias

from .common import SinusoidalNoiseEmbedding


class LatentFlowMatching(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        model_dim=128,
        latent_dim=8,
        emb_size=1280,
        heads=8,
        conditioning_depth=6,
        noise_emb_dim=16,
        device=torch.device("cuda"),
        protein_mixer_depth=4,
        mean=0.0,
        std=1.0,
        latent_mean=0.0,
        latent_std=1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.device = device
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.mean, self.std = mean, std

        self.p_uncond = 0.2
        self.null_feature_bind = nn.Parameter(torch.randn(model_dim))
        self.null_feature_tgt = nn.Parameter(torch.randn(model_dim))

        self.noise_embedding = SinusoidalNoiseEmbedding(dim=noise_emb_dim)
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_emb_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.vae_proj = nn.Linear(hidden_dim, model_dim)
        self.latent_proj = nn.Linear(latent_dim, model_dim)
        self.residue_proj = nn.Linear(emb_size, model_dim)

        self.protein_mixer = CustomTransformer(
            dim=model_dim,
            heads=heads,
            mlp_ratio=4,
            depth=protein_mixer_depth,
        )

        self.prediction_head = nn.Linear(model_dim, latent_dim)

        c_pair = model_dim // 2
        self.pair_embedding = PairEmbeddingInit(
            c_pair=c_pair,
            num_aa=21,
            max_relpos=32,
            symmetric=True,
        )
        self.triangle = TriangleModule(
            c_pair=c_pair,
            c_hidden=model_dim,
            n_blocks=8,
        )
        self.to_bias = nn.Linear(model_dim, 1)
        self.conditioner = DiT(
            dim=model_dim,
            depth=conditioning_depth,
            heads=heads * 2,
            out_dim=model_dim,
        )

        self.gaussian_encoder = GaussianSmearing(start=0.0, stop=mean, num_gaussians=64)
        num_centers = self.gaussian_encoder.num_gaussians
        self.distance_mlp = nn.Sequential(
            nn.Linear(num_centers, num_centers),
            nn.LayerNorm(num_centers),
            nn.SiLU(),
            nn.Linear(num_centers, 1),
        )

    def _get_cond_features(self, bind_H, target_H, force_null=False):
        B = bind_H.shape[0]
        drop = force_null or (self.training and random.random() < self.p_uncond)
        if drop:
            M = bind_H.shape[1]
            N = target_H.shape[1]
            null_bind = self.null_feature_bind.view(1, 1, -1).expand(B, M, -1)
            null_tgt = self.null_feature_tgt.view(1, 1, -1).expand(B, N, -1)
            return null_bind, null_tgt

        return bind_H, target_H

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

    def get_cplx_dists(self, binder: Features, target: Features):
        cplx_X = torch.cat([binder.X, target.X], dim=1)
        cplx_internal_dist = torch.cdist(cplx_X[:, :, 1], cplx_X[:, :, 1])

        cplx_dist_H = self.distance_mlp(
            self.gaussian_encoder(cplx_internal_dist)
        ).squeeze(-1)
        return cplx_dist_H

    def predict_z1(
        self,
        binder: Features,
        target: Features,
        zt: Tensor,
        t: Tensor,
        bind_cond: Tensor,
        target_cond: Tensor,
        guidance_scale: float = 5.0,
        mix=False,
    ):
        """
        Predicts the clean data Z1 directly.
        Corresponds to F(Z(t), t, c) in Eq 11.
        """

        if mix:
            bind_cond, target_cond = self.mix_cond_features(
                binder, target, bind_cond, target_cond
            )

        t_embed = self.noise_embedding(t).to(self.device)
        t_embed = self.noise_proj(t_embed).unsqueeze(1)
        z1_cond = self._forward_backbone(
            binder, target, zt, bind_cond, target_cond, t_embed
        )

        if guidance_scale == 1.0:
            return z1_cond

        null_bind, null_tgt = self._get_cond_features(
            bind_cond, target_cond, force_null=True
        )
        z1_uncond = self._forward_backbone(
            binder, target, zt, null_bind, null_tgt, t_embed
        )

        return z1_uncond + guidance_scale * (z1_cond - z1_uncond)

    def get_velocity(
        self, binder: Features, target: Features, z_t, t, bind_cond, target_cond
    ):
        """
        Predicts the velocity (vector field) directly.
        Corresponds to F(Z(t), t, c) in Eq 11.
        """
        z1_pred = self.predict_z1(binder, target, z_t, t, bind_cond, target_cond)
        z1_pred = z1_pred.detach()

        velocity = (z1_pred - z_t) / (1.0 - t + 1e-4).view(-1, 1, 1)

        return velocity

    def _forward_backbone(
        self, binder: Features, target: Features, zt, binder_H, target_H, t_embed
    ):
        _M, _N = binder_H.size(1), target_H.size(1)

        cplx_z = torch.cat([binder_H, target_H], dim=1)

        pair_emb = binder_H.unsqueeze(2) * target_H.unsqueeze(1)

        bias = self.to_bias(pair_emb).squeeze(-1)

        bind_dist_H, tgt_dist_H = self.get_internal_dists(binder, target)
        cplx_dists = bias

        cplx_attn_mask = get_cplx_attn_bias(
            binder,
            target,
            inter_dists=cplx_dists,
            bind_dists=bind_dist_H,
            target_dists=tgt_dist_H,
        )

        cplx_z = self.protein_mixer(cplx_z, attn_mask=cplx_attn_mask)

        h_zt = self.latent_proj(zt)
        h_zt = self.conditioner(
            h_zt,
            cplx_z,
            t=t_embed.squeeze(1),
        )

        return self.prediction_head(h_zt)

    def mix_cond_features(
        self, binder: Features, target: Features, bind_H: Tensor, target_H: Tensor
    ):

        bind_cond, target_cond = self.vae_proj(bind_H), self.vae_proj(target_H)
        bind_res_S, target_res_S = binder.S, target.S
        bind_res_S, target_res_S = (
            self.residue_proj(bind_res_S),
            self.residue_proj(target_res_S),
        )
        bind_cond = bind_cond + bind_res_S
        target_cond = target_cond + target_res_S

        return bind_cond, target_cond

    def forward(
        self,
        binder: Features,
        target: Features,
        z1: Tensor,
        bind_cond: Tensor,
        target_cond: Tensor,
        res_mask: Tensor = None,
        guidance_scale: float = 1.0,
    ):
        """
        Flow Matching Loss (Eq 12).
        Minimizes || (Z1 - Zt)/(1-t) - (Pred - Zt)/(1-t) ||^2
        which simplifies to weighted MSE: || Z1 - Pred ||^2 / (1-t)^2
        """
        B = z1.size(0)

        bind_cond, target_cond = self.mix_cond_features(
            binder, target, bind_cond, target_cond
        )

        t = torch.rand(B, device=self.device).clamp(max=0.95)

        z0 = torch.randn_like(z1)
        z0 = z0 - z0.mean(dim=0, keepdim=True)
        z1_norm = (z1 - self.latent_mean) / self.latent_std

        t_view = t.view(B, 1, 1)
        zt = (1 - t_view) * z0 + t_view * z1_norm

        bind_cond_in, target_cond_in = self._get_cond_features(bind_cond, target_cond)
        z1_pred = self.predict_z1(
            binder, target, zt, t, bind_cond_in, target_cond_in, guidance_scale
        )

        weight = 1.0 / ((1.0 - t_view) ** 2)

        if res_mask is not None:
            res_mask_f = res_mask.float()
            mse = (z1_pred - z1_norm) ** 2
            weighted_mse = mse * weight

            loss = (weighted_mse * res_mask_f).sum() / res_mask_f.sum().clamp(min=1.0)
        else:
            mse = (z1_pred - z1_norm) ** 2
            loss = (mse * weight).mean()

        return loss

    def euler_step(
        self,
        zt: Tensor,
        z1_pred: Tensor,
        t_curr: float,
        dt: float,
    ):
        """
        Perform a single Euler integration step.
        Z(t+dt) = Z(t) + dt * (Z' - Z(t
        ) / (1 - t)
        """

        denom = 1.0 - t_curr
        if denom < 1e-5:
            update = z1_pred - zt
        else:
            vector_field = (z1_pred - zt) / denom
            update = vector_field * dt
        zt = zt + update
        return zt

    def sample(
        self,
        binder,
        target,
        bind_H: Tensor,
        target_H: Tensor,
        num_samples: int = 1,
        steps: int = 50,
        guidance_scale: float = 1.0,
        verbose=False,
    ):
        """
        Euler Integration using Algorithm 3 from the PDF.
        """
        if num_samples < 1 or steps < 1:
            raise ValueError("num_samples and steps must be positive")
        device = self.device
        M, N = bind_H.size(1), target_H.size(1)

        zt = torch.randn(num_samples, M + N, self.latent_dim, device=device)
        if num_samples > 1:
            zt = zt - zt.mean(dim=0, keepdim=True)
        zt = (zt - self.latent_mean) / self.latent_std

        dt = 1.0 / steps
        times = torch.linspace(0.0, 1.0, steps + 1, device=device)

        bind_H, target_H = self.mix_cond_features(binder, target, bind_H, target_H)

        if verbose:
            ctx = tqdm(range(steps))
        else:
            ctx = range(steps)

        for i in ctx:
            t_curr = times[i]
            t_batch = torch.full((num_samples,), t_curr, device=device)

            z1_pred = self.predict_z1(
                binder,
                target,
                zt,
                t_batch,
                bind_H,
                target_H,
                guidance_scale=guidance_scale,
            )

            zt = self.euler_step(zt, z1_pred, t_curr, dt)

        zt = zt * self.latent_std + self.latent_mean
        return zt
