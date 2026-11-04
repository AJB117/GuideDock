import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import Features
from latent_utils import get_pairwise_cplx_attn_mask
from models.latent_modules import (
    CustomCrossAttentionTransformer,
    CustomTransformer,
    GaussianSmearing,
    ResidualBlock,
    FusionAttentionModule,
    MLP,
    FiLM,
    DiTTransformer,
)


class SinusoidalNoiseEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode="constant")
        return emb


class LatentDDPM(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        model_dim=128,
        latent_dim=8,
        timesteps=500,
        heads=8,
        conditioning_depth=1,
        noise_emb_dim=16,
        device=torch.device("cuda"),
        loss_target="noise",
        protein_mixer_depth=2,
        mixer_depths=(2, 2),
        mean=0.0,
        latent_mean=0.0,  # per dim
        latent_std=1.0,  # per dim
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.device = device
        self.loss_target = loss_target
        self.latent_mean = latent_mean
        self.latent_std = latent_std

        # Simple β schedule
        #! diffusion
        self.betas = torch.linspace(1e-4, 2e-3, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.ac = torch.cumprod(self.alphas, dim=0)
        # self.register_buffer("betas", torch.linspace(1e-4, 2e-3, timesteps))
        # self.register_buffer("alphas", 1.0 - self.betas)
        # self.register_buffer("ac", torch.cumprod(self.alphas, dim=0))  # alpha_cumprod

        #! neural network
        self.noise_embedding = SinusoidalNoiseEmbedding(dim=noise_emb_dim)
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_emb_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.latent_proj = nn.Linear(latent_dim, model_dim)
        self.cond_proj = nn.Linear(hidden_dim + model_dim, model_dim)

        self.protein_mixer = CustomTransformer(
            dim=model_dim,
            heads=heads,
            mlp_ratio=4,
            depth=protein_mixer_depth,
        )

        # # Mixer layers before and after cross-attention
        # self.pre_mixer = nn.Sequential(
        #     nn.LayerNorm(latent_dim),
        #     CustomTransformer(
        #         dim=latent_dim,
        #         heads=heads,
        #         mlp_ratio=4,
        #         depth=mixer_depths[0],
        #     ),
        # )

        # self.post_mixer = nn.Sequential(
        #     nn.LayerNorm(latent_dim),
        #     CustomTransformer(
        #         dim=latent_dim,
        #         heads=heads,
        #         mlp_ratio=4,
        #         depth=mixer_depths[1],
        #     ),
        # )

        # Cross-attention layer for conditioning on protein features
        # self.cross_attn_layers = CustomCrossAttentionTransformer(
        #     dim=latent_dim,
        #     heads=heads,
        #     mlp_ratio=4,
        #     depth=conditioning_depth,
        # )

        self.score_head = nn.Linear(model_dim, latent_dim)

        self.conditioner = FusionAttentionModule(
            latent_dim=model_dim,
            heads=heads,
            depth=conditioning_depth,
            dropout=0.0,
        )

        # self.conditioner = CustomCrossAttentionTransformer(
        #     dim=model_dim,
        #     heads=heads,
        #     mlp_ratio=4,
        #     depth=conditioning_depth,
        # )

        self.gaussian_encoder = GaussianSmearing(start=0.0, stop=mean, num_gaussians=32)
        num_centers = self.gaussian_encoder.num_gaussians
        self.distance_mlp = nn.Sequential(
            # nn.Linear(num_centers, 1),
            nn.Linear(num_centers, num_centers // 2),
            nn.SiLU(),
            nn.Linear(num_centers // 2, 1),
        )

    def make_ddim_timesteps(self, ddim_steps: int, device=None):
        """
        Choose a subset of timesteps for DDIM sampling.
        Returns a 1D tensor of long indices of length ddim_steps, ascending.
        """
        if device is None:
            device = self.device

        # Linearly spaced timesteps from [0, T-1]
        # (classic DDIM-style schedule)
        return torch.linspace(
            0, self.timesteps - 1, steps=ddim_steps, dtype=torch.long, device=device
        )

    def embed_dists(self, binder: Features, target: Features, attn_mask: Tensor):
        bind_internal_dist = torch.cdist(binder.X[:, :, 1], binder.X[:, :, 1])
        target_internal_dist = torch.cdist(target.X[:, :, 1], target.X[:, :, 1])

        bind_dist_emb = self.gaussian_encoder(bind_internal_dist)
        target_dist_emb = self.gaussian_encoder(target_internal_dist)

        bind_dist_H = self.distance_mlp(bind_dist_emb).squeeze(-1)
        target_dist_H = self.distance_mlp(target_dist_emb).squeeze(-1)

        bind_dist_bias = bind_dist_H.unsqueeze(1)
        target_dist_bias = target_dist_H.unsqueeze(1)

        attn_mask[:, :, : binder.X.size(1), : binder.X.size(1)] = bind_dist_bias
        attn_mask[:, :, binder.X.size(1) :, binder.X.size(1) :] = target_dist_bias

        return attn_mask

    def condition(
        self,
        binder: Features,
        target: Features,
        zt: Tensor,
        bind_H: Tensor,
        tgt_H: Tensor,
        t_embed: Tensor,
    ):
        """
        Condition the latent zt on the protein features using cross-attention.
            zt: Noisy latent at time t, shape (B, M+N, latent_dim).
            bind_H: Binder protein features, shape (B, M, hidden_dim).
            tgt_H: Target protein features, shape (B, N, hidden_dim).
            t_embed: Time embedding, shape (B, 1, model_dim).
        """
        t_cplx = t_embed.expand(-1, zt.size(1), -1)  # (B, M+N, model_dim)
        cplx_z = torch.cat([bind_H, tgt_H], dim=1)  # [B, M+N, hidden_dim]
        cplx_z = torch.cat([cplx_z, t_cplx], dim=-1)  # (B, M+N, hidden_dim + model_dim)
        cplx_z = self.cond_proj(cplx_z)  # (B, M+N, model_dim)

        attn_mask, _ = get_pairwise_cplx_attn_mask(binder, target)  # (B, M+N, M+N)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, M+N, M+N)

        attn_mask = self.embed_dists(binder, target, attn_mask)

        cplx_z = self.protein_mixer(
            cplx_z, t=t_cplx, attn_mask=attn_mask
        )  # (B, M+N, model_dim)
        zt = self.latent_proj(zt)  # (B, M+N, model_dim)

        t_zt = t_embed.expand(-1, zt.size(1), -1)  # (B, M+N, model_dim)
        cplx_z = self.conditioner(zt, cplx_z, t=t_zt, attn_mask=attn_mask)

        return cplx_z  # (B, M+N, latent_dim)

        # cplx_z = torch.cat([bind_H, tgt_H], dim=1)  # [B, M+N, hidden_dim]
        # cplx_z = self.cond_proj(cplx_z)  # (B, M+N, latent_dim)
        # cplx_z = self.protein_mixer(cplx_z)  # (B, M+N, latent_dim)
        # cplx_z = self.film1(cplx_z, t_embed)

        # # Mixer before cross-attention
        # zt = self.pre_mixer[0](zt)
        # zt_pre = self.pre_mixer[1](zt)
        # zt_pre = self.film2(zt_pre, t_embed)
        # zt_pre = zt_pre + zt

        # # Cross-attention conditioning
        # zt_cross = self.cross_attn_layers(zt_pre, cplx_z)

        # # Mixer after cross-attention
        # zt = self.post_mixer[0](zt_cross)
        # zt = self.post_mixer[1](zt)
        # zt = self.film3(zt, t_embed)
        # zt = zt + zt_cross

        # return zt  # (B, M+N, latent_dim)

    def score(self, zt: Tensor):
        """
        Score function to predict noise or x0.
            zt: Noisy latent at time t, shape (B, M+N, model_dim).
        """
        score = self.score_head(zt)
        return score

    def q_sample(
        self,
        z0: Tensor,
        t: Tensor,
        noise=None,
    ):
        """
        Sample from q(z_t | z_0)
            z0: Original latent, shape (B, latent_dim).
            t: Time step tensor, shape (B,).
            noise: Optional noise to add, shape (B, latent_dim).
        """
        if noise is None:
            noise = torch.randn_like(z0)
        a = self.ac[t].view(-1, 1, 1)

        return torch.sqrt(a) * z0 + torch.sqrt(1 - a) * noise

    def predict_noise(
        self,
        binder: Features,
        target: Features,
        zt: Tensor,
        t: Tensor,
        bind_H: Tensor,
        target_H: Tensor,
    ):
        """
        Predict the noise added at time step t.
            zt: Noisy latent at time t, shape (B, M+N, latent_dim).
            t: Time step tensor, shape (B,).
            bind_H: Binder protein features, shape (B, M, hidden_dim).
            target_H: Target protein features, shape (B, N, hidden_dim).
            res_mask: Residue mask, shape (B, M+N).
        """
        # t_embed = t.float().unsqueeze(1) / self.timesteps
        t_embed = t.float().unsqueeze(1)
        t_embed = self.noise_embedding(t).to(self.device)
        t_embed = self.noise_proj(t_embed)
        t_embed = t_embed.unsqueeze(1)  # (B, 1, model_dim)

        zt_cond = self.condition(binder, target, zt, bind_H, target_H, t_embed)

        return self.score(zt_cond)

    def p_sample_ddim(
        self,
        binder: Features,
        target: Features,
        zt: torch.Tensor,  # (B, M+N, latent_dim)
        t: torch.Tensor,  # (B,) current timestep indices
        t_prev: torch.Tensor,  # (B,) next (previous) timestep indices in DDIM schedule
        bind_H: torch.Tensor,
        target_H: torch.Tensor,
        eta: float = 0.0,  # 0 = deterministic DDIM, >0 adds noise
    ):
        """
        DDIM update: p(z_{t_prev} | z_t) using the noise predictor.
        """
        # Predict epsilon (noise)
        eps = self.predict_noise(
            binder, target, zt, t, bind_H, target_H
        )  # (B, M+N, latent_dim)

        # alpha_cumprod at t and t_prev
        ac_t = self.ac[t].view(-1, 1, 1)  # (B,1,1)
        ac_prev = self.ac[t_prev].view(-1, 1, 1)  # (B,1,1)

        # Predict x0 from zt and eps
        # z_t = sqrt(ac_t) * x0 + sqrt(1 - ac_t) * eps
        x0_pred = (zt - torch.sqrt(1.0 - ac_t) * eps) / torch.sqrt(ac_t + 1e-8)

        if eta == 0.0:
            # Deterministic DDIM (no extra noise)
            z_prev = torch.sqrt(ac_prev) * x0_pred + torch.sqrt(1.0 - ac_prev) * eps
            return z_prev

        # Stochastic DDIM variant
        sigma = eta * torch.sqrt(
            (1.0 - ac_prev) / (1.0 - ac_t) * (1.0 - ac_t / ac_prev + 1e-8)
        )  # (B,1,1)

        noise = torch.randn_like(zt)
        # "direction" term
        dir_xt = torch.sqrt(torch.clamp(1.0 - ac_prev - sigma**2, min=0.0)) * eps

        z_prev = torch.sqrt(ac_prev) * x0_pred + dir_xt + sigma * noise
        return z_prev

    def p_sample(
        self,
        binder: Features,
        target: Features,
        zt: Tensor,
        t: Tensor,
        bind_H: Tensor,
        target_H: Tensor,
    ):
        """
        Sample from p(z_{t-1} | z_t)
            zt: Noisy latent at time t, shape (B, latent_dim).
            t: Time step tensor, shape (B,).
            bind_H: Binder protein features, shape (B, M, hidden_dim).
            target_H: Target protein features, shape (B, N, hidden_dim).
        """
        beta = self.betas[t].view(-1, 1, 1)
        a = self.alphas[t].view(-1, 1, 1)
        ac = self.ac[t].view(-1, 1, 1)

        pred = self.predict_noise(binder, target, zt, t, bind_H, target_H)

        if self.loss_target == "x0":
            mean = torch.sqrt(ac) * zt + torch.sqrt(1 - ac) * pred
            return mean
        else:
            mean = (1 / torch.sqrt(a)) * (zt - beta * pred / torch.sqrt(1 - ac))

        if t[0] == 0:
            return mean

        noise = torch.randn_like(zt)
        return mean + torch.sqrt(beta) * noise

    @torch.no_grad()
    def sample_ddim(
        self,
        binder: Features,
        target: Features,
        bind_H: torch.Tensor,
        target_H: torch.Tensor,
        num_samples: int = 1,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ):
        """
        DDIM sampling from p(z_0 | binder, target).

        binder/target: Features
        bind_H: (B, M, hidden_dim)
        target_H: (B, N, hidden_dim)
        num_samples: number of samples in the batch
        ddim_steps: number of DDIM steps (e.g. 20, 50, 100)
        eta: 0.0 for deterministic DDIM, >0 for stochastic DDIM
        """
        device = self.device
        M, N = bind_H.size(1), target_H.size(1)

        # Start from Gaussian noise
        zt = torch.randn(num_samples, M + N, self.latent_dim, device=device)

        # center and scale latents (same as in your training/sample)
        zt = (zt - zt.mean(dim=1, keepdim=True)) / (zt.std(dim=1, keepdim=True) + 1e-5)

        # Build DDIM timestep schedule (ascending)
        ddim_timesteps = self.make_ddim_timesteps(
            ddim_steps, device=device
        )  # (ddim_steps,)

        # Loop backwards over DDIM timesteps
        for i in reversed(range(ddim_steps)):
            t_scalar = ddim_timesteps[i].item()
            t = torch.full((num_samples,), t_scalar, device=device, dtype=torch.long)

            if i == 0:
                # Last step: we can directly predict x0 and stop
                eps = self.predict_noise(binder, target, zt, t, bind_H, target_H)
                ac_t = self.ac[t].view(-1, 1, 1)
                x0_pred = (zt - torch.sqrt(1.0 - ac_t) * eps) / torch.sqrt(ac_t + 1e-8)
                zt = x0_pred  # z_0
                break

            t_prev_scalar = ddim_timesteps[i - 1].item()
            t_prev = torch.full(
                (num_samples,), t_prev_scalar, device=device, dtype=torch.long
            )

            zt = self.p_sample_ddim(
                binder,
                target,
                zt,
                t,
                t_prev,
                bind_H,
                target_H,
                eta=eta,
            )

        return zt

    @torch.no_grad()
    def sample(
        self,
        binder: Features,
        target: Features,
        bind_H: Tensor,
        target_H: Tensor,
        num_samples=1,
    ):
        """
        Generate samples from the DDPM model conditioned on protein features.
            bind_H: Binder protein features, shape (M, hidden_dim).
            target_H: Target protein features, shape (N, hidden_dim).
            num_samples: Number of samples to generate.
        """
        M, N = bind_H.size(1), target_H.size(1)
        zt = torch.randn(num_samples, M + N, self.latent_dim).to(self.device)  # z0

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full(
                (num_samples,), t, device=self.device, dtype=torch.long
            )
            zt = self.p_sample(binder, target, zt, t_batch, bind_H, target_H)

        return zt  # zT

    def forward(
        self,
        binder: Features,
        target: Features,
        z: Tensor,  # (B, M+N, latent_dim)
        bind_H: Tensor,  # (B, M, hidden_dim)
        target_H: Tensor,  # (B, N, hidden_dim)
        step: int,
        res_mask: Tensor = None,  # (B, M+N, 1) float/bool, 1 = valid
    ):
        """
        Compute DDPM training loss on latents.

        z:        (B, M+N, latent_dim)
        bind_H:   (B, M, hidden_dim)
        target_H: (B, N, hidden_dim)
        res_mask: (B, M+N, 1) optional residue mask (1 = valid, 0 = padded)
        """
        B = z.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=self.device).long()

        noise = torch.randn_like(z)

        # center and scale latents
        z = (z - self.latent_mean) / self.latent_std

        zt = self.q_sample(z, t, noise=noise)

        noise_pred = self.predict_noise(binder, target, zt, t, bind_H, target_H)

        if self.loss_target == "noise":
            if res_mask is not None:
                # res_mask: (B, M+N, 1)
                res_mask_f = res_mask.float()
                diff2 = (noise_pred - noise) ** 2
                diff2 = diff2 * res_mask_f
                loss = diff2.sum() / res_mask_f.sum().clamp(min=1.0)
            else:
                loss = F.mse_loss(noise_pred, noise)
        elif self.loss_target == "x0":
            # If you keep x0 target, remember noise_pred is still epsilon.
            a = self.ac[t].view(-1, 1, 1)
            z0_pred = (zt - torch.sqrt(1 - a) * noise_pred) / torch.sqrt(a)
            if res_mask is not None:
                res_mask_f = res_mask.float()
                diff2 = (z0_pred - z) ** 2
                diff2 = diff2 * res_mask_f
                loss = diff2.sum() / res_mask_f.sum().clamp(min=1.0)
            else:
                loss = F.mse_loss(z0_pred, z)
        else:
            raise ValueError(f"Unknown loss target: {self.loss_target}")

        return loss
