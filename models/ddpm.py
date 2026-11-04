# Code largely based on:
# https://github.com/lucidrains/denoising-diffusion-pytorch

import pdb
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce
import warnings

from .ddpm_utils import (
    default,
    extract,
    linear_beta_schedule,
    cosine_beta_schedule,
    center_zero,
    assert_center_zero,
)
from .energy import AllAtomEnergyModel


class GaussianDiffusion(nn.Module):
    """DDPM model with Gaussian noise."""

    def __init__(
        self,
        model,
        timesteps=50000,
        loss_type="l2",
        objective="pred_noise",
        beta_schedule="cosine",
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        norm_factor=1,  # scale input, recommended: scale by variance
        loss_weights="ones",  # ones, score_matching
    ):
        super().__init__()
        self.dims = 3
        self.model: AllAtomEnergyModel = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.objective = objective

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.norm_factor = norm_factor

        if self.model.resolution == "CA":
            self.sel, self.bb = 1, 0
        elif self.model.resolution == "backbone":
            self.sel, self.bb = 4, 1
        elif self.model.resolution == "all_atom":
            self.sel, self.bb = 14, 1


        def register_buffer(name, val):
            """
            Helper function to register buffer from float64 to float32
            """
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        if loss_weights == "ones":
            # calculate p2 reweighting
            register_buffer(
                "p2_loss_weight",
                (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
                ** -p2_loss_weight_gamma,
            )
        elif loss_weights == "score_matching":
            # calculate pe reweighting
            unnormalized = 1.0 / (1 - alphas_cumprod)
            normalized = unnormalized / sum(unnormalized) * len(alphas_cumprod)
            register_buffer(
                "p2_loss_weight",
                unnormalized,
            )
        elif "higheruntil_" in loss_weights:
            # calculate pe reweighting
            threshold = int(loss_weights.split("_")[1])
            weight_1 = len(alphas_cumprod) / (threshold)
            weight_2 = len(alphas_cumprod) / (len(alphas_cumprod) - threshold)
            register_buffer(
                "p2_loss_weight",
                torch.Tensor(
                    [weight_1] * threshold
                    + [weight_2] * (len(alphas_cumprod) - threshold)
                ),
            )
        elif "lower_bound" in loss_weights:
            clamp_val = int(loss_weights.split("_")[2])
            unnormalized = (1.0 / ((1 - alphas_cumprod) * (1 - betas))).clip(
                0, clamp_val
            )
            normalized = unnormalized / sum(unnormalized) * len(betas)
            register_buffer(
                "p2_loss_weight",
                normalized,
            )
        else:
            raise Exception(f"Wrong loss_weights: {loss_weights}")

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict input molecule form noisy molecule.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Calculate posterior of forward process.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def q_mean_variance(self, x_start, t):
        """
        Posterior of forward process of x_T given x_0, used in assert_normal_kl.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    @torch.no_grad()
    def assert_normal_kl(self, x_start, mask, t, eps=1e-3):
        """
        Check if the KL divergence between posterior q(x_T|x_0) and prior p(x_T)
        is close to zero. Basically checks if we have enough diffusion time steps.
        """
        # assert_center_zero(x_start, mask)
        mean1, _, logvar1 = self.q_mean_variance(x_start, t)
        logvar1 = logvar1.squeeze()
        mean2, logvar2 = torch.zeros_like(mean1), torch.zeros_like(logvar1)
        meandifsq = ((mean1 - mean2) ** 2).sum(dim=(-2, -1))
        normal_kl = 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + meandifsq * torch.exp(-logvar2)
        )
        assert normal_kl.abs().max().item() <= eps, (
            f"Normal KL check at T failed, max value: {normal_kl.abs().max().item()}"
        )

    def p_mean_variance(self, binder, target, mask, t):
        """
        Get mean and variance of approximated posterior from model.
        """
        binder_X = binder[0].view(
            binder[0].shape[0], -1, 3
        )
        # assert_center_zero(binder_X, mask)
        binder = (
            binder_X.view(1, binder[0][0].shape[0]//self.sel, self.sel, 3),
            *binder[1:]
        )
        model_output = self.model.gaussian_forward(
            binder,
            target,
            1.0 * t / self.num_timesteps,
            self.model.resolution
        ).view(binder_X.shape[0], -1, 3)
        # model_output = center_zero(model_output, mask)

        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(binder_X, t=t, noise=model_output)
            x_start = center_zero(x_start, mask)
        elif self.objective == "pred_x0":
            x_start = model_output
        else:
            raise ValueError(f"unknown objective {self.objective}")

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=binder_X, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, binder, target, mask, t):
        """
        Single sample from model given (noisy) molecule x and timestep t.
        """
        b = binder[0][0].shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(binder, target, t=t, mask=mask)
        noise = torch.randn_like(binder[0].view(
            b, -1, 3
        ))
        noise = center_zero(noise, mask)
        # no noise when t == 0
        # nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # pdb.set_trace()
        # nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * 2))
        return model_mean * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, binder, target, shape, mask):
        """
        Loop over diffusion timesteps to go from noise to molecule.
        """
        device = self.betas.device

        b = shape[0]
        mol = center_zero(torch.randn(shape, device=device))

        for j, i in enumerate(reversed(range(0, self.num_timesteps))):
            mol = self.p_sample(
                binder, target, mask, torch.full((b,), i, device=device, dtype=torch.long)
            )
            if (mol.max() > 1000) or (mol.min() < -1000):
                warnings.warn("Large molecule encountered in sampling")
                mol = torch.clamp(mol, min=-1000, max=1000)
            mol = center_zero(mol)
        # assert_center_zero(mol, mask)

        return mol

    @torch.no_grad()
    def sample(self, binder, target, mask, batch_size):
        """
        Sample from model.
        num_atoms: 1-D LongTensor [batch_size] giving atom-counts
                OR python list/tuple.
        batch_size: int, number of molecules to sample.

        """
        # dims = self.dims
        # return self.p_sample_loop((batch_size, num_atoms, dims)) * self.norm_factor
        device = self.betas.device
        num_atoms = [len(bind[0]) for bind in binder]
        N_max = max(num_atoms)
        # build (B,N_max,3) gaussian noise and (B,N_max) mask
        mol = torch.randn(batch_size, N_max*4, 3, device=device)
        mol = center_zero(mol, mask)
        # p_sample_loop rewritten to carry the mask through
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            mol = self.p_sample((mol, *binder[1:]), target, mask, t)
            mol = torch.where(mask.unsqueeze(-1), mol, torch.zeros_like(mol))
            mol = center_zero(mol, mask)

        return mol * self.norm_factor, mask  # (B,N_max,3), (B,N_max)

    def q_sample(self, binder, mask, t, noise=None):
        """
        Sample noisy molecule from forward process.
        """
        x_start, *_ = binder
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise = center_zero(noise, mask)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        """
        Loss function.
        """
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    #     def p_losses(self, x_start, t, noise=None):
    #         """
    #         Calculate loss from model.
    #         """
    #         noise = default(noise, lambda: torch.randn_like(x_start))
    #         noise = center_zero(noise)

    #         x = self.q_sample(x_start=x_start, t=t, noise=noise)
    #         x = center_zero(x)
    #         model_out = self.model(
    #             x,
    #             1.0 * t / self.num_timesteps,
    # )
    #         model_out = center_zero(model_out)

    #         if self.objective == "pred_noise":
    #             target = noise
    #         elif self.objective == "pred_x0":
    #             target = x_start
    #         else:
    #             raise ValueError(f"unknown objective {self.objective}")

    #         loss = self.loss_fn(model_out, target, reduction="none")
    #         loss = reduce(loss, "b ... -> b (...)", "mean")

    #         return loss.mean()

    def p_losses(self, binder, target, t, mask, noise=None):
        B, N_num_bb, num_coords = binder[0].shape # (B, N * num_bb, 3)
        num_bb = self.sel
        N = N_num_bb // num_bb  # number of atoms

        noise = default(noise, lambda: torch.randn_like(binder[0]))
        x = self.q_sample(binder, mask, t, noise=noise)
        noised_binder = (x.view(B, N, num_bb, num_coords), *binder[1:])
        time_steps = torch.tensor(1.0 * t / self.num_timesteps).to(self.device).view(-1, 1)
        model_out = self.model.gaussian_forward(
            noised_binder, target, time_steps, resolution="backbone", return_nodes=True
        )  # if your model needs it
        model_out = model_out.view(B, N * num_bb, 3) * mask.unsqueeze(-1)
        ground_truth = noise if self.objective == "pred_noise" else binder
        loss = self.loss_fn(model_out, ground_truth, reduction="none")
        loss = (loss.mean(-1) * mask).sum() / mask.sum()  # mask-weighted
        return loss

    # def forward(self, mol, *args, t_diff_range=None, **kwargs):
    #     mol = center_zero(mol) / self.norm_factor
    #     assert_center_zero(mol)
    #     b, n, d, device, num_atoms, dims, T = (
    #         mol.shape[0],
    #         mol.shape[1],
    #         mol.shape[2],
    #         mol.device,
    #         self.num_atoms,
    #         self.dims,
    #         self.num_timesteps - 1,
    #     )
    #     assert (
    #         n == num_atoms and d == dims
    #     ), f"Molecule shape must be {(num_atoms, dims)}"

    #     t = torch.multinomial(self.p2_loss_weight, b, replacement=True).long()
    #     self.assert_normal_kl(
    #         x_start=mol, t=torch.full((b,), T, device=device, dtype=torch.long)
    #     )
    #     return self.p_losses(mol, t, *args, **kwargs)

    def forward(self, binder, target, mask, *args, **kwargs):
        """
        mol  : (B,N_max,3) padded coordinates
        mask : (B,N_max)   1=real atom, 0=pad
        """
        binder_X, *_ = binder
        B, N, num_bb, num_coords = binder_X.shape # (B, N, 4, 3)

        binder_X = binder_X.view(B, N * num_bb, num_coords)
        mask = mask.view(B, N * num_bb)
        # binder_X = center_zero(binder_X / self.norm_factor, mask)
        b = binder_X.shape[0]
        t = torch.multinomial(self.p2_loss_weight, b, replacement=True).long()
        # binder_X = binder_X.view(B, N, num_bb, num_coords)

        centered_binder = (binder_X, *binder[1:])  # keep other binder parts unchanged
        self.assert_normal_kl(binder_X, mask, t.new_full((b,), self.num_timesteps - 1))
        loss = self.p_losses(centered_binder, target, t, mask=mask, *args, **kwargs)
        return loss
