import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from scipy.spatial.transform import Rotation

from .modules import (
    FiLM,
    GLU,
    SinusoidalEmbedding,
    GaussianFourierProjection,
    ScalarHead,
    ForceHead,
    StatPooling
)
from utils import sample_vec, score_vec, score_norm
from .frame import *

torch.autograd.set_detect_anomaly(True)

def get_indices(resolution):
    sel, bb = 14, 1
    if resolution == "CA":
        sel, bb = 1, 0
    elif resolution == "backbone":
        sel, bb = 4, 1
    elif resolution == "all_atom":
        sel, bb = 14, 1
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    return sel, bb


@torch.jit.script
def rotate(X, w, sel: int):
    B, N = X.size(0), X.size(1)
    X = X.reshape(B, N * sel, 3)
    w = w.unsqueeze(1).expand_as(X)  # [B,N,3]
    c = w.norm(p=2, dim=-1, keepdim=True)  # [B,N,1]
    c1 = torch.sin(c) / c.clamp(min=1e-6)
    c2 = (1 - torch.cos(c)) / (c**2).clamp(min=1e-6)
    X = X + c1 * cross_product(w, X) + c2 * cross_product(w, cross_product(w, X))
    return X.view(B, N, sel, 3)


class AllAtomEnergyModel(nn.Module):
    def __init__(self, args, device, resolution):
        super(AllAtomEnergyModel, self).__init__()
        self.mse_loss = nn.MSELoss()

        # Noise type: "se3" (default, original behavior) or "gaussian" (per-atom isotropic)
        self.noise_type = getattr(args, "noise_type", "se3").lower()
        if self.noise_type not in {"se3", "gaussian"}:
            raise ValueError("noise_type must be 'se3' or 'gaussian'")

        hidden_size = args.hidden_size

        self.threshold = args.threshold
        self.encoder = AllAtomEncoder(args)
        hsize = (
            hidden_size // 2
            if args.encoder not in ("gt", "flash_ipa", "sa")
            else hidden_size
        )
        self.W_o = nn.Sequential(
            nn.Linear(hsize, hsize),
            nn.ReLU(),
            nn.Linear(hsize, hsize),
        )
        self.U_o = nn.Sequential(
            nn.Linear(hsize, hsize),
            nn.ReLU(),
            nn.Linear(hsize, hsize),
        )
        self.eye = torch.eye(3).unsqueeze(0).cuda()
        self.device = device
        self.noise_scaler = args.noise_scaler
        self.resolution = resolution
        self.bb = 0 if self.resolution == "CA" else 1

        if self.resolution == "all_atom":
            self.residue_size = 14
            self.bb = 1
        elif self.resolution == "backbone":
            self.residue_size = 4
            self.bb = 1
        elif self.resolution == "CA":
            self.residue_size = 1
            self.bb = 0
        else:
            raise ValueError(f"Unsupported resolution: {self.resolution}")

        self.non_conservative = args.non_conservative

        # (SE(3) score heads)
        self.tr_scalar_head = ScalarHead(args, depth=3)
        self.tr_scalar_head.compile()
        self.rot_scalar_head = ScalarHead(args)
        self.rot_scalar_head.compile()
        self.noise_conditioner = SinusoidalEmbedding(args.noise_emb_dim)
        self.noise_conditioner.compile()

        self.coarse_W_o = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.coarse_U_o = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            # nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, hidden_size),
            # nn.Linear(hidden_size, 3),
            # GLU(hidden_size, args, act="relu"),
            nn.Linear(hidden_size, hidden_size),
        )
        self.node_decoder.compile()

        if self.non_conservative:
            self.tr_force_head = ForceHead(args, depth=3)
            self.rot_force_head = ForceHead(args)

        self.film_layer_1 = FiLM(hidden_size, args)
        self.noise_proj = nn.Sequential(
            nn.Linear(args.noise_emb_dim * 2, hidden_size),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        self.noise_proj.compile()
        self.norm_list = []
        self.norm_diff_list = []
        self.cos_sim_list = []

        # self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.zeros_(m.weight)
                nn.init.xavier_uniform_(m.weight)
                # nn.init.orthogonal_(m.weight)

    # -------------------
    # Noise helpers
    # -------------------
    def get_noise_terms(self, binder, eps_tr, eps_rot):
        """Return noise terms depending on self.noise_type.

        For noise_type == 'se3':
            - w:      Bx3 rotation vectors (per-structure)
            - hat_t:  Bx3 translations (per-structure)
            - backbone_score: target SE(3) rotational score (Bx3)

        For noise_type == 'gaussian':
            - w:      zeros (unused)
            - hat_t:  BxNxselx3 per-atom Gaussian noise scaled by eps_tr
            - backbone_score: zeros (unused)
        """
        true_X = binder[0]
        B = true_X.size(0)
        sel = self.residue_size

        if self.noise_type == "se3":
            # Rotational noise & target rotational score (per-structure)
            w = torch.zeros(B, 3)
            backbone_score = torch.zeros(B, 3)
            for i in range(B):
                w_ = torch.from_numpy(sample_vec(eps_rot[i].item()))
                backbone_score_ = score_vec(eps_rot[i].item(), w_)
                w[i] = w_
                backbone_score[i] = backbone_score_

            # Translation noise (per-structure)
            hat_t = torch.from_numpy(
                np.random.normal(0, eps_tr.cpu().numpy(), (B, 3))
            ).float()

            return (
                w.cuda(true_X.device),
                hat_t.cuda(true_X.device),
                eps_tr,
                backbone_score.cuda(true_X.device),
            )

        # Gaussian per-atom noise
        # hat_t shape matches coordinates
        eps_tr_broadcast = eps_tr.view(-1, 1, 1, 1).to(true_X)
        hat_t = torch.randn_like(true_X).to(true_X) * eps_tr_broadcast
        w = torch.zeros(true_X.size(0), 3, device=true_X.device, dtype=true_X.dtype)
        backbone_score = torch.zeros_like(w)

        return w, hat_t, eps_tr, backbone_score

    def add_noise(self, binder, w, hat_t, resolution):
        true_X, _, bind_A, _, _ = binder

        if self.noise_type == "gaussian":
            # Per-atom Gaussian perturbation
            # Respect atom presence mask
            sel, bb = get_indices(self.resolution)
            bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
            atom_mask = (bind_A > 0).float().unsqueeze(-1)
            bind_X = true_X + hat_t * atom_mask
            bind_mask = bind_A[:, :, bb].clamp(max=1).float()
            return bind_X, bind_mask

        # SE(3) (original behavior): global rotate + translate around backbone center
        # if resolution == "backbone":
        #     bb = 1
        # elif resolution == "CA":
        #     bb = 0
        sel, bb = get_indices(resolution)

        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        bind_mask = bind_A[:, :, bb].clamp(max=1).float()

        # center = self.mean(true_X[:, :, bb], bind_mask)
        
        B = true_X.shape[0]
        center = true_X.view(B, -1, 3).sum(dim=1, keepdim=True)[:, None, ...] # [B, 1, 1, 3]
        nonzero = bind_mask.sum(dim=1, keepdim=True)[:, None, None, :] # [B, 1, 1, 1]
        center = center / ( nonzero * sel )

        bind_X = true_X - center
        # bind_X = rotate(X=bind_X, w=w, sel=sel) + hat_t[:, None, None, :]
        bind_X = bind_X + hat_t[:, None, None, :]
        bind_X = bind_X + center  # move back
        bind_X = bind_X * (bind_A > 0).float().unsqueeze(-1)

        return bind_X, bind_mask

    # -------------------
    # Model forward & utilities
    # -------------------
    def predict(
        self, binder, target, resolution, time_steps, eps_tr, return_energy=False, return_nodes=False
    ):
        # time_steps = self.noise_proj(time_steps)

        bind_X, bind_S, bind_A, bind_O, bind_V = binder
        tgt_X, tgt_S, tgt_A, tgt_O, tgt_V = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        sel, _ = get_indices(self.resolution)

        h1, h2 = self.encoder(
            (bind_X, bind_S, bind_A, bind_O, bind_V),
            (tgt_X, tgt_S, tgt_A, tgt_O, tgt_V),
            time_steps=time_steps,
            eps_tr=eps_tr,
        )  # [B,N+M,14,H]
        bind_h = h1.view(B, N * sel, -1)
        tgt_h = h2.view(B, M * sel, -1)

        if return_nodes:
            return bind_h, tgt_h

        # mean pooling to get per-protein scores
        bind_h = bind_h.mean(dim=1)
        tgt_h = tgt_h.mean(dim=1)

        combined = torch.cat((bind_h, tgt_h), dim=-1)
        # combined = bind_h
        f = self.node_decoder(combined)
        f = f + time_steps
        until = N

        return f, until

    def get_forces(self, force, until, bind_X, bind_mask, time_steps):
        # force_h = torch.cat((force, time_steps), dim=-1)
        tr_force, rot_force = (
            # self.tr_force_head(force),
            # self.rot_force_head(force),
            self.tr_force_head(force, time_steps),
            self.rot_force_head(force, time_steps)
        )
        return tr_force, rot_force


    def forward(
        self, binder, target, eps_tr, eps_rot, resolution=None, time_steps=None
    ):
        # time_steps = self.noise_conditioner(time_steps)
        time_emb = self.noise_conditioner(time_steps)
        tr_emb = self.noise_conditioner(eps_tr)
        rot_emb = self.noise_conditioner(eps_rot)
        time_steps = torch.cat((tr_emb, rot_emb), dim=-1)
        time_steps = self.noise_proj(time_steps)
        # Branch on noise type
        if self.noise_type == "gaussian":
            # Per-atom Gaussian perturbation training (overdamped Langevin)
            true_X, bind_S, bind_A, bind_O, bind_V = binder
            tgt_X, tgt_S, tgt_A, tgt_O, tgt_V = target

            w, hat_t, eps, _ = self.get_noise_terms(binder, eps_tr, eps_rot)
            noised_bind_X, _ = self.add_noise(binder, w, hat_t, self.resolution)
            noised_binder = (noised_bind_X, bind_S, bind_A, bind_O, bind_V)

            if not self.non_conservative:
                noised_bind_X = noised_bind_X.requires_grad_()

            # Predict forces (per-atom)
            force, until = self.predict(
                noised_binder,
                (tgt_X, tgt_S, tgt_A, tgt_O, tgt_V),
                resolution or self.resolution,
                time_steps,
                eps_tr,
            )

            # Compute masked per-atom score-matching loss
            B = bind_S.size(0)
            sel, bb = get_indices(self.resolution)
            bind_A_eff = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
            atom_mask = (bind_A_eff > 0).float().unsqueeze(-1)

            eps_scalar = eps_tr.view(B, 1, 1, 1).to(force)
            # target score for Gaussian: -hat_t / eps^2, but Gaussian-forward used force*eps vs -hat_t/eps
            # We stick with that convention for stability
            diff = (force[:, :until] * eps_scalar) - (-hat_t / eps_scalar**2)
            per_elem = (diff**2) * atom_mask[:, :until]
            # average per-sample
            denom = atom_mask[:, :until].reshape(B, -1).sum(dim=1).clamp_min(1.0)
            loss = per_elem.reshape(B, -1).sum(dim=1) / denom
            return loss

        # === SE(3) path (original behavior) ===
        rotation_force, hat_t, eps, backbone_score = self.get_noise_terms(
            binder, eps_tr, eps_rot
        )
        noised_bind_X, bind_mask = self.add_noise(
            binder, rotation_force, hat_t, self.resolution
        )
        noised_binder = (noised_bind_X, *binder[1:])

        _, bb = get_indices(self.resolution)

        bind_X, bind_S, bind_A, bind_O, bind_V = noised_binder

        if not self.non_conservative:
            bind_X.requires_grad_()

        tgt_X, tgt_S, tgt_A, tgt_O, tgt_V = target

        force, until = self.predict(
            (bind_X, bind_S, bind_A, bind_O, bind_V),
            (tgt_X, tgt_S, tgt_A, tgt_O, tgt_V),
            resolution or self.resolution,
            time_steps,
            eps_tr,
        )

        rot_score_norm = score_norm(eps_rot)
        translation_score = -hat_t / eps_tr**2

        tr_force, rot_force = self.get_forces(
            force, until, bind_X, bind_mask, time_steps
        )

        pred_rot_score, pred_tr_score = self.predict_score(
            tr_force, rot_force, eps_tr, eps_rot, time_steps
        )

        tr_loss = ((pred_tr_score - translation_score) ** 2 * eps_tr**2).mean(dim=-1)
        # tr_loss = ((pred_tr_score - translation_score) ** 2).mean(dim=-1)

        rot_loss = (((pred_rot_score - backbone_score) / rot_score_norm) ** 2).mean(
            dim=-1
        )

        # rot_loss = (
        #     (pred_rot_score - backbone_score) ** 2
        # ).mean(dim=-1)

        torch._assert(torch.isfinite(rot_loss).all(), "rot_loss has NaNs/Inf")
        torch._assert(torch.isfinite(tr_loss).all(), "tr_loss has NaNs/Inf")

        loss = 0.5 * rot_loss + 0.5 * tr_loss
        return loss

    # -------------------
    # Sampling
    # -------------------
    def sample(
        self,
        binder,
        target,
        T,
        alpha_,
        eta_start,
        batch_size,
        noise_schedule,
        resolution="backbone",
    ):
        self.eval()
        sel, bb = get_indices(self.resolution)

        true_X, bind_S, bind_A, bind_O, bind_V = binder
        tgt_X, tgt_S, tgt_A, tgt_O, tgt_V = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        bind_mask = bind_A[:, :, bb].clamp(max=1).float()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        # Initialize binder coordinates
        bind_X = true_X.clone()
        if not self.non_conservative:
            bind_X = bind_X.requires_grad_()

        B = batch_size  # Update batch size
        alpha_start = alpha_

        time_steps = np.linspace(1, 0, T + 1)[:-1]
        trajectory = [bind_X.clone().detach().cpu().numpy()]

        with torch.set_grad_enabled(not self.non_conservative):
            for t_idx in range(T):
                cur_t = time_steps[t_idx]
                if t_idx == T - 1:
                    dt = cur_t
                else:
                    dt = cur_t - time_steps[t_idx + 1]

                # time_step = torch.tensor(dt).float().view(1, 1).cuda()

                if self.noise_scaler == "linear":
                    eta = self.get_linear_scale(dt, T, eta_start)
                elif self.noise_scaler == "cosine":
                    eta = self.get_cosine_scale(dt, T, eta_start)
                    alpha = self.get_linear_scale(dt, T, alpha_start)
                else:  # default to uniform
                    eta = eta_start

                if self.noise_type == "gaussian":
                    # Per-atom Langevin updates using predicted forces directly
                    force, until, energy = self.predict(
                        (bind_X, bind_S, bind_A, bind_O, bind_V),
                        (tgt_X, tgt_S, tgt_A, tgt_O, tgt_V),
                        resolution,
                        time_step,
                        return_energy=True,
                    )

                    tr_s, _ = noise_schedule(dt, dt)
                    tr_s = torch.tensor(tr_s).float().cuda(bind_X.device).view(1, 1)

                    tr_scale = torch.sqrt(
                        torch.log(
                            torch.tensor(noise_schedule.tr_s_max / noise_schedule.tr_s_min)
                        )
                    )
                    tr_g = tr_s * tr_scale

                    epsilon_t = torch.normal(0, 1, size=(batch_size, N, sel, 3)).cuda(
                        device=self.device
                    )

                    delta_X = (tr_g**2) * dt * force[:, :until]
                    delta_X = delta_X + (tr_g * math.sqrt(dt) * epsilon_t)

                    bind_X = bind_X + delta_X
                    bind_X = bind_X * atom_mask

                    if not self.non_conservative:
                        bind_X = bind_X.detach().requires_grad_()

                    continue  # next step

                # === SE(3) sampling (original path) ===
                tr_s, rot_s = noise_schedule(cur_t, cur_t)
                tr_s, rot_s = (
                    torch.tensor(tr_s).float().cuda(bind_X.device).view(1, 1),
                    torch.tensor(rot_s).float().cuda(bind_X.device).view(1, 1),
                )
                # tr_s, rot_s = tr_s.float().cuda(bind_X.device).view(1, 1), rot_s.float().cuda(bind_X.device).view(1, 1)

                time_step = self.noise_conditioner(
                    torch.tensor(cur_t).float().view(1, 1)
                ).to(bind_X.device)
                tr_emb = self.noise_conditioner(tr_s)
                rot_emb = self.noise_conditioner(rot_s)
                time_step = torch.cat((tr_emb, rot_emb), dim=-1)
                time_step = self.noise_proj(time_step)

                out = self.predict(
                    (bind_X, bind_S, bind_A, bind_O, bind_V),
                    (tgt_X, tgt_S, tgt_A, tgt_O, tgt_V),
                    resolution,
                    time_step,
                    tr_s,
                    return_energy=not self.non_conservative,
                )
                if self.non_conservative:
                    force, until = out
                else:
                    force, until, energy = out

                tr_force, rot_force = self.get_forces(
                    force, until, bind_X, bind_mask, time_step
                )
                if self.non_conservative:
                    energy = tr_force.norm() + rot_force.norm()

                # Center and compute scores
                # center = self.mean(bind_X[:, :until, bb], bind_mask)  # [B, 3]
                center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[:, None, ...]
                nonzero = bind_mask.sum(dim=1, keepdim=True)[:, None, None, :] # [B, 1, 1, 1]
                center = center / (nonzero * sel)

                bind_X_centered = bind_X - center  # [B, N, sel, 3]
                # bind_X_centered = bind_X

                rot_score, tr_score = self.predict_score(
                    tr_force, rot_force, tr_s, rot_s, time_step
                )

                # print(tr_score.detach(), rot_score.detach())

                tr_scale = torch.sqrt(
                    torch.log(
                        torch.tensor(noise_schedule.tr_s_max / noise_schedule.tr_s_min)
                    )
                )
                tr_g = tr_s * tr_scale

                rot_scale = torch.sqrt(
                    torch.log(
                        torch.tensor(noise_schedule.rot_s_max / noise_schedule.rot_s_min)
                    )
                )
                rot_g = 2 * rot_s * rot_scale

                # Generate noise
                if t_idx == T - 1:
                    epsilon_t = torch.zeros(batch_size, 3).cuda(device=self.device)
                    epsilon_w = torch.zeros(batch_size, 3).cuda(device=self.device)
                else:
                    epsilon_t = torch.normal(0, 1, size=(batch_size, 3)).cuda(
                        device=self.device
                    )
                    epsilon_w = torch.normal(0, 1, size=(batch_size, 3)).cuda(
                        device=self.device
                    )

                # Updates
                delta_t = tr_g**2 * dt * tr_score
                delta_t = delta_t + (tr_g * math.sqrt(dt) * epsilon_t)

                delta_w = rot_g**2 * dt * rot_score
                delta_w = delta_w + (rot_g * math.sqrt(dt) * epsilon_w)

                # temp sampling
                # temp_sampling = 0.3
                # temp_sigma_data_tr = 0.593
                # temp_sigma_data_rot = 0.228
                # temp_psi = 0.216

                # tr_sigma_data = np.exp(
                #     temp_sigma_data_tr * np.log(noise_schedule.tr_s_max)
                #     + (1 - temp_sigma_data_tr) * np.log(noise_schedule.tr_s_min)
                # )
                # lambda_tr = (tr_sigma_data + tr_s) / (tr_sigma_data + tr_s / temp_sampling)
                # delta_t = (
                #     tr_g**2 * dt * (lambda_tr + temp_sampling * temp_psi / 2) * tr_score
                #     + tr_g * np.sqrt(dt * (1 + temp_psi)) * epsilon_t
                # )

                # rot_sigma_data = np.exp(
                #     temp_sigma_data_rot * np.log(noise_schedule.rot_s_max)
                #     + (1 - temp_sigma_data_rot) * np.log(noise_schedule.rot_s_min)
                # )
                # lambda_rot = (rot_sigma_data + rot_s) / (
                #     rot_sigma_data + rot_s / temp_sampling
                # )
                # delta_w = (
                #     rot_g**2 * dt * (lambda_rot + temp_sampling * temp_psi / 2) * rot_score
                #     + rot_g * np.sqrt(dt * (1 + temp_psi)) * epsilon_w
                # )

                # Apply translation and backbone rotation
                # bind_X_rotated = (
                #     rotate(X=bind_X_centered, w=delta_w, sel=sel)
                #     + delta_t[:, None, None, :]
                # )
                bind_X_rotated = bind_X_centered + delta_t[:, None, None, :]
                bind_X = bind_X_rotated + center

                # Apply atom mask
                bind_X = bind_X * atom_mask

                # Ensure bind_X requires gradient for next iteration
                if not self.non_conservative:
                    bind_X = bind_X.detach().requires_grad_()
                else:
                    bind_X = bind_X.detach()

                trajectory.append(bind_X.clone().detach().cpu().numpy())

        # Return the final binder coordinates as samples
        bind_X = bind_X.detach()

        # if self.non_conservative:
        #     return bind_X, bind_S, bind_A, None, trajectory

        return (bind_X, bind_S, bind_A, None), trajectory, energy.item()

    # -------------------
    # (legacy) Gaussian forward utility kept for reference
    # -------------------
    def gaussian_forward_(self, binder, target, step):
        sel = self.residue_size
        bb = self.bb

        true_X, bind_S, bind_A, bind_O, _ = binder
        tgt_X, tgt_S, tgt_A, tgt_O, _ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        bind_mask = bind_A[:, :, bb].clamp(max=1).float()
        tgt_mask = tgt_A[:, :, bb].clamp(max=1).float()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        eps = np.random.uniform(0.1, 1.0, size=B)
        eps = torch.tensor(eps).float().cuda(device=self.device)[:, None, None, None]
        hat_t = torch.randn_like(true_X).cuda(device=self.device) * eps
        bind_X = true_X + hat_t * atom_mask
        bind_X = bind_X.requires_grad_()

        mask_2D = (bind_A > 0).float().view(B, N * sel, 1) * (tgt_A > 0).float().view(
            B, 1, M * sel
        )
        D = (bind_X.view(B, N * sel, 1, 3) - tgt_X.view(B, 1, M * sel, 3)).norm(
            dim=-1
        )  # [B,N*14,M*14]
        mask_2D = mask_2D * (D < self.threshold).float()

        h = self.encoder(
            (bind_X, bind_S, bind_A, bind_O),
            (tgt_X, tgt_S, tgt_A, tgt_O),
        )  # [B,N+M,14,H]

        if self.non_conservative:
            force = h
        else:
            bind_h = self.W_o(h[:, :N]).view(B, N * sel, -1)
            tgt_h = self.U_o(h[:, N:]).view(B, M * sel, -1)

            energy = torch.matmul(bind_h, tgt_h.transpose(1, 2))  # [B,N*sel,M*sel]
            energy = (energy * mask_2D).sum(dim=(1, 2))  # [B]

            force = torch.autograd.grad(energy.sum(), bind_X, create_graph=True)[0]

        loss = F.mse_loss(force[:, :N] * eps, -hat_t / eps, reduction="none")
        loss = (loss * atom_mask).sum() / atom_mask.sum()
        return loss

    def get_linear_scale(self, t, T, initial_eta, s=0.008):
        """Linear noise schedule"""
        return initial_eta * (1 - t / T)

    def get_cosine_scale(self, t, T, initial_eta, s=0.008):
        """Cosine noise schedule"""
        t_scaled = t / T
        cosine_factor = np.cos(((t_scaled + s) / (1 + s) * np.pi * 0.5)) ** 2
        return initial_eta * cosine_factor

    def randomly_translate_and_rotate(self, binder, target, tr_s_max):
        true_X, _, bind_A, _, _ = binder
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        # Initialize binder coordinates
        bind_X = true_X.clone()
        # center = self.mean(bind_X[:, :, bb], bind_A[:, :, bb])
        # center = bind_X.mean(dim=(1, 2))  # [B,3]
        B = bind_X.shape[0]
        center = bind_X.view(B, -1, 3).mean(dim=1, keepdim=True)[:, None, ...]

        centered_X = bind_X - center
        # centered_X = bind_X
        # noise_w = torch.from_numpy(Rotation.random().as_matrix()).to(centered_X.device).float()
        # centered_X = centered_X @ noise_w

        noise_t = torch.normal(0, tr_s_max, size=(1, 3)).to(centered_X.device)
        bind_X = centered_X + noise_t[:, None, None, :] + center
        # bind_X = centered_X + noise_t[:, None, None, :]

        return bind_X * atom_mask

    def compute_clash_energy(self, bind_X, tgt_X, bind_A, tgt_A):
        """
        Compute repulsive energy for overlapping atoms
        """
        B, N, sel, _ = bind_X.shape
        M = tgt_X.shape[1]

        # Reshape for pairwise distance computation
        bind_coords = bind_X.view(B, N * sel, 3)  # [B, N*sel, 3]
        tgt_coords = tgt_X.view(B, M * sel, 3)  # [B, M*sel, 3]

        # Create masks for valid atoms
        bind_mask = (bind_A > 0).float().view(B, N * sel)  # [B, N*sel]
        tgt_mask = (tgt_A > 0).float().view(B, M * sel)  # [B, M*sel]

        # Compute pairwise distances
        bind_coords_exp = bind_coords.unsqueeze(2)  # [B, N*sel, 1, 3]
        tgt_coords_exp = tgt_coords.unsqueeze(1)  # [B, 1, M*sel, 3]

        distances = torch.norm(
            bind_coords_exp - tgt_coords_exp, dim=-1
        )  # [B, N*sel, M*sel]

        # Create interaction mask
        interaction_mask = bind_mask.unsqueeze(2) * tgt_mask.unsqueeze(
            1
        )  # [B, N*sel, M*sel]

        # Soft repulsive potential (modified Lennard-Jones)
        # E = penalty * (threshold/r)^12 for r < threshold
        clash_mask = (distances < 5) & (distances > 1e-6)
        clash_distances = distances * clash_mask.float() + 5 * (~clash_mask).float()

        repulsion = 0.001 * torch.pow(5 / clash_distances, 12)
        repulsion = repulsion * clash_mask.float() * interaction_mask

        clash_energy = repulsion.sum(dim=(1, 2))  # [B]
        return clash_energy

    def compute_interior_penalty(self, bind_X, tgt_X, bind_A, tgt_A):
        """
        Penalize binder atoms that are inside the target protein
        """
        B, N, sel, _ = bind_X.shape
        M = tgt_X.shape[1]

        # Get centers of mass
        bind_coords = bind_X.view(B, N * sel, 3)
        tgt_coords = tgt_X.view(B, M * sel, 3)

        bind_mask = (bind_A > 0).float().view(B, N * sel)
        tgt_mask = (tgt_A > 0).float().view(B, M * sel)

        # Compute target center and radius
        tgt_com = (tgt_coords * tgt_mask.unsqueeze(-1)).sum(dim=1) / tgt_mask.sum(
            dim=1, keepdim=True
        )  # [B, 3]
        tgt_distances_to_com = torch.norm(
            tgt_coords - tgt_com.unsqueeze(1), dim=-1
        )  # [B, M*sel]
        tgt_radius = (tgt_distances_to_com * tgt_mask).max(dim=1)[0]  # [B]

        # Check if binder atoms are inside target
        bind_distances_to_tgt_com = torch.norm(
            bind_coords - tgt_com.unsqueeze(1), dim=-1
        )  # [B, N*sel]
        inside_mask = (bind_distances_to_tgt_com < tgt_radius.unsqueeze(1)) & (
            bind_mask > 0
        )

        # Penalty for being inside (exponential decay from center)
        interior_penalty = 0.001 * torch.exp(
            -bind_distances_to_tgt_com / tgt_radius.unsqueeze(1)
        )
        interior_penalty = (interior_penalty * inside_mask.float()).sum(dim=1)  # [B]

        return interior_penalty

    def compute_bias_energy(self, bind_X, tgt_X, bind_A, tgt_A):
        """
        Compute total bias energy combining clash and interior penalties
        """
        clash_energy = self.compute_clash_energy(bind_X, tgt_X, bind_A, tgt_A)
        interior_energy = self.compute_interior_penalty(bind_X, tgt_X, bind_A, tgt_A)
        return clash_energy + interior_energy
