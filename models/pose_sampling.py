from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch import Tensor

from utils import Features, NoiseSchedule, get_resolution_indices, rotate

if TYPE_CHECKING:
    from train_contact_diff import LitContactDiff


class PoseSamplingMixin:
    def sample(
        self,
        binder: Features,
        target: Features,
        orig_binder: Features,
        T: int,
        alpha_: float,
        eta_start: float,
        batch_size: int,
        noise_schedule: NoiseSchedule,
        resolution: str = "backbone",
        binder_guide: Optional[Tensor] = None,
        D_guide: Optional[Tensor] = None,
        tr_grad_weight: float = 1.0,
        rot_grad_weight: float = 1.0,
        uncertainty: Optional[Tensor] = None,
        dist_model: Optional[LitContactDiff] = None,
        binder_cond: Optional[Tensor] = None,
        target_cond: Optional[Tensor] = None,
        cutoff_threshold: float = 50.0,
    ):
        self.eval()
        sel, bb = get_resolution_indices(self.resolution)

        initial_X, bind_S, bind_A, *_ = binder
        tgt_X, tgt_S, tgt_A, *_ = target

        B, _N, _M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A = bind_A * (initial_X.norm(dim=-1) > 1e-4).long()
        bind_mask = bind_A[:, :, bb].clamp(max=1).float()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        bind_X = initial_X.clone()
        if not self.non_conservative:
            bind_X = bind_X.requires_grad_()

        B = batch_size

        time_steps = np.linspace(1, 0, T + 1)[:-1]
        trajectory = [bind_X.clone().detach().cpu().numpy()]

        w_grad, t_grad = (
            torch.zeros(B, 3).cuda(bind_X.device),
            torch.zeros(B, 3).cuda(bind_X.device),
        )

        for t_idx in range(T):
            cur_t = time_steps[t_idx]
            if t_idx == T - 1:
                dt = cur_t
            else:
                dt = cur_t - time_steps[t_idx + 1]

            tr_s, rot_s = noise_schedule(cur_t, cur_t)

            if True:
                if D_guide is not None:
                    t_grad, w_grad = self.get_distance_potential_spectral_mask(
                        binder, target, D_guide, cutoff_threshold
                    )

                    delta_t, delta_w = (
                        tr_grad_weight * t_grad,
                        rot_grad_weight * w_grad,
                    )

                    center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[
                        :, None, ...
                    ]
                    nonzero = bind_mask.sum(dim=1, keepdim=True)[:, None, None, :]
                    center = center / (nonzero * sel)
                    bind_X = bind_X - center
                    bind_X = rotate(X=bind_X, w=delta_w, sel=sel)
                    bind_X = bind_X - delta_t[:, None, None, :]
                    bind_X = bind_X + center
                    bind_X = bind_X * atom_mask

            tr_s = tr_s.repeat(B)
            rot_s = rot_s.repeat(B)

            tr_s, rot_s = (
                torch.tensor(tr_s)
                .float()
                .cuda(bind_X.device)
                .view(B, 1)
                .requires_grad_(),
                torch.tensor(rot_s)
                .float()
                .cuda(bind_X.device)
                .view(B, 1)
                .requires_grad_(),
            )

            with torch.no_grad(), torch.inference_mode():
                tr_emb = self.noise_conditioner(tr_s)
                rot_emb = self.noise_conditioner(rot_s)
                noise_emb = torch.cat((tr_emb, rot_emb), dim=-1)
                noise_emb = self.noise_proj(noise_emb)

                out = self.predict(
                    binder,
                    target,
                    noise_emb,
                    tr_s,
                    return_energy=not self.non_conservative,
                )

            if self.non_conservative:
                force, until = out
            else:
                force, until, energy = out

            with torch.no_grad(), torch.inference_mode():
                tr_force, rot_force = self.get_forces(
                    force, until, bind_X, bind_mask, noise_emb
                )

            if self.non_conservative:
                energy = tr_force.norm() + rot_force.norm()

            center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[:, None, ...]
            nonzero = bind_mask.sum(dim=1, keepdim=True)[:, None, None, :]
            center = center / (nonzero * sel)

            bind_X_centered = bind_X - center

            with torch.no_grad():
                rot_score, tr_score = self.predict_score(
                    tr_force, rot_force, tr_s, rot_s, noise_emb
                )

            rot_score = rot_score.detach().requires_grad_()
            tr_score = tr_score.detach().requires_grad_()

            tr_g, rot_g = self.get_diff_coeffs(noise_schedule, tr_s, rot_s)

            delta_t, delta_w = self.get_updates(
                tr_g, rot_g, dt, tr_score, rot_score, t_idx
            )
            delta_t = delta_t.requires_grad_()
            delta_w = delta_w.requires_grad_()

            bind_X_centered = bind_X_centered.requires_grad_()
            bind_X_rotated = (
                rotate(X=bind_X_centered, w=delta_w, sel=sel)
                + delta_t[:, None, None, :]
                + center
            )

            bind_X = bind_X_rotated * atom_mask

            binder = Features(bind_X, *binder[1:])

            w_grad = w_grad.detach()
            t_grad = t_grad.detach()
            bind_X = bind_X.detach()

            if not self.non_conservative:
                bind_X = bind_X.detach().requires_grad_()
            else:
                bind_X = bind_X.detach()

            trajectory.append(bind_X.clone().detach().cpu().numpy())

        bind_X = bind_X.detach()

        return (bind_X, bind_S, bind_A, None), trajectory, energy.item()

    def sample_fk(
        self,
        binder: Features,
        target: Features,
        orig_binder: Features,
        T: int,
        alpha_: float,
        eta_start: float,
        batch_size: int,
        noise_schedule: NoiseSchedule,
        resolution: str = "backbone",
        binder_guide: Optional[Tensor] = None,
        D_guide: Optional[Tensor] = None,
        tr_grad_weight: float = 1.0,
        rot_grad_weight: float = 1.0,
        uncertainty: Optional[Tensor] = None,
        dist_model: Optional[LitContactDiff] = None,
        binder_cond: Optional[Tensor] = None,
        target_cond: Optional[Tensor] = None,
        cutoff_threshold=50.0,
        num_particles=8,
    ):
        self.eval()
        sel, bb = get_resolution_indices(self.resolution)

        initial_X, bind_S, bind_A, *_ = binder
        tgt_X, tgt_S, tgt_A, *_ = target

        B, _N, _M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A = bind_A * (initial_X.norm(dim=-1) > 1e-4).long()
        bind_mask = bind_A[:, :, bb].clamp(max=1).float()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        bind_X = initial_X.clone()
        if not self.non_conservative:
            bind_X = bind_X.requires_grad_()

        B = num_particles

        time_steps = np.linspace(1, 0, T + 1)[:-1]
        trajectory = [bind_X.clone().detach().cpu().numpy()]

        w_grad, t_grad = (
            torch.zeros(B, 3).cuda(bind_X.device),
            torch.zeros(B, 3).cuda(bind_X.device),
        )
        device = bind_X.device

        prev_energy = None

        for t_idx in range(T):
            cur_t = time_steps[t_idx]
            if t_idx == T - 1:
                dt = cur_t
            else:
                dt = cur_t - time_steps[t_idx + 1]

            tr_s, rot_s = noise_schedule(cur_t, cur_t)

            tr_s = tr_s.repeat(B)
            rot_s = rot_s.repeat(B)

            tr_s, rot_s = (
                torch.tensor(tr_s)
                .float()
                .cuda(bind_X.device)
                .view(B, 1)
                .requires_grad_(),
                torch.tensor(rot_s)
                .float()
                .cuda(bind_X.device)
                .view(B, 1)
                .requires_grad_(),
            )

            with torch.no_grad(), torch.inference_mode():
                tr_emb = self.noise_conditioner(tr_s)
                rot_emb = self.noise_conditioner(rot_s)
                noise_emb = torch.cat((tr_emb, rot_emb), dim=-1)
                noise_emb = self.noise_proj(noise_emb)

                out = self.predict(
                    binder,
                    target,
                    noise_emb,
                    tr_s,
                    return_energy=not self.non_conservative,
                )

            if self.non_conservative:
                force, until = out
            else:
                force, until, energy = out

            with torch.no_grad(), torch.inference_mode():
                tr_force, rot_force = self.get_forces(
                    force, until, bind_X, bind_mask, noise_emb
                )

            if self.non_conservative:
                energy = tr_force.norm() + rot_force.norm()

            center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[:, None, ...]
            nonzero = bind_mask.sum(dim=1, keepdim=True)[:, None, None, :]
            center = center / (nonzero * sel)

            bind_X_centered = bind_X - center

            with torch.no_grad():
                rot_score, tr_score = self.predict_score(
                    tr_force, rot_force, tr_s, rot_s, noise_emb
                )

            rot_score = rot_score.detach().requires_grad_()
            tr_score = tr_score.detach().requires_grad_()

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

            delta_t = tr_g**2 * dt * tr_score
            delta_t = delta_t + (tr_g * math.sqrt(dt) * epsilon_t)
            delta_t = delta_t.detach().requires_grad_(True)

            delta_w = rot_g**2 * dt * rot_score
            delta_w = delta_w + (rot_g * math.sqrt(dt) * epsilon_w)
            delta_w = delta_w.detach().requires_grad_(True)

            bind_X_centered = bind_X_centered.requires_grad_()

            bind_X_rotated = (
                rotate(X=bind_X_centered, w=delta_w, sel=sel)
                + delta_t[:, None, None, :]
            )

            bind_X = bind_X_rotated + center

            bind_X = bind_X * atom_mask
            energy = torch.zeros(B, device=device, dtype=bind_X.dtype)

            if t_idx % 2 == 0:
                if D_guide is not None:
                    tr_grad, rot_grad, energy = (
                        self.get_distance_potential_spectral_mask(
                            binder,
                            target,
                            D_guide,
                            cutoff_threshold=cutoff_threshold,
                            return_energy=True,
                        )
                    )
                    delta_t, delta_w = (
                        tr_grad_weight * tr_grad,
                        rot_grad_weight * rot_grad,
                    )
                    center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[
                        :, None, ...
                    ]
                    nonzero = bind_mask.sum(dim=1, keepdim=True)[:, None, None, :]
                    center = center / (nonzero * sel)
                    bind_X = bind_X - center
                    bind_X = rotate(X=bind_X, w=delta_w, sel=sel)
                    bind_X = bind_X - delta_t[:, None, None, :] + center
                    bind_X = bind_X * atom_mask

                if prev_energy is None:
                    logG = torch.zeros((B), device=device, dtype=energy.dtype)
                else:
                    logG = -1 * (energy - prev_energy)

                    eps_tr = tr_s * torch.randn(B, 3, device=device)
                    eps_rot = rot_s * torch.randn(B, 3, device=device)
                    ll_tr = (eps_tr**2 - (eps_tr + delta_t) ** 2).sum() / (2 * tr_s**2)
                    ll_rot = (eps_rot**2 - (eps_rot + delta_w) ** 2).sum() / (
                        2 * rot_s**2
                    )
                    logG = logG + ll_tr.flatten() + ll_rot.flatten()

                probs = torch.nan_to_num(
                    torch.softmax(logG, dim=0), nan=0.0, posinf=0.0, neginf=0.0
                )
                Z = probs.sum()
                if Z <= 0:
                    probs = torch.ones_like(probs) / probs.numel()
                else:
                    probs = probs / Z

                idx = torch.multinomial(probs, num_particles, replacement=True)

                if t_idx != 0:
                    bind_X = bind_X[idx]

                prev_energy = energy

            binder = Features(bind_X, *binder[1:])

            w_grad = w_grad.detach()
            t_grad = t_grad.detach()
            bind_X = bind_X.detach()

            if not self.non_conservative:
                bind_X = bind_X.detach().requires_grad_()
            else:
                bind_X = bind_X.detach()

            trajectory.append(bind_X.clone().detach().cpu().numpy())

        bind_X = bind_X.detach()
        idx = torch.multinomial(probs, 1, replacement=True)

        bind_X = bind_X[idx].squeeze(1)

        return (bind_X, bind_S, bind_A, None), trajectory, 0
