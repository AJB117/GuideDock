from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from utils import Features, get_resolution_indices, sample_vec, score_norm, score_vec


class PoseDiffusionMixin:
    def get_noise_terms(self, binder: Features, tr_s: Tensor, rot_s: Tensor):
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
        true_X = binder.X
        B = true_X.size(0)

        w = torch.zeros(B, 3)
        backbone_score = torch.zeros(B, 3)
        for i in range(B):
            w_ = torch.from_numpy(sample_vec(rot_s[i].item()))
            backbone_score_ = score_vec(rot_s[i].item(), w_)
            w[i] = w_
            backbone_score[i] = backbone_score_

        hat_t = torch.from_numpy(
            np.random.normal(0, tr_s.cpu().numpy(), (B, 3))
        ).float()

        return (
            w.cuda(true_X.device),
            hat_t.cuda(true_X.device),
            tr_s,
            backbone_score.cuda(true_X.device),
        )

    def predict_score(
        self,
        tr_force: Tensor,
        rot_force: Tensor,
        eps_tr: Tensor,
        eps_rot: Tensor,
        noise_emb: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        tr_norm = tr_force.norm(dim=-1, keepdim=True)
        rot_norm = rot_force.norm(dim=-1, keepdim=True)

        tr_scale = self.tr_scalar_head(tr_norm, noise_emb)
        rot_scale = self.rot_scalar_head(rot_norm, noise_emb)

        rot_score = (rot_force / (rot_norm + 1e-7)) * rot_scale
        tr_score = (tr_force / (tr_norm + 1e-7)) * tr_scale

        tr_score = tr_score / eps_tr
        rot_score = rot_score * score_norm(eps_rot)

        return rot_score, tr_score

    def forward(
        self,
        binder: Features,
        target: Features,
        tr_s: Tensor,
        rot_s: Tensor,
        resolution: Optional[str] = None,
        time_steps: Tensor = None,
        get_loss_dict: bool = False,
    ):
        time_steps_ = time_steps
        tr_emb = self.noise_conditioner(tr_s)
        rot_emb = self.noise_conditioner(rot_s)
        noise_emb = torch.cat((tr_emb, rot_emb), dim=-1)
        noise_emb = self.noise_proj(noise_emb)

        rotation_force, hat_t, eps, backbone_score = self.get_noise_terms(
            binder, tr_s, rot_s
        )
        noised_bind_X, bind_mask = self.add_noise(
            binder, rotation_force, hat_t, self.resolution
        )
        noised_binder = Features(noised_bind_X, *binder[1:])

        _, bb = get_resolution_indices(self.resolution)

        bind_X = noised_binder.X

        if not self.non_conservative:
            bind_X.requires_grad_()

        force, until, bind_H, tgt_H = self.predict(
            noised_binder,
            target,
            noise_emb,
            tr_s,
            return_nodes=True,
        )

        rot_score_norm = score_norm(rot_s)
        translation_score = -hat_t / tr_s**2

        tr_force, rot_force = self.get_forces(
            force,
            until,
            bind_X,
            bind_mask,
            noise_emb,
        )

        pred_rot_score, pred_tr_score = self.predict_score(
            tr_force,
            rot_force,
            tr_s,
            rot_s,
            noise_emb,
        )

        tr_loss = (pred_tr_score - translation_score) ** 2 * tr_s**2
        time_decay = (1 - torch.exp(-time_steps_)) / torch.exp(-time_steps_ / 2)
        tr_loss = time_decay * tr_loss
        tr_loss = tr_loss.mean(dim=-1)

        rot_loss = (((pred_rot_score - backbone_score) / rot_score_norm) ** 2).mean(
            dim=-1
        )

        torch._assert(torch.isfinite(rot_loss).all(), "rot_loss has NaNs/Inf")
        torch._assert(torch.isfinite(tr_loss).all(), "tr_loss has NaNs/Inf")

        bb_loss_mask = time_steps_ < 0.2
        bb_loss = (
            self.coord_loss(binder, bind_H, tgt_H.mean(dim=1), bind_mask)
            * bb_loss_mask.float()
        )

        contact_loss = torch.tensor(0.0).cuda()
        f1 = torch.tensor(0.0).cuda()

        loss = (
            0.35 * rot_loss
            + 0.35 * tr_loss
            + 0.2 * bb_loss.mean(dim=-1)
            + 0.2 * contact_loss
        ).mean()

        if get_loss_dict:
            loss_dict = {
                "total_loss": loss,
                "rot_loss": rot_loss.mean(),
                "tr_loss": tr_loss.mean(),
                "bb_loss": bb_loss.mean(),
                "contact_loss": contact_loss.mean(),
                "contact_f1": f1.mean(),
            }
            return loss, loss_dict

        return loss

    def get_diff_coeffs(self, noise_schedule, tr_s, rot_s):

        tr_scale = torch.sqrt(
            torch.log(torch.tensor(noise_schedule.tr_s_max / noise_schedule.tr_s_min))
        )
        tr_g = tr_s * tr_scale

        rot_scale = torch.sqrt(
            torch.log(torch.tensor(noise_schedule.rot_s_max / noise_schedule.rot_s_min))
        )
        rot_g = 2 * rot_s * rot_scale

        return tr_g, rot_g

    def get_updates(self, tr_g, rot_g, dt, tr_score, rot_score, t_idx):
        epsilon_t = torch.normal(0, 1, size=tr_score.size()).cuda(device=self.device)
        epsilon_w = torch.normal(0, 1, size=rot_score.size()).cuda(device=self.device)

        if t_idx == 0:
            epsilon_t = epsilon_t * 0.0
            epsilon_w = epsilon_w * 0.0

        delta_t = tr_g**2 * dt * tr_score
        delta_t = delta_t + (tr_g * math.sqrt(dt) * epsilon_t)

        delta_w = rot_g**2 * dt * rot_score
        delta_w = delta_w + (rot_g * math.sqrt(dt) * epsilon_w)

        return delta_t, delta_w
