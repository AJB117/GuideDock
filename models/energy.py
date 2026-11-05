from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch import Tensor

from utils import get_resolution_indices

from .common import FiLM
from .modules import ForceHead, ScalarHead, SinusoidalEmbedding

if TYPE_CHECKING:
    from train_contact_diff import LitContactDiff
from utils import Features, NoiseSchedule, rotate, sample_vec, score_norm, score_vec

from .frame import AllAtomEncoder

torch.autograd.set_detect_anomaly(True)


def focal_loss(inputs, targets, alpha=0.999, gamma=2.0, reduction="none"):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduction == "mean":
        return F_loss.mean()
    elif reduction == "sum":
        return F_loss.sum()
    else:
        return F_loss


class FARigidModel(nn.Module):
    def __init__(self):
        super(FARigidModel, self).__init__()

    def mean(self, X, mask):
        return (X * mask[..., None]).sum(dim=1) / mask[..., None].sum(dim=1).clamp(
            min=1e-6
        )

    def inertia(self, X, mask):
        inner = (X**2).sum(dim=-1)
        inner = inner[..., None, None] * torch.eye(3).to(X)[None, None, ...]
        outer = X.unsqueeze(2) * X.unsqueeze(3)
        inertia = (inner - outer) * mask[..., None, None]
        return 0.1 * inertia.sum(dim=1)


class AllAtomEnergyModel(FARigidModel):
    def __init__(self, args, device, resolution, dist_model: LitContactDiff):
        super(AllAtomEnergyModel, self).__init__()
        self.mse_loss = nn.MSELoss()

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
            nn.Linear(hsize, hsize // 2),
            nn.SiLU(),
            nn.Linear(hsize // 2, hsize // 2),
        )
        self.U_o = nn.Sequential(
            nn.Linear(hsize, hsize // 2),
            nn.SiLU(),
            nn.Linear(hsize // 2, hsize // 2),
        )
        self.eye = torch.eye(3).unsqueeze(0).cuda()
        self.use_sidechain = not args.nosidechain
        self.device = device
        self.encoder.ops = self.encoder.ops.cuda()
        self.noise_scaler = args.noise_scaler
        self.resolution = resolution
        self.bb = 0 if self.resolution == "CA" else 1
        self.dist_model = dist_model

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

        self.tr_scalar_head = ScalarHead(args, depth=2)
        self.rot_scalar_head = ScalarHead(args)

        self.noise_conditioner = SinusoidalEmbedding(args.noise_emb_dim)

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
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        if self.non_conservative:
            self.tr_force_head = ForceHead(args, depth=3)

            self.rot_force_head = ForceHead(args, depth=3)

        self.film_layer_1 = FiLM(hidden_size, args)
        self.noise_proj = nn.Sequential(
            nn.Linear(args.noise_emb_dim * 2, hidden_size),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm_list = []
        self.norm_diff_list = []
        self.cos_sim_list = []

        self.backbone_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 3),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

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

    def add_noise(self, binder: Features, w: Tensor, hat_t: Tensor, resolution: str):
        true_X, bind_A = binder.X, binder.A

        sel, bb = get_resolution_indices(resolution)

        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        bind_mask = bind_A[:, :, bb].clamp(max=1).float()

        B = true_X.shape[0]
        center = true_X.view(B, -1, 3).sum(dim=1, keepdim=True)[:, None, ...]
        nonzero = bind_mask.sum(dim=1, keepdim=True)[:, None, None, :]
        center = center / (nonzero * sel)

        bind_X = true_X - center
        bind_X = rotate(X=bind_X, w=w, sel=sel) + hat_t[:, None, None, :]

        bind_X = bind_X + center
        bind_X = bind_X * (bind_A > 0).float().unsqueeze(-1)

        return bind_X, bind_mask

    def predict(
        self,
        binder: Features,
        target: Features,
        noise_emb: Tensor,
        eps_tr: Tensor,
        return_energy: bool = False,
        return_nodes: bool = False,
    ):

        bind_X, bind_S, bind_A, *_ = binder
        tgt_X, _, tgt_A, *_ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        sel, _ = get_resolution_indices(self.resolution)

        if self.non_conservative:
            h1, h2 = self.encoder(
                binder,
                target,
                time_steps=noise_emb,
                eps_tr=eps_tr,
            )
            bind_h = h1.reshape(B, N, -1)
            tgt_h = h2.reshape(B, M, -1)

            bind_h_pooled = bind_h.mean(dim=1)
            tgt_h_pooled = tgt_h.mean(dim=1)

            combined = torch.cat((bind_h_pooled, tgt_h_pooled), dim=-1)
            f = self.node_decoder(combined)
            f = f + noise_emb
            until = N
        else:
            h1, h2 = self.encoder(
                binder,
                target,
                time_steps=noise_emb,
                eps_tr=eps_tr,
            )

            bind_h = self.W_o(h1).view(B, N * sel, -1)
            tgt_h = self.U_o(h2).view(B, M * sel, -1)

            mask_2D = (bind_A > 0).float().view(B, N * sel, 1) * (
                tgt_A > 0
            ).float().view(B, 1, M * sel)
            D = (bind_X.view(B, N * sel, 1, 3) - tgt_X.view(B, 1, M * sel, 3)).norm(
                dim=-1
            )
            mask_2D = mask_2D * (D < 40 + 3 * eps_tr.view(-1, 1, 1)).float()

            energy = torch.matmul(bind_h, tgt_h.transpose(1, 2))
            energy = (energy * mask_2D).sum(dim=(1, 2))

            f = torch.autograd.grad(
                energy.sum(),
                bind_X,
                create_graph=True,
                allow_unused=True,
                materialize_grads=True,
            )[0]
            until = f.shape[1]

        if return_nodes:
            return f, until, bind_h, tgt_h

        if return_energy:
            return f, until, energy

        return f, until

    def get_forces(
        self,
        force: Tensor,
        until: int,
        bind_X: Tensor,
        bind_mask: Tensor,
        noise_emb: Tensor,
    ):
        if self.non_conservative:
            tr_force, rot_force = (
                self.tr_force_head(force, noise_emb),
                self.rot_force_head(force, noise_emb),
            )
            return tr_force, rot_force

        _, bb = get_resolution_indices(self.resolution)

        tr_force = self.mean(force[:, :until, bb], bind_mask)

        center = self.mean(bind_X[:, :, bb], bind_mask)
        bind_X = bind_X - center[:, None, None, :]
        G = torch.cross(bind_X[:, :, bb], force[:, :until, bb], dim=-1)
        G = (G * bind_mask[..., None]).sum(dim=1)
        I = self.inertia(bind_X[:, :, bb], bind_mask)
        rot_force = torch.linalg.solve(
            I.detach() + torch.eye(3, device=I.device) * 1e-4, G
        )

        return tr_force, rot_force

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

    def contact_loss(
        self,
        binder: Features,
        target: Features,
        bind_H: Tensor,
        tgt_H: Tensor,
        bind_mask: Tensor,
        contact_threshold: float = 5.0,
    ):
        """
        Contact loss between binder and target proteins. Predicts the binding interface.
        """
        true_X, bind_S, *_ = binder
        tgt_X, _, tgt_A, *_ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        sel, bb = get_resolution_indices(self.resolution)

        contact_pred = torch.matmul(
            self.W_o(bind_H.reshape(B * N, -1)).reshape(B, N, -1),
            self.U_o(tgt_H.reshape(B * M, -1)).reshape(B, M, -1).transpose(-1, -2),
        ).view(B, N, M)

        D = (true_X[:, :, 1].view(B, N, 1, 3) - tgt_X[:, :, 1].view(B, 1, M, 3)).norm(
            dim=-1
        )

        contact_true = (D < contact_threshold).float()
        contact_mask = bind_mask.view(B, N, 1) * (tgt_A[:, :, 1] > 0).float().view(
            B, 1, M
        )

        contact_true = contact_true * contact_mask
        loss = focal_loss(contact_pred, contact_true, reduction="none")
        denom = contact_mask.reshape(B, -1).sum(dim=1).clamp_min(1.0)
        loss = loss.reshape(B, -1).sum(dim=1) / denom

        contact_accuracy = (
            (contact_pred > 0).float() == contact_true
        ).float() * contact_mask
        contact_accuracy = contact_accuracy.reshape(B, -1).sum(dim=1) / denom

        tp = ((contact_pred > 0).float() * contact_true).float() * contact_mask
        fp = ((contact_pred > 0).float() * (1 - contact_true)).float() * contact_mask
        fn = ((1 - (contact_pred > 0).float()) * contact_true).float() * contact_mask
        tp_sum = tp.reshape(B, -1).sum(dim=1)
        fp_sum = fp.reshape(B, -1).sum(dim=1)
        fn_sum = fn.reshape(B, -1).sum(dim=1)
        precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        recall = tp_sum / (tp_sum + fn_sum + 1e-8)
        F1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return loss, contact_accuracy, F1

    def coord_loss(
        self, binder: Features, bind_H: Tensor, tgt_H_mean: Tensor, mask: Tensor
    ):
        true_X, bind_S, *_ = binder

        B, N = bind_S.size(0), bind_S.size(1)
        sel, bb = get_resolution_indices(self.resolution)

        tgt_H_mean = tgt_H_mean.unsqueeze(1).expand(-1, N, -1)

        pred_backbone = self.backbone_predictor(
            torch.cat((bind_H, tgt_H_mean), dim=-1).view(B * N, -1)
        )

        pred_backbone = pred_backbone.view(B, N, 3)

        true_X = true_X[:, :, bb, :]

        diff = ((pred_backbone - true_X) ** 2).sum(dim=1) / (
            sel * mask.sum(dim=1).clamp_min(1.0)[..., None]
        )
        loss = diff.mean(dim=-1)

        return loss

    def torque(self, binder_coords, guidance_grad):
        """
        binder_coords: (B, N, 3)
        guidance_grad: (B, N, 3)
        returns: (B, 3) rotational gradients (torques)
        """

        com = binder_coords.mean(dim=1, keepdim=True)

        rel = binder_coords - com

        torque_per_point = torch.cross(rel, guidance_grad, dim=-1)

        tau = torque_per_point.sum(dim=1)

        return tau

    def _masked_mean(self, x, mask, dim):

        w = mask.unsqueeze(-1)
        denom = w.sum(dim=dim, keepdim=True).clamp_min(1e-8)
        return (x * w).sum(dim=dim, keepdim=True) / denom

    def nere_se3_from_point_grads(
        self,
        coords,
        dL_dcoords,
        mask=None,
        dt=0.1,
        damping=1e-4,
    ):
        """
        Convert per-point gradients into global SE(3) updates (t_step, w_step) using NERE physics.

        We interpret force as -∂L/∂x so that stepping along (t_step, w_step) reduces L.
        """
        B, N, _ = coords.shape
        device = coords.device
        eye = torch.eye(3, device=device).unsqueeze(0).expand(B, 3, 3)

        if mask is None:
            mask = torch.ones(B, N, device=device)

        mu = self._masked_mean(coords, mask, dim=1)
        r = coords - mu

        f = -dL_dcoords
        f = f * mask.unsqueeze(-1)

        tau = torch.cross(r, f, dim=-1).sum(dim=1)

        r2 = (r * r).sum(dim=-1)
        term1 = r2[:, :, None, None] * eye[:, None, :, :]
        term2 = r[:, :, :, None] * r[:, :, None, :]
        I = (term1 - term2) * mask[:, :, None, None]
        I = I.sum(dim=1)

        w_step = torch.linalg.solve(
            I.detach() + damping * eye, tau.unsqueeze(-1)
        ).squeeze(-1)

        w_step = w_step * dt

        return w_step

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
            energy = 0

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

    def get_distance_potential_spectral_mask(
        self,
        binder: Features,
        target: Features,
        guide_D: Tensor,
        cutoff_threshold=50.0,
        return_energy=False,
    ):
        """
        Potential guidance but using the spectrum of the full distance matrix
        to mask problematic distances
        """
        binder_mask = (binder.A[:, :, self.bb] > 0).float()
        target_mask = (target.A[:, :, self.bb] > 0).float()

        _, M, N = guide_D.shape
        device = guide_D.device
        dtype = guide_D.dtype

        B = binder.X.shape[0]

        inter_dists_bind = torch.cdist(binder.X[:, :, self.bb], binder.X[:, :, self.bb])
        inter_dists_bind = inter_dists_bind * (
            binder_mask.unsqueeze(1) * binder_mask.unsqueeze(1)
        )
        inter_dists_target = torch.cdist(
            target.X[:, :, self.bb], target.X[:, :, self.bb]
        )
        inter_dists_target = inter_dists_target * (
            target_mask.unsqueeze(1) * target_mask.unsqueeze(1)
        )

        M_N = M + N
        D = torch.zeros(B, M_N, M_N, device=device)

        D[:, :M, :M] = inter_dists_bind
        D[:, M:, M:] = inter_dists_target
        D[:, :M, M:] = guide_D
        D[:, M:, :M] = guide_D.transpose(1, 2)

        I = torch.eye(M_N, device=device, dtype=dtype).unsqueeze(0).expand(B, M_N, M_N)
        ones = torch.ones((B, M_N, M_N), device=device, dtype=dtype) / float(M_N)
        J = I - ones

        D2 = D * D
        G = -0.5 * torch.bmm(torch.bmm(J, D2), J)

        try:
            evals, evecs = torch.linalg.eigh(G)
            evals = torch.flip(evals, dims=[1])
            evecs = torch.flip(evecs, dims=[2])

            evals_top = evals[:, :3].clamp_min(0.0)
            evecs_top = evecs[:, :, :3]

            X0 = evecs_top * torch.sqrt(evals_top.unsqueeze(1) + 1e-6)

            X0 = X0 - X0.mean(dim=1, keepdim=True)

            recon = torch.cdist(X0, X0)
            diag_indices = torch.arange(D.shape[-1]).to(device)
            D[:, diag_indices, diag_indices] = 0
            residual = (D - recon) / (D + 1e-4)
            weights = 1 - residual
            binder_weights = weights[:, :M, M:].unsqueeze(-1)
        except:
            binder_weights = torch.ones(B, M, N, 1).to(binder.X.device)

        binder_ca, target_ca = binder.X[:, :, self.bb, :], target.X[:, :, self.bb, :]
        r = binder_ca.unsqueeze(2) - target_ca.unsqueeze(1)

        curr_D = torch.cdist(binder.X[:, :, self.bb, :], target.X[:, :, self.bb, :])
        diff_D = (curr_D - guide_D.cuda()).unsqueeze(-1)

        r = r * diff_D
        r = r / (curr_D.unsqueeze(-1).clamp_min(1e-6))

        dist_mask = (guide_D < cutoff_threshold).float()
        num_valid_receptors = dist_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        r = r * dist_mask.unsqueeze(-1)

        bind_grad = (r * binder_weights).sum(dim=2) / num_valid_receptors
        tr_grad = 2 * bind_grad.sum(dim=1) / (bind_grad != 0).sum(1).clamp_min(1.0)
        rot_grad = self.nere_se3_from_point_grads(
            coords=binder_ca,
            dL_dcoords=bind_grad,
            mask=binder_mask,
            dt=0.1,
        )

        if return_energy:
            denom = dist_mask.sum(dim=(1, 2)).clamp_min(1.0)
            energy = (diff_D.squeeze(-1).pow(2) * dist_mask).sum(dim=(1, 2)) / denom

            return 0.1 * tr_grad, 5 * rot_grad, energy

        return 0.1 * tr_grad, 5 * rot_grad

    def randomly_translate_and_rotate(
        self, binder: Features, target: Features, tr_s_max: int
    ):
        true_X, _, bind_A, *_ = binder
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        bind_X = true_X.clone()
        B = bind_X.shape[0]
        center = bind_X.view(B, -1, 3).mean(dim=1, keepdim=True)[:, None, ...]

        centered_X = bind_X - center
        noise_w = (
            torch.from_numpy(Rotation.random(B).as_matrix())
            .to(centered_X.device)
            .float()
        )

        centered_X = torch.bmm(centered_X.view(B, -1, 3), noise_w).view_as(centered_X)

        noise_t = torch.normal(0, tr_s_max, size=(1, 3)).to(centered_X.device)
        bind_X = centered_X + noise_t[:, None, None, :] + center

        return bind_X * atom_mask
