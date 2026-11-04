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
from torch import Tensor
from torch.func import jvp
from typing import Optional
from .modules import (
    FiLM,
    GLU,
    SinusoidalEmbedding,
    GaussianFourierProjection,
    ScalarHead,
    ForceHead,
    StatPooling,
    AttentionPooling,
)
from train_contact_diff import LitContactDiff
from utils import sample_vec, score_vec, score_norm, Features, kabsch, kabsch_torch
from .frame import *

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


# def cross_product(a, b):
#     return torch.cross(a, b, dim=-1)


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


# def rotate(X, w, sel: int):
#     B, N = X.size(0), X.size(1)
#     X = X.reshape(B, N * sel, 3)
#     w = w.unsqueeze(1).expand_as(X)  # [B,N,3]
#     c = w.norm(p=2, dim=-1, keepdim=True)  # [B,N,1]
#     c1 = torch.sin(c) / c.clamp(min=1e-6)
#     c2 = (1 - torch.cos(c)) / (c**2).clamp(min=1e-6)
#     X = X + c1 * cross_product(w, X) + c2 * cross_product(w, cross_product(w, X))
#     return X.view(B, N, sel, 3)


class FARigidModel(nn.Module):
    def __init__(self):
        super(FARigidModel, self).__init__()

    def mean(self, X, mask):
        return (X * mask[..., None]).sum(dim=1) / mask[..., None].sum(dim=1).clamp(
            min=1e-6
        )

    def inertia(self, X, mask):
        inner = (X**2).sum(dim=-1)
        inner = (
            inner[..., None, None] * torch.eye(3).to(X)[None, None, ...]
        )  # [B,N,3,3]
        outer = X.unsqueeze(2) * X.unsqueeze(3)  # [B,N,3,3]
        inertia = (inner - outer) * mask[..., None, None]
        return 0.1 * inertia.sum(dim=1)  # [B,3,3]


class AllAtomEnergyModel(FARigidModel):
    def __init__(self, args, device, resolution, dist_model: LitContactDiff):
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
            nn.Linear(hsize, hsize // 2),
            # nn.BatchNorm1d(hsize // 2),
            nn.SiLU(),
            nn.Linear(hsize // 2, hsize // 2),
        )
        self.U_o = nn.Sequential(
            nn.Linear(hsize, hsize // 2),
            # nn.BatchNorm1d(hsize // 2),
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

        # (SE(3) score heads)
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
            # nn.Linear(hidden_size, hidden_size),
            # nn.SiLU(),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Linear(hidden_size, 3),
            # GLU(hidden_size, args, act="relu"),
            nn.Linear(hidden_size, hidden_size),
        )

        if self.non_conservative:
            self.tr_force_head = ForceHead(args, depth=3)
            # self.rot_force_head = ForceHead(args, depth=3)
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
                # nn.init.zeros_(m.weight)
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                # nn.init.orthogonal_(m.weight)

    # -------------------
    # Noise helpers
    # -------------------
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
        sel = self.residue_size

        # Rotational noise & target rotational score (per-structure)
        w = torch.zeros(B, 3)
        backbone_score = torch.zeros(B, 3)
        for i in range(B):
            w_ = torch.from_numpy(sample_vec(rot_s[i].item()))
            backbone_score_ = score_vec(rot_s[i].item(), w_)
            w[i] = w_
            backbone_score[i] = backbone_score_

        # Translation noise (per-structure)
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
        center = true_X.view(B, -1, 3).sum(dim=1, keepdim=True)[
            :, None, ...
        ]  # [B, 1, 1, 3]
        nonzero = bind_mask.sum(dim=1, keepdim=True)[:, None, None, :]  # [B, 1, 1, 1]
        center = center / (nonzero * sel)

        bind_X = true_X - center
        bind_X = rotate(X=bind_X, w=w, sel=sel) + hat_t[:, None, None, :]
        # bind_X = bind_X + hat_t[:, None, None, :]
        bind_X = bind_X + center  # move back
        bind_X = bind_X * (bind_A > 0).float().unsqueeze(-1)

        return bind_X, bind_mask

    # -------------------
    # Model forward & utilities
    # -------------------
    def predict(
        self,
        binder: Features,
        target: Features,
        noise_emb: Tensor,
        eps_tr: Tensor,
        return_energy: bool = False,
        return_nodes: bool = False,
    ):
        # time_steps = self.noise_proj(time_steps)

        bind_X, bind_S, bind_A, *_ = binder
        tgt_X, _, tgt_A, *_ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        sel, _ = get_indices(self.resolution)

        if self.non_conservative:
            h1, h2 = self.encoder(
                binder,
                target,
                time_steps=noise_emb,
                eps_tr=eps_tr,
            )  # [B,N+M,14,H]
            bind_h = h1.reshape(B, N, -1)
            tgt_h = h2.reshape(B, M, -1)

            # mean pooling to get per-protein scores
            bind_h_pooled = bind_h.mean(dim=1)
            tgt_h_pooled = tgt_h.mean(dim=1)
            # bind_h_pooled = self.bind_attn_pooling(bind_h)
            # tgt_h_pooled = self.tgt_attn_pooling(tgt_h)

            combined = torch.cat((bind_h_pooled, tgt_h_pooled), dim=-1)
            f = self.node_decoder(combined)
            f = f + noise_emb
            until = N
        else:
            h1, h2 = self.encoder(
                binder,
                target,
                time_steps=time_steps,
                eps_tr=eps_tr,
            )  # [B,N+M,14,H]

            bind_h = self.W_o(h1).view(B, N * sel, -1)
            tgt_h = self.U_o(h2).view(B, M * sel, -1)

            mask_2D = (bind_A > 0).float().view(B, N * sel, 1) * (
                tgt_A > 0
            ).float().view(B, 1, M * sel)
            D = (bind_X.view(B, N * sel, 1, 3) - tgt_X.view(B, 1, M * sel, 3)).norm(
                dim=-1
            )  # [B,N*sel,M*sel]
            mask_2D = mask_2D * (D < 40 + 3 * eps_tr.view(-1, 1, 1)).float()
            # mask_2D = mask_2D * (D < self.threshold).float()
            energy = torch.matmul(bind_h, tgt_h.transpose(1, 2))  # [B,N*sel,M*sel]
            energy = (energy * mask_2D).sum(dim=(1, 2))  # [B]

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

        _, bb = get_indices(self.resolution)

        # Translation force (per-structure mean across backbone)
        tr_force = self.mean(force[:, :until, bb], bind_mask)

        # Backbone torque
        center = self.mean(bind_X[:, :, bb], bind_mask)
        bind_X = bind_X - center[:, None, None, :]
        G = torch.cross(bind_X[:, :, bb], force[:, :until, bb], dim=-1)
        G = (G * bind_mask[..., None]).sum(dim=1)  # [B,3] angular momentum
        I = self.inertia(bind_X[:, :, bb], bind_mask)  # [B,3,3] inertia matrix
        try:
            rot_force = torch.linalg.solve(
                I.detach() + torch.eye(3, device=I.device) * 1e-4, G
            )  # angular velocity
        except Exception:
            print("Inertia matrix is singular, using pseudo-inverse.")
            pdb.set_trace()

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
        B = binder.X.size(0)
        time_steps_ = time_steps
        tr_emb = self.noise_conditioner(tr_s)
        rot_emb = self.noise_conditioner(rot_s)
        noise_emb = torch.cat((tr_emb, rot_emb), dim=-1)
        noise_emb = self.noise_proj(noise_emb)

        # === SE(3) path (original behavior) ===
        rotation_force, hat_t, eps, backbone_score = self.get_noise_terms(
            binder, tr_s, rot_s
        )
        noised_bind_X, bind_mask = self.add_noise(
            binder, rotation_force, hat_t, self.resolution
        )
        noised_binder = Features(noised_bind_X, *binder[1:])

        _, bb = get_indices(self.resolution)

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

        bb_loss_mask = time_steps_ < 0.2  # 0.25?
        bb_loss = (
            self.coord_loss(binder, bind_H, tgt_H.mean(dim=1), bind_mask)
            * bb_loss_mask.float()
        )

        # contact_loss, accuracy, f1 = self.contact_loss(
        #     binder, target, bind_H, tgt_H, bind_mask
        # )
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
        sel, bb = get_indices(self.resolution)

        # contact_pred = torch.matmul(self.W_o(bind_H.reshape(B*N*sel, -1)).reshape(B, N*sel, -1), self.U_o(tgt_H.reshape(B*M*sel, -1)).reshape(B, M*sel, -1).transpose(-1, -2)).view(B, N*sel, M*sel)
        contact_pred = torch.matmul(
            self.W_o(bind_H.reshape(B * N, -1)).reshape(B, N, -1),
            self.U_o(tgt_H.reshape(B * M, -1)).reshape(B, M, -1).transpose(-1, -2),
        ).view(B, N, M)
        # bind_mask = bind_mask.repeat_interleave(sel, dim=1)

        D = (true_X[:, :, 1].view(B, N, 1, 3) - tgt_X[:, :, 1].view(B, 1, M, 3)).norm(
            dim=-1
        )
        # D = (true_X.view(B, N * sel, 1, 3) - tgt_X.view(B, 1, M * sel, 3)).norm(dim=-1)
        contact_true = (D < contact_threshold).float()
        contact_mask = bind_mask.view(B, N, 1) * (tgt_A[:, :, 1] > 0).float().view(
            B, 1, M
        )
        # contact_mask = (bind_mask.view(B, N * sel, 1) * (tgt_A > 0).float().view(B, 1, M * sel))
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
        sel, bb = get_indices(self.resolution)

        # tgt_H_mean = tgt_H_mean.unsqueeze(1).expand(-1, N * sel, -1)
        tgt_H_mean = tgt_H_mean.unsqueeze(1).expand(-1, N, -1)

        # pred_backbone = self.backbone_predictor(torch.cat((bind_H, tgt_H_mean), dim=-1).view(B*N*sel, -1))
        pred_backbone = self.backbone_predictor(
            torch.cat((bind_H, tgt_H_mean), dim=-1).view(B * N, -1)
        )
        # pred_backbone = pred_backbone.view(B, N * sel, 3)
        pred_backbone = pred_backbone.view(B, N, 3)
        # true_X = true_X.view(B, N * sel, 3)
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

        # Compute center of mass per batch: (B, 1, 3)
        com = binder_coords.mean(dim=1, keepdim=True)

        # Relative coordinates r_i = x_i - c  → (B, N, 3)
        rel = binder_coords - com

        # Torque per point: r_i × g_i  → (B, N, 3)
        torque_per_point = torch.cross(rel, guidance_grad, dim=-1)

        # Sum over points → (B, 3)
        tau = torque_per_point.sum(dim=1)

        return tau

    def _masked_mean(self, x, mask, dim):
        # mask: (B, N) float {0,1}
        w = mask.unsqueeze(-1)  # (B,N,1)
        denom = w.sum(dim=dim, keepdim=True).clamp_min(1e-8)
        return (x * w).sum(dim=dim, keepdim=True) / denom

    def nere_se3_from_point_grads(
        self,
        coords,  # (B, N, 3)
        dL_dcoords,  # (B, N, 3)
        mask=None,  # (B, N) float {0,1} optional
        dt=0.1,  # NERE uses small Δt in ω ≈ I^{-1} τ Δt:contentReference[oaicite:6]{index=6}
        damping=1e-4,  # stabilizer for inertia solve
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

        # center of mass μ (masked)
        mu = self._masked_mean(coords, mask, dim=1)  # (B,1,3)
        r = coords - mu  # (B,N,3)

        # force f_i = -∂L/∂x_i
        f = -dL_dcoords  # (B,N,3)
        f = f * mask.unsqueeze(-1)

        # torque τ = Σ r_i × f_i:contentReference[oaicite:8]{index=8}
        tau = torch.cross(r, f, dim=-1).sum(dim=1)  # (B,3)

        # inertia I = Σ (||r||^2 I - r r^T):contentReference[oaicite:9]{index=9}
        r2 = (r * r).sum(dim=-1)  # (B,N)
        term1 = r2[:, :, None, None] * eye[:, None, :, :]  # (B,N,3,3)
        term2 = r[:, :, :, None] * r[:, :, None, :]  # (B,N,3,3)
        I = (term1 - term2) * mask[:, :, None, None]
        I = I.sum(dim=1)  # (B,3,3)

        # ω ≈ I^{-1} τ Δt:contentReference[oaicite:10]{index=10}
        w_step = torch.linalg.solve(
            I.detach() + damping * eye, tau.unsqueeze(-1)
        ).squeeze(-1)

        w_step = w_step * dt  # (B,3)

        return w_step

    def get_diff_coeffs(self, noise_schedule, tr_s, rot_s):
        #! Scales
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

        # ODE Updates
        delta_t = tr_g**2 * dt * tr_score
        delta_t = delta_t + (tr_g * math.sqrt(dt) * epsilon_t)
        # delta_t = delta_t.detach().requires_grad_(True)

        delta_w = rot_g**2 * dt * rot_score
        delta_w = delta_w + (rot_g * math.sqrt(dt) * epsilon_w)
        # delta_w = delta_w.detach().requires_grad_(True)

        # # temp sampling
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

        return delta_t, delta_w

    def proj_parallel(self, g, s, eps=1e-8):
        ss = (s * s).sum(dim=-1, keepdim=True).clamp(min=eps)
        return ((g * s).sum(dim=-1, keepdim=True) / ss) * s

    # -------------------
    # Sampling
    # -------------------
    def sample_(
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
        uncertainty: Optional[Tensor] = None,
        dist_model: Optional[LitContactDiff] = None,
        binder_cond: Optional[Tensor] = None,
        target_cond: Optional[Tensor] = None,
        D_guide: Optional[Tensor] = None,
    ):
        self.eval()
        sel, bb = get_indices(self.resolution)

        true_X, bind_S, bind_A, *_ = binder
        tgt_X, tgt_S, tgt_A, *_ = target

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

        # binder_mds = self.get_mds(orig_binder, target)
        # energy = torch.tensor(0.0)

        # with torch.set_grad_enabled(not self.non_conservative):

        # if binder_guide is not None:
        # mds = self.get_mds_guidance(binder, target, binder_guide, uncertainty)

        # mds = self.get_mds(orig_binder, target)

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

            #! Move in the direction of the mds potential
            # if binder_guide is not None:
            #     with torch.enable_grad():
            #         mds_force = self.get_guidance(binder, mds)
            #         mds_torque = self.torque(
            #             binder.X[:, :, 1].view(B, -1, 3), mds_force.view(B, -1, 3)
            #         )

            # bind_X = bind_X - mds_force.view(B, -1, 3).mean(dim=1) * dt

            # bind_X = binder_mds.view(B, N, sel, 3).float()

            # === SE(3) sampling (original path) ===
            tr_s, rot_s = noise_schedule(cur_t, cur_t)

            tr_s = tr_s.repeat(B)
            rot_s = rot_s.repeat(B)

            tr_s, rot_s = (
                torch.tensor(tr_s).float().cuda(bind_X.device).view(B, 1),
                torch.tensor(rot_s).float().cuda(bind_X.device).view(B, 1),
            )

            with torch.no_grad() as X, torch.inference_mode() as Y:
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

            with torch.no_grad() as X, torch.inference_mode() as Y:
                tr_force, rot_force = self.get_forces(
                    force, until, bind_X, bind_mask, noise_emb
                )

            if self.non_conservative:
                energy = tr_force.norm() + rot_force.norm()

            #! Center and compute scores
            center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[:, None, ...]
            nonzero = bind_mask.sum(dim=1, keepdim=True)[
                :, None, None, :
            ]  # [B, 1, 1, 1]
            center = center / (nonzero * sel)

            bind_X_centered = bind_X - center  # [B, N, sel, 3]

            with torch.no_grad() as X, torch.inference_mode() as Y:
                rot_score, tr_score = self.predict_score(
                    tr_force, rot_force, tr_s, rot_s, noise_emb
                )
            tr_score = tr_score + dt**2 * t_grad.view(B, -1, 3).mean(dim=1)
            rot_score = rot_score + dt**2 * w_grad.view(B, 3)

            #! Scales
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

                # ODE Updates
                delta_t = tr_g**2 * dt * tr_score
                delta_t = delta_t + (tr_g * math.sqrt(dt) * epsilon_t)
                delta_t = delta_t.detach().requires_grad_(True)

                delta_w = rot_g**2 * dt * rot_score
                delta_w = delta_w + (rot_g * math.sqrt(dt) * epsilon_w)
                delta_w = delta_w.detach().requires_grad_(True)

                # # temp sampling
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
                bind_X_centered = bind_X_centered.requires_grad_()

                bind_X_rotated = (
                    rotate(X=bind_X_centered, w=delta_w, sel=sel)
                    + delta_t[:, None, None, :]
                )

                # bind_X_rotated = bind_X_centered + delta_t[:, None, None, :]
                bind_X = bind_X_rotated + center
                # Apply atom mask
                bind_X = bind_X * atom_mask

                binder = Features(bind_X, *binder[1:])

                #! Compute latent guidance
                x_t = torch.cdist(
                    # binder.X[:, :, bb, :].view(B, -1, 3),
                    bind_X[:, :, bb, :].view(B, -1, 3),
                    target.X[:, :, bb, :].view(B, -1, 3),
                )

                # if cur_t < 0.2:
                #     try:
                #         w_grad, t_grad = self.get_latent_guidance(
                #             x_t,
                #             torch.tensor(cur_t).float().cuda(bind_X.device).view(B),
                #             binder,
                #             target,
                #             binder_cond,
                #             target_cond,
                #             dist_model,
                #             delta_w,
                #             delta_t,
                #         )
                #     except RuntimeError as e:
                #         print(
                #             f"RuntimeError in get_latent_guidance at t={cur_t}: {e}. Setting guidance to zero."
                #         )
                #         w_grad = torch.zeros(B, 3).cuda(bind_X.device)
                #         t_grad = torch.zeros(B, 3).cuda(bind_X.device)
                # else:
                w_grad = torch.zeros(B, 3).cuda(bind_X.device)
                t_grad = torch.zeros(B, 3).cuda(bind_X.device)

                # Detach everything so far
                w_grad = w_grad.detach()
                t_grad = t_grad.detach()
                bind_X = bind_X.detach()

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
        sel, bb = get_indices(self.resolution)

        initial_X, bind_S, bind_A, *_ = binder
        tgt_X, tgt_S, tgt_A, *_ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A = bind_A * (initial_X.norm(dim=-1) > 1e-4).long()
        bind_mask = bind_A[:, :, bb].clamp(max=1).float()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        # Initialize binder coordinates
        bind_X = initial_X.clone()
        if not self.non_conservative:
            bind_X = bind_X.requires_grad_()

        B = batch_size  # Update batch size
        alpha_start = alpha_

        time_steps = np.linspace(1, 0, T + 1)[:-1]
        trajectory = [bind_X.clone().detach().cpu().numpy()]

        # binder_mds = self.get_mds(orig_binder, target)
        # energy = torch.tensor(0.0)

        # with torch.set_grad_enabled(not self.non_conservative):

        # if binder_guide is not None:
        # mds = self.get_mds_guidance(binder, target, binder_guide, uncertainty)

        # mds = self.get_mds(orig_binder, target)

        w_grad, t_grad = (
            torch.zeros(B, 3).cuda(bind_X.device),
            torch.zeros(B, 3).cuda(bind_X.device),
        )

        # bind_X_init = bind_X.clone()
        # pred_X = self.coords_from_distances(binder, target, binder_guide)
        # pred_X = self.align_by_receptor(binder, target, pred_X)
        # pred_binder_X = pred_X[:, :N, :]
        # pred_target_X = pred_X[:, N:, :]
        # binder = Features(pred_binder_X.cuda(), *binder[1:])
        for t_idx in range(T):
            cur_t = time_steps[t_idx]
            if t_idx == T - 1:
                dt = cur_t
            else:
                dt = cur_t - time_steps[t_idx + 1]

            tr_s, rot_s = noise_schedule(cur_t, cur_t)
            #! Move in the direction of the mds potential
            # t_grad, w_grad = self.get_coord_potential(binder, target, binder_guide)
            if True:
                if D_guide is not None:
                    # t_grad, w_grad = self.get_distance_potential(binder, target, D_guide)
                    t_grad, w_grad = self.get_distance_potential_spectral_mask(
                        binder, target, D_guide, cutoff_threshold
                    )
                    # rot_score = rot_score + 0.01 * w_grad
                    # tr_score = tr_score + t_grad

                    # Re-apply updates with guidance
                    # tr_s, rot_s = noise_schedule(cur_t, cur_t)
                    # tr_g, rot_g = self.get_diff_coeffs(noise_schedule, tr_s, rot_s)
                    # delta_t, delta_w = self.get_updates(tr_g, rot_g, dt, t_grad, w_grad, t_idx)
                    # delta_t, delta_w = 0.25 * t_grad, w_grad
                    # rot_g, tr_g = self.get_diff_coeffs(noise_schedule, tr_s, rot_s)

                    delta_t, delta_w = (
                        tr_grad_weight * t_grad,
                        rot_grad_weight * w_grad,
                    )

                    # delta_t, delta_w = tr_g * tr_grad_weight * (1-dt)**2 * t_grad, rot_g * rot_grad_weight * (1-dt)**2 * w_grad

                    # delta_w = delta_w + dt ** 2 * w_grad
                    # delta_t = delta_t + dt ** 2 * t_grad

                    # now move bind_X according to w_grad and t_grad
                    # center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[
                    #     :, None, ...
                    # ]  # [B, 1, 1, 3]
                    # nonzero = bind_mask.sum(dim=1, keepdim=True)[
                    #     :, None, None, :
                    # ]  # [B, 1, 1, 1]
                    # center = center / (nonzero * sel) # [B, 1, 1, 3]
                    # bind_X_centered = bind_X - center  # [B, N, sel, 3]
                    # bind_X_rotated = (
                    #     rotate(X=bind_X_centered, w=w_grad, sel=sel)
                    # )

                    # bind_X_rotated = bind_X_centered + delta_t[:, None, None, :]
                    # bind_X = bind_X_rotated + center
                    # # # Apply atom mask
                    # bind_X = bind_X * atom_mask
                    # binder = Features(bind_X, *binder[1:])

                    #! apply T before diffusion
                    center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[
                        :, None, ...
                    ]
                    nonzero = bind_mask.sum(dim=1, keepdim=True)[
                        :, None, None, :
                    ]  # [B, 1, 1, 1]
                    center = center / (nonzero * sel)
                    bind_X = bind_X - center  # [B, N, sel, 3]
                    bind_X = rotate(X=bind_X, w=delta_w, sel=sel)
                    bind_X = bind_X - delta_t[:, None, None, :]
                    bind_X = bind_X + center
                    bind_X = bind_X * atom_mask

                # if binder_guide is not None:
                #     with torch.enable_grad():
                #         mds_force = self.get_guidance(binder, mds)
                #         mds_torque = self.torque(
                #             binder.X[:, :, 1].view(B, -1, 3), mds_force.view(B, -1, 3)
                #         )

                # bind_X = bind_X - mds_force.view(B, -1, 3).mean(dim=1) * dt

                # bind_X = binder_mds.view(B, N, sel, 3).float()

            # === SE(3) sampling (original path) ===

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

            with torch.no_grad() as X, torch.inference_mode() as Y:
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

            with torch.no_grad() as X, torch.inference_mode() as Y:
                tr_force, rot_force = self.get_forces(
                    force, until, bind_X, bind_mask, noise_emb
                )

            if self.non_conservative:
                energy = tr_force.norm() + rot_force.norm()

            #! Center and compute scores
            center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[:, None, ...]
            nonzero = bind_mask.sum(dim=1, keepdim=True)[
                :, None, None, :
            ]  # [B, 1, 1, 1]
            center = center / (nonzero * sel)

            bind_X_centered = bind_X - center  # [B, N, sel, 3]

            with torch.no_grad():
                rot_score, tr_score = self.predict_score(
                    tr_force, rot_force, tr_s, rot_s, noise_emb
                )

            rot_score = rot_score.detach().requires_grad_()
            tr_score = tr_score.detach().requires_grad_()

            #! add grad to score components
            # rot_score = rot_score + w_grad
            # tr_score = tr_score + t_grad

            # Get diffusion coefficients
            tr_g, rot_g = self.get_diff_coeffs(noise_schedule, tr_s, rot_s)

            # ODE updates
            delta_t, delta_w = self.get_updates(
                tr_g, rot_g, dt, tr_score, rot_score, t_idx
            )
            delta_t = delta_t.requires_grad_()
            delta_w = delta_w.requires_grad_()

            # Apply translation and backbone rotation
            bind_X_centered = bind_X_centered.requires_grad_()
            bind_X_rotated = (
                rotate(X=bind_X_centered, w=delta_w, sel=sel)
                + delta_t[:, None, None, :]
                + center
            )

            # bind_X_rotated = bind_X_rotated + delta_t[:, None, None, :]
            # bind_X = bind_X_rotated + center
            # Apply atom mask
            bind_X = bind_X_rotated * atom_mask

            binder = Features(bind_X, *binder[1:])

            w_grad = w_grad.detach()
            t_grad = t_grad.detach()
            bind_X = bind_X.detach()

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
        sel, bb = get_indices(self.resolution)

        initial_X, bind_S, bind_A, *_ = binder
        tgt_X, tgt_S, tgt_A, *_ = target

        B, N, M = bind_S.size(0), bind_S.size(1), tgt_X.size(1)
        bind_A = bind_A * (initial_X.norm(dim=-1) > 1e-4).long()
        bind_mask = bind_A[:, :, bb].clamp(max=1).float()
        tgt_A = tgt_A * (tgt_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        # Initialize binder coordinates
        bind_X = initial_X.clone()
        if not self.non_conservative:
            bind_X = bind_X.requires_grad_()

        B = num_particles  # Update batch size

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

            # === SE(3) sampling (original path) ===

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

            with torch.no_grad() as X, torch.inference_mode() as Y:
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

            with torch.no_grad() as X, torch.inference_mode() as Y:
                tr_force, rot_force = self.get_forces(
                    force, until, bind_X, bind_mask, noise_emb
                )

            if self.non_conservative:
                energy = tr_force.norm() + rot_force.norm()

            #! Center and compute scores
            center = bind_X.view(B, -1, 3).sum(dim=1, keepdim=True)[:, None, ...]
            nonzero = bind_mask.sum(dim=1, keepdim=True)[
                :, None, None, :
            ]  # [B, 1, 1, 1]
            center = center / (nonzero * sel)

            bind_X_centered = bind_X - center  # [B, N, sel, 3]

            with torch.no_grad():
                rot_score, tr_score = self.predict_score(
                    tr_force, rot_force, tr_s, rot_s, noise_emb
                )

            rot_score = rot_score.detach().requires_grad_()
            tr_score = tr_score.detach().requires_grad_()

            # Get diffusion coefficients
            # tr_g, rot_g = self.get_diff_coeffs(noise_schedule, tr_s, rot_s)
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

            # ODE updates
            # delta_t, delta_w = self.get_updates(
            #     tr_g, rot_g, dt, tr_score, rot_score, t_idx
            # )
            # delta_t = delta_t.requires_grad_()
            # delta_w = delta_w.requires_grad_()
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

                # ODE Updates
                delta_t = tr_g**2 * dt * tr_score
                delta_t = delta_t + (tr_g * math.sqrt(dt) * epsilon_t)
                delta_t = delta_t.detach().requires_grad_(True)

                delta_w = rot_g**2 * dt * rot_score
                delta_w = delta_w + (rot_g * math.sqrt(dt) * epsilon_w)
                delta_w = delta_w.detach().requires_grad_(True)

            # Apply translation and backbone rotation
            bind_X_centered = bind_X_centered.requires_grad_()

            bind_X_rotated = (
                rotate(X=bind_X_centered, w=delta_w, sel=sel)
                + delta_t[:, None, None, :]
            )

            # bind_X_rotated = bind_X_centered + delta_t[:, None, None, :]
            bind_X = bind_X_rotated + center
            # Apply atom mask
            bind_X = bind_X * atom_mask
            energy = 0

            if t_idx % 2 == 0:
                if D_guide is not None:
                    # for k in range(2):
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
                    nonzero = bind_mask.sum(dim=1, keepdim=True)[
                        :, None, None, :
                    ]  # [B, 1, 1, 1]
                    center = center / (nonzero * sel)
                    bind_X = bind_X - center
                    bind_X = rotate(X=bind_X, w=delta_w, sel=sel)
                    bind_X = bind_X - delta_t[:, None, None, :] + center
                    bind_X = bind_X * atom_mask
                    # binder = Features(bind_X, *binder[1:])

                if prev_energy is None:
                    logG = torch.zeros((B), device=device, dtype=energy.dtype)
                else:
                    # DIFFERENCE potential with reward=-energy:
                    # logG = λ(r_t - r_prev) = -λ(E_t - E_prev)
                    # logG = -0.001 * (energy - prev_energy)

                    #! works!
                    logG = -1 * (energy - prev_energy)

                    # compute likelihood correction
                    eps_tr = tr_s * torch.randn(B, 3, device=device)
                    eps_rot = rot_s * torch.randn(B, 3, device=device)
                    ll_tr = (eps_tr**2 - (eps_tr + delta_t) ** 2).sum() / (
                        2 * tr_s**2
                    )
                    ll_rot = (eps_rot**2 - (eps_rot + delta_w) ** 2).sum() / (
                        2 * rot_s**2
                    )
                    logG = logG + ll_tr.flatten() + ll_rot.flatten()

                #! probs = F.softmax(logG, dim=0)  # (B), higher prob = lower energy
                probs = torch.nan_to_num(torch.softmax(logG, dim=0), nan=0.0, posinf=0.0, neginf=0.0)
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

            # Detach everything so far
            w_grad = w_grad.detach()
            t_grad = t_grad.detach()
            bind_X = bind_X.detach()

            # Ensure bind_X requires gradient for next iteration
            if not self.non_conservative:
                bind_X = bind_X.detach().requires_grad_()
            else:
                bind_X = bind_X.detach()

            trajectory.append(bind_X.clone().detach().cpu().numpy())

        # Return the final binder coordinates as samples. Pick the lowest energy
        bind_X = bind_X.detach()
        idx = torch.multinomial(probs, 1, replacement=True)
        # bind_X = bind_X[torch.argmin(logG)].unsqueeze(0)
        bind_X = bind_X[idx].squeeze(1)

        # if self.non_conservative:
        #     return bind_X, bind_S, bind_A, None, trajectory
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
        binder_mask = (binder.A[:, :, self.bb] > 0).float()  # (B, M)
        target_mask = (target.A[:, :, self.bb] > 0).float()  # (B, N)

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

        # Gram matrix: G = -0.5 * J * (D^2) * J
        D2 = D * D
        G = -0.5 * torch.bmm(torch.bmm(J, D2), J)

        try:
            evals, evecs = torch.linalg.eigh(G)
            evals = torch.flip(evals, dims=[1])  # descending: (B, N)
            evecs = torch.flip(evecs, dims=[2])  # match: (B, N, N)

            # Take top-dim components, clamp negatives to 0 for init
            evals_top = evals[:, :3].clamp_min(0.0)  # (B, dim)
            evecs_top = evecs[:, :, :3]  # (B, N, dim)

            X0 = evecs_top * torch.sqrt(evals_top.unsqueeze(1) + 1e-6)

            # Center to remove translation drift
            X0 = X0 - X0.mean(dim=1, keepdim=True)

            # Compute residual
            recon = torch.cdist(X0, X0)
            diag_indices = torch.arange(D.shape[-1]).to(device)
            D[:, diag_indices, diag_indices] = 0
            residual = (D - recon) / (D + 1e-4)
            weights = 1 - residual
            binder_weights = weights[:, :M, M:].unsqueeze(-1)  # (B, N, M, 1)
        except:
            binder_weights = torch.ones(B, M, N, 1).to(binder.X.device)

        binder_ca, target_ca = binder.X[:, :, self.bb, :], target.X[:, :, self.bb, :]
        r = binder_ca.unsqueeze(2) - target_ca.unsqueeze(1)  # (B, N, M, 3)

        curr_D = torch.cdist(
            binder.X[:, :, self.bb, :], target.X[:, :, self.bb, :]
        )  # (B, N, M)
        diff_D = (curr_D - guide_D.cuda()).unsqueeze(-1)  # (B, N, M, 1)

        r = r * diff_D  # (B, N, M, 3)
        r = r / (curr_D.unsqueeze(-1).clamp_min(1e-6))  # (B, N, M, 3)

        # Mask out distances outside the interface
        dist_mask = (guide_D < cutoff_threshold).float()  # (B, N, M)
        num_valid_receptors = dist_mask.sum(dim=2, keepdim=True).clamp_min(
            1.0
        )  # (B, N, 1)
        r = r * dist_mask.unsqueeze(-1)

        bind_grad = (r * binder_weights).sum(dim=2) / num_valid_receptors  # (B, N, 3)
        tr_grad = 2 * bind_grad.sum(dim=1) / (bind_grad != 0).sum(1).clamp_min(1.0)
        rot_grad = self.nere_se3_from_point_grads(
            coords=binder_ca,
            dL_dcoords=bind_grad,  # (B, N, 3)
            mask=binder_mask,
            dt=0.1,
        )  # (B, 3)

        if return_energy:
            denom = dist_mask.sum(dim=(1, 2)).clamp_min(1.0)  # (B,)
            energy = (diff_D.squeeze(-1).pow(2) * dist_mask).sum(
                dim=(1, 2)
            ) / denom  # (B,)

            return 0.1 * tr_grad, 5 * rot_grad, energy

        return 0.1 * tr_grad, 5 * rot_grad

    def get_distance_potential(
        self, binder: Features, target: Features, guide_D: Tensor
    ):
        binder_ca, target_ca = binder.X[:, :, self.bb, :], target.X[:, :, self.bb, :]
        r = binder_ca.unsqueeze(2) - target_ca.unsqueeze(1)  # (B, N, M, 3)

        curr_D = torch.cdist(
            binder.X[:, :, self.bb, :], target.X[:, :, self.bb, :]
        )  # (B, N, M)
        diff_D = (curr_D - guide_D.cuda()).unsqueeze(-1)  # (B, N, M, 1)

        r = r * diff_D  # (B, N, M, 3)
        r = r / (curr_D.unsqueeze(-1).clamp_min(1e-6))  # (B, N, M, 3)

        binder_mask = (binder.A[:, :, self.bb] > 0).float()  # (B, N)
        target_mask = (target.A[:, :, self.bb] > 0).float()  # (B, M)

        # mask out outlier distances
        # top_75 = torch.quantile(guide_D, 0.75, dim=-1, keepdim=True)  # (B,1)
        # dist_mask = (curr_D < top_75.cuda()).float()  # (B, N, M)

        # mask = binder_mask.unsqueeze(-1) * target_mask.unsqueeze(1)  # (B, N, M)
        # mask = binder_mask.unsqueeze(-1) * dist_mask  # (B, N, M)
        # r = r * mask.unsqueeze(-1)  # (B, N, M, 3)

        bind_grad = r.sum(dim=2) / target_mask.sum(1, keepdim=True).clamp_min(
            1.0
        )  # (B, N, 3)
        tr_grad = (
            2 * bind_grad.sum(dim=1) / binder_mask.sum(1, keepdim=True).clamp_min(1.0)
        )

        rot_grad = self.nere_se3_from_point_grads(
            coords=binder_ca,
            dL_dcoords=bind_grad,  # (B, N, 3)
            mask=binder_mask,
            dt=0.1,
        )  # (B, 3)

        # return 1 / 2 * tr_grad, 9 * rot_grad

        return 0.1 * tr_grad, 5 * rot_grad

    def get_coord_potential(
        self, binder: Features, target: Features, binder_guide: Tensor
    ):
        # mds_coords = coords_from_distances(binder, target, guide_D)  # (B, M+N, 3)
        # mds_coords = align_by_receptor(binder, target, mds_coords)  # (B, M+N, 3)

        # mds_binder = mds_coords[:, : binder.X.shape[1], :].cuda()  # (B, N, 3)
        curr_X = binder.X[:, :, self.bb, :]  # (B, N, 3)

        mds_binder = binder_guide.cuda()
        r = mds_binder - curr_X  # (B, N, 3)

        binder_mask = (binder.A[:, :, self.bb] > 0).float()  # (B, N)
        r = r * binder_mask.unsqueeze(-1)
        tr_grad = 2 * r.sum(dim=1) / binder_mask.sum(1, keepdim=True).clamp_min(1.0)

        rot_grad = self.nere_se3_from_point_grads(
            coords=curr_X,
            dL_dcoords=r,
            mask=binder_mask,
            dt=1.0,
        )  # (B, 3)

        return tr_grad, rot_grad

    def get_sds_guidance(
        self,
        tau: Tensor,  # [B], in [0, 0.95]
        binder: Features,
        target: Features,
        binder_cond: Tensor,
        target_cond: Tensor,
        dist_model: LitContactDiff,
        delta_w: Tensor,  # [B,3], requires_grad=True
        delta_t: Tensor,  # [B,3], requires_grad=True
    ):
        B = binder.X.shape[0]
        M = binder_cond.shape[1]
        device = tau.device

        # --- 1) Build the distance matrix x from the current coordinates ---
        bb = 1  # you use bb index for backbone atom in cdist; keep
        bind_X = binder.X
        bind_X = bind_X[:, :, bb, :].view(B, -1, 3).detach().requires_grad_(True)

        x = torch.cdist(
            bind_X,
            target.X[:, :, bb, :].view(B, -1, 3),
        )  # [B, Nb, Nt]

        # --- 2) Encode distances to latent tokens z (this is your "data latent") ---
        with torch.enable_grad():
            z1 = dist_model.vae.vae.get_latents(x, binder, target)  # [B, L, latent_dim]

        fm = dist_model.diffusion_model
        z1 = (z1 - fm.latent_mean) / (fm.latent_std + 1e-6)

        t = (0.3 - 0.7) * torch.rand(
            B, device=device
        ) + 0.8  # random time in [0.2, 0.8]
        # t_view = tau.view(B, 1, 1).float()  # broadcast over tokens/latent_dim
        z0 = torch.randn_like(z1)
        z0 = z0 - z0.mean(dim=0, keepdim=True)

        t_view = t.view(B, 1, 1).float()  # broadcast over tokens/latent_dim

        z_t = (1.0 - t_view) * z0 + t_view * z1  # [B, L, latent_dim]

        # --- 3) Predict Z1 at (z_t, t) ---
        z1_hat = fm.predict_z1(
            binder=binder,
            target=target,
            zt=z_t,
            t=tau,
            bind_cond=binder_cond,
            target_cond=target_cond,
            guidance_scale=5.0,
            mix=True,
        ).detach()

        # weight = min(t_view**2, (1 - t_view) ** 2)
        z1_hat_bind = z1_hat[:, :M, :]
        z1_bind = z1[:, :M, :]
        loss = ((z1_hat_bind - z1_bind) ** 2).sum(dim=1).mean()
        loss.backward()

        # w_grad = delta_w.grad  # [B,3]
        # t_grad = delta_t.grad  # [B,3]

        # return w_grad, t_grad
        grad = bind_X.grad

        # mask = torch.ones(bind_X.shape[:2], device=bind_X.device)
        # t_step, w_step = self.nere_se3_from_point_grads(
        #     coords=bind_X,
        #     dL_dcoords=grad,
        #     mask=mask,
        #     dt=0.1,
        # )

        t_grad = grad.view(B, -1, 3).mean(
            dim=1
        )  # approximate translation gradient as mean over points
        w_grad = self.torque(binder.X[:, :, bb].view(B, -1, 3), grad.view(B, -1, 3))

        return w_grad, t_grad

    def get_latent_guidance(
        self,
        tau: Tensor,  # [B], in [0, 0.95]
        binder: Features,
        target: Features,
        binder_cond: Tensor,
        target_cond: Tensor,
        dist_model: LitContactDiff,
        rot_score: Tensor,  # [B,3], requires_grad=True
        tr_score: Tensor,  # [B,3], requires_grad=True
        lambda0: float = 1.0,
        guidance_scale: float = 5.0,  # CFG inside dist model if you want
        clip_norm: float = 10.0,
        eps: float = 1e-6,
    ):
        """
        Returns (w_grad, t_grad) each [B,3] to be injected into your rot_score/tr_score.
        Implements: g_theta = (dz_tau/dtheta)^T * s_tau(z_tau),
        where s_tau is a Tweedie/score surrogate from Z1 prediction.
        """

        B = tau.shape[0]
        device = tau.device

        # --- 1) Build the distance matrix x from the current coordinates ---
        bb = 1  # you use bb index for backbone atom in cdist; keep consistent with your code
        x = torch.cdist(
            binder.X[:, :, bb, :].view(B, -1, 3),
            target.X[:, :, bb, :].view(B, -1, 3),
        )  # [B, Nb, Nt]

        # --- 2) Encode distances to latents z (this is your "data latent") ---
        s_latent_avg = torch.zeros(
            B,
            binder.X.size(1) + target.X.size(1),
            dist_model.diffusion_model.latent_dim,
            device=device,
        )
        K = 1
        # for k in range(K):

        # IMPORTANT: this must be differentiable w.r.t. x
        binder_expand = binder.expand(K)
        target_expand = target.expand(K)

        z = dist_model.vae.vae.get_latents(
            x.repeat_interleave(K, dim=0), binder_expand, target_expand
        )  # [B, L, latent_dim] typically

        # Normalize the same way your flow model trains internally
        fm = dist_model.diffusion_model
        z = (z - fm.latent_mean) / (fm.latent_std + eps)

        # --- 3) Construct z_tau consistent with your flow-matching training path ---
        tau_view = tau.view(B, 1, 1).float()  # broadcast over tokens/latent_dim
        tau_view = tau_view.repeat_interleave(K, dim=0)

        z0 = torch.randn_like(z)
        # Match your model’s training choice if you kept it:
        z0 = z0 - z0.mean(dim=0, keepdim=True)

        z_tau = (1.0 - tau_view) * z0 + tau_view * z  # [B, L, latent_dim]
        # --- 4) Predict Z1 at (z_tau, tau) ---
        with torch.no_grad() as X, torch.inference_mode() as Y:
            z1_hat = fm.predict_z1(
                binder=binder_expand,
                target=target_expand,
                zt=z_tau,
                t=tau,
                bind_cond=binder_cond.repeat_interleave(K, dim=0),
                target_cond=target_cond.repeat_interleave(K, dim=0),
                guidance_scale=guidance_scale,
                mix=True,
            )

        # Stop-grad through teacher for stability (SDS-style)
        z1_hat = z1_hat.detach()

        # --- 5) Tweedie/score surrogate in latent space ---
        # s_tau(z_tau) ≈ ∇_{z_tau} log p_tau(z_tau)
        denom = (1.0 - tau_view).clamp(min=1e-4)
        s_latent = (tau_view * z1_hat - z_tau) / (denom**2)
        # s_latent = (z_tau - z1_hat) / denom

        s_latent_avg = s_latent.mean(dim=0)  # average over K samples

        # Optional: detach s_latent so this becomes a pure VJP (no Hessian terms)
        s_latent = s_latent.detach()

        # --- 6) Pull back to (delta_w, delta_t) via VJP ---
        # Equivalent to autograd.grad(outputs=z_tau, inputs=(delta_w,delta_t), grad_outputs=s_latent)
        # w_grad, t_grad = torch.autograd.grad(
        #     outputs=z_tau,
        #     inputs=(rot_score, tr_score),
        #     grad_outputs=s_latent_avg,
        #     retain_graph=True,
        #     create_graph=False,
        #     allow_unused=False,
        # )  # each [B,3]
        loss = -(s_latent_avg * z_tau).sum()
        loss.backward()
        w_grad = rot_score.grad  # [B,3]
        t_grad = tr_score.grad  # [B,3]

        # --- 7) Lambda scaling: compensate (1-tau)^(-2) blow-up and keep stable magnitude ---
        # Theory-motivated base: lambda(t) ∝ (1 - tau)^2 / tau
        lam = lambda0 * ((1.0 - tau).clamp(min=1e-4) ** 2) / (tau.clamp(min=1e-3))
        lam = lam.view(B, 1)

        w_grad = lam * w_grad
        t_grad = lam * t_grad

        # # --- 8) Norm clip (prevents rare explosions) ---
        # w_norm = torch.norm(w_grad, dim=-1, keepdim=True).clamp(min=eps)
        # t_norm = torch.norm(t_grad, dim=-1, keepdim=True).clamp(min=eps)
        # w_grad = w_grad * torch.clamp(clip_norm / w_norm, max=1.0)
        # t_grad = t_grad * torch.clamp(clip_norm / t_norm, max=1.0)

        return w_grad, t_grad

    def get_latent_guidance_(
        self,
        x_t: Tensor,
        t: Tensor,
        binder: Features,
        target: Features,
        binder_cond: Tensor,
        target_cond: Tensor,
        dist_model: LitContactDiff,
        delta_w: Tensor,
        delta_t: Tensor,
    ):
        """
        x_t: distance matrix at time t: [B, N_binder, N_target]
        Returns: guidance gradient: [B, N_binder, 3]
        """
        z_t = dist_model.vae.vae.get_latents(x_t, binder, target)
        velocity = dist_model.diffusion_model.get_velocity(
            binder, target, z_t, t, binder_cond, target_cond
        )  # [B, latent_dim]

        # def decode_latents(z):
        #     x_recon = dist_model.vae.vae.decode(
        #         z, binder, target, upscale=True
        #     )  # [B, N_binder, N_target]
        #     return x_recon
        obj = (z_t * velocity).sum()

        grads = torch.autograd.grad(
            outputs=obj,
            inputs=(delta_w, delta_t),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        w_grad, t_grad = grads  # [B,3], [B,3]

        return w_grad, t_grad

    def get_linear_scale(self, t, T, initial_eta, s=0.008):
        """Linear noise schedule"""
        return initial_eta * (1 - t / T)

    def get_cosine_scale(self, t, T, initial_eta, s=0.008):
        """Cosine noise schedule"""
        t_scaled = t / T
        cosine_factor = np.cos(((t_scaled + s) / (1 + s) * np.pi * 0.5)) ** 2
        return initial_eta * cosine_factor

    def randomly_translate_and_rotate(
        self, binder: Features, target: Features, tr_s_max: int
    ):
        true_X, _, bind_A, *_ = binder
        bind_A = bind_A * (true_X.norm(dim=-1) > 1e-4).long()
        atom_mask = (bind_A > 0).float().unsqueeze(-1)

        # Initialize binder coordinates
        bind_X = true_X.clone()
        B = bind_X.shape[0]
        center = bind_X.view(B, -1, 3).mean(dim=1, keepdim=True)[:, None, ...]

        centered_X = bind_X - center
        noise_w = (
            torch.from_numpy(Rotation.random(B).as_matrix())
            .to(centered_X.device)
            .float()
        )
        # centered_X = centered_X @ noise_w
        centered_X = torch.bmm(centered_X.view(B, -1, 3), noise_w).view_as(centered_X)

        noise_t = torch.normal(0, tr_s_max, size=(1, 3)).to(centered_X.device)
        bind_X = centered_X + noise_t[:, None, None, :] + center

        return bind_X * atom_mask

    def get_mds(self, binder: Features, target: Features):
        B = binder.X.size(0)
        flattened_binder_X = binder.X[:, :, 1, :].view(B, -1, 3).cuda()
        flattened_target_X = target.X[:, :, 1, :].view(B, -1, 3).cuda()

        # D_inter = torch.cdist(
        #     flattened_binder_X, flattened_target_X
        # ).double()  # [B, N_binder, N_target]

        # z, _, _ = self.vae.get_latents(D_inter, binder, target)

        cplx_X = torch.cat(
            [flattened_binder_X, flattened_target_X], dim=1
        )  # [B, N_binder + N_target, 3]

        D = torch.cdist(
            cplx_X, cplx_X
        ).double()  # [B, N_binder + N_target, N_binder + N_target]

        mds_solution = torch.from_numpy(
            batched_solve_mds(D.cpu().numpy(), dim=3)[0]
        ).cuda()  # [B, N_binder + N_target, 3]

        # D_pred = self.vae.decode(z, binder, target, upscale=True) # [B, N_binder, N_target]

        # mask_2D = binder_mask.view(B, -1, 1) * target_mask.view(B, 1, -1)  # [B, N_binder, N_target]

        # binder_mds = mds_solution[
        #     : flattened_binder_X.size(1), :
        # ]  # [B, N_binder, 3]
        sel, _ = get_indices(self.resolution)
        binder_mds = torch.from_numpy(
            mds_solution[: flattened_binder_X.size(1), :].double().cpu().numpy()
        ).unsqueeze(0)  # [B, N_binder, 3]

        binder_aligned, _, _ = kabsch(
            binder_mds.view(1, -1, 3),
            binder.X[:, :, 1, :].view(1, -1, 3).cpu().double(),
        )  # [B, N_binder, 3]

        return binder_aligned.reshape(B, -1, 3).float().cuda()

    def get_mds_guidance(
        self,
        binder: Features,
        target: Features,
        binder_guide: Tensor,
        uncertainty: Optional[Tensor] = None,  # [B, N_binder, N_target]
    ):
        """
        Compute MDS solution given binder_guide and then return the guidance gradient.
        """
        D = binder_guide  # [B, N_binder + N_target, N_binder + N_target]
        B = binder.X.size(0)
        flattened_binder_X = binder.X[:, :, 1, :].view(B, -1, 3).cuda()
        flattened_target_X = target.X[:, :, 1, :].view(B, -1, 3).cuda()
        # uncertainty = uncertainty.float()

        mds_solution = torch.from_numpy(
            batched_solve_mds(D.cpu().numpy(), dim=3)[0]
        ).cuda()  # [B, N_binder, 3]
        sel, _ = get_indices(self.resolution)
        binder_mds = torch.from_numpy(
            mds_solution[: flattened_binder_X.size(1), :].double().cpu().numpy()
        ).unsqueeze(0)  # [B, N_binder, 3]
        target_mds = torch.from_numpy(
            mds_solution[flattened_binder_X.size(1) :, :].double().cpu().numpy()
        ).unsqueeze(0)  # [B, N_target, 3]

        #! binder-aligned
        binder_aligned, _, _ = kabsch(
            binder_mds.view(1, -1, 3),
            flattened_binder_X.view(1, -1, 3).cpu().double(),
        )  # [B, N_binder, 3]

        #! target-aligned
        # _, R, t = kabsch(
        #     target_mds.view(1, -1, 3),
        #     flattened_target_X.view(1, -1, 3).cpu().double
        # )  # [B, N_target, 3]
        # binder_aligned = (binder_mds @ R) + t  # [B, N_binder, 3]

        #! complex-aligned
        # aligned, _, _ = kabsch(
        #     mds_solution.view(1, -1, 3).cpu().double(),
        #     torch.cat([flattened_binder_X, flattened_target_X], dim=1).cpu().double(),
        # )  # [B, N_binder, 3]

        #! complex-aligned, weigh by uncertainty
        # if uncertainty is not None:
        #     # uncertainty: [B, N_binder//4, N_target//4]
        #     conf = 1.0 / (uncertainty + 1e-8)  # [B, Nb//4, Nt//4]

        #     # point weights on coarse grids
        #     # binder coarse weight: mean confidence to targets
        #     w_b_coarse = conf.mean(dim=2)  # [B, Nb//4]
        #     # target coarse weight: mean confidence to binders
        #     w_t_coarse = conf.mean(dim=1)  # [B, Nt//4]

        #     w_b = w_b_coarse  # [B, Nb]
        #     w_t = w_t_coarse  # [B, Nt]

        #     # Build complex-level weights: [B, Nb+Nt]
        #     w_complex = torch.cat([w_b, w_t], dim=1).to(flattened_binder_X.device)

        #     # Current complex coords (target of alignment): [B, Nb+Nt, 3]
        #     cur_complex = torch.cat([flattened_binder_X, flattened_target_X], dim=1).to(
        #         torch.float32
        #     )

        #     # MDS complex coords (source): [B, Nb+Nt, 3]
        #     # mds_solution is already [B, Nb+Nt, 3] on CUDA, but may be double; cast to float
        #     mds_complex = mds_solution.to(cur_complex.dtype)

        #     # Weighted rigid alignment of MDS complex onto current complex
        #     aligned, _, _ = weighted_kabsch_torch(
        #         mds_complex.unsqueeze(0).double(),
        #         cur_complex.double(),
        #         w_complex.double(),
        #     )

        # else:
        #     # fallback: unweighted complex-aligned
        #     aligned, _, _ = kabsch_torch(
        #         mds_solution.to(torch.float32),
        #         torch.cat([flattened_binder_X, flattened_target_X], dim=1).to(
        #             torch.float32
        #         ),
        #     )

        # binder_aligned = aligned[:, : flattened_binder_X.size(1), :]

        binder_aligned = binder_aligned.reshape(B, -1, 3).float().cuda()
        return binder_aligned

    def get_mds_guidance_(
        self,
        binder: Features,
        target: Features,
        binder_guide: Tensor,
        uncertainty: Optional[Tensor] = None,  # [B, Nb_coarse, Nt_coarse]
    ):
        """
        Target-anchored, uncertainty-weighted MDS guidance.

        Returns:
            binder_aligned: [B, Nb, 3]  (binder CA coords in target frame)
        """

        device = binder.X.device
        eps = 1e-6

        # ---- extract CA coordinates ----
        binder_ca = binder.X[:, :, 1, :].contiguous().double()  # [B, Nb, 3]
        target_ca = target.X[:, :, 1, :].contiguous().double()  # [B, Nt, 3]

        B, Nb, _ = binder_ca.shape
        Nt = target_ca.shape[1]

        # ---- solve MDS on provided distance matrix ----
        # binder_guide: [B, Nb+Nt, Nb+Nt]
        mds_np = batched_solve_mds(binder_guide.cpu().numpy(), dim=3)
        mds = torch.from_numpy(mds_np).to(device)  # [B, Nb+Nt, 3]

        binder_mds = mds[:, :Nb, :]  # [B, Nb, 3]
        target_mds = mds[:, Nb : Nb + Nt, :]  # [B, Nt, 3]

        # ---- compute target weights from uncertainty ----
        if uncertainty is not None:
            # inverse-uncertainty (precision) weighting
            w_pair = 1.0 / (uncertainty + eps)  # [B, Nb_c, Nt_c]

            # per-target coarse weights
            w_t_coarse = w_pair.mean(dim=1)  # [B, Nt_c]

            # expand coarse -> CA
            group = 4  # assumes 4 residues per coarse bead

            def expand_coarse(wc, L):
                Bc, K = wc.shape
                w = wc.repeat_interleave(group, dim=1)
                if w.size(1) < L:
                    pad = L - w.size(1)
                    w = torch.cat([w, w[:, -1:].expand(Bc, pad)], dim=1)
                return w[:, :L]

            w_t = expand_coarse(w_t_coarse, Nt)

        else:
            w_t = torch.ones((B, Nt), device=device)

        # apply target atom mask (CA present)
        tgt_mask = (target.A[:, :, 1] > 0).float()
        w_t = w_t * tgt_mask

        # normalize
        w_t = w_t / (w_t.sum(dim=1, keepdim=True) + eps)

        # ---- weighted alignment: target_mds -> target_ca ----
        R, muP, muQ = weighted_kabsch_torch(
            target_mds.double(),
            target_ca.double(),
            w_t.double(),
        )

        # ---- apply same transform to binder ----
        binder_aligned = (binder_mds - muP) @ R + muQ  # [B, Nb, 3]

        return binder_aligned

    def get_guidance(self, binder: Features, mds: Tensor):
        # potential = ((binder_aligned - binder_X) * binder_mask.unsqueeze(-1)) ** 2  # [B, N_binder, 3]
        # potential = potential.sum(dim=-1)
        binder_X = binder.X[:, :, 1, :].view(binder.X.size(0), -1, 3)
        binder_mask = binder.A[:, :, 1].view(binder.X.size(0), -1).clamp(max=1).float()

        potential_grad = 2 * (binder_X - mds) * binder_mask.unsqueeze(-1)

        return potential_grad.view_as(binder.X[:, :, 1, :]).detach().float()


def weighted_kabsch_torch(P, Q, w, eps=1e-6):
    """
    Weighted rigid alignment P -> Q
    P, Q: [B, N, 3]
    w:    [B, N] (nonnegative)
    Returns:
      R:    [B, 3, 3]
      muP:  [B, 1, 3]
      muQ:  [B, 1, 3]
    """
    B, N, _ = P.shape

    w = w.clamp_min(0.0)
    w = w / (w.sum(dim=1, keepdim=True) + eps)  # normalize

    muP = (P * w[..., None]).sum(dim=1, keepdim=True)
    muQ = (Q * w[..., None]).sum(dim=1, keepdim=True)

    Pc = P - muP
    Qc = Q - muQ

    # weighted covariance
    H = Pc.transpose(1, 2) @ (Qc * w[..., None])  # [B,3,3]

    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.transpose(-2, -1).double()

    # reflection fix
    det = torch.det(V @ U.transpose(-2, -1))
    D = torch.eye(3, device=P.device).unsqueeze(0).repeat(B, 1, 1).double()
    D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)

    R = V @ D @ U.transpose(-2, -1)

    return R, muP, muQ
