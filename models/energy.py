# Credit for base code to: https://github.com/wengong-jin/DSMBind

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils import Features, get_resolution_indices

from .common import FiLM
from .distance_guidance import DistanceGuidanceMixin
from .frame import AllAtomEncoder
from .modules import ForceHead, ScalarHead, SinusoidalEmbedding
from .pose_diffusion import PoseDiffusionMixin
from .pose_sampling import PoseSamplingMixin
from .rigid import FARigidModel

if TYPE_CHECKING:
    from train_contact_diff import LitContactDiff

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


class AllAtomEnergyModel(
    PoseDiffusionMixin, PoseSamplingMixin, DistanceGuidanceMixin, FARigidModel
):
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
