# Credit for base code to: https://github.com/wengong-jin/DSMBind

from __future__ import annotations

import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch import Tensor

from utils import Features, get_resolution_indices, rotate


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
