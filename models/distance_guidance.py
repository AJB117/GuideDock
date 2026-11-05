from __future__ import annotations

import torch
from torch import Tensor

from utils import Features


class DistanceGuidanceMixin:
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
