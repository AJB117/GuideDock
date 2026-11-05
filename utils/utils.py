from copy import deepcopy
from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from tqdm import tqdm

from .constants import ALPHABET


def idx_to_seq(seq_idx: torch.Tensor):
    """Convert sequence indices to string representation.

    Args:
        seq_idx (torch.Tensor): Tensor of shape [B, L] containing sequence indices.

    Returns:
        List[str]: List of sequences as strings.
    """
    B, L = seq_idx.size()
    seqs = []
    for b in range(B):
        seq = "".join([ALPHABET[seq_idx[b, l]] for l in range(L)])
        if seq == "#":
            continue
        seqs.append(seq)
    return "".join(seqs)


class Features(NamedTuple):
    X: torch.Tensor
    S: torch.Tensor
    A: torch.Tensor
    V: torch.Tensor
    O: torch.Tensor
    T: torch.Tensor
    C: torch.Tensor
    Si: torch.Tensor
    pdbs: List[str]

    def expand(self, B_new):
        B_old = self.X.size(0)
        assert B_new % B_old == 0, "B_new must be multiple of B_old"
        repeat_times = B_new // B_old
        return Features(
            X=self.X.repeat_interleave(repeat_times, dim=0),
            S=self.S.repeat_interleave(repeat_times, dim=0),
            A=self.A.repeat_interleave(repeat_times, dim=0),
            V=self.V.repeat_interleave(repeat_times, dim=0),
            O=self.O.repeat_interleave(repeat_times, dim=0),
            T=self.T.repeat_interleave(repeat_times, dim=0),
            C=self.C.repeat_interleave(repeat_times, dim=0),
            Si=self.Si.repeat_interleave(repeat_times, dim=0),
            pdbs=self.pdbs * repeat_times,
        )

    def to(self, device):
        return Features(
            X=self.X.to(device),
            S=self.S.to(device),
            A=self.A.to(device),
            V=self.V.to(device),
            O=self.O.to(device),
            T=self.T.to(device),
            C=self.C.to(device),
            Si=self.Si.to(device),
            pdbs=self.pdbs,
        )


class NoiseSchedule:
    """
    Transforms t into scaled sigmas
    """

    def __init__(self, tr_s_min, tr_s_max, rot_s_min, rot_s_max):
        self.tr_s_min = tr_s_min
        self.tr_s_max = tr_s_max
        self.rot_s_max = rot_s_max
        self.rot_s_min = rot_s_min

    def __call__(self, t_tr, t_rot):
        """
        Convert from time to (scaled) sigma space
        """
        tr_s = self.tr_s_min ** (1 - t_tr) * self.tr_s_max**t_tr
        rot_s = self.rot_s_min ** (1 - t_rot) * self.rot_s_max**t_rot

        return tr_s, rot_s


# Credit for base code to: https://github.com/wengong-jin/DSMBind
def compute_dihedrals(X, eps=1e-7):

    X = X.unsqueeze(0)
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

    dX = X[:, 1:, :] - X[:, :-1, :]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    u_0 = U[:, 2:, :]

    n_2 = F.normalize(torch.linalg.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.linalg.cross(u_1, u_0), dim=-1)

    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

    D = F.pad(D, (3, 0), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    phi, psi, omega = torch.unbind(D, -1)

    D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
    nan_indices = torch.isnan(D_features)

    D_features[nan_indices] = 0.0
    return D_features


def recursive_to(obj, device):
    """Recursively move tensors in a nested structure to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(
            torch.device("cuda", device) if isinstance(device, int) else device
        )
    elif isinstance(obj, list):
        return [recursive_to(item, device) for item in obj]
    elif isinstance(obj, Features):
        return Features(*(recursive_to(field, device) for field in obj))
    elif isinstance(obj, tuple):
        return tuple(recursive_to(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    else:
        return obj


def get_cplx_attn_bias(
    binder: Features,
    target: Features,
    cplx_dists: Optional[Tensor] = None,
    bind_dists: Optional[Tensor] = None,
    target_dists: Optional[Tensor] = None,
    inter_dists: Optional[Tensor] = None,
):
    """
    Compute attention bias for complex (binder + target)
    based on masks and optional distance biases. Invalid positions get large negative bias.
    """
    bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()
    tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()
    cplx_mask = torch.cat([bind_mask, tgt_mask], dim=1)
    pair_mask = cplx_mask.unsqueeze(2) & cplx_mask.unsqueeze(1)

    attn_bias = (~pair_mask).float() * -1e9

    if cplx_dists is not None:
        attn_bias = attn_bias + cplx_dists
    elif bind_dists is not None and target_dists is not None:
        M = binder.X.size(1)
        attn_bias[:, :M, :M] = attn_bias[:, :M, :M] + bind_dists
        attn_bias[:, M:, M:] = attn_bias[:, M:, M:] + target_dists

        if inter_dists is not None:
            attn_bias[:, :M, M:] = attn_bias[:, :M, M:] + inter_dists
            attn_bias[:, M:, :M] = attn_bias[:, M:, :M] + inter_dists.transpose(1, 2)

    attn_bias = attn_bias.unsqueeze(1)

    return attn_bias


class EMACallback(Callback):
    def __init__(self, decay=0.9999, every_n_steps: int = 10):
        self.decay = decay
        self.every_n_steps = every_n_steps
        self.module_pair_list = []

    def on_pretrain_routine_start(self, trainer, pl_module) -> None:
        def forward_wrapper(module, org, ema):
            def forward(*args, **kwargs):
                return org(*args, **kwargs) if module.training else ema(*args, **kwargs)

            return forward

        modules = list(
            filter(
                lambda x: len(list(x[1].parameters())) > 0,
                pl_module.named_children(),
            )
        )

        for name, module in modules:
            ema_module = deepcopy(module)
            self.module_pair_list.append((ema_module, module))
            pl_module.add_module(f"EMA_{name}", ema_module)
            module.forward_bc = module.forward
            module.forward = forward_wrapper(
                module, module.forward_bc, ema_module.forward
            )

    def on_after_backward(self, trainer, pl_module) -> None:
        step = trainer.global_step

        if step % self.every_n_steps != 0:
            return

        for ema_module, module in self.module_pair_list:
            self._update(
                ema_module,
                module,
                update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m,
            )

    def _update(self, ema_module, module, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                ema_module.state_dict().values(), module.state_dict().values()
            ):
                ema_v.copy_(update_fn(ema_v, model_v))


def get_resolution_indices(resolution):
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


def get_latent_mean_std(
    dataloader, vae, cache_path="latent_mean_std.pt", recompute=False
):
    if not recompute:
        try:
            stats = torch.load(cache_path)
            return stats["mean"], stats["std"]
        except FileNotFoundError:
            pass
    all_latents = []
    for binder, target in tqdm(dataloader):
        binder = recursive_to(binder, vae.device)
        target = recursive_to(target, vae.device)
        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()
        distances = torch.cdist(binder.X[:, :, 1], target.X[:, :, 1])
        with torch.no_grad():
            latent = vae.vae.get_latents(distances, binder, target)
            latent = latent * torch.cat([bind_mask, tgt_mask], dim=1).unsqueeze(-1)
            all_latents.append(latent[latent.abs().sum(dim=-1) > 0])
    all_latents = torch.cat(all_latents, dim=0)
    mean, std = all_latents.mean(dim=0), all_latents.std(dim=0)
    torch.save({"mean": mean, "std": std}, cache_path)
    return mean, std
