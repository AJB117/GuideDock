import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
from utils import Features


def get_single_attn_bias(
    binder: Features,
    target: Features,
    bind_dists: Optional[Tensor] = None,
    target_dists: Optional[Tensor] = None,
):
    """
    Compute attention bias for individual binder and target chains.
    based on masks and optional distance biases. Invalid positions get large negative bias.
    """
    bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()
    target_mask = target.A[:, :, 1].clamp(max=1.0).bool()

    bind_dist_mask = bind_mask.unsqueeze(2) & bind_mask.unsqueeze(1)
    tgt_dist_mask = target_mask.unsqueeze(2) & target_mask.unsqueeze(1)

    bind_attn_bias = (~bind_dist_mask).float() * -1e9
    tgt_attn_bias = (~tgt_dist_mask).float() * -1e9

    if bind_dists is not None:
        bind_attn_bias = bind_attn_bias + bind_dists
        tgt_attn_bias = tgt_attn_bias + target_dists

    # for multi-head attention
    bind_attn_bias = bind_attn_bias.unsqueeze(1)
    tgt_attn_bias = tgt_attn_bias.unsqueeze(1)

    return bind_attn_bias, tgt_attn_bias  # (B, 1, M, M), (B, 1, N, N)


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
    cplx_mask = torch.cat([bind_mask, tgt_mask], dim=1)  # (B, M+N)
    pair_mask = cplx_mask.unsqueeze(2) & cplx_mask.unsqueeze(1)  # (B, M+N, M+N)

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

    # for multi-head attention
    attn_bias = attn_bias.unsqueeze(1)

    return attn_bias  # (B, 1, M+N, M+N), (B, 1, M+N, M+N)


def get_pairwise_cplx_attn_mask(binder: Features, target: Features):
    bind_mask = binder.A[:, :, 1].clamp(max=1.0).bool()
    tgt_mask = target.A[:, :, 1].clamp(max=1.0).bool()
    cplx_mask = torch.cat([bind_mask, tgt_mask], dim=1)  # (B, M+N)
    pair_mask = cplx_mask.unsqueeze(2) & cplx_mask.unsqueeze(1)  # (B, M+N, M+N)

    attn_mask = (~pair_mask).float() * -1e9

    return attn_mask, pair_mask


def pad_coords(binder_ca, target_ca, max_rows, max_cols):
    binder_ca = F.pad(binder_ca, (0, 0, 0, max_rows - binder_ca.size(1)))
    target_ca = F.pad(target_ca, (0, 0, 0, max_cols - target_ca.size(1)))
    return binder_ca, target_ca


def pad_masks(bind_A, target_A, max_rows, max_cols):
    bind_A = F.pad(bind_A, (0, 0, 0, max_rows - bind_A.size(1)))
    target_A = F.pad(target_A, (0, 0, 0, max_cols - target_A.size(1)))
    return bind_A, target_A


def compute_class_weights(dataloader, bin_edges):
    """Compute class weights inversely proportional to bin frequencies."""
    bin_counts = torch.zeros(len(bin_edges) - 1)

    for binder, target in dataloader:
        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]
        D = torch.cdist(binder_ca, target_ca)
        valid_mask = binder.A[:, :, 1].bool().unsqueeze(2) & target.A[
            :, :, 1
        ].bool().unsqueeze(1)

        distances = D[valid_mask].view(-1).flatten()

        bins = torch.bucketize(distances, bin_edges)  # Bin indices

        for b in range(len(bin_edges) - 1):
            bin_counts[b] += (bins == b).sum().item()

    total_counts = bin_counts.sum().item()
    class_weights = total_counts / (len(bin_edges) - 1) / bin_counts
    class_weights = class_weights / class_weights.sum() * (len(bin_edges) - 1)

    normalized_weights = class_weights / class_weights.sum()

    return normalized_weights


def compute_contact_metrics(
    pred_bins: torch.Tensor, true_bins: torch.Tensor, pair_mask: torch.Tensor
):
    """Compute accuracy / recall / F1 on binned contacts under a pairwise mask.

    pred_bins, true_bins: (B, M, N) integer tensors
    pair_mask: (B, M, N) bool tensor where True means *valid* (non-padded) position.
    """
    mask = pair_mask.bool()

    # Flatten to 1D using only valid positions
    y_true = true_bins[mask].view(-1).cpu().numpy()
    y_pred = pred_bins[mask].view(-1).cpu().numpy()

    if y_true.size == 0:
        return 0.0, 0.0, 0.0

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, rec, f1


def get_mean_std_distance(dataloader, which_dists="all", use_logscale=False):
    max_binder_size = 0
    max_target_size = 0

    all_distances = []
    all_stds = []

    print("Computing mean and std of distances...")

    try:
        mean, std = torch.load("mean_std_distance.pt")
        print(f"Loaded precomputed mean: {mean}, std: {std}")
        return mean, std
    except:
        for binder, target in tqdm(dataloader):
            binder_ca = binder.X[:, :, 1]
            target_ca = target.X[:, :, 1]

            if which_dists == "between":
                cplx_ca = torch.cat((binder_ca, target_ca), dim=1)
                D = torch.cdist(cplx_ca, cplx_ca)  # [M+N, M+N]
            elif which_dists == "all":
                D = torch.cdist(binder_ca, target_ca)
                valid_mask = binder.A[:, :, 1].bool().unsqueeze(2) & target.A[
                    :, :, 1
                ].bool().unsqueeze(1)
            else:
                raise ValueError(f"Unknown which_dists: {which_dists}")

            valid_mask = D > 0
            valid_distances = D[valid_mask] + 1e-8

            if use_logscale:
                valid_distances = valid_distances.log()

            mean_distance = valid_distances.mean().item()
            std_distance = valid_distances.std().item()

            all_distances.append(mean_distance)
            all_stds.append(std_distance)

            binder_size = binder_ca.size(1)
            target_size = target_ca.size(1)

            max_binder_size = max(max_binder_size, binder_size)
            max_target_size = max(max_target_size, target_size)

        print(f"Max binder size: {max_binder_size}, Max target size: {max_target_size}")

        N = len(all_distances)

        return sum(all_distances) / N, sum(all_stds) / N


def get_latent_mean_std(dataloader, vae):
    all_means = []
    all_stds = []

    for binder, target in dataloader:
        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]
        D = torch.cdist(binder_ca, target_ca)
        valid_mask = binder.A[:, :, 1].bool().unsqueeze(2) & target.A[
            :, :, 1
        ].bool().unsqueeze(1)

        latent = vae(
            D, binder, target, valid_mask, step=0
        )  # Assuming 'vae.encode' method exists

        mean_latent = latent.mean().item()
        std_latent = latent.std().item()

        all_means.append(mean_latent)
        all_stds.append(std_latent)

    N = len(all_means)

    return sum(all_means) / N, sum(all_stds) / N
