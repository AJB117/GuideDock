import pdb as debugger
import torch
import numpy as np
from utils import *


class ComplexLoader:
    def __init__(self, dataset, batch_tokens, field="epitope_seq"):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i][field]) for i in range(self.size)]
        self.batch_tokens = batch_tokens
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            batch.append(ix)
            if size * (len(batch) + 1) > self.batch_tokens:
                clusters.append(batch)
                batch = []

        self.clusters = clusters
        if len(batch) > 0:
            clusters.append(batch)

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def featurize(
    batch, name, vocab=ALPHABET, device=None, patch_size=50, resolution="all_atom"
):
    B = len(batch)
    L_max = max([len(b[name + "_seq"]) for b in batch])
    if resolution == "all_atom":
        sel = 14
    elif resolution == "backbone":
        sel = 4
    elif resolution == "CA":
        sel = 1
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    # L_max = patch_size
    X = torch.zeros([B, L_max, sel, 3], device=device)
    S = torch.zeros([B, L_max], device=device).long()
    A = torch.zeros([B, L_max, sel], device=device).long()
    V = torch.zeros([B, L_max, 6], device=device)
    T = torch.zeros([B, L_max, sel, 1], device=device) # spline torsion
    C = torch.zeros([B, L_max, sel, 1], device=device) # spline curvature
    Si = torch.zeros([B, L_max], device=device).long()

    O = torch.zeros([B, L_max], device=device).long()

    # Build the batch
    for i, b in enumerate(batch):
        l = min(len(b[name + "_seq"]), L_max)
        indices = torch.tensor(
            [vocab.index(a) for a in b[name + "_seq"]], device=device
        )
        S[i, :l] = indices[:l]
        X[i, :l] = b[name + "_coords"][:l]
        A[i, :l] = b[name + "_atypes"][:l]
        V[i, :l] = b[name + "_dihedrals"][:l]
        O[i, :l] = indices[:l]
        T[i, :l] = b[name + "_spline_torsion"][:l]
        C[i, :l] = b[name + "_spline_curvature"][:l]

    V = V.repeat_interleave(sel, dim=1)
    return (X, S, A, V, O, T, Si, C)


def featurize_docking(
    batch, name, vocab=ALPHABET, device=None, patch_size=50, resolution="all_atom"
):
    if resolution == "all_atom":
        sel = 14
    elif resolution == "backbone":
        sel = 4
    elif resolution == "CA":
        sel = 1
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    B = len(batch)
    L_max = max([len(b[name + "_seq"]) for b in batch])
    # L_max = patch_size
    X = torch.zeros([B, L_max, sel, 3], device=device)
    X_bnd = torch.zeros([B, L_max, sel, 3], device=device)
    S = torch.zeros([B, L_max], device=device).long()
    A = torch.zeros([B, L_max, sel], device=device).long()
    V = torch.zeros([B, L_max, 6], device=device)
    T = torch.zeros([B, L_max, sel, 1], device=device) # spline torsion
    C = torch.zeros([B, L_max, sel, 1], device=device) # spline curvature

    O = torch.zeros([B, L_max], device=device).long()

    # Build the batch
    for i, b in enumerate(batch):
        l = min(len(b[name + "_seq"]), L_max)
        b[name + "_seq"] = b[name + "_seq"].replace("X", "#")
        indices = torch.tensor(
            [vocab.index(a) for a in b[name + "_seq"]], device=device
        )
        S[i, :l] = indices[:l]
        X[i, :l] = b[name + "_coords"][:l]
        # X_bnd[i, :l] = b[name + "_coords_bnd"][:l]
        X_bnd[i, :l] = b[name + "_coords"][:l]
        A[i, :l] = b[name + "_atypes"][:l]
        V[i, :l] = b[name + "_dihedrals"][:l]
        O[i, :l] = indices[:l]
        # T[i, :l] = b[name + "_spline_torsion"][:l]
        # C[i, :l] = b[name + "_spline_curvature"][:l]

    # Repeat dihedrals by sel, [B, L_max, 6] -> [B, L_max * sel, 6]
    V = V.repeat_interleave(sel, dim=1)

    return (X, S, A, V, O, X_bnd, T, C)
