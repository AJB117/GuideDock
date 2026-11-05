import torch

from utils import ALPHABET, get_resolution_indices


def featurize(
    batch, name, vocab=ALPHABET, device=None, patch_size=50, resolution="all_atom"
):
    B = len(batch)
    L_max = max([len(b[name + "_seq"]) for b in batch])
    sel, _ = get_resolution_indices(resolution)

    X = torch.zeros([B, L_max, sel, 3], device=device)
    S = torch.zeros([B, L_max], device=device).long()
    A = torch.zeros([B, L_max, sel], device=device).long()
    V = torch.zeros([B, L_max, 6], device=device)
    T = torch.zeros([B, L_max, sel, 1], device=device)
    C = torch.zeros([B, L_max, sel, 1], device=device)
    Si = torch.zeros([B, L_max], device=device).long()

    O = torch.zeros([B, L_max], device=device).long()

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
