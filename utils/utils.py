import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import esm
from tqdm import tqdm
from einops import rearrange
from torch_geometric.nn import radius_graph
from .constants import ALPHABET
from typing import NamedTuple, List


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
    X: torch.Tensor  # coordinates [B, L, K, 3]
    S: torch.Tensor  # sequence embeddings/indices [B, L] 0-21
    A: torch.Tensor  # atom types [B, L, K] 0-4
    V: torch.Tensor  # dihedral features [B, L, 6]
    O: torch.Tensor  # original indices [B, L]
    T: torch.Tensor  # spline torsion [B, L]
    C: torch.Tensor  # spline curvature [B, L]
    Si: torch.Tensor  # sequence indices
    pdbs: List[str]  # list of pdb ids
    # S_ESM2: torch.Tensor  # ESM2 embeddings [B, L, D]

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
            # S_ESM2=self.S_ESM2.repeat_interleave(repeat_times, dim=0) if self.S_ESM2 is not None else None,
        )

    def to(self, device):
        return Features(
            X=self.X.cuda(device),
            S=self.S.cuda(device),
            A=self.A.cuda(device),
            V=self.V.cuda(device),
            O=self.O.cuda(device),
            T=self.T.cuda(device),
            C=self.C.cuda(device),
            Si=self.Si.cuda(device),
            pdbs=self.pdbs,
            # S_ESM2=self.S_ESM2.cuda(device) if self.S_ESM2 is not None else None,
        )

    # def swap_S(self):
    #     return Features(
    #         X=self.X,
    #         S=self.S_ESM2,
    #         A=self.A,
    #         V=self.V,
    #         O=self.O,
    #         T=self.T,
    #         C=self.C,
    #         Si=self.Si,
    #         pdbs=self.pdbs,
    #         S_ESM2=self.S,
    #     )

def to_batched_spherical_coordinates(X):
    """
    X: [B, N, 3]
    return [B, N, 3] (r, theta, phi)
    """
    r = torch.norm(X, dim=-1) + 1e-8
    theta = torch.acos(torch.clamp(X[..., 2] / r, -1 + 1e-7, 1 - 1e-7))  # polar angle
    phi = torch.atan2(X[..., 1], X[..., 0])  # azimuthal angle
    return torch.stack((r, theta, phi), dim=-1)


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
        # tr_s = self.tr_s_max + 0.5 * (self.tr_s_min - self.tr_s_max) * (1 + math.cos(math.pi * t_tr))
        # rot_s = self.rot_s_max + 0.5 * (self.rot_s_min - self.rot_s_max) * (1 + math.cos(math.pi * t_rot))

        return tr_s, rot_s


def matrix_to_angles(R):
    """
    Convert rotation matrix to Euler angles (ZYX convention)
    R: [B, 3, 3]
    return: [B, 3] (alpha, beta, gamma)
    """
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = sy < 1e-6
    alpha = torch.where(
        singular,
        torch.atan2(-R[:, 1, 2], R[:, 1, 1]),
        torch.atan2(R[:, 1, 0], R[:, 0, 0]),
    )
    beta = torch.where(
        singular,
        torch.atan2(-R[:, 2, 0], sy),
        torch.atan2(-R[:, 2, 0], sy),
    )
    gamma = torch.where(
        singular,
        torch.zeros_like(alpha),
        torch.atan2(R[:, 2, 1], R[:, 2, 2]),
    )
    return torch.stack((alpha, beta, gamma), dim=-1)


def kabsch_torch(A, B):
    """
    Aligns point cloud A to B using the Kabsch algorithm.
    A, B: [B, N, 3]

    Returns:
        A_aligned: [B, N, 3]
        R: [B, 3, 3]
        t: [B, 1, 3]
    """
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(0)
    # Compute centroids
    a_mean = A.mean(dim=1, keepdim=True)  # [B, 1, 3]
    b_mean = B.mean(dim=1, keepdim=True)  # [B, 1, 3]

    # Centered coordinates
    A_c = A - a_mean  # [B, N, 3]
    B_c = B - b_mean  # [B, N, 3]

    # Covariance matrix: H = A_c^T B_c
    H = torch.bmm(A_c.transpose(1, 2), B_c)  # [B, 3, 3]

    # SVD
    # torch.linalg.svd returns U, S, Vh where Vh = V^T
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.transpose(-2, -1)

    # Reflection correction: ensure det(R) = +1
    det = torch.det(V) * torch.det(U)  # [B]
    reflect_mask = det < 0

    if reflect_mask.any():
        # Flip last column of V for batches where reflection occurs
        V[reflect_mask, :, -1] *= -1

    # Rotation
    R = torch.bmm(V, U.transpose(1, 2))  # [B, 3, 3]

    # Translation
    # t = b_mean - R * a_mean
    t = b_mean - torch.bmm(a_mean, R.transpose(1, 2))  # [B, 1, 3]

    # Apply transform to A
    A_aligned = torch.bmm(A, R.transpose(1, 2)) + t  # [B, N, 3]

    return A_aligned, R, t


def kabsch_best(A, B, eps=1e-8):
    """
    A, B: (B, N, 3)
    Returns: A_aligned, R, t  where R may have det +1 or -1 (best fit).
    """
    a_mean = A.mean(dim=1, keepdim=True)
    b_mean = B.mean(dim=1, keepdim=True)
    A_c = A - a_mean
    B_c = B - b_mean

    H = torch.bmm(A_c.transpose(1, 2), B_c)  # (B,3,3)
    # torch.linalg.svd is more stable than torch.svd
    U, S, Vh = torch.linalg.svd(H)  # U:(B,3,3), Vh:(B,3,3)
    V = Vh.transpose(1, 2)

    # Candidate 1: allow whatever det happens (could be reflection)
    R1 = torch.bmm(V, U.transpose(1, 2))

    # Candidate 2: force proper rotation (det +1)
    det1 = torch.det(R1)
    D = torch.eye(3, device=A.device).unsqueeze(0).repeat(A.size(0), 1, 1)
    D[:, 2, 2] = torch.sign(det1 + eps)  # if det1<0 this flips last axis
    R2 = torch.bmm(torch.bmm(V, D), U.transpose(1, 2))

    def apply(R):
        t = b_mean - torch.bmm(R, a_mean.transpose(1, 2)).transpose(1, 2)
        A_al = torch.bmm(R, A.transpose(1, 2)).transpose(1, 2) + t
        return A_al, t

    A1, t1 = apply(R1)
    A2, t2 = apply(R2)

    # Pick lower RMSD on receptor
    err1 = ((A1 - B) ** 2).sum(dim=(1, 2))
    err2 = ((A2 - B) ** 2).sum(dim=(1, 2))
    use1 = (err1 <= err2).view(-1, 1, 1)

    R = torch.where(use1, R1, R2)
    t = torch.where(use1, t1, t2)
    A_aligned = torch.where(use1, A1, A2)

    return A_aligned, R, t


def mds_kabsch(A, B):
    a_mean = A.mean(dim=1, keepdims=True)
    b_mean = B.mean(dim=1, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = torch.bmm(A_c.transpose(1, 2), B_c)  # [B, 3, 3]
    U, S, V = torch.svd(H)

    det = torch.det(torch.bmm(V, U.transpose(1, 2)))
    D = torch.eye(3, device=H.device).unsqueeze(0).repeat(H.size(0), 1, 1)
    D[:, 2, 2] = torch.sign(det)

    # Rotation matrix
    R = torch.bmm(torch.bmm(V, D), U.transpose(1, 2))
    # Translation vector
    t = b_mean - torch.bmm(R, a_mean.transpose(1, 2)).transpose(1, 2)
    A_aligned = torch.bmm(R, A.transpose(1, 2)).transpose(1, 2) + t
    return A_aligned, R, t


def kabsch(A, B):
    """
    Aligns point cloud A to B using the Kabsch algorithm.
    """
    a_mean = A.mean(dim=1, keepdims=True)
    b_mean = B.mean(dim=1, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = torch.bmm(A_c.transpose(1, 2), B_c).to(torch.float32)  # [B, 3, 3]
    # U, S, V = torch.svd(H)
    U, S, V = torch.linalg.svd(H)
    # Flip
    sign = torch.det(U) * torch.det(V) < 0.0
    if sign.any():
        S[sign] = S[sign] * (-1)
        U[sign, :] = U[sign, :] * (-1)
    # Rotation matrix
    R = torch.bmm(V, U.transpose(1, 2))  # [B, 3, 3]
    # Translation vector
    t = b_mean - torch.bmm(R, a_mean.transpose(1, 2)).transpose(1, 2)
    A_aligned = torch.bmm(R, A.transpose(1, 2)).transpose(1, 2) + t
    return A_aligned, R, t


# X: [B, N, 4, 3], R: [B, 3, 3], t: [B, 3]
def rigid_transform(X, R, t):
    B, N, L = X.size(0), X.size(1), X.size(2)
    X = X.reshape(B, N * L, 3)
    X = torch.bmm(R, X.transpose(1, 2)).transpose(1, 2) + t
    return X.view(B, N, L, 3)


# A: [B, N, 3], B: [B, N, 3], mask: [B, N]
def compute_rmsd(A, B, mask):
    A_aligned, _, _ = kabsch(A, B)
    rmsd = ((A_aligned - B) ** 2).sum(dim=-1)
    rmsd = torch.sum(rmsd * mask, dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return rmsd.sqrt()


# A: [B, N, 3], B: [B, N, 3], mask: [B, N]
def compute_rmsd_no_align(A, B, mask):
    rmsd = ((A - B) ** 2).sum(dim=-1)
    rmsd = torch.sum(rmsd * mask, dim=-1) / (mask.sum(dim=-1) + 1e-6)
    return rmsd.sqrt()


def eig_coord(X, mask):
    D, mask_2D = self_square_dist(X, torch.ones_like(mask))
    return eig_coord_from_dist(D)


def eig_coord_from_dist(D):
    M = (D[:, :1, :] + D[:, :, :1] - D) / 2
    L, V = torch.linalg.eigh(M)
    L = torch.diag_embed(L)
    X = torch.matmul(V, L.clamp(min=0).sqrt())
    return X[:, :, -3:].detach()


def inner_square_dist(X, mask):
    L = mask.size(2)
    dX = X.unsqueeze(2) - X.unsqueeze(3)  # [B,N,1,L,3] - [B,N,L,1,3]
    mask_2D = mask.unsqueeze(2) * mask.unsqueeze(3)
    mask_2D = mask_2D * (1 - torch.eye(L)[None, None, :, :]).to(mask_2D)
    D = torch.sum(dX**2, dim=-1)
    return D * mask_2D, mask_2D


def self_square_dist(X, mask):
    X = X[:, :, 1]
    dX = X.unsqueeze(1) - X.unsqueeze(2)  # [B, 1, N, 3] - [B, N, 1, 3]
    D = torch.sum(dX**2, dim=-1)
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, 1, N] x [B, N, 1]
    mask_2D = mask_2D * (1 - torch.eye(mask.size(1))[None, :, :]).to(mask_2D)
    return D, mask_2D


def cross_square_dist(X, Y, xmask, ymask):
    X, Y = X[:, :, 1], Y[:, :, 1]
    dxy = X.unsqueeze(2) - Y.unsqueeze(1)  # [B, N, 1, 3] - [B, 1, M, 3]
    D = torch.sum(dxy**2, dim=-1)
    mask_2D = xmask.unsqueeze(2) * ymask.unsqueeze(1)  # [B, N, 1] x [B, 1, M]
    return D, mask_2D


def full_square_dist(X, Y, XA, YA, contact=False, remove_diag=False):
    B, N, M, L = X.size(0), X.size(1), Y.size(1), Y.size(2)
    X = X.view(B, N * L, 3)
    Y = Y.view(B, M * L, 3)
    dxy = X.unsqueeze(2) - Y.unsqueeze(1)  # [B, NL, 1, 3] - [B, 1, ML, 3]
    D = torch.sum(dxy**2, dim=-1)
    D = D.view(B, N, L, M, L)
    D = D.transpose(2, 3).reshape(B, N, M, L * L)

    xmask = XA.clamp(max=1).float().view(B, N * L)
    ymask = YA.clamp(max=1).float().view(B, M * L)
    mask = xmask.unsqueeze(2) * ymask.unsqueeze(1)  # [B, NL, 1] x [B, 1, ML]
    mask = mask.view(B, N, L, M, L)
    mask = mask.transpose(2, 3).reshape(B, N, M, L * L)
    if remove_diag:
        mask = mask * (1 - torch.eye(N)[None, :, :, None]).to(mask)

    if contact:
        D = D + 1e6 * (1 - mask)
        return D.amin(dim=-1), mask.amax(dim=-1)
    else:
        return D, mask


def compute_directions(X, mask):
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
    dX = X[:, 1:, :] - X[:, :-1, :]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    u_0 = U[:, 2:, :]

    return torch.cat((u_2, u_1, u_0), 2)


def compute_dihedrals(X, eps=1e-7):
    # First 3 coordinates are N, CA, C
    X = X.unsqueeze(0)
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

    # Shifted slices of unit vectors
    dX = X[:, 1:, :] - X[:, :-1, :]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:, :-2, :]
    u_1 = U[:, 1:-1, :]
    u_0 = U[:, 2:, :]

    # Backbone normals
    n_2 = F.normalize(torch.linalg.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.linalg.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

    D = F.pad(D, (3, 0), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    phi, psi, omega = torch.unbind(D, -1)

    # Lift angle representations to the circle
    D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
    nan_indices = torch.isnan(D_features)

    # if nan_indices.any():
    #     pdb.set_trace()

    D_features[nan_indices] = 0.0  # Replace NaNs with zeros
    return D_features


def expansion(theta, sigma, L=2000):  # the summation term only
    p = 0
    for l in range(L):
        p += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * sigma**2)
            * np.sin(theta * (l + 1 / 2))
            / np.sin(theta / 2)
        )
    return p


def density(expansion, theta):
    density = expansion * (1 - np.cos(theta)) / np.pi
    density = np.clip(density, 0, 1000)
    return density / density.sum()


def score(exp, theta, sigma, L=2000):
    dSigma = 0
    for l in range(L):
        hi = np.sin(theta * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(theta * (l + 1 / 2))
        lo = np.sin(theta / 2)
        dlo = 1 / 2 * np.cos(theta / 2)
        dSigma += (
            (2 * l + 1)
            * np.exp(-l * (l + 1) * sigma**2)
            * (lo * dhi - hi * dlo)
            / (lo**2)
        )
    return dSigma / exp + np.sin(theta) / (1 - np.cos(theta))


def load_esm_embedding(data, fields, device, pretrained_weights="./pretrained_weights"):
    torch.hub.set_dir(pretrained_weights)
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.cuda(device=device)
    model.eval()
    embedding = {}
    with torch.no_grad():
        for f in fields:
            seqs = [d[f] for d in data if d[f] not in embedding]
            for s in tqdm(sorted(set(seqs))):
                batch_labels, batch_strs, batch_tokens = batch_converter([(s, s)])
                batch_tokens = batch_tokens.cuda(device=device)
                results = model(batch_tokens, repr_layers=[36], return_contacts=False)
                embedding[s] = results["representations"][36][0, 1 : len(s) + 1].cpu()
    return embedding


def recursive_to(obj, device):
    """Recursively move tensors in a nested structure to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.cuda(device)
    elif isinstance(obj, list):
        return [recursive_to(item, device) for item in obj]
    elif isinstance(obj, Features):
        return Features(*(recursive_to(field, device) for field in obj))
    elif isinstance(obj, tuple):
        return tuple(recursive_to(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    else:
        return obj  # Return other types unchanged


def get_graph(coords, valid_radius=15.0, max_neighbors=10000):
    B, N, device = coords.shape[0], coords.shape[1], coords.device

    edge_idx = radius_graph(
        x=coords.reshape(B * N, 3),
        r=valid_radius,
        batch=torch.arange(B, device=device).repeat_interleave(N),
        loop=False,
        max_num_neighbors=max_neighbors,
    )
    batch = torch.arange(B, device=device).repeat_interleave(N)

    return edge_idx, batch
