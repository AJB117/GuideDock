import warnings

import numpy as np
import torch
from scipy.interpolate import splev, splprep
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def randomly_rotate_and_translate(coords: Tensor, tr_max=20, mask=None):
    """
    Randomly rotate and translate a batch of coordinates.
    coords: (B, N, 3)
    tr_max: maximum translation in Angstroms
    mask: (B, N) boolean mask indicating valid points to consider for centering
    """
    B, _N = coords.size(0), coords.size(1)

    if mask is not None:
        masked_coords = coords * mask.unsqueeze(-1)
        center = masked_coords.sum(dim=1) / (
            mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        )
    else:
        center = coords.mean(dim=1)

    coords = coords - center.unsqueeze(1)
    rot_mats = torch.from_numpy(R.random(B).as_matrix()).cuda()

    rotated_coords = torch.bmm(coords.double(), rot_mats.transpose(1, 2))

    translations = (torch.rand(B, 3).cuda() - 0.5) * 2 * tr_max
    translated_coords = rotated_coords + translations.unsqueeze(1)

    return translated_coords.float()


def cross_product(a, b):
    with torch.enable_grad():
        return torch.cross(a, b, dim=-1)


def rotate(X, w, sel: int):
    with torch.enable_grad():
        B, N = X.size(0), X.size(1)
        X = X.reshape(B, N * sel, 3)
        w = w.unsqueeze(1).expand_as(X)
        c = w.norm(p=2, dim=-1, keepdim=True)
        c1 = torch.sin(c) / c.clamp(min=1e-6)
        c2 = (1 - torch.cos(c)) / (c**2).clamp(min=1e-6)
        X = X + c1 * cross_product(w, X) + c2 * cross_product(w, cross_product(w, X))
        return X.view(B, N, sel, 3)


def compute_spline_torsion(tck, u):
    """
    Compute torsion at points along the spline.

    Parameters:
    -----------
    tck : tuple
        Spline representation from splprep
    u : np.ndarray
        Parameter values where to compute torsion

    Returns:
    --------
    torsion : np.ndarray
        Torsion values at each point
    """

    dx, dy, dz = splev(u, tck, der=1)

    ddx, ddy, ddz = splev(u, tck, der=2)

    dddx, dddy, dddz = splev(u, tck, der=3)

    r_prime = np.array([dx, dy, dz]).T
    r_double_prime = np.array([ddx, ddy, ddz]).T
    r_triple_prime = np.array([dddx, dddy, dddz]).T

    cross = np.cross(r_prime, r_double_prime)

    dot = np.einsum("ij,ij->i", cross, r_triple_prime)

    cross_norm_sq = np.linalg.norm(cross, axis=1) ** 2 + 1e-10

    torsion = dot / cross_norm_sq

    return torsion


def compute_spline_curvature(tck, u):
    """
    Compute curvature at points along the spline.

    Parameters:
    -----------
    tck : tuple
        Spline representation from splprep
    u : np.ndarray
        Parameter values where to compute curvature

    Returns:
    --------
    curvature : np.ndarray
        Curvature values at each point
    """

    dx, dy, dz = splev(u, tck, der=1)

    ddx, ddy, ddz = splev(u, tck, der=2)

    r_prime = np.array([dx, dy, dz]).T
    r_double_prime = np.array([ddx, ddy, ddz]).T

    cross = np.cross(r_prime, r_double_prime)

    cross_norm = np.linalg.norm(cross, axis=1)
    r_prime_norm = np.linalg.norm(r_prime, axis=1)

    curvature = cross_norm / (r_prime_norm**3 + 1e-10)

    return curvature


def parametrize_backbone_spline(coords, s=0, k=3, use_all_atoms=False):
    """
    Parametrize a spline along the protein backbone.

    Parameters:
    -----------
    coords : torch.Tensor or np.ndarray
        Shape [M, K, 3] where M is number of residues, K is number of backbone atoms
        CA atoms should be at index 0
    s : float
        Smoothing parameter (0 = no smoothing, interpolates all points)
    k : int
        Degree of the spline (1=linear, 2=quadratic, 3=cubic)
    use_all_atoms : bool
        If True, use all backbone atoms. If False, use only CA atoms (default)

    Returns:
    --------
    dict with keys:
        'ca_coords': CA atom coordinates [M, 3]
        'backbone_coords': All backbone coordinates [M*K, 3] (if use_all_atoms=True)
        'spline_coords': Evaluated spline coordinates [num_points, 3]
        'tck': Spline representation (for further computation)
        'u': Parameter values for spline evaluation
    """

    N = coords.shape[0]
    if use_all_atoms:
        N = N * coords.shape[1]

    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()

    ca_coords = coords[:, 1, :]

    if use_all_atoms:
        backbone_coords = coords.reshape(-1, 3)

        try:
            tck, u = splprep(
                [backbone_coords[:, 0], backbone_coords[:, 1], backbone_coords[:, 2]],
                s=s,
                k=k,
            )
        except Exception as E:
            jitter_mag = 1e-5
            warnings.warn(
                f"spline construction error: {E}, adding {jitter_mag} jitter to coords"
            )

            jitter = jitter_mag * np.random.randn(*backbone_coords.shape)
            backbone_coords += jitter
            tck, u = splprep(
                [backbone_coords[:, 0], backbone_coords[:, 1], backbone_coords[:, 2]],
                s=s,
                k=k,
            )
    else:
        backbone_coords = ca_coords
        tck, u = splprep([ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2]], s=s, k=k)

    u_fine = np.linspace(0, 1, N)

    return tck, u_fine


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


@torch.compile
def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)

    mat = torch.cat([e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1)
    return mat
