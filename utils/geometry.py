import torch
import numpy as np
import pdb
import warnings
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from typing import Tuple, Optional

from utils import Features, kabsch_best


def randomly_rotate_and_translate(coords: Tensor, tr_max=20, mask=None):
    """
    Randomly rotate and translate a batch of coordinates.
    coords: (B, N, 3)
    tr_max: maximum translation in Angstroms
    mask: (B, N) boolean mask indicating valid points to consider for centering
    """
    B, N = coords.size(0), coords.size(1)

    if mask is not None:
        masked_coords = coords * mask.unsqueeze(-1)  # (B, N, 3)
        center = masked_coords.sum(dim=1) / (
            mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        )  # (B, 3)
    else:
        center = coords.mean(dim=1)  # (B, 3)

    coords = coords - center.unsqueeze(1)  # (B, N, 3)
    rot_mats = torch.from_numpy(R.random(B).as_matrix()).cuda()  # (B, 3, 3)

    rotated_coords = torch.bmm(coords.double(), rot_mats.transpose(1, 2))  # (B, N, 3)

    # Random translation
    translations = (torch.rand(B, 3).cuda() - 0.5) * 2 * tr_max  # (B, 3)
    translated_coords = rotated_coords + translations.unsqueeze(1)  # (B, N, 3)

    return translated_coords.float()


def coords_from_distances(
    binder: Features, target: Features, pred_distances: Tensor, relax_iters=0
):
    binder_internal_dist = torch.cdist(
        binder.X[:, :, 1], binder.X[:, :, 1]
    )  # (B, M, M)
    target_internal_dist = torch.cdist(
        target.X[:, :, 1], target.X[:, :, 1]
    )  # (B, N, N)

    full_distances = torch.zeros(
        binder_internal_dist.size(0),
        binder_internal_dist.size(1) + target_internal_dist.size(1),
        binder_internal_dist.size(1) + target_internal_dist.size(1),
        device=pred_distances.device,
    )
    M, N = binder_internal_dist.size(1), target_internal_dist.size(1)
    full_distances[:, :M, :M] = binder_internal_dist
    full_distances[:, M:, M:] = target_internal_dist

    full_distances[:, :M, M:] = pred_distances
    full_distances[:, M:, :M] = pred_distances.transpose(1, 2)
    full_distances = full_distances.double().cpu()

    # coords = batched_solve_mds(full_distances)  # (B, M+N, 3)
    if relax_iters == 0:
        coords = batched_solve_mds_weighted(full_distances, refine_iters=0)
        return coords.float()

    W = make_block_weights(
        B=binder_internal_dist.size(0),
        M=M,
        N=N,
        w_internal=1.0,
        w_cross=0.2,
        device=full_distances.device,
    )
    coords = batched_solve_mds_weighted(
        full_distances, W=W, refine_iters=200, lr=5e-2, verbose=True
    )

    # binder_coords = torch.from_numpy(coords[:, :M, :]).float()

    # return torch.from_numpy(coords).float()
    return coords.float()


def align_by_receptor(binder: Features, target: Features, pred_coords: Tensor):
    target_ca = target.X[:, :, 1]  # (B, N, 3)
    B, N = target_ca.size(0), target_ca.size(1)

    pred_target_coords = pred_coords[:, -N:, :]  # (B, N, 3)

    _, R, t = kabsch_best(pred_target_coords.cpu(), target_ca.cpu())
    aligned_coords = (torch.bmm(R, pred_coords.transpose(1, 2)).transpose(1, 2)) + t

    # CA only, so now we align binder.X to aligned_coords
    # aligned_binder_coords = aligned_coords[:, : binder.X.size(1), :].cuda()

    # aligned_binder_coords, R, t = kabsch(
    #     binder.X[:, :, 1].to(torch.float32), aligned_binder_coords.to(torch.float32)
    # )
    # aligned_binder_coords = (
    #     torch.bmm(R, binder.X.float().view(B, -1, 3).transpose(1, 2)).transpose(1, 2)
    # ) + t

    # return aligned_binder_coords
    return aligned_coords


def cross_product(a, b):
    with torch.enable_grad():
        return torch.cross(a, b, dim=-1)


def rotate(X, w, sel: int):
    with torch.enable_grad():
        B, N = X.size(0), X.size(1)
        X = X.reshape(B, N * sel, 3)
        w = w.unsqueeze(1).expand_as(X)  # [B,N,3]
        c = w.norm(p=2, dim=-1, keepdim=True)  # [B,N,1]
        c1 = torch.sin(c) / c.clamp(min=1e-6)
        c2 = (1 - torch.cos(c)) / (c**2).clamp(min=1e-6)
        X = X + c1 * cross_product(w, X) + c2 * cross_product(w, cross_product(w, X))
        return X.view(B, N, sel, 3)


@torch.no_grad()
def _classical_mds_init(
    D: torch.Tensor,
    dim: int = 3,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Classical (Torgerson) MDS init from distances.

    Args:
        D: (B, N, N) distances, assumed symmetric with zeros on diagonal.
        dim: embedding dimension
    Returns:
        X0: (B, N, dim) coordinates (centered)
        evals: (B, N) eigenvalues of the Gram matrix (descending)
    """
    B, N, _ = D.shape
    device = D.device
    dtype = D.dtype

    # Double-centering matrix J = I - 1/n 11^T
    I = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, N, N)
    ones = torch.ones((B, N, N), device=device, dtype=dtype) / float(N)
    J = I - ones

    # Gram matrix: G = -0.5 * J * (D^2) * J
    D2 = D * D
    G = -0.5 * torch.bmm(torch.bmm(J, D2), J)

    # Symmetric eigendecomposition
    # torch.linalg.eigh returns ascending eigenvalues
    evals, evecs = torch.linalg.eigh(G)
    evals = torch.flip(evals, dims=[1])  # descending: (B, N)
    evecs = torch.flip(evecs, dims=[2])  # match: (B, N, N)

    # Take top-dim components, clamp negatives to 0 for init
    evals_top = evals[:, :dim].clamp_min(0.0)  # (B, dim)
    evecs_top = evecs[:, :, :dim]  # (B, N, dim)

    X0 = evecs_top * torch.sqrt(evals_top.unsqueeze(1) + eps)

    # Center to remove translation drift
    X0 = X0 - X0.mean(dim=1, keepdim=True)
    return X0, evals


@torch.enable_grad()
def batched_solve_mds_weighted(
    D: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    dim: int = 3,
    refine_iters: int = 200,
    lr: float = 5e-2,
    flat_ratio_thresh: float = 0.5,
    min_third_eig: float = 1e-8,
    eps: float = 1e-8,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Batched MDS with optional weighted stress minimization refinement for "flat" cases.

    Args:
        D: (B, N, N) target distances (symmetric, diag=0).
        W: (B, N, N) nonnegative weights. If None, uniform weights are used.
           Tip: set W[:, i, i] = 0, and optionally downweight noisy blocks.
        dim: output dimension (3).
        refine_iters: number of stress-min steps for flat batches.
        lr: Adam learning rate for refinement.
        flat_ratio_thresh: if lambda3/lambda2 is below this, treat as flat.
        min_third_eig: also treat as flat if lambda3 < this.
        eps: numerical stability epsilon.
        verbose: prints how many batches were refined.
    Returns:
        X: (B, N, dim) coordinates (centered).
    """
    if D.ndim != 3 or D.shape[-1] != D.shape[-2]:
        raise ValueError(f"D must be (B,N,N). Got {tuple(D.shape)}")

    B, N, _ = D.shape
    device = D.device
    dtype = D.dtype

    if W is None:
        W_use = None
    else:
        if W.shape != D.shape:
            raise ValueError(
                f"W must match D shape. Got W={tuple(W.shape)} vs D={tuple(D.shape)}"
            )
        W_use = W.to(device=device, dtype=dtype)
        # Ensure diagonal is zero weight
        diag = torch.arange(N, device=device)
        W_use[:, diag, diag] = 0.0
        # Make sure weights are nonnegative
        if (W_use < 0).any():
            raise ValueError("W must be nonnegative.")

    # 1) Classical MDS init
    X0, evals = _classical_mds_init(D.to(dtype), dim=dim, eps=eps)

    # 2) Flatness detection (needs at least dim>=3)
    if dim < 3:
        return X0

    lam2 = evals[:, 1].clamp_min(0.0)
    lam3 = evals[:, 2].clamp_min(0.0)
    ratio = lam3 / (lam2 + eps)
    flat = (lam3 < min_third_eig) | (ratio < flat_ratio_thresh)

    if not flat.any() or refine_iters <= 0:
        return X0

    if verbose:
        print(
            f"[batched_solve_mds_weighted] Refining {int(flat.sum().item())}/{B} flat batches..."
        )

    # 3) Weighted stress refinement for flat batches only
    idx = flat.nonzero(as_tuple=False).squeeze(1)
    D_ref = D.index_select(0, idx).to(dtype)
    X_ref = X0.index_select(0, idx).detach().clone().requires_grad_(True)

    if W_use is None:
        W_ref = None
        # mask to drop diagonal from uniform loss
        diag = torch.arange(N, device=device)
        mask = torch.ones((1, N, N), device=device, dtype=dtype)
        mask[:, diag, diag] = 0.0
    else:
        W_ref = W_use.index_select(0, idx)

    opt = torch.optim.Adam([X_ref], lr=lr)

    for _ in range(refine_iters):
        dist = torch.cdist(X_ref, X_ref).clamp_min(eps)  # (b, N, N)
        diff = dist - D_ref

        if W_ref is None:
            # Uniform weights, ignore diagonal
            loss = (mask * diff * diff).sum() / (mask.sum() + eps)
        else:
            loss = (W_ref * diff * diff).sum() / (W_ref.sum() + eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Re-center every step (translation is unidentifiable)
        with torch.no_grad():
            X_ref -= X_ref.mean(dim=1, keepdim=True)

    X = X0.clone()
    X[idx] = X_ref.detach()
    return X


# ---- Optional helper: build block weights for binder/target stitched distances ----
def make_block_weights(
    B: int,
    M: int,
    N: int,
    w_internal: float = 1.0,
    w_cross: float = 0.3,
    device=None,
) -> torch.Tensor:
    """
    Build (B, M+N, M+N) weights: internal blocks get w_internal, cross gets w_cross, diag=0.
    """
    K = M + N
    W = torch.full((B, K, K), w_cross, device=device)
    W[:, :M, :M] = w_internal
    W[:, M:, M:] = w_internal
    diag = torch.arange(K, device=device)
    W[:, diag, diag] = 0.0
    # Symmetrize (just in case)
    W = 0.5 * (W + W.transpose(1, 2))
    return W


def batched_solve_mds(D, dim=3, jitter=0.0):
    """
    Solve classical MDS using eigendecomposition.

    Parameters:
    -----------
    D : np.ndarray
        Pairwise distance matrix of shape [B, N, N]
    dim : int
        Target dimension for embedding (default: 3)

    Returns:
    --------
    X : np.ndarray
        Coordinates in target dimension of shape [N, dim]
    """
    B, N, _ = D.shape
    # D = D + jitter * np.random.randn(B, N, N)

    # Centering matrix
    H = np.eye(N) - np.ones((N, N)) / N  # [N, N]
    H = np.expand_dims(H, 0).repeat(B, axis=0)  # [B, N, N]

    # Double centering
    # D_squared = D**2
    # B_matrix = -0.5 * H @ D_squared @ H  # [B, N, N]
    D_squared = D**2  # [B, N, N]

    # Row / column / global means
    row_mean = D_squared.mean(axis=-1, keepdims=True)  # [B, N, 1]
    col_mean = D_squared.mean(axis=-2, keepdims=True)  # [B, 1, N]
    grand_mean = D_squared.mean(axis=(-1, -2), keepdims=True)  # [B, 1, 1]

    # Stable double-centering
    B_matrix = -0.5 * (D_squared - row_mean - col_mean + grand_mean)  # [B, N, N]

    # (Optional) enforce symmetry to kill tiny asymmetries
    B_matrix = 0.5 * (B_matrix + np.swapaxes(B_matrix, -1, -2))

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B_matrix)  # [B, N], [B, N, N]

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals, axis=-1)[:, ::-1].copy()
    eigvals = np.take_along_axis(eigvals, idx, axis=-1)
    eigvecs = np.take_along_axis(eigvecs, idx[:, None, :], axis=-1)

    eigvals[eigvals < 0] = 0  # or a tolerance-based cutoff
    # Select top 'dim' eigenvalues and eigenvectors
    # L = np.sqrt(np.maximum(eigvals[:, :dim], 0))  # [B, dim]
    L = np.sqrt(eigvals[:, :dim])
    V = eigvecs[:, :, :dim]  # [B, N, dim]

    # Compute coordinates
    X = V * L[:, np.newaxis, :]  # [B, N, dim]

    return X  # [B, N, dim]


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
    # First derivative (tangent)
    dx, dy, dz = splev(u, tck, der=1)

    # Second derivative
    ddx, ddy, ddz = splev(u, tck, der=2)

    # Third derivative
    dddx, dddy, dddz = splev(u, tck, der=3)

    # Compute torsion: τ = (r' × r'') · r''' / |r' × r''|²
    r_prime = np.array([dx, dy, dz]).T
    r_double_prime = np.array([ddx, ddy, ddz]).T
    r_triple_prime = np.array([dddx, dddy, dddz]).T

    # Cross product
    cross = np.cross(r_prime, r_double_prime)

    # Dot product with third derivative
    dot = np.einsum("ij,ij->i", cross, r_triple_prime)

    # Norm squared of cross product
    cross_norm_sq = np.linalg.norm(cross, axis=1) ** 2 + 1e-10  # avoid division by zero

    # Torsion
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
    # First derivative (tangent)
    dx, dy, dz = splev(u, tck, der=1)

    # Second derivative
    ddx, ddy, ddz = splev(u, tck, der=2)

    # Compute curvature: κ = |r' × r''| / |r'|³
    r_prime = np.array([dx, dy, dz]).T
    r_double_prime = np.array([ddx, ddy, ddz]).T

    # Cross product
    cross = np.cross(r_prime, r_double_prime)

    # Norms
    cross_norm = np.linalg.norm(cross, axis=1)
    r_prime_norm = np.linalg.norm(r_prime, axis=1)

    # Curvature
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
    # Convert to numpy if torch tensor
    N = coords.shape[0]
    if use_all_atoms:
        N = N * coords.shape[1]

    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()

    # Extract CA atoms (first backbone atom index)
    ca_coords = coords[:, 1, :]  # Shape: [M, 3]

    if use_all_atoms:
        # Flatten all backbone atoms in sequence
        # Shape: [M, K, 3] -> [M*K, 3]
        backbone_coords = coords.reshape(-1, 3)

        # Parametrize the spline through all backbone atoms
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
        # Use only CA atoms
        backbone_coords = ca_coords
        tck, u = splprep([ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2]], s=s, k=k)

    # Evaluate spline at finer resolution
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
    v1 = p1 - center  # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center  # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)  # (N, L, 3)

    mat = torch.cat(
        [e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)], dim=-1
    )  # (N, L, 3, 3_index)
    return mat


def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)
    t = t.reshape(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)
    p = torch.matmul(R.transpose(-1, -2), (q - t))  # (N, L, 3, *)
    p = p.transpose(-1, -2).reshape(
        q_size
    )  # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)  # (N, L, 3, *)
    q = q.transpose(-1, -2).reshape(
        p_size
    )  # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q


def pairwise_directions(pos_atoms):
    R = construct_3d_basis(pos_atoms[:, :, 1], pos_atoms[:, :, 2], pos_atoms[:, :, 0])

    pos_N = pos_atoms[:, :, 0]  # (N, L, 3)
    pos_CA = pos_atoms[:, :, 1]
    pos_C = pos_atoms[:, :, 2]
    pos_O = pos_atoms[:, :, 3]

    N, L = pos_atoms.shape[:2]
    dire_CA = global_to_local(
        R, pos_CA[:, :, None].expand(N, L, L, 3), pos_CA[:, None, :].expand(N, L, L, 3)
    )
    dire_C = global_to_local(
        R, pos_CA[:, :, None].expand(N, L, L, 3), pos_C[:, None, :].expand(N, L, L, 3)
    )
    dire_O = global_to_local(
        R, pos_CA[:, :, None].expand(N, L, L, 3), pos_O[:, None, :].expand(N, L, L, 3)
    )
    dire_N = global_to_local(
        R, pos_CA[:, :, None].expand(N, L, L, 3), pos_N[:, None, :].expand(N, L, L, 3)
    )

    pair_direct_feat = torch.cat([dire_CA, dire_C, dire_O, dire_N], dim=-1)
    return pair_direct_feat
