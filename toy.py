import math, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn


# ---------- geometry bits ----------
def center_of_mass(x, weights=None):
    # x: (B, P, 3); weights: (B, P) or (P,)
    if weights is None:
        return x.mean(dim=1, keepdim=True)
    w = weights
    if w.dim() == 1:
        w = w[None, :]
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    return (x * w[..., None]).sum(dim=1, keepdim=True)


def kabsch(P, Q, weights=None):
    """
    Find R, t s.t. R @ P_i + t ≈ Q_i   (least squares)
    P,Q: (B, P, 3)   weights: (B,P) or None
    Returns: R (B,3,3), t (B,3)
    """
    if weights is None:
        muP = P.mean(dim=1, keepdim=True)
        muQ = Q.mean(dim=1, keepdim=True)
        Pw = P - muP
        Qw = Q - muQ
        H = torch.einsum("bpi,bpj->bij", Pw, Qw)  # (B,3,3)
    else:
        w = weights
        if w.dim() == 1:
            w = w[None, :]
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        muP = (P * w[..., None]).sum(dim=1, keepdim=True)
        muQ = (Q * w[..., None]).sum(dim=1, keepdim=True)
        Pw = P - muP
        Qw = Q - muQ
        H = torch.einsum("bpi,bpj,bp->bij", Pw, Qw, w)  # weighted covariance

    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(-1, -2)
    R = V @ U.transpose(-1, -2)
    # Enforce proper rotation (det=+1)
    det = torch.det(R)
    mask = (det < 0).float()
    # Flip last column of V when needed
    V_fix = V.clone()
    V_fix[:, :, 2] = V_fix[:, :, 2] * (1 - 2 * mask)[:, None]
    R = V_fix @ U.transpose(-1, -2)
    t = (muQ - (R @ muP.transpose(-1, -2)).transpose(-1, -2)).squeeze(1)  # (B,3)
    return R, t


def skew(v):
    """
    v: (..., 3)  ->  K: (..., 3, 3)
    Batched skew-symmetric matrix builder supporting arbitrary leading dims.
    """
    O = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    O[..., 0, 1] = -v[..., 2]
    O[..., 0, 2] = v[..., 1]
    O[..., 1, 0] = v[..., 2]
    O[..., 1, 2] = -v[..., 0]
    O[..., 2, 0] = -v[..., 1]
    O[..., 2, 1] = v[..., 0]
    return O


def so3_exp(omega, eps=1e-8):
    """
    omega: (..., 3) axis-angle  ->  R: (..., 3, 3)
    Numerically stable Rodrigues with small-angle series; fully broadcastable.
    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)  # (...,1)
    k = omega / (theta + eps)  # (...,3)
    K = skew(k)  # (...,3,3)

    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(
        *omega.shape[:-1], 3, 3
    )  # (...,3,3)

    s = torch.sin(theta)  # (...,1)
    c = torch.cos(theta)  # (...,1)
    small = theta < 1e-4

    A = torch.where(small, 1 - (theta**2) / 6, s / (theta + eps))  # (...,1)
    B = torch.where(small, 0.5 - (theta**2) / 24, (1 - c) / (theta**2 + eps))  # (...,1)

    return I + A[..., None] * K + B[..., None] * (K @ K)  # (...,3,3)


def so3_log(R):
    # R: (B,3,3) -> omega: (B,3) in [-pi, pi]
    B = R.shape[0]
    I = torch.eye(3, device=R.device, dtype=R.dtype).expand(B, 3, 3)
    trace = torch.clamp(
        (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1 + 1e-6, 1 - 1e-6
    )  # (B,)
    theta = torch.acos(trace)  # (B,)
    # For general case:
    W = (R - R.transpose(-1, -2)) / (2 * torch.sin(theta).clamp_min(1e-6)).unsqueeze(
        -1
    ).unsqueeze(-1)
    omega = torch.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], dim=-1) * theta.unsqueeze(
        -1
    )
    # Small-angle fallback via series
    small = (theta < 1e-3).float().unsqueeze(-1)
    omega_series = (
        torch.stack(
            [R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]],
            dim=-1,
        )
        / 2
    )
    return small * omega_series + (1 - small) * omega


def left_apply(R, t, X):
    # Apply SE(3) pose (R,t) to points X: (B,P,3)
    return (R @ X.transpose(-1, -2)).transpose(-1, -2) + t[:, None, :]


# ---------- schedules ----------
def sigma_rot(t):  # t in [0,1], radians
    # linear to ~180° at t=1
    return torch.pi * t


def sigma_trans(t):  # t in [0,1], Angstrom
    return 30.0 * t


# ---------- main utilities ----------
def center_on_receptor(rec_coords, bind_coords, use_ca=True):
    """
    rec_coords: (B, N, K, 3), bind_coords: (B, M, K, 3)
    """
    if use_ca:
        rec_pts = rec_coords[:, :, 1, :]  # CA at index 1
    else:
        rec_pts = rec_coords.reshape(rec_coords.shape[0], -1, 3)
    com = center_of_mass(rec_pts)  # (B,1,3)
    return rec_coords - com[:, None, None, :], bind_coords - com[:, None, None, :]


def binder_points(binder, use_ca=True):
    # binder: (B, M, K, 3) -> (B, P, 3)
    if use_ca:
        return binder[:, :, 1, :]
    return binder.reshape(binder.shape[0], -1, 3)


def diffuse_binder_coords(bind_clean_pts, t):
    """
    bind_clean_pts: (B, P, 3) in receptor-centered frame
    t: (B,) or scalar tensor in [0,1]
    """
    B = bind_clean_pts.shape[0]
    if t.ndim == 0:
        t = t * torch.ones(B, device=bind_clean_pts.device)
    sig_r = sigma_rot(t).unsqueeze(-1)  # (B,1)
    sig_t = sigma_trans(t).unsqueeze(-1)  # (B,1)
    # sample noise
    omega = torch.randn(B, 3, device=bind_clean_pts.device) * sig_r
    delta = torch.randn(B, 3, device=bind_clean_pts.device) * sig_t
    Rn = so3_exp(omega)  # (B,3,3)
    X_noisy = left_apply(Rn, delta, bind_clean_pts)
    return X_noisy, omega, delta


def supervision_from_noisy(
    bind_noisy_pts, bind_clean_pts, weights=None, t=None, eps_targets=True
):
    """
    Returns labels for either epsilon-prediction or 'delta pose' regression.
    """
    R_delta, t_delta = kabsch(bind_noisy_pts, bind_clean_pts, weights=weights)
    if eps_targets:
        # Map to 'noise' space with schedules (like epsilon for DDPM)
        assert t is not None
        if t.ndim == 0:
            t = t * torch.ones(bind_noisy_pts.shape[0], device=bind_noisy_pts.device)
        sig_r = sigma_rot(t).unsqueeze(-1)
        sig_t = sigma_trans(t).unsqueeze(-1)
        xi = so3_log(R_delta) / (sig_r + 1e-8)  # (B,3)
        eta = t_delta / (sig_t + 1e-8)  # (B,3)
        return xi, eta  # rotational and translational 'eps'
    else:
        # Direct delta pose (axis-angle + translation) targets
        xi = so3_log(R_delta)  # (B,3)
        eta = t_delta  # (B,3)
        return xi, eta


@torch.no_grad()
def make_toy_docking_dataset(
    B=32,  # batch size
    N=128,  # receptor residues
    M=64,  # binder residues
    K=3,  # atoms per residue (N, CA, C) with CA at index 1
    step=3.8,  # target CA–CA spacing (Å)
    device="cpu",
    seed=None,
    contact_dist=4.0,  # target closest approach (Å) around interface
    frame_on_receptor=True,  # subtract receptor COM from both structures
):
    """
    Returns:
      rec_coords:  (B, N, K, 3)
      bind_coords: (B, M, K, 3)
      R_gt:        (B, 3, 3)   # binder->receptor rotation
      t_gt:        (B, 3)      # binder->receptor translation
    Notes:
      - Atom order is (N, CA, C), CA at index 1 (matches your utils).
      - Geometry is toy but stable; good for debugging rigid docking code.
    """
    if seed is not None:
        torch.manual_seed(seed)

    B = int(B)
    N = int(N)
    M = int(M)
    K = int(K)
    assert K >= 3, "This toy generator expects K>=3 (N, CA, C)."

    def _smooth_dirs(dirs, iters=3):
        # dirs: (B, L, 3) -> smoothed along sequence (cheap 1D blur)
        for _ in range(iters):
            prev = dirs
            dirs = prev.clone()
            dirs[:, 1:-1, :] = (
                prev[:, :-2, :] + 2 * prev[:, 1:-1, :] + prev[:, 2:, :]
            ) / 4.0
        return dirs

    def _make_ca_chain(B, L, step, device):
        # random smooth directions, normalized, then cumulative sum to get positions
        dirs = torch.randn(B, L, 3, device=device)
        dirs = _smooth_dirs(dirs)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        incs = dirs * step
        ca = incs.cumsum(dim=1)  # (B, L, 3), first near origin
        # recenter to zero-mean for numerical neatness
        ca = ca - ca.mean(dim=1, keepdim=True)
        return ca

    def _add_N_CA_C_from_CA(ca):
        # ca: (B, L, 3) -> coords: (B, L, 3(atoms), 3) with order (N, CA, C)
        B, L, _ = ca.shape
        # estimate tangent using finite differences
        t = torch.zeros_like(ca)
        t[:, 1:-1, :] = ca[:, 2:, :] - ca[:, :-2, :]
        t[:, 0, :] = ca[:, 1, :] - ca[:, 0, :]
        t[:, -1, :] = ca[:, -1, :] - ca[:, -2, :]
        t = t / (t.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        # simple colinear N/C placement along tangent (toy geometry)
        d_N, d_C = 1.46, 1.53  # typical bond lengths; just for scale
        Np = ca - d_N * t
        Cp = ca + d_C * t
        coords = torch.stack([Np, ca, Cp], dim=2)  # (B, L, 3, 3)
        return coords

    # --- 1) Build independent receptor & binder backbones in their own frames ---
    rec_ca_local = _make_ca_chain(B, N, step, device)  # (B, N, 3)
    bind_ca_local = _make_ca_chain(B, M, step, device)  # (B, M, 3)
    rec_coords = _add_N_CA_C_from_CA(rec_ca_local)  # (B, N, 3, 3)
    bind_coords_0 = _add_N_CA_C_from_CA(bind_ca_local)  # (B, M, 3, 3)

    # --- 2) Sample a random rigid transform for binder -> receptor ---
    # random axis-angle with uniform axis and angle in [0, π]
    axis = torch.randn(B, 3, device=device)
    axis = axis / (axis.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    angle = torch.rand(B, 1, device=device) * torch.pi
    omega = axis * angle  # (B, 3)

    # Use your so3_exp if already defined in scope; otherwise inline Rodrigues
    def _skew(v):
        O = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
        O[:, 0, 1], O[:, 0, 2] = -v[:, 2], v[:, 1]
        O[:, 1, 0], O[:, 1, 2] = v[:, 2], -v[:, 0]
        O[:, 2, 0], O[:, 2, 1] = -v[:, 1], v[:, 0]
        return O

    # def _so3_exp(omega):
    #     theta = omega.norm(dim=-1, keepdim=True)  # (B,1)
    #     k = omega / (theta.clamp_min(1e-8))
    #     K = _skew(k)
    #     I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(
    #         omega.size(0), 3, 3
    #     )
    #     s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    #     A = torch.where(
    #         theta[..., None] < 1e-4, 1 - (theta**2) / 6, s / (theta.clamp_min(1e-8))
    #     )
    #     Bc = torch.where(
    #         theta[..., None] < 1e-4,
    #         0.5 - (theta**2) / 24,
    #         (1 - c) / ((theta**2).clamp_min(1e-8)),
    #     )
    #     return I + A * K + Bc * (K @ K)

    R_gt = so3_exp(omega)  # (B, 3, 3)

    # rotate binder atoms
    B_, M_, K_, _ = bind_coords_0.shape
    bind_flat = bind_coords_0.reshape(B_, M_ * K_, 3)  # (B, MK, 3)
    bind_rot = (R_gt @ bind_flat.transpose(-1, -2)).transpose(-1, -2)  # (B, MK, 3)
    bind_rot = bind_rot.reshape(B_, M_, K_, 3)

    # --- 3) Translate binder near a random receptor site with ~contact_dist ---
    batch_idx = torch.arange(B, device=device)
    anchor_idx = torch.randint(0, N, (B,), device=device)
    anchor = rec_coords[batch_idx, anchor_idx, 1, :]  # CA at site, (B,3)

    bind_ca_rot = bind_rot[:, :, 1, :]  # (B, M, 3)
    bind_com = bind_ca_rot.mean(dim=1)  # simple COM (uniform)

    rand_dir = torch.randn(B, 3, device=device)
    rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # initial translation to bring binder COM near anchor with small offset
    offset = contact_dist + 2.0 * torch.rand(B, 1, device=device)  # 4..6 Å
    t_gt = anchor - bind_com + rand_dir * offset  # (B,3)

    bind_pos = bind_rot + t_gt[:, None, None, :]  # (B, M, K, 3)

    # small push-out if any CA–CA pairs overlap too much
    rec_ca = rec_coords[:, :, 1, :]  # (B, N, 3)
    d = torch.cdist(rec_ca, bind_pos[:, :, 1, :])  # (B, N, M)
    dmin = d.view(B, -1).min(dim=1).values  # (B,)
    extra = (3.0 - dmin).clamp_min(0.0) + 0.0  # aim ≥3 Å
    if extra.max() > 0:
        bind_pos = bind_pos + rand_dir[:, None, None, :] * extra[:, None, None, None]
        t_gt = t_gt + rand_dir * extra[:, None]

    # --- 4) Optional receptor-centering frame (matches your utilities) ---
    if frame_on_receptor:
        # subtract receptor COM (using CA by convention)
        rec_ca_com = rec_ca.mean(dim=1, keepdim=True)  # (B,1,3)
        rec_coords = rec_coords - rec_ca_com[:, None, None, :]
        bind_pos = bind_pos - rec_ca_com[:, None, None, :]

    # Final pack
    bind_coords = bind_pos  # (B, M, K, 3)
    return (
        rec_coords.to(device),
        bind_coords.to(device),
        R_gt.to(device),
        t_gt.to(device),
    )


# --------------------------
# Time embedding (sin/cos)
# --------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), steps=half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t):  # t in [0,1], shape (B,)
        # Map to angles (B, half)
        x = t[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, dim)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


# --------------------------
# Tiny PointNet-style denoiser
# --------------------------
class PoseDenoiser(nn.Module):
    """
    Inputs:
      X_noisy: (B, P, 3)   binder points at time t
      t:       (B,)        diffusion time in [0,1]
    Output:
      xi_pred:  (B,3)  rotational epsilon
      eta_pred: (B,3)  translational epsilon
    """

    def __init__(self, t_dim=64, hidden=128):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(t_dim)
        in_dim = 3 + t_dim  # [xyz | t-embed]
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),  # 3 for xi, 3 for eta
        )

    def forward(self, X_noisy, t):
        B, P, _ = X_noisy.shape
        t_emb = self.t_embed(t).unsqueeze(1).expand(B, P, -1)
        feats = torch.cat([X_noisy, t_emb], dim=-1)  # (B,P,3+t_dim)
        per_point = self.point_mlp(feats)  # (B,P,H)
        global_feat, _ = torch.max(per_point, dim=1)  # (B,H)
        out = self.head(global_feat)  # (B,6)
        xi_pred, eta_pred = out[:, :3], out[:, 3:]
        return xi_pred, eta_pred


# --------------------------
# Training utilities
# --------------------------
@torch.no_grad()
def make_batch(B=32, N=128, M=64, device="cpu"):
    rec, bind, R_gt, t_gt = make_toy_docking_dataset(
        B=B, N=N, M=M, device=device, seed=None
    )
    # receptor-centered frame already applied by the generator
    clean_pts = binder_points(bind)  # (B,M,3)
    rec_pts = (
        binder_points(rec) if rec.dim() == 4 else rec
    )  # not used in this minimal baseline
    return rec, bind, clean_pts


def loss_step(model, clean_pts, device):
    B = clean_pts.shape[0]
    # sample time per-batch-element
    t = torch.rand(B, device=device)
    # forward diffuse
    X_noisy, _, _ = diffuse_binder_coords(clean_pts, t)  # uses sigma_rot/trans
    # epsilon supervision
    xi_tgt, eta_tgt = supervision_from_noisy(X_noisy, clean_pts, t=t, eps_targets=True)
    # predict
    xi_pred, eta_pred = model(X_noisy, t)
    # MSE losses
    loss_rot = F.mse_loss(xi_pred, xi_tgt)
    loss_trans = F.mse_loss(eta_pred, eta_tgt)
    loss = loss_rot + loss_trans
    return loss, {"rot": loss_rot.detach(), "trans": loss_trans.detach()}


def train(model, steps=1000, lr=3e-4, batch_size=64, device="cpu", N=128, M=64):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    hist = {"loss": [], "rot": [], "trans": []}
    model.train()
    for it in range(1, steps + 1):
        rec, bind, clean_pts = make_batch(B=batch_size, N=N, M=M, device=device)
        loss, logs = loss_step(model, clean_pts, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        hist["loss"].append(loss.item())
        hist["rot"].append(logs["rot"].item())
        hist["trans"].append(logs["trans"].item())
        if it % max(steps // 10, 1) == 0:
            print(
                f"[{it:4d}/{steps}] loss={loss.item():.4f} (rot {logs['rot']:.4f}, trans {logs['trans']:.4f})"
            )
    return hist


# --------------------------
# Deterministic iterative denoiser (sampling)
# --------------------------
@torch.no_grad()
def iterative_denoise(model, bind_clean_pts, steps=50, device="cpu"):
    """
    Start from a random pose at t=1 and iteratively move toward the clean pose
    using the model's predicted residual transform per timestep.
    Returns: X_pred (B,P,3)
    """
    model.eval()
    B = bind_clean_pts.shape[0]
    # initialize with a single noisy sample at t=1
    t1 = torch.ones(B, device=device)
    X, _, _ = diffuse_binder_coords(bind_clean_pts, t1)  # (B,P,3)
    # schedule from 1 -> 0 (inclusive)
    ts = torch.linspace(1.0, 0.0, steps, device=device)
    Xk = X.clone()
    for t in ts:
        tt = torch.full((B,), float(t), device=device)
        xi_pred, eta_pred = model(Xk, tt)  # eps
        # Convert back to a residual pose (noisy -> clean) at this t
        omega = xi_pred * sigma_rot(tt).unsqueeze(-1)  # (B,3)
        delta = eta_pred * sigma_trans(tt).unsqueeze(-1)  # (B,3)
        R_delta = so3_exp(omega)  # (B,3,3)
        Xk = left_apply(R_delta, delta, Xk)  # move toward clean
    return Xk  # predicted clean binder points


# --------------------------
# Evaluation & plotting
# --------------------------
@torch.no_grad()
def rmsd(A, B):  # (B,P,3) vs (B,P,3)
    return torch.sqrt(((A - B) ** 2).sum(dim=-1).mean(dim=-1))


@torch.no_grad()
def kabsch_align_to(A, B):  # best-fit R,t to map A->B, return transformed A
    R, t = kabsch(A, B)
    return left_apply(R, t, A)


def plot_training_curves(hist):
    plt.figure()
    plt.plot(hist["loss"], label="total")
    plt.plot(hist["rot"], label="rot")
    plt.plot(hist["trans"], label="trans")
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _set_equal_3d(ax, X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    max_range = (x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
    R = max(max_range).item()
    cx, cy, cz = x.mean().item(), y.mean().item(), z.mean().item()
    ax.set_xlim(cx - R / 2, cx + R / 2)
    ax.set_ylim(cy - R / 2, cy + R / 2)
    ax.set_zlim(cz - R / 2, cz + R / 2)


def visualize_pose(
    rec_coords, bind_clean_pts, bind_pred_pts, example=0, title="Pred vs GT"
):
    # Extract CA traces
    rec_ca = rec_coords[example, :, 1, :].cpu()  # (N,3)
    clean = bind_clean_pts[example].cpu()  # (P,3)
    pred = bind_pred_pts[example].cpu()  # (P,3)
    # Best-align prediction to GT for RMSD reporting only (not for display overlay)
    pred_aligned = kabsch_align_to(pred.unsqueeze(0), clean.unsqueeze(0))[0]
    print("RMSD (Å):", float(rmsd(pred_aligned.unsqueeze(0), clean.unsqueeze(0))[0]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        rec_ca[:, 0], rec_ca[:, 1], rec_ca[:, 2], s=2, alpha=0.5, label="Receptor CA"
    )
    ax.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=8, alpha=0.9, label="Binder GT")
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=8, alpha=0.9, label="Binder Pred")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal_3d(ax, torch.cat([rec_ca, clean, pred], dim=0))
    ax.legend()
    plt.tight_layout()
    plt.show()


# --------------------------
# Quick demo script
# --------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # Model
    model = PoseDenoiser(t_dim=64, hidden=128)

    # Train briefly on toy data
    hist = train(
        model,
        steps=800,  # keep small for a quick run
        lr=3e-4,
        batch_size=64,
        device=device,
        N=128,
        M=64,
    )

    # Plot losses
    plot_training_curves(hist)

    # Make a fresh test batch
    rec, bind, clean_pts = make_batch(B=8, N=128, M=64, device=device)

    # Sample (iterative denoise) starting from a random pose at t=1
    pred_pts = iterative_denoise(model, clean_pts, steps=60, device=device)

    # Visualize one example overlaid on receptor + GT
    visualize_pose(
        rec_coords=rec,
        bind_clean_pts=clean_pts,
        bind_pred_pts=pred_pts,
        example=0,
        title="Toy docking — predicted vs ground-truth",
    )


# ============================================================
# VP schedule (DDPM-style)
# ============================================================
def make_beta_schedule(T=1000, kind="cosine", beta_start=1e-4, beta_end=2e-2):
    """
    Returns dict with:
      betas:      (T,)
      alphas:     (T,)
      alpha_bar:  (T+1,) with alpha_bar[0]=1, alpha_bar[t]=prod_{i=1..t} alphas[i]
    """
    if kind == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
    elif kind == "cosine":
        # Improved DDPM cosine schedule (Nichol & Dhariwal), converted to discrete betas
        s = 0.008
        t = torch.linspace(0, T, T + 1)
        f = lambda x: torch.cos(((x / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = (f(t) / f(torch.tensor(0.0))).clamp(min=1e-8, max=1.0)  # (T+1,)
        # Convert to per-step alphas/betas
        alphas = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-8, max=1.0)
        betas = 1.0 - alphas
        # Rebuild alpha_bar to be consistent and explicit (size T+1, index 0..T)
        alpha_bar = torch.cat([torch.ones(1), torch.cumprod(alphas, dim=0)])
        return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}
    else:
        raise ValueError("kind must be 'linear' or 'cosine'")

    alphas = 1.0 - betas
    alpha_bar = torch.cat([torch.ones(1), torch.cumprod(alphas, dim=0)])
    return {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}


# Convenience: gather alpha_bar[s] for a batch of indices s (1..T)
def gather_alpha_bar(alpha_bar, s_idx):
    # alpha_bar is length T+1, with alpha_bar[0]=1
    return alpha_bar[s_idx]


# ============================================================
# Training with VP schedule (ε-targets on SE(3) residuals)
# ============================================================
@torch.no_grad()
def vp_forward_sample_pose(
    clean_pts, s_idx, alpha_bar, sigma_rot_max=math.pi, sigma_trans_max=30.0
):
    """
    clean_pts: (B,P,3); s_idx: (B,) integers in [1..T]
    Produces X_s via a single transform whose marginal matches the VP std at step s:
      σ_s = σ_max * sqrt(1 - alpha_bar[s])
    Returns: X_s, eps_r, eps_t, σ_r, σ_t
    """
    B = clean_pts.shape[0]
    ab_s = gather_alpha_bar(alpha_bar, s_idx)  # (B,)
    sig_r = sigma_rot_max * torch.sqrt(1.0 - ab_s)  # (B,)
    sig_t = sigma_trans_max * torch.sqrt(1.0 - ab_s)  # (B,)
    eps_r = torch.randn(B, 3, device=clean_pts.device, dtype=clean_pts.dtype)
    eps_t = torch.randn(B, 3, device=clean_pts.device, dtype=clean_pts.dtype)
    omega = eps_r * sig_r[:, None]  # (B,3)
    delta = eps_t * sig_t[:, None]  # (B,3)
    Rn = so3_exp(omega)  # (B,3,3)
    Xs = left_apply(Rn, delta, clean_pts)  # (B,P,3)
    return Xs, eps_r, eps_t, sig_r, sig_t


def supervision_from_noisy_vp(
    bind_noisy_pts, bind_clean_pts, sigma_rot_s, sigma_trans_s, weights=None
):
    """
    Like your supervision_from_noisy(eps_targets=True) but takes explicit σ values
    (per-sample) from the VP schedule instead of a continuous t.
    """
    R_delta, t_delta = kabsch(bind_noisy_pts, bind_clean_pts, weights=weights)
    xi = so3_log(R_delta) / (sigma_rot_s[:, None] + 1e-8)  # (B,3)
    eta = t_delta / (sigma_trans_s[:, None] + 1e-8)  # (B,3)
    return xi, eta


def loss_step_vp(
    model, clean_pts, schedule, sigma_rot_max=math.pi, sigma_trans_max=30.0
):
    """
    Sample a random discrete step s, synthesize X_s with the VP marginal,
    compute ε-targets (xi, eta), and return MSE.
    """
    B = clean_pts.shape[0]
    T = schedule["betas"].numel()
    device = clean_pts.device
    # Sample per-example step s in [1..T]
    s_idx = torch.randint(1, T + 1, (B,), device=device)
    # Forward sample
    Xs, _, _, sig_r, sig_t = vp_forward_sample_pose(
        clean_pts, s_idx, schedule["alpha_bar"], sigma_rot_max, sigma_trans_max
    )
    # Targets in ε-space (noisy -> clean residual normalized by σ_s)
    xi_tgt, eta_tgt = supervision_from_noisy_vp(Xs, clean_pts, sig_r, sig_t)
    # Predict
    t_embed = s_idx.float() / float(T)  # map to [0,1] for time embedding
    xi_pred, eta_pred = model(Xs, t_embed)
    # Loss
    loss_rot = F.mse_loss(xi_pred, xi_tgt)
    loss_trans = F.mse_loss(eta_pred, eta_tgt)
    return loss_rot + loss_trans, {
        "rot": loss_rot.detach(),
        "trans": loss_trans.detach(),
    }


def train_vp(
    model,
    steps=2000,
    lr=3e-4,
    batch_size=64,
    device="cpu",
    N=128,
    M=64,
    T=400,
    schedule_kind="cosine",
    sigma_rot_max=math.pi,
    sigma_trans_max=30.0,
):
    schedule = make_beta_schedule(T=T, kind=schedule_kind)
    for k in schedule:
        schedule[k] = schedule[k].to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = {"loss": [], "rot": [], "trans": []}
    model.to(device).train()

    for it in range(1, steps + 1):
        _, _, clean_pts = make_toy_docking_dataset(
            B=batch_size, N=N, M=M, device=device
        )
        clean_pts = binder_points(
            _[1] if False else _[0]
        )  # dummy to hint shape; ignored
        # we actually want binder points from the generator call:
        # The call above returns (rec_coords, bind_coords, R_gt, t_gt)
        # so rebuild clean_pts correctly:
        rec_coords, bind_coords, _, _ = make_toy_docking_dataset(
            B=batch_size, N=N, M=M, device=device
        )
        clean_pts = binder_points(bind_coords)  # (B,M,3)

        loss, logs = loss_step_vp(
            model, clean_pts, schedule, sigma_rot_max, sigma_trans_max
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        hist["loss"].append(loss.item())
        hist["rot"].append(logs["rot"].item())
        hist["trans"].append(logs["trans"].item())
        if it % max(steps // 10, 1) == 0:
            print(
                f"[{it:4d}/{steps}] loss={loss.item():.4f} (rot {logs['rot']:.4f}, trans {logs['trans']:.4f})"
            )

    return schedule, hist


# ============================================================
# Stochastic reverse process (VP SDE discretization)
# ============================================================
@torch.no_grad()
def sample_vp(
    model,
    clean_pts,
    schedule,
    steps_T=None,
    sigma_rot_max=math.pi,
    sigma_trans_max=30.0,
    device="cpu",
):
    """
    Starts from X_T ~ q(X_T | X_0) (i.e., a heavily noised pose of the clean binder),
    then runs reverse steps with stochasticity: small SE(3) updates with drift+noise.

    Returns X_0_pred (B,P,3)
    """
    model.eval()
    betas = schedule["betas"].to(device)
    alpha_bar = schedule["alpha_bar"].to(device)
    T = betas.numel() if steps_T is None else int(steps_T)
    assert 1 <= T <= betas.numel()

    B = clean_pts.shape[0]
    # Initialize at t=T from the VP marginal
    s_init = torch.full((B,), T, device=device, dtype=torch.long)
    X = vp_forward_sample_pose(
        clean_pts, s_init, alpha_bar, sigma_rot_max, sigma_trans_max
    )[0]

    # Reverse loop
    for s in range(T, 0, -1):
        bs = betas[s - 1]  # scalar
        t_embed = torch.full((B,), float(s) / float(T), device=device)
        xi_pred, eta_pred = model(X, t_embed)  # (B,3), (B,3)

        # Drift term: -β_s * σ_max * ε_pred  (dimensionally correct in Lie algebra)
        omega_mean = -bs * sigma_rot_max * xi_pred  # (B,3)
        delta_mean = -bs * sigma_trans_max * eta_pred  # (B,3)

        # Stochasticity (set to zero at final step if you want a deterministic final step)
        z_r = torch.randn_like(xi_pred) if s > 1 else torch.zeros_like(xi_pred)
        z_t = torch.randn_like(eta_pred) if s > 1 else torch.zeros_like(eta_pred)
        omega = omega_mean + torch.sqrt(bs) * sigma_rot_max * z_r
        delta = delta_mean + torch.sqrt(bs) * sigma_trans_max * z_t

        R_step = so3_exp(omega)
        X = left_apply(R_step, delta, X)

    return X  # predicted clean binder points at step 0


# ============================================================
# Simple plotting helpers (unchanged)
# ============================================================
def plot_training_curves(hist):
    plt.figure()
    plt.plot(hist["loss"], label="total")
    plt.plot(hist["rot"], label="rot")
    plt.plot(hist["trans"], label="trans")
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.title("Training loss (VP)")
    plt.legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def rmsd(A, B):  # (B,P,3)
    return torch.sqrt(((A - B) ** 2).sum(dim=-1).mean(dim=-1))


@torch.no_grad()
def kabsch_align_to(A, B):  # (B,P,3) -> transformed A
    R, t = kabsch(A, B)
    return left_apply(R, t, A)


def _set_equal_3d(ax, X):
    import numpy as np

    X = X.detach().cpu().numpy()
    mins, maxs = X.min(axis=0), X.max(axis=0)
    c = (mins + maxs) / 2
    R = (maxs - mins).max() / 2
    ax.set_xlim(c[0] - R, c[0] + R)
    ax.set_ylim(c[1] - R, c[1] + R)
    ax.set_zlim(c[2] - R, c[2] + R)


def visualize_pose(
    rec_coords, bind_clean_pts, bind_pred_pts, example=0, title="VP-DDPM: Pred vs GT"
):
    rec_ca = rec_coords[example, :, 1, :].cpu()
    clean = bind_clean_pts[example].cpu()
    pred = bind_pred_pts[example].cpu()
    pred_aln = kabsch_align_to(pred.unsqueeze(0), clean.unsqueeze(0))[0]
    print("RMSD (Å):", float(rmsd(pred_aln.unsqueeze(0), clean.unsqueeze(0))[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        rec_ca[:, 0], rec_ca[:, 1], rec_ca[:, 2], s=2, alpha=0.5, label="Receptor CA"
    )
    ax.scatter(clean[:, 0], clean[:, 1], clean[:, 2], s=8, alpha=0.9, label="Binder GT")
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=8, alpha=0.9, label="Binder Pred")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal_3d(ax, torch.cat([rec_ca, clean, pred], dim=0))
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # Tiny model from previous message
    model = PoseDenoiser(t_dim=64, hidden=128)

    # Train on toy data with VP schedule
    schedule, hist = train_vp(
        model,
        steps=1200,
        lr=3e-4,
        batch_size=64,
        device=device,
        N=128,
        M=64,
        T=400,  # fewer steps for speed
        schedule_kind="cosine",
        sigma_rot_max=math.pi,
        sigma_trans_max=30.0,
    )

    plot_training_curves(hist)

    # Test batch
    rec_coords, bind_coords, _, _ = make_toy_docking_dataset(
        B=8, N=128, M=64, device=device
    )
    clean_pts = binder_points(bind_coords)

    # Stochastic VP sampling (start at t=T, denoise to 0)
    pred_pts = sample_vp(
        model,
        clean_pts,
        schedule,
        steps_T=400,  # must be <= schedule T
        sigma_rot_max=math.pi,
        sigma_trans_max=30.0,
        device=device,
    )

    # Visualize
    visualize_pose(
        rec_coords,
        clean_pts,
        pred_pts,
        example=0,
        title="Toy docking (VP-DDPM) — predicted vs ground-truth",
    )
