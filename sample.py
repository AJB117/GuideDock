import os
import torch
import numpy as np
from models import AllAtomEnergyModel
from utils import (
    Meter_Unbound_Bound,
    NoiseSchedule,
    recursive_to,
    save_trajectory_to_pdb,
    idx_to_seq,
    Features,
)
from tqdm import tqdm


@torch.enable_grad()
def evaluate_samples(
    args,
    device,
    val_loader,
    model: AllAtomEnergyModel,
    noise_schedule: NoiseSchedule,
    save_dir: str,
    step: int,
    vae,
    use_ca=False,
):
    model.eval()

    max_len = 500
    meter = Meter_Unbound_Bound()
    gt_binders, gt_targets = [], []
    binders, targets = [], []
    for batch in tqdm(val_loader):
        binder, target = batch

        binder: Features
        target: Features

        binder_bound_coords, tgt_bound_coords = binder.X, target.X

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        binder = recursive_to(binder, device)
        target = recursive_to(target, device)

        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()
        pair_mask = (bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)).to(device)
        D = torch.cdist(binder_ca, target_ca).to(device)

        B = binder.X.size(0)

        binder_cond, target_cond = None, None

        if args.use_guidedock and not args.use_gt_D:
            binder_expanded, target_expanded = (
                binder.expand(args.num_D_samples),
                target.expand(args.num_D_samples),
            )

            binder_cond, target_cond = model.dist_model.vae.vae.get_diff_conditioning(
                binder_expanded, target_expanded
            )

            with torch.no_grad() as X, torch.inference_mode() as Y:
                if not args.use_vae_regressor:
                    steps = 25
                else:
                    steps = 1
                z = model.dist_model.diffusion_model.sample(
                    binder_expanded,
                    target_expanded,
                    binder_cond,
                    target_cond,
                    num_samples=args.num_D_samples,
                    steps=steps,
                    guidance_scale=10.0,
                    verbose=False,
                )

                x_recon_phys = model.dist_model.vae.vae.decode(
                    z, binder_expanded, target_expanded, upscale=True
                )
                x_recon_phys = x_recon_phys.mean(dim=0, keepdim=True)
                print(
                    f"{binder.pdbs[0]}: MSE of reconstructed distance matrix: {torch.mean((x_recon_phys - D) ** 2).item():.4f}"
                )

        M, N = binder.X.size(1), target.X.size(1)

        gt_binders.append(binder)
        gt_targets.append(target)

        best_crmsd = float("inf")
        best_energy = float("inf")
        best_lrmsd = float("inf")
        best_binder_pred = None
        best_coords = None

        use_energy_selection = False

        gt_target_coords = target.X.cpu()
        gt_binder_coords = binder.X.cpu()

        if use_ca:
            gt_target_coords = gt_target_coords[:, :, 1, :]
            gt_binder_coords = gt_binder_coords[:, :, 1, :]

        target_idxs = torch.arange(gt_target_coords.shape[0], device=device).cpu()
        binder_idxs = torch.arange(gt_binder_coords.shape[0], device=device).cpu()

        for k in range(args.top_k):
            for _ in range(1):
                num_particles = args.num_particles
                binder_X_noised = model.randomly_translate_and_rotate(
                    binder.expand(num_particles),
                    target.expand(num_particles),
                    args.tr_s_max,
                )
                noised_binder = Features(
                    binder_X_noised, *binder.expand(num_particles)[1:]
                )

                noised_binder, binder, target = (
                    recursive_to(noised_binder, device),
                    recursive_to(binder, device),
                    recursive_to(target, device),
                )

                if args.use_guidedock and not args.use_gt_D:
                    D_guide = x_recon_phys
                elif args.use_guidedock and args.use_gt_D:
                    D_guide = D
                else:
                    D_guide = None

                output = model.sample_fk(
                    noised_binder,
                    target.expand(num_particles),
                    binder.expand(num_particles),
                    args.langevin_T,
                    0,
                    0,
                    1,
                    noise_schedule=noise_schedule,
                    uncertainty=None,
                    binder_cond=binder_cond,
                    target_cond=target_cond,
                    D_guide=D_guide,
                    tr_grad_weight=args.tr_grad_weight,
                    rot_grad_weight=args.rot_grad_weight,
                    num_particles=num_particles,
                    cutoff_threshold=args.cutoff_threshold,
                )

                if args.non_conservative:
                    binder_pred_k = output[0]
                    energy = None
                    trajectory = output[-1]

                binder_pred_k, trajectory, energy = output[0][0], output[-2], output[-1]

                energy = 1.0

            if energy is not None:
                if energy > 0:
                    energy = -energy

            if use_ca:
                binder_coords = binder_pred_k[:, :, 1, :]
            else:
                binder_coords = binder_pred_k.clone().cpu()

            target_coords = gt_target_coords.clone().cpu()

            nonzero_target_idxs = (
                target[0][0].sum(dim=-1).sum(dim=-1).nonzero(as_tuple=True)[0].cpu()
            )
            nonzero_binder_idxs = (
                binder_pred_k[0].sum(dim=-1).sum(dim=-1).nonzero(as_tuple=True)[0].cpu()
            )

            crmsd = None
            try:
                temp_meter = Meter_Unbound_Bound()
                lrmsd, irmsd, crmsd, dockq = temp_meter.update_rmsd(
                    binder_coords.reshape(-1, 3),
                    target_coords.reshape(-1, 3),
                    gt_binder_coords.reshape(-1, 3),
                    gt_target_coords.reshape(-1, 3),
                )

            except Exception as e:
                print(f"Sample {k + 1}/{args.top_k}: Error calculating CRMSD - {e}")
                continue

            is_better = False
            if use_energy_selection and energy is not None:
                if energy < best_energy:
                    is_better = True
                    best_energy = energy
                    best_crmsd = crmsd
                    best_lrmsd = lrmsd
            else:
                if crmsd is not None and crmsd < best_crmsd:
                    is_better = True
                    best_crmsd = crmsd
                    best_lrmsd = lrmsd

            if is_better:
                best_binder_pred = binder_pred_k
                best_coords = (
                    binder_coords,
                    target_coords,
                    nonzero_binder_idxs,
                    nonzero_target_idxs,
                )
                best_trajectory = trajectory

            k += 1

        if use_energy_selection:
            print(
                f"Selected best conformation with Energy: {best_energy:.3f} and CRMSD: {best_crmsd:.3f} and LRMSD: {best_lrmsd:.3f}"
            )
        else:
            print(
                f"Selected best conformation with CRMSD: {best_crmsd:.3f} and LRMSD: {best_lrmsd:.3f}"
            )
        if best_coords is None:
            continue

        binder_pred = best_binder_pred
        binder_coords, target_coords, nonzero_binder_idxs, nonzero_target_idxs = (
            best_coords
        )
        binder = Features(binder_pred, *binder[1:])

        binders.append(binder_pred)
        targets.append(target)

        binder_seq = idx_to_seq(binder.Si)
        tgt_seq = idx_to_seq(target.Si)
        pdb_str = binder.pdbs[0]

        if args.save_trajectory:
            save_trajectory_to_pdb(
                np.concatenate(best_trajectory, axis=0),
                binder_seq,
                os.path.join(
                    args.save_dir, f"step_{step}", "trajectory", binder.pdbs[0] + ".pdb"
                ),
            )
            print(
                f"Saving trajectory to {os.path.join(args.save_dir, f'step_{step}', 'trajectory', pdb_str + '.pdb')}"
            )

        try:
            lrmsd, irmsd, crmsd, dockq = meter.update_rmsd(
                binder_coords.reshape(-1, 3),
                target_coords.reshape(-1, 3),
                gt_binder_coords.reshape(-1, 3),
                gt_target_coords.reshape(-1, 3),
            )
            print(
                f"lrmsd: {lrmsd:.3f}, irmsd: {irmsd:.3f}, crmsd: {crmsd:.3f}, dockq: {dockq:.3f}; running interface RMSD: {np.mean(meter.interface_rmsd_list):.3f}, complex RMSD: {np.mean(meter.complex_rmsd_list):.3f}, DockQ: {np.mean(meter.dockQ_list):.3f}, num failed: {meter.bad_case}"
            )
        except Exception as e:
            print(e)
            continue

    avg_lrmsd, avg_crmsd, avg_irmsd, num_failed = meter.summarize(
        reduction_rmsd="mean", include_failed=True
    )
    std_lrmsd, std_crmsd, std_irmsd, std_dockq = meter.summarize(
        reduction_rmsd="std", include_failed=True
    )
    median_lrmsd, median_crmsd, median_irmsd, median_dockq = meter.summarize(
        reduction_rmsd="median", include_failed=True
    )
    avg_dockq, median_dockq, std_dockq = meter.dockQ(include_failed=True)

    clean_avg_lrmsd, clean_avg_crmsd, clean_avg_irmsd, clean_avg_dockq = (
        meter.summarize(reduction_rmsd="mean")
    )
    clean_std_lrmsd, clean_std_crmsd, clean_std_irmsd, clean_std_dockq = (
        meter.summarize(reduction_rmsd="std")
    )
    clean_median_lrmsd, clean_median_crmsd, clean_median_irmsd, clean_median_dockq = (
        meter.summarize(reduction_rmsd="median")
    )
    clean_avg_dockq, clean_median_dockq, clean_std_dockq = meter.dockQ()

    num_failed = meter.bad_case
    print(
        f"Avg ligand RMSD: {avg_lrmsd:.3f} ± {std_lrmsd:.3f}, Avg interface RMSD: {avg_irmsd:.3f} ± {std_irmsd:.3f}, Avg complex RMSD: {avg_crmsd:.3f} ± {std_crmsd:.3f}, Avg DockQ: {avg_dockq:.3f} ± {std_dockq:.3f}, num failed: {num_failed}"
    )
    print(
        f"Clean Avg ligand RMSD: {clean_avg_lrmsd:.3f} ± {clean_std_lrmsd:.3f}, Clean Avg interface RMSD: {clean_avg_irmsd:.3f} ± {clean_std_irmsd:.3f}, Clean Avg complex RMSD: {clean_avg_crmsd:.3f} ± {clean_std_crmsd:.3f}, Clean Avg DockQ: {clean_avg_dockq:.3f} ± {clean_std_dockq:.3f}"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.save_meter:
        torch.save(
            {
                "binders": binders,
                "targets": targets,
                "meter": meter,
            },
            f"{save_dir}/val_{step}.pt",
        )

    return avg_lrmsd, avg_irmsd, avg_crmsd, avg_dockq
