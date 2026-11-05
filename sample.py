import os

import numpy as np
import torch
from tqdm import tqdm

from models import AllAtomEnergyModel
from utils import (
    Features,
    Meter_Unbound_Bound,
    NoiseSchedule,
    idx_to_seq,
    recursive_to,
    save_trajectory_to_pdb,
)


@torch.no_grad()
def sample_distance_map(args, model, binder, target, conditioning, independent=False):
    steps = getattr(args, "distance_steps", None)
    if steps is None:
        steps = 1 if args.use_vae_regressor else (100 if independent else 25)
    guidance_scale = getattr(args, "distance_guidance_scale", None)
    if guidance_scale is None:
        guidance_scale = 2.0 if independent else 10.0
    latent = model.dist_model.diffusion_model.sample(
        binder,
        target,
        *conditioning,
        num_samples=binder.X.size(0),
        steps=steps,
        guidance_scale=guidance_scale,
        verbose=False,
    )
    distances = model.dist_model.vae.vae.decode(latent, binder, target, upscale=True)
    return distances.mean(dim=0, keepdim=True)


def sample_pose(args, model, binder, target, noise_schedule, guidance, conditioning):
    particles = args.num_particles
    expanded_binder = binder.expand(particles)
    expanded_target = target.expand(particles)
    initial_coords = model.randomly_translate_and_rotate(
        expanded_binder, expanded_target, args.tr_s_max
    )
    initial_binder = Features(initial_coords, *expanded_binder[1:])
    output = model.sample_fk(
        initial_binder,
        expanded_target,
        expanded_binder,
        args.langevin_T,
        0,
        0,
        1,
        noise_schedule=noise_schedule,
        uncertainty=None,
        binder_cond=conditioning[0],
        target_cond=conditioning[1],
        D_guide=guidance,
        tr_grad_weight=args.tr_grad_weight,
        rot_grad_weight=args.rot_grad_weight,
        num_particles=particles,
        cutoff_threshold=args.cutoff_threshold,
    )
    return output[0][0].detach().cpu(), output[-2], output[-1]


def coordinate_metrics(meter, binder_coords, target_coords, binder_gt, target_gt):
    return meter.update_rmsd(
        binder_coords.reshape(-1, 3),
        target_coords.reshape(-1, 3),
        binder_gt.reshape(-1, 3),
        target_gt.reshape(-1, 3),
    )


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
    return_all=False,
):
    if args.top_k < 1 or args.num_particles < 1 or args.num_D_samples < 1:
        raise ValueError("Sample and particle counts must be positive")
    if getattr(args, "distance_steps", None) is not None and args.distance_steps < 1:
        raise ValueError("distance_steps must be positive")
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    independent = getattr(args, "resample_distance_per_pose", False)
    save_all = getattr(args, "save_all_samples", False) or independent or return_all
    all_results, binders, targets = [], [], []
    meter = Meter_Unbound_Bound()

    for binder, target in tqdm(val_loader):
        if binder.X.size(0) != 1 or target.X.size(0) != 1:
            raise ValueError("Docking evaluation requires one complex per batch")
        binder, target = recursive_to((binder, target), device)
        pdb = binder.pdbs[0]
        binder_gt, target_gt = binder.X.cpu(), target.X.cpu()
        if use_ca:
            binder_gt, target_gt = binder_gt[:, :, 1], target_gt[:, :, 1]
        true_distances = torch.cdist(binder.X[:, :, 1], target.X[:, :, 1])
        conditioning = (None, None)
        guidance = true_distances if args.use_guidedock and args.use_gt_D else None
        if args.use_guidedock and not args.use_gt_D:
            distance_samples = 1 if independent else args.num_D_samples
            distance_binder = binder.expand(distance_samples)
            distance_target = target.expand(distance_samples)
            with torch.no_grad():
                conditioning = model.dist_model.vae.vae.get_diff_conditioning(
                    distance_binder, distance_target
                )
            if not independent:
                guidance = sample_distance_map(
                    args, model, distance_binder, distance_target, conditioning
                )

        candidates, best = [], None
        for k in range(args.top_k):
            if independent and args.use_guidedock and not args.use_gt_D:
                guidance = sample_distance_map(
                    args,
                    model,
                    distance_binder,
                    distance_target,
                    conditioning,
                    independent=True,
                )
            prediction, trajectory, energy = sample_pose(
                args, model, binder, target, noise_schedule, guidance, conditioning
            )
            coords = prediction[:, :, 1] if use_ca else prediction
            try:
                lrmsd, irmsd, crmsd, dockq = coordinate_metrics(
                    Meter_Unbound_Bound(), coords, target_gt, binder_gt, target_gt
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                print(f"{pdb}: sample {k + 1}/{args.top_k} metrics failed: {error}")
                continue
            if not np.isfinite([lrmsd, irmsd, crmsd, dockq]).all():
                print(f"{pdb}: sample {k + 1}/{args.top_k} has nonfinite metrics")
                continue
            candidate = {
                "k": k,
                "binder_pred": prediction,
                "trajectory": trajectory,
                "lrmsd": float(lrmsd),
                "irmsd": float(irmsd),
                "crmsd": float(crmsd),
                "dockq": float(dockq),
                "energy": float(energy) if energy is not None else None,
            }
            if save_all:
                candidates.append(candidate)
            if best is None or candidate["crmsd"] < best["crmsd"]:
                best = candidate
            print(
                f"[{pdb}] k={k + 1}/{args.top_k}: CRMSD={crmsd:.3f}, LRMSD={lrmsd:.3f}, IRMSD={irmsd:.3f}, DockQ={dockq:.3f}"
            )

        if save_all:
            torch.save(
                candidates, os.path.join(save_dir, f"{pdb}_k_results_step{step}.pt")
            )
            all_results.append({"pdb": pdb, "results_k": candidates})
        if best is None:
            continue
        prediction = best["binder_pred"]
        coords = prediction[:, :, 1] if use_ca else prediction
        coordinate_metrics(meter, coords, target_gt, binder_gt, target_gt)
        binders.append(prediction)
        targets.append(target)
        if args.save_trajectory:
            save_trajectory_to_pdb(
                np.concatenate(best["trajectory"], axis=0),
                idx_to_seq(binder.Si),
                os.path.join(args.save_dir, f"step_{step}", "trajectory", pdb + ".pdb"),
            )
        print(f"[{pdb}] selected CRMSD={best['crmsd']:.3f}, LRMSD={best['lrmsd']:.3f}")

    if save_all:
        torch.save(all_results, os.path.join(save_dir, f"all_results_step{step}.pt"))
    if args.save_meter:
        torch.save(
            {"binders": binders, "targets": targets, "meter": meter},
            os.path.join(save_dir, f"val_{step}.pt"),
        )
    if return_all:
        return all_results
    if not meter.complex_rmsd_list_all:
        return float("inf"), float("inf"), float("inf"), 0.0
    lrmsd, crmsd, irmsd, _ = meter.summarize(reduction_rmsd="mean", include_failed=True)
    dockq, _, _ = meter.dockQ(include_failed=True)
    print(
        f"Mean LRMSD={lrmsd:.3f}, IRMSD={irmsd:.3f}, CRMSD={crmsd:.3f}, DockQ={dockq:.3f}; failed={meter.bad_case}"
    )
    return lrmsd, irmsd, crmsd, dockq
