import os
import torch
import pdb as debugger
import numpy as np
from models import AllAtomEnergyModel
from datasets import DockingDataset
from utils import (
    save_structures,
    Meter_Unbound_Bound,
    NoiseSchedule,
    recursive_to,
    save_trajectory_to_pdb,
    idx_to_seq,
    Features,
    batched_solve_mds,
    kabsch,
    randomly_rotate_and_translate
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

    all_complex_results = []  # <- stores per-complex k results

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

        D_gt = torch.cdist(binder_ca, target_ca).to(device)

        # swap if using ESM
        if binder.S_ESM2 is not None and target.S_ESM2 is not None:
            binder = binder.swap_S()
            target = target.swap_S()

        # conditioning (compute once)
        binder_cond, target_cond = None, None
        if args.use_guidedock:
            binder_cond, target_cond = model.dist_model.vae.vae.get_diff_conditioning(
                binder, target
            )

        # swap back
        if binder.S_ESM2 is not None and target.S_ESM2 is not None:
            binder = binder.swap_S()
            target = target.swap_S()

        # GT coords
        gt_target_coords = target.X.cpu()
        gt_binder_coords = binder.X.cpu()

        if use_ca:
            gt_target_coords = gt_target_coords[:, :, 1, :]
            gt_binder_coords = gt_binder_coords[:, :, 1, :]

        pdb_str = binder.pdbs[0]

        # ===============================
        # 🔥 SAMPLE K DISTANCE MAPS
        # ===============================
        results_k = []

        for k in range(args.top_k):

            # ---- SAMPLE DISTANCE MAP ----
            if args.use_guidedock and not args.use_gt_D:
                with torch.no_grad():
                    z = model.dist_model.diffusion_model.sample(
                        binder,
                        target,
                        binder_cond,
                        target_cond,
                        num_samples=1,
                        steps=100,
                        guidance_scale=2.0,
                        verbose=False,
                    )

                    x_recon_phys = model.dist_model.vae.vae.decode(
                        z, binder, target, upscale=True
                    )  # [1, Nb, Nt]

                D_guide = x_recon_phys

            elif args.use_guidedock and args.use_gt_D:
                D_guide = D_gt
            else:
                D_guide = None

            # ---- RANDOM INIT ----
            num_particles = args.num_particles
            binder_X_noised = model.randomly_translate_and_rotate(
                binder.expand(num_particles),
                target.expand(num_particles),
                args.tr_s_max,
            )

            noised_binder = Features(
                binder_X_noised, *binder.expand(num_particles)[1:]
            )

            noised_binder, binder_dev, target_dev = (
                recursive_to(noised_binder, device),
                recursive_to(binder, device),
                recursive_to(target, device),
            )

            # ---- RUN DOCKING ----
            output = model.sample_fk(
                noised_binder,
                target_dev.expand(num_particles),
                binder_dev.expand(num_particles),
                args.langevin_T,
                0,
                0,
                1,
                noise_schedule=noise_schedule,
                binder_cond=binder_cond,
                target_cond=target_cond,
                D_guide=D_guide,
                tr_grad_weight=args.tr_grad_weight,
                rot_grad_weight=args.rot_grad_weight,
                num_particles=num_particles,
                cutoff_threshold=args.cutoff_threshold,
            )

            binder_pred_k, trajectory, energy = output[0][0], output[-2], output[-1]

            # ---- PREP COORDS ----
            if use_ca:
                binder_coords = binder_pred_k[:, :, 1, :]
            else:
                binder_coords = binder_pred_k.clone().cpu()

            target_coords = gt_target_coords.clone().cpu()

            # ---- METRICS ----
            temp_meter = Meter_Unbound_Bound()
            lrmsd, irmsd, crmsd, dockq = temp_meter.update_rmsd(
                binder_coords.reshape(-1, 3),
                target_coords.reshape(-1, 3),
                gt_binder_coords.reshape(-1, 3),
                gt_target_coords.reshape(-1, 3),
            )

            # ---- STORE RESULT ----
            results_k.append(
                {
                    "k": k,
                    "binder_pred": binder_pred_k.detach().cpu(),
                    "trajectory": trajectory,
                    "lrmsd": float(lrmsd),
                    "irmsd": float(irmsd),
                    "crmsd": float(crmsd),
                    "dockq": float(dockq),
                    "energy": float(energy) if energy is not None else None,
                }
            )

            print(
                f"[{pdb_str}] k={k+1}/{args.top_k} | "
                f"CRMSD={crmsd:.3f}, LRMSD={lrmsd:.3f}, IRMSD={irmsd:.3f}, DockQ={dockq:.3f}"
            )

        # ===============================
        # 🔥 SAVE ALL K RESULTS
        # ===============================
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(
            results_k,
            os.path.join(save_dir, f"{pdb_str}_k_results_step{step}.pt"),
        )

        # store globally
        all_complex_results.append(
            {
                "pdb": pdb_str,
                "results_k": results_k,
            }
        )

    # ===============================
    # 🔥 SAVE EVERYTHING (optional)
    # ===============================
    torch.save(
        all_complex_results,
        os.path.join(save_dir, f"all_results_step{step}.pt"),
    )

    return all_complex_results