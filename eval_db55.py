import argparse
import os
import torch
import pdb as debugger
import numpy as np
from models import AllAtomEnergyModel
from ema_pytorch import EMA
from datasets import DockingDataset
from utils import (
    load_esm_embedding,
    save_structures,
    save_trajectory_to_pdb,
    Meter_Unbound_Bound,
    NoiseSchedule,
    recursive_to,
    Features,
    batched_solve_mds,
    kabsch
)
from tqdm import tqdm


def eval_db55(
    args,
    gpu,
    ckpt_path,
    model_type,
    save_dir,
    test_data=None,
    embedding=None,
    precomputed_path=None,
):
    device = torch.device(f"cuda:{gpu}")

    assets = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema_model_ckpt = assets['ema']
    model_ckpt = assets['model']
    model_args = assets['args']

    model_args = argparse.Namespace(**model_args)
    model_args.use_swiglu = True
    model_args.noise_scaler = args.noise_scaler
    noise_schedule = NoiseSchedule(
        model_args.tr_s_min,
        model_args.tr_s_max,
        model_args.rot_s_min,
        model_args.rot_s_max,
    )
    model = AllAtomEnergyModel(model_args, device=device, resolution=args.resolution).cuda(device=device)
    model.load_state_dict(model_ckpt)
    # model_ema: EMA = EMA(model, beta=0.999)
    # model_ema.load_state_dict(ema_model_ckpt)

    # model = torch.compile(model)
    model.eval()

    max_len = 500
    if test_data is None:
        test_data = DockingDataset(
            "data/db55/data.pkl",
            max_len,
            patch_strategy="com",
            resolution=args.resolution,
        )
    # if embedding is None:
    #     embedding = load_esm_embedding(
    #         test_data.data,
    #         ["binder_full", "target_full"],
    #         device=device,
    #         pretrained_weights=args.pretrained_weights,
    #     )
    if embedding is not None:
        embedding = torch.load(embedding, map_location=torch.device("cpu"))

    meter = Meter_Unbound_Bound()
    gt_binders, gt_targets = [], []
    binders, targets = [], []
    for idx, entry in enumerate(tqdm(test_data.data)):
        entries = [entry]
        binder, target, binder_bound_coords, tgt_bound_coords = (
            DockingDataset.make_local_batch(
                entries,
                embedding,
                model_args,
                "binder",
                "target",
                device=device,
                patch_size=max_len,
                resolution=args.resolution,
            )
        )
        gt_binders.append(binder)
        gt_targets.append(target)

        # Repeat sampling top_k times and evaluate each
        best_crmsd = float("inf")
        best_energy = float("inf")
        best_binder_pred = None
        best_coords = None
        
        # Determine selection criterion based on oracle flag
        # use_energy_selection = not args.non_conservative
        use_energy_selection = False

        gt_target_coords = entry["target_full_coords"].cpu()
        gt_binder_coords = entry["binder_full_coords"].cpu()
        target_idxs = torch.arange(
            gt_target_coords.shape[0], device=device
        ).cpu()
        binder_idxs = torch.arange(gt_binder_coords.shape[0], device=device).cpu()

        # with model_ema.average_parameters():
    # for k in range(args.top_k):
    #     for _ in range(5):
    #         binder_ = model.randomly_translate_and_rotate(binder, target, model_args.tr_s_max)
    #         binder = Features(binder_, *binder[1:])

    #         binder, target = recursive_to(binder, device), recursive_to(target, device)
    #         # Sample one conformation
    #         output = model.sample(
    #             binder,
    #             target,
    #             args.langevin_T,
    #             args.langevin_step_size,
    #             args.langevin_noise,
    #             len(entries),
    #             noise_schedule=noise_schedule,
    #         )

    #         if args.non_conservative:
    #             binder_pred_k = output[0]
    #             energy = None
    #         else:
    #             binder_pred_k, trajectory, energy = output[0][0], output[-2], output[-1]
            
    #         if energy != 0:
    #             break

    #     if energy > 0:
    #         energy = -energy

        flattened_binder_X = binder.X.view(1, -1, 3)
        flattened_target_X = target.X.view(1, -1, 3)
        cplx_X = torch.cat([flattened_binder_X, flattened_target_X], dim=1) # [1, N_binder + N_target, 3]

        D = torch.cdist(
            cplx_X, cplx_X
        ).double() # [1, N_binder + N_target, N_binder + N_target]

        num_binder = flattened_binder_X.shape[1]
        noise_scale = 1.0
        noise = noise_scale * 0 * torch.randn(
            D.shape[1], D.shape[2], device=D.device
        ).relu()
        D_noise = D.clone()
        D_noise[0, :num_binder, num_binder:] += noise[:num_binder, num_binder:]
        D_noise[0, num_binder:, :num_binder] += noise[num_binder:, :num_binder]
        D = D_noise

        mds_solution = batched_solve_mds(D.cpu().numpy(), dim=3)[0]  # [1, N_binder + N_target, 3]

        mds_binder = torch.from_numpy(mds_solution[: flattened_binder_X.shape[1], :]).unsqueeze(0)  # [1, N_binder, 3]
        mds_binder = mds_binder.view(1, -1, 4, 3)

        # align the mds binder and target to the ground truth
        aligned_mds_binder, _, _= kabsch(mds_binder[0], gt_binder_coords.double())

        binder_pred_k = aligned_mds_binder.unsqueeze(0).float()
        trajectory = [aligned_mds_binder.unsqueeze(0).float()]

        # Prepare coordinates for evaluation
        target_coords = gt_target_coords.clone().cpu()
        binder_coords = gt_binder_coords.clone().cpu()

        nonzero_target_idxs = (
            target[0][0].sum(dim=-1).sum(dim=-1).nonzero(as_tuple=True)[0].cpu()
        )
        nonzero_binder_idxs = (
            binder_pred_k[0].sum(dim=-1).sum(dim=-1).nonzero(as_tuple=True)[0].cpu()
        )

        target_coords[target_idxs[nonzero_target_idxs]] = target[0][0][
            nonzero_target_idxs
        ].cpu()
        binder_coords[binder_idxs[nonzero_binder_idxs]] = binder_pred_k[0][
            nonzero_binder_idxs
        ].cpu()
        k=1
        energy = None

        # Calculate metrics for this conformation
        crmsd = None
        try:
            # Create a temporary meter just for evaluation
            temp_meter = Meter_Unbound_Bound()
            lrmsd, irmsd, crmsd, dockq = temp_meter.update_rmsd(
                binder_coords.reshape(-1, 3),
                target_coords.reshape(-1, 3),
                gt_binder_coords.reshape(-1, 3),
                gt_target_coords.reshape(-1, 3),
            )

            if use_energy_selection and energy is not None:
                print(f"Sample {k + 1}/{args.top_k}: energy = {energy:.3f}, crmsd = {crmsd:.3f}")
            else:
                print(f"Sample {k + 1}/{args.top_k}: crmsd = {crmsd:.3f}")

        except Exception as e:
            print(f"Sample {k + 1}/{args.top_k}: Error calculating CRMSD - {e}")
            continue

        # Select best conformation based on oracle setting
        is_better = False
        if use_energy_selection and energy is not None:
            # Use energy-based selection (lower energy is better)
            if energy < best_energy:
                is_better = True
                best_energy = energy
                best_crmsd = crmsd
        else:
            # Use CRMSD-based selection (lower CRMSD is better)
            if crmsd is not None and crmsd < best_crmsd:
                is_better = True
                best_crmsd = crmsd
        
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

        # Print selection result
        if use_energy_selection:
            print(f"Selected best conformation with Energy: {best_energy:.3f} and CRMSD: {best_crmsd:.3f}")
        else:
            print(f"Selected best conformation with CRMSD: {best_crmsd:.3f}")

        # Use the best conformation for further processing
        binder_pred = best_binder_pred
        binder_coords, target_coords, nonzero_binder_idxs, nonzero_target_idxs = (
            best_coords
        )
        # binder = (binder_pred, binder[1], binder[2], binder[3], None)
        binder = Features(binder_pred, *binder[1:])

        binders.append(binder_pred)
        targets.append(target)


        #! viz - save structures using the best conformation
        save_trajectory_to_pdb(
            np.concatenate(best_trajectory, axis=0),
            [entry['binder_full'][i] for i in binder_idxs[nonzero_binder_idxs].cpu()],
            os.path.join(args.save_dir, "trajectory", entry["pdb"] + '.pdb'),
        )
        print(f"Saving trajectory to {os.path.join(args.save_dir, 'trajectory', entry['pdb'] + '.pdb')}")

        save_structures(
            binder.Si,
            target.Si,
            binder_pred[0].cpu(),
            target[0][0].cpu(),
            os.path.join(args.save_dir, "predicted"),
            entry["pdb"],
            binder_idx=binder_idxs[nonzero_binder_idxs].cpu(),
            target_idx=target_idxs[nonzero_target_idxs].cpu(),
            note="site_predicted",
        )
        # save_structures(
        #     entry,
        #     binder_bound_coords[0].cpu(),
        #     tgt_bound_coords[0].cpu(),
        #     os.path.join(args.save_dir, "gt"),
        #     entry["pdb"],
        #     binder_idx=binder_idxs[nonzero_binder_idxs].cpu(),
        #     target_idx=target_idxs[nonzero_target_idxs].cpu(),
        #     note="site_gt",
        # )
        # save_structures(
        #     entry,
        #     gt_binder_coords.cpu(),
        #     gt_target_coords.cpu(),
        #     os.path.join(args.save_dir, "gt"),
        #     entry["pdb"],
        #     binder_idx=None,
        #     target_idx=None,
        #     note="gt",
        # )

        # Update the main meter with the best conformation
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

    avg_lrmsd = np.mean(meter.ligand_rmsd_list)
    std_lrmsd = np.std(meter.ligand_rmsd_list)
    avg_irmsd = np.mean(meter.interface_rmsd_list)
    std_irmsd = np.std(meter.interface_rmsd_list)
    avg_cmrsd = np.mean(meter.complex_rmsd_list)
    std_cmrsd = np.std(meter.complex_rmsd_list)
    avg_dockq = np.mean(meter.dockQ_list)
    std_dockq = np.std(meter.dockQ_list)
    num_failed = meter.bad_case
    print(
        f"Avg ligand RMSD: {avg_lrmsd:.3f} ± {std_lrmsd:.3f}, Avg interface RMSD: {avg_irmsd:.3f} ± {std_irmsd:.3f}, Avg complex RMSD: {avg_cmrsd:.3f} ± {std_cmrsd:.3f}, Avg DockQ: {avg_dockq:.3f} ± {std_dockq:.3f}, num failed: {num_failed}"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(
        {
            "binders": binders,
            "targets": targets,
            "meter": meter,
        },
        f"{save_dir}/db55_results_{args.modifier}.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--encoder", type=str)
    parser.add_argument(
        "--model_type",
        choices=["all_atom", "fa", "faformer", "decoy"],
        default="allatom",
    )
    parser.add_argument("--embedding", type=str)
    parser.add_argument(
        "--ckpt",
        default="ckpts/all_nm_swiglu_kaiming_u_opt_adam_fflayer2_gaussian_silu_noffn_hdim_256_skv24_pkv6_ipa_depth1_head1_psz50/model-all_atom-L_1-lr_0.0001-clip_1.0-epoch_2-all_nm_swiglu_kaiming_u_opt_adam_fflayer2_gaussian_silu_noffn_hdim_256_skv24_pkv6_ipa_depth1_head1_psz50-dout-0.1.ckpt.best",
    )
    parser.add_argument("--save_dir", default="ckpts/tmp")
    parser.add_argument("--noise_scaler", default="cosine")
    parser.add_argument(
        "--precomputed",
        type=str,
        help="Path to precomputed results file (.pt) containing binders, targets, and meter",
    )
    parser.add_argument("--modifier", type=str)
    parser.add_argument("--resolution", type=str, default="all_atom")
    parser.add_argument(
        "--langevin_T", type=int, default=100, help="number of Langevin steps to run"
    )
    parser.add_argument(
        "--langevin_step_size",
        type=float,
        default=0.025,
        help="step size for Langevin dynamics",
    )
    parser.add_argument(
        "--langevin_noise",
        type=float,
        default=0.001,
        help="noise for Langevin dynamics",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="./pretrained_weights",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Number of conformations to sample before selecting the best one"
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="If set with top_k and not non_conservative, select best conformation based on energy instead of CRMSD"
    )
    parser.add_argument(
        "--non_conservative",
        action="store_true",
        help="Use non-conservative sampling mode"
    )

    args = parser.parse_args()
    gpu, ckpt, model_type, save_dir = (
        args.gpu,
        args.ckpt,
        args.model_type,
        args.save_dir,
    )

    eval_db55(
        args,
        gpu,
        ckpt,
        model_type,
        save_dir,
        precomputed_path=None,
        embedding=args.embedding,
    )