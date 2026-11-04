#! parallel
import copy
import pdb
import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.utils as nn_utils
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict

import wandb
from ema_pytorch import EMA

from datasets import (
    build_dataset,
    make_protein_collate_fn,
    build_dataloader,
)
from utils import NoiseSchedule, recursive_to
from args import parse_args
from models import AllAtomEnergyModel
from sample import evaluate_samples

# set random seeds for reproducibility
torch.manual_seed(1212010)
np.random.seed(1212010)
random.seed(1212010)

torch.set_float32_matmul_precision("high")


def sd(obj):
    return obj.module.state_dict() if hasattr(obj, "module") else obj.state_dict()


def setup_environment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.use_wandb:
        wandb.init(project="ppi_docking", name=args.exp_name, config=vars(args))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    return device


def get_dips_train_val_test_keys():
    train_keys = set([x for x in open("data/DIPS-Het/train.txt").read().splitlines()])
    val_keys = set([x for x in open("data/DIPS-Het/val.txt").read().splitlines()])
    test_keys = set([x for x in open("data/DIPS-Het/test.txt").read().splitlines()])
    return train_keys, val_keys, test_keys


def load_data(args, device=None):
    train_ds, val_ds, sample_val, test_ds = build_dataset(args)
    if args.embedding:
        print(f"Using embedding from {args.embedding}")
        embedding = torch.load(args.embedding, map_location=torch.device("cpu"))
    else:
        embedding = None

    collate = make_protein_collate_fn(
        embedding=embedding, args=args, device=device
    )

    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pad_last_to_full=True,
        infinite=True,
        num_workers=8,
        train_mode=True,
    )
    val_loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        pad_last_to_full=True,
        infinite=False,
        num_workers=8,
    )
    val_sample_loader = build_dataloader(
        train_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
        pad_last_to_full=True,
        infinite=False,
        num_workers=0,
    )
    # test_loader = build_dataloader(
    #     test_ds,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     collate_fn=collate,
    #     pad_last_to_full=True,
    #     infinite=False,
    #     num_workers=4
    # )

    return train_loader, val_loader, val_sample_loader


def init_model(args, device):
    model = AllAtomEnergyModel(args, device, resolution=args.resolution)
    coarse_model = AllAtomEnergyModel(args, device, resolution="CA")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Initialized model with resolution {args.resolution} and {num_params} parameters"
    )

    return model.to(device), coarse_model.to(device)


def init_optimizer(parameters, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=args.lr, fused=True, weight_decay=1e-5)
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0, fused=True)
    elif args.optimizer == "radam":
        return torch.optim.RAdam(parameters, lr=args.lr, weight_decay=0.0)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(
            parameters, lr=args.lr, momentum=0.9, weight_decay=0.0, fused=True
        )
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")


def make_inf_iterator(data, args):
    while True:
        # random.shuffle(data)
        for i in range(0, len(data), args.batch_size):
            # batch = data[i: i + args.batch_size]
            batch = [copy.deepcopy(data[i]) for _ in range(args.batch_size)]
            curr_bs = len(batch)
            if curr_bs < args.batch_size:
                batch.extend(random.choices(data, k=args.batch_size - curr_bs))
            yield batch


def save_ckpt(
    path, step, best_vals, args, optimizer, scaler, ema, model_dp, model_coarse_dp
):
    ckpt = {
        "step": step,
        "model": (
            model_dp.module.state_dict()
            if isinstance(model_dp, nn.DataParallel)
            else model_dp.state_dict()
        ),
        "model_coarse": (
            model_coarse_dp.module.state_dict()
            if isinstance(model_coarse_dp, nn.DataParallel)
            else model_coarse_dp.state_dict()
        ),
        "ema": ema.state_dict(),  # wrapper state (contains shadow model weights)
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        # "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best": best_vals,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_ckpt(path, ema, model_dp, model_coarse_dp, optimizer=None, scaler=None):
    data = torch.load(path, map_location="cpu")

    (
        model_dp.module if isinstance(model_dp, nn.DataParallel) else model_dp
    ).load_state_dict(data["model"])
    (
        model_coarse_dp.module
        if isinstance(model_coarse_dp, nn.DataParallel)
        else model_coarse_dp
    ).load_state_dict(data["model_coarse"])
    ema.load_state_dict(data["ema"])

    if optimizer is not None and "optimizer" in data and data["optimizer"] is not None:
        optimizer.load_state_dict(data["optimizer"])
    if scaler is not None and "scaler" in data and data["scaler"] is not None:
        scaler.load_state_dict(data["scaler"])

    step = data.get("step", 0)
    best_vals = data.get("best", {})
    args_saved = data.get("args", None)
    return step, best_vals, args_saved


def train_one_step_dp(
    model_dp: nn.Module,
    model_coarse_dp: nn.Module,
    data_iterator,
    embedding: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    args,
    device,
    noise_schedule: NoiseSchedule,
    scaler: GradScaler,
    fixed_t=None,
    step=0,
):
    model_dp.train()
    model_coarse_dp.train()

    binder, target = next(data_iterator)
    binder, target = recursive_to(binder, device), recursive_to(target, device)

    rot_s = torch.zeros(args.batch_size, 1, device=device)
    tr_s = torch.zeros(args.batch_size, 1, device=device)
    time_steps = torch.zeros(args.batch_size, 1, device=device)

    for idx in range(args.batch_size):
        t = fixed_t if fixed_t is not None else np.random.uniform()
        time_steps[idx] = t
        t_tr, t_rot = t, t
        tr_s_, rot_s_ = noise_schedule(t_tr, t_rot)
        rot_s[idx] = rot_s_
        tr_s[idx] = tr_s_

    use_amp = scaler is not None and scaler.is_enabled()
    # mask = torch.arange(args.batch_size, device=device).unsqueeze(-1).repeat(1, binder[0].view(args.batch_size, -1, 3).shape[1])
    with autocast("cuda", enabled=use_amp):
        loss, loss_dict = model_dp(
            binder, target, tr_s, rot_s, args.resolution, time_steps, get_loss_dict=True
        )

    return loss, loss_dict


@torch.no_grad()
def validate_with_ema(
    ema_dp: nn.Module,
    model_coarse_dp: nn.Module,
    data,
    embedding: Dict[str, torch.Tensor],
    args,
    noise_schedule: NoiseSchedule,
    device,
    fixed_t=0.1,
):
    ema_dp.eval()
    model_coarse_dp.eval()

    total_loss = 0.0
    total_data = 0

    # for i in iters:
    for batch in tqdm(data):
        binder, target = batch
        curr_bs = binder.X.shape[0]
        binder, target = recursive_to(binder, device), recursive_to(target, device)

        rot_s = torch.zeros(curr_bs, 1, device=device)
        tr_s = torch.zeros(curr_bs, 1, device=device)
        time_steps = torch.zeros(curr_bs, 1, device=device)

        for idx in range(curr_bs):
            t = fixed_t if fixed_t is not None else np.random.uniform()
            time_steps[idx] = t
            t_tr, t_rot = t, t
            tr_s_, rot_s_ = noise_schedule(t_tr, t_rot)
            rot_s[idx] = rot_s_
            tr_s[idx] = tr_s_

        loss = ema_dp(binder, target, tr_s, rot_s, args.resolution, time_steps).mean()
        total_loss += loss.item()
        total_data += curr_bs

    return total_loss / total_data if total_data else float("inf")


def main():
    args = parse_args()
    device = setup_environment(args)

    if torch.cuda.is_available():
        # keep linalg kernels loaded in main thread (perf warm-up)
        torch.linalg.inv(torch.eye(1, device="cuda"))

    # Init models
    model_raw, model_coarse_raw = init_model(args, device)

    # DataParallel wrapping mirrors the first script
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model_dp = nn.DataParallel(model_raw).to(device)
        model_coarse_dp = nn.DataParallel(model_coarse_raw).to(device)
        parallel_batches = torch.cuda.device_count()
    else:
        model_dp = model_raw.to(device)
        model_coarse_dp = model_coarse_raw.to(device)
        print("Using single GPU or CPU")

    # EMA wrapper over the MODULE (not params), like your first script
    ema = EMA(
        model_dp if isinstance(model_dp, nn.DataParallel) else model_raw,
        beta=args.ema_decay,
        update_every=10,
    ).to(device)
    ema_model = ema.ema_model

    # DP for EMA model too
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        ema_dp = nn.DataParallel(ema_model).to(device)
    else:
        ema_dp = ema_model.to(device)

    # Optimizer (all params: fine + coarse)
    all_parameters = list(model_raw.parameters())
    optimizer = init_optimizer(all_parameters, args)

    # Optional scheduler
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.num_steps, eta_min=0.0)
    scheduler = None

    # AMP scaler
    use_amp = getattr(args, "use_amp", True)
    scaler = GradScaler("cuda", enabled=use_amp and torch.cuda.is_available())

    # Load dataset
    train_data, val_data, sample_val_data = load_data(args, device)
    embedding = (
        torch.load(args.embedding, map_location=torch.device("cpu"))
        if args.embedding is not None
        else None
    )

    # Resume (supports new EMA-style checkpoints)
    begin_step = 0
    best_val_loss = float("inf")
    best_val_crmsd = float("inf")

    if getattr(args, "ckpt", None):
        print(f"Loading checkpoint: {args.ckpt}")
        begin_step, best_vals, args_saved = load_ckpt(
            args.ckpt,
            ema,
            model_dp,
            model_coarse_dp,
            optimizer=optimizer,
            scaler=scaler,
        )
        if best_vals:
            best_val_loss = best_vals.get("loss", best_val_loss)
            best_val_crmsd = best_vals.get("crmsd", best_val_crmsd)
        if args_saved is not None:
            # Not overwriting args; but you could reconcile if needed
            print("Loaded saved args for reference (not overriding current CLI args).")

    # Iterator & progress
    train_iterator = train_data
    pbar = range(begin_step, args.num_steps)
    if args.verbose:
        pbar = tqdm(pbar, desc="Training")

    noise_schedule = NoiseSchedule(
        args.tr_s_min, args.tr_s_max, args.rot_s_min, args.rot_s_max
    )

    gradient_accumulate_every = getattr(args, "gradient_accumulate_every", 1)
    patience_counter = 0
    patience = args.patience
    fixed_t = None  # or set a float to fix time

    for step in pbar:
        # optional debug dumps like your original script
        if step > 500:
            if hasattr(model_raw, "cos_sim_list"):
                torch.save(model_raw.cos_sim_list, "cos_sim_list.pt")
            if hasattr(model_raw, "norm_diff_list"):
                torch.save(model_raw.norm_diff_list, "norm_diff_list.pt")
            if hasattr(model_raw, "norm_list"):
                torch.save(model_raw.norm_list, "norm_list.pt")

        # forward pass
        loss, loss_dict = train_one_step_dp(
            model_dp,
            model_coarse_dp,
            train_iterator,
            embedding,
            optimizer,
            args,
            device,
            noise_schedule,
            scaler,
            fixed_t=fixed_t,
            step=step,
        )
        loss = loss.mean()

        # backward (AMP + accumulation)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss / gradient_accumulate_every).backward()
            scaler.unscale_(optimizer)
        else:
            (loss / gradient_accumulate_every).backward()

        unclipped_grad_norm = None
        if (step + 1) % gradient_accumulate_every == 0:
            max_norm = args.clip_norm if getattr(args, "clip_norm", -1) != -1 else 1e6
            unclipped_grad_norm = nn_utils.clip_grad_norm_(
                model_dp.parameters(), max_norm
            )

            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # EMA update matches first script
            ema.update()

            if scheduler is not None:
                scheduler.step()

        # logging
        if args.use_wandb:
            log_dict = {
                "unclipped_grad_norm": (
                    unclipped_grad_norm.item()
                    if unclipped_grad_norm is not None
                    and hasattr(unclipped_grad_norm, "item")
                    else (
                        float(unclipped_grad_norm)
                        if unclipped_grad_norm is not None
                        else None
                    )
                ),
                "step": step,
            }
            log_dict |= {
                f"train_{k}": v.detach().item() if hasattr(v, "detach") else v
                for k, v in loss_dict.items()
            }
            if step % 20 == 0:
                norms_per_module = {
                    name: param.grad.norm().detach().item()
                    if param.grad is not None
                    else None
                    for name, param in model_dp.named_parameters()
                }
                log_dict |= {
                    f"norms/{k}": v for k, v in norms_per_module.items() if v is not None
                }
            wandb.log(log_dict)

        optimizer.zero_grad()

        if args.verbose:
            num_trained = (step + 1) * args.batch_size
            colored_description = f"\033[92mStep {step}, samples: {num_trained}, Loss: {loss.item():.4f}\033[0m"
            if isinstance(pbar, tqdm):
                pbar.set_description(colored_description)
            else:
                print(colored_description)

        # evaluation & checkpointing
        if (step + 1) % args.eval_every == 0:
            ema_dp.eval()
            val_loss = validate_with_ema(
                ema_dp,
                model_coarse_dp,
                val_data,
                embedding,
                args,
                noise_schedule,
                device,
                fixed_t=fixed_t,
            )
            val_lrmsd, val_irmsd, val_crmsd, val_dockq = evaluate_samples(
                args,
                device,
                sample_val_data,
                embedding,
                model_dp,
                ema_dp,
                noise_schedule,
                args.save_dir,
                step,
            )
            if val_dockq == 0.0:
                continue

            print(
                f"\033[92mStep {step}, val_loss={val_loss:.4f}, "
                f"val_lrmsd={val_lrmsd:.4f}, val_irmsd={val_irmsd:.4f}, "
                f"val_crmsd={val_crmsd:.4f}, val_dockq={val_dockq:.4f}\033[0m"
            )

            if args.use_wandb:
                wandb.log(
                    {
                        "val_loss": val_loss,
                        "val_lrmsd": val_lrmsd,
                        "val_irmsd": val_irmsd,
                        "val_crmsd": val_crmsd,
                        "val_dockq": val_dockq,
                        "step": step + 1,
                    }
                )

            # save last
            save_ckpt(
                os.path.join(args.save_dir, "last.pt"),
                step + 1,
                {"loss": best_val_loss, "crmsd": best_val_crmsd},
                args,
                optimizer,
                scaler,
                ema,
                model_dp,
                model_coarse_dp,
            )

            # track bests
            improved_loss = val_loss < best_val_loss
            if improved_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.4f}")

            improved_crmsd = val_crmsd < best_val_crmsd
            if improved_crmsd:
                best_val_crmsd = val_crmsd
                patience_counter = 0

                # save best-by-crmsd
                save_ckpt(
                    os.path.join(args.save_dir, f"best_step_{step + 1}.pt"),
                    step + 1,
                    {"loss": best_val_loss, "crmsd": best_val_crmsd},
                    args,
                    optimizer,
                    scaler,
                    ema,
                    model_dp,
                    model_coarse_dp,
                )
                save_ckpt(
                    os.path.join(args.save_dir, "best.pt"),
                    step + 1,
                    {"loss": best_val_loss, "crmsd": best_val_crmsd},
                    args,
                    optimizer,
                    scaler,
                    ema,
                    model_dp,
                    model_coarse_dp,
                )
                print(f"New best validation avg CRMSD: {val_crmsd:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {step + 1} steps")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    print(f"Best validation CRMSD: {best_val_crmsd:.4f}")
                    break

    print("Training completed!")
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
