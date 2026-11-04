import pdb
import os
import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as L
import matplotlib.pyplot as plt
import numpy as np

from pprint import pprint
from torch import Tensor
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
from typing import Optional
from copy import deepcopy

from models.distance_fm import LatentFlowMatching
from utils import (
    Features,
    recursive_to,
    save_structures_ca,
    batched_solve_mds,
    make_block_weights,
    batched_solve_mds_weighted,
    idx_to_seq,
    kabsch,
    mds_kabsch,
    kabsch_best,
    Meter_Unbound_Bound,
)

from datasets import (
    build_dataset,
    make_protein_collate_fn,
    build_dataloader,
)
from train_contact_vae import LitContactVAE


def coords_from_distances(binder: Features, target: Features, pred_distances: Tensor):
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


class EMACallback(Callback):
    def __init__(self, decay=0.9999, every_n_steps: int = 10):
        self.decay = decay
        self.every_n_steps = every_n_steps
        self.module_pair_list = []

    def on_pretrain_routine_start(self, trainer, pl_module) -> None:
        def forward_wrapper(module, org, ema):
            def forward(*args, **kwargs):
                return org(*args, **kwargs) if module.training else ema(*args, **kwargs)

            return forward

        modules = list(
            filter(
                lambda x: len(list(x[1].parameters())) > 0,
                pl_module.named_children(),
            )
        )

        for name, module in modules:
            ema_module = deepcopy(module)
            self.module_pair_list.append((ema_module, module))
            pl_module.add_module(f"EMA_{name}", ema_module)
            module.forward_bc = module.forward
            module.forward = forward_wrapper(
                module, module.forward_bc, ema_module.forward
            )

    def on_after_backward(self, trainer, pl_module) -> None:
        step = trainer.global_step

        # Only update EMA every N steps
        if step % self.every_n_steps != 0:
            return

        for ema_module, module in self.module_pair_list:
            self._update(
                ema_module,
                module,
                update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m,
            )

    def _update(self, ema_module, module, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                ema_module.state_dict().values(), module.state_dict().values()
            ):
                ema_v.copy_(update_fn(ema_v, model_v))


def get_latent_mean_std(dataloader, vae):
    all_latents = []
    # try:
    #     # per_dim_mean, per_dim_std = torch.load("latent_mean_std_all.pt").values()
    #     per_dim_mean, per_dim_std = torch.load("latent_mean_std.pt").values()
    #     print("Loaded precomputed latent mean and std.")
    # except FileNotFoundError:
    print("Computing latent mean and std over training set...")
    for binder, target in tqdm(dataloader):
        binder = recursive_to(binder, vae.device)
        target = recursive_to(target, vae.device)

        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()  # (B, M)
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        # Distances (Å)
        x = torch.cdist(binder_ca, target_ca)

        # Forward pass through VAE to get latents
        with torch.no_grad():
            # [B, M+N, latent_dim]
            z  = vae.vae.get_latents(x, binder, target)
            z = (
                z * torch.cat([bind_mask, tgt_mask], dim=1).unsqueeze(-1).float()
            )  # Mask out padded residues

            non_zero_latents = z[
                z.abs().sum(dim=-1) > 0
            ]  # [num_residues, latent_dim]
            all_latents.append(non_zero_latents)

    all_latents = torch.cat(all_latents, dim=0)  # [total_residues, latent_dim]

    per_dim_mean = all_latents.mean(dim=0)
    per_dim_std = all_latents.std(dim=0)

    torch.save(
        {"mean": per_dim_mean, "std": per_dim_std},
        "latent_mean_std_all_esm2.pt",
    )

    return per_dim_mean, per_dim_std


class PPIDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for PPI docking."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets and embedding."""
        self.train_ds, self.val_ds, self.sample_val_ds, self.test_ds = build_dataset(
            self.args
        )

        if self.args.embedding:
            print(f"Using embedding from {self.args.embedding}")
            self.embedding = torch.load(
                self.args.embedding, map_location=torch.device("cpu")
            )
            # self.embedding.share_memory_()

        self.collate = make_protein_collate_fn(
            embedding=self.embedding,
            args=self.args,
            device=None,  # Lightning handles device placement
        )

    def train_dataloader(self):
        return build_dataloader(
            self.train_ds,
            batch_size=self.args.diffusion_batch_size,
            shuffle=True,
            collate_fn=self.collate,
            pad_last_to_full=False,
            infinite=False,
            num_workers=8,
            train_mode=False,
        )

    def val_dataloader(self):
        return build_dataloader(
            self.val_ds,
            batch_size=self.args.diffusion_batch_size,
            shuffle=False,
            collate_fn=self.collate,
            pad_last_to_full=False,
            infinite=False,
            num_workers=8,
        )

    def test_dataloader(self):
        return build_dataloader(
            self.test_ds,
            batch_size=self.args.diffusion_batch_size,
            shuffle=False,
            collate_fn=self.collate,
            pad_last_to_full=False,
            infinite=False,
            num_workers=8,
        )


class LitContactDiff(L.LightningModule):
    def __init__(
        self,
        args,
        mean: float,
        std: float,
        vae: LitContactVAE,
        latent_mean: Tensor,
        latent_std: Tensor,
        lr=1e-4,
        hidden_size=128,
        latent_dim=8,
        model_dim=128,
        embedding_size=1280,
        heads=8,
        conditioning_depth=6,
        noise_emb_dim=64,
        protein_mixer_depth=4,
    ):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.vae: LitContactVAE = vae
        self.vae.eval()

        for p in self.vae.parameters():
            p.requires_grad = False

        self.lr = lr
        # self.diffusion_model: LatentDDPM = LatentDDPM(
        self.diffusion_model: LatentFlowMatching = LatentFlowMatching(
            hidden_dim=hidden_size,
            model_dim=model_dim,
            latent_dim=latent_dim,
            emb_size=embedding_size,
            # timesteps=args.num_steps,
            heads=heads,
            conditioning_depth=conditioning_depth,
            noise_emb_dim=noise_emb_dim,
            device=torch.device("cuda"),
            # loss_target=args.diffusion_target,
            protein_mixer_depth=protein_mixer_depth,
            # mixer_depths=args.mixer_depths,
            mean=mean,
            std=std,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )
        self.rotations = []
        self.translation_norms = [0]
        self.meter = Meter_Unbound_Bound()

    def forward(
        self,
        binder: Features,
        target: Features,
        z: Tensor,  # (B, M+N, latent_dim)
        bind_H: Tensor,  # (B, M, hidden_dim)
        target_H: Tensor,  # (B, N, hidden_dim)
        step: int,
        res_mask: Tensor = None,  # (B, M+N, 1) float/bool, 1 = valid)
    ):
        return self.diffusion_model(
            binder, target, z, bind_H, target_H, step, res_mask=res_mask
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.diffusion_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.vae_epochs, eta_min=1e-7
        )
        return [optimizer], [scheduler]

    # -------------------------------
    # TRAINING STEP
    # -------------------------------
    # def training_step(self, batch, batch_idx):
    #     binder, target = batch

    #     binder_ca = binder.X[:, :, 1]
    #     target_ca = target.X[:, :, 1]

    #     # Pad coords and masks
    #     binder_ca, target_ca = pad_coords(
    #         binder_ca, target_ca, self.max_rows, self.max_cols
    #     )

    #     # Distances (Å)
    #     x = torch.cdist(binder_ca, target_ca)

    #     # Forward pass through VAE (always return reconstruction so we can compute metrics)
    #     with torch.no_grad():
    #         z, binder_H, target_H = self.vae.vae.get_latents(x, binder, target)

    #     loss = self.diffusion_model(z, binder_H, target_H, self.global_step)

    #     self.log("train/loss", loss, prog_bar=True, on_step=True)

    #     return loss
    def training_step(self, batch, batch_idx):
        binder, target = batch

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        # Per-residue masks
        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()  # (B, M)
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()  # (B, N)

        # Pairwise mask (B, M, N)
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)

        # Distances (Å)
        x = torch.cdist(binder_ca, target_ca)  # (B, M, N)

        # Forward pass through VAE to get latents & protein features
        with torch.no_grad():
            z = self.vae.vae.get_latents(x, binder, target)
            binder_cond, target_cond = self.vae.vae.get_diff_conditioning(
                binder, target
            )

        # Residue mask for latents: concat binder & target masks
        res_mask = torch.cat([bind_mask, tgt_mask], dim=1).unsqueeze(-1).float()
        # res_mask: (B, M+N, 1)

        loss = self.diffusion_model(
            binder,
            target,
            z,
            binder_cond,
            target_cond,
            # self.global_step,
            res_mask=res_mask,
            guidance_scale=5.0
        )

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        return loss

    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx):
        binder, target = batch

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        B = binder_ca.size(0)

        # Pad masks based on A

        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()  # (B, M)
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()  # (B, N)
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)  # (B, M, N)

        # Ground-truth distances (Å)
        x = torch.cdist(binder_ca, target_ca)  # (B, M, N)

        # Always ask VAE protein encoder to return latents
        with torch.no_grad() as X, torch.inference_mode() as Y:
            z = self.vae.vae.get_latents(x, binder, target)
            binder_cond, target_cond = self.vae.vae.get_diff_conditioning(
                binder, target
            )

        # Residue mask for latents
        res_mask = torch.cat([bind_mask, tgt_mask], dim=1).unsqueeze(-1).float()

        with torch.no_grad() as X, torch.inference_mode() as Y:
            val_loss = self.diffusion_model(
                binder,
                target,
                z,
                binder_cond,
                target_cond,
                # self.global_step,
                res_mask=res_mask,
            )
        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Save visualization for batch 0
        # if batch_idx == 3:
        with torch.no_grad() as X, torch.inference_mode() as Y:
            samples = self.diffusion_model.sample(
                binder,
                target,
                binder_cond,
                target_cond,
                B,
                steps=args.num_steps,
                guidance_scale=5.0,
            )

            # Decode samples
            x_recon_phys = self.vae.vae.decode(
                samples, binder, target, upscale=True
            )  # (B, M, N)

            x_recon_vae = self.vae.vae.decode(z, binder, target, upscale=True)

        # Masked MSE over valid residue pairs only
        mse_gt = F.mse_loss(
            x_recon_phys[pair_mask],
            x[pair_mask],
        )
        mse_vae = F.mse_loss(
            x_recon_vae[pair_mask],
            x[pair_mask],
        )

        # Maksed MSE over contact (within 8A) pairs only
        within_8_mask = x < 8.0
        mse_gt_8A = F.mse_loss(
            x_recon_phys[pair_mask & within_8_mask],
            x[pair_mask & within_8_mask],
        )

        # NDCG for top-k contacts
        top_k = 50
        ndcg_scores = []
        for i in range(B):
            true_dists = x[i][pair_mask[i]]  # (num_valid_pairs,)
            pred_dists = x_recon_phys[i][pair_mask[i]]  # (num_valid_pairs,)

            if true_dists.size(0) < top_k:
                continue

            true_sorted, true_indices = torch.topk(
                -true_dists, k=top_k
            )  # smallest distances
            pred_sorted = pred_dists[true_indices]

            # Compute DCG
            gains = 1.0 / torch.log2(torch.arange(2, top_k + 2, device=true_dists.device).float())
            ideal_gains = gains  # since true_sorted is perfectly ranked

            dcg = (gains / (pred_sorted.argsort().argsort() + 2).float()).sum()
            idcg = ideal_gains.sum()

            ndcg = dcg / idcg
            ndcg_scores.append(ndcg.item())

        avg_ndcg = np.mean(ndcg_scores) if len(ndcg_scores) > 0 else 0.0
        self.log("val/ndcg_top50", avg_ndcg, prog_bar=True, on_step=True, on_epoch=True)

        self.log("val/mse_batch1", mse_gt, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "val/mse_vae_batch1",
            mse_vae,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "val/mse_8A_batch1",
            mse_gt_8A,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        os.makedirs(
            f"{self.args.save_path}/{self.args.exp_name}/diff_outputs",
            exist_ok=True,
        )
        generated = x_recon_phys.cpu()

        x_vis = x.clone()
        if self.args.target_mode == "contact":
            x_vis = self.vae.bin_distances(x_vis)

        # Optional: zero out padded regions before plotting
        x_vis = x_vis.clone()
        gen_vis = generated.clone()
        x_vis[~pair_mask] = 0.0
        gen_vis[~pair_mask] = 0.0

        Bv = x_vis.size(0)
        # plt.figure(figsize=(15, 5))
        # plt.suptitle(f"(VAE o Diffusion) Reconstructions @ step {self.global_step}")
        # for i in range(Bv):
        #     plt.subplot(2, Bv, i + 1)
        #     plt.imshow(x_vis[i].float().cpu().numpy())
        #     plt.title(f"Original {i}")
        #     plt.axis("off")

        #     min_value = 0
        #     max_value = torch.max(gen_vis[i]).item()

        #     colorbar_ticks = [min_value, max_value / 2, max_value]
        #     cbar = plt.colorbar(ticks=colorbar_ticks)
        #     cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in colorbar_ticks])

        #     plt.subplot(2, Bv, Bv + i + 1)
        #     plt.imshow(gen_vis[i].float().cpu().numpy())
        #     plt.title(f"Recon {i}")
        #     plt.axis("off")
        #     cbar = plt.colorbar(ticks=colorbar_ticks)
        #     cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in colorbar_ticks])

        # Global min/max from ground-truth distances (masked)
        # fig = plt.figure(figsize=(15, 5))
        # plt.suptitle(f"(VAE ∘ Diffusion) Reconstructions @ step {self.global_step}")

        # pred_coords = coords_from_distances(binder, target, generated)  # (B, M+N, 3)
        # pred_coords = align_by_receptor(binder, target, pred_coords)

        # pred_binder_coords = pred_coords[:, : binder_ca.size(1), :]  # (B, M, 3)
        # pred_binder_coords = pred_coords.view(B, -1, 4, 3)[:, :, 1, :]
        # pred_binder_coords = pred_binder_coords[:, : binder_ca.size(1), :].view(
        #     B, -1, 3
        # )

        # pred_target_coords = pred_coords[:, binder_ca.size(1) :, :]  # (B, N, 3)
        # for i in range(Bv):
        #     binder_seq = idx_to_seq(binder.Si[i].view(1, -1))
        #     target_seq = idx_to_seq(target.Si[i].view(1, -1))

        #     # save_structures_ca(
        #     #     binder_seq,
        #     #     target_seq,
        #     #     pred_binder_coords[i],
        #     #     pred_target_coords[i],
        #     #     f"{self.args.save_path}/{self.args.exp_name}/diff_outputs/step_{self.global_step}_batch1.pdb",
        #     #     binder.pdbs[i],
        #     #     "",
        #     # )

            # binder_mask = binder.A[i, :, 1] > 0
            # target_mask = target.A[i, :, 1] > 0
            # # binder_mask = binder.A[i].view(-1) > 0
            # # target_mask = target.A[i].view(-1) > 0

        #     lrmsd, irmsd, crmsd, dockq = self.meter.update_rmsd(
        #         pred_binder_coords[i][binder_mask.cpu()].view(-1, 3).cpu(),
        #         # target.X[i][:, 1, :][target_mask].view(-1, 3).cpu(),
        #         # binder.X[i][:, 1, :][binder_mask].view(-1, 3).cpu(),
        #         # target.X[i][:, 1, :][target_mask].view(-1, 3).cpu(),
        #         target.X[i][:, 1, :].view(-1, 3)[target_mask].cpu(),
        #         binder.X[i][:, 1, :].view(-1, 3)[binder_mask].cpu(),
        #         target.X[i][:, 1, :].view(-1, 3)[target_mask].cpu(),
        #     )

        #     summary_stats = self.meter.get_summary_stats(reduction_rmsd="std")
        #     dockq_mean = summary_stats["including_failed"]["dockq_mean"]
        #     dockq_median = summary_stats["including_failed"]["dockq_median"]
        #     dockq_std = summary_stats["including_failed"]["dockq_std"]

        #     print(
        #         f"lrmsd: {lrmsd:.3f}, irmsd: {irmsd:.3f}, crmsd: {crmsd:.3f}, dockq: {dockq:.3f}"
        #     )
        #     print(f"RMSD stats after step {self.global_step}, batch {i}:")
        #     print(
        #         f"{dockq_mean:.3f} +/- {dockq_std:.3f} (mean +/- std); median: {dockq_median:.3f}"
        #     )
        #     print("-----------------------------------------------------")
        #     # print(f"RMSD stats after step {self.global_step}, batch {i}:")
        #     # pprint(summary_stats)

        # im = None
        # for i in range(Bv):
        #     gt_vals = x[i][pair_mask[i]]

        #     vmin = gt_vals.min().item()
        #     vmax = gt_vals.max().item()

        #     # Ground-truth distances
        #     ax = plt.subplot(3, Bv, i + 1)
        #     im = ax.imshow(
        #         x_vis[i].float().cpu().numpy(),
        #         vmin=vmin,
        #         vmax=vmax,
        #     )
        #     ax.set_title(f"Original {i}")
        #     ax.axis("off")
        #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        #     # Diffusion + VAE reconstructions
        #     ax = plt.subplot(3, Bv, Bv + i + 1)
        #     ax.imshow(
        #         gen_vis[i].float().cpu().numpy(),
        #         vmin=vmin,
        #         vmax=vmax,
        #     )
        #     ax.set_title(f"Recon {i}")
        #     ax.axis("off")
        #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        #     # VAE reconstructions with ground-truth latents
        #     ax = plt.subplot(3, Bv, 2 * Bv + i + 1)
        #     ax.imshow(
        #         x_recon_vae[i].float().cpu().numpy(),
        #         vmin=vmin,
        #         vmax=vmax,
        #     )

        #     ax.set_title(f"VAE Recon {i}")
        #     ax.axis("off")
        #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.savefig(
            f"{self.args.save_path}/{self.args.exp_name}/diff_outputs/step_{self.global_step}.png"
        )
        plt.close()

        self.log("test/loss", mse_gt, prog_bar=True, on_step=True)

    def on_test_end(self):
        summary_stats = self.meter.get_summary_stats(reduction_rmsd="std")

        pprint(summary_stats)

    def validation_step(self, batch, batch_idx):
        binder, target = batch

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        B = binder_ca.size(0)

        # Pad masks based on A

        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()  # (B, M)
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()  # (B, N)
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)  # (B, M, N)

        # Ground-truth distances (Å)
        x = torch.cdist(binder_ca, target_ca)  # (B, M, N)

        # Always ask VAE protein encoder to return latents
        z = self.vae.vae.get_latents(x, binder, target)
        binder_cond, target_cond = self.vae.vae.get_diff_conditioning(binder, target)

        # Residue mask for latents
        res_mask = torch.cat([bind_mask, tgt_mask], dim=1).unsqueeze(-1).float()

        val_loss = self.diffusion_model(
            binder,
            target,
            z,
            binder_cond,
            target_cond,
            # self.global_step,
            res_mask=res_mask,
        )
        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

        # Save visualization for batch 0
        if batch_idx == 3:
            samples = self.diffusion_model.sample(
                binder,
                target,
                binder_cond,
                target_cond,
                B,
                steps=args.num_steps,
                guidance_scale=5.0,
            )

            # Decode samples
            x_recon_phys = self.vae.vae.decode(
                samples, binder, target, upscale=True
            )  # (B, M, N)

            x_recon_vae = self.vae.vae.decode(z, binder, target, upscale=True)

            # Masked MSE over valid residue pairs only
            mse_gt = F.mse_loss(
                x_recon_phys[pair_mask],
                x[pair_mask],
            )
            mse_vae = F.mse_loss(
                x_recon_vae[pair_mask],
                x[pair_mask],
            )
            self.log(
                "val/mse_batch1", mse_gt, prog_bar=True, on_step=False, on_epoch=True
            )
            self.log(
                "val/mse_vae_batch1",
                mse_vae,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            os.makedirs(
                f"{self.args.save_path}/{self.args.exp_name}/diff_outputs",
                exist_ok=True,
            )
            generated = x_recon_phys.cpu()

            x_vis = x.clone()
            if self.args.target_mode == "contact":
                x_vis = self.vae.bin_distances(x_vis)

            # Optional: zero out padded regions before plotting
            x_vis = x_vis.clone()
            gen_vis = generated.clone()
            x_vis[~pair_mask] = 0.0
            gen_vis[~pair_mask] = 0.0

            Bv = x_vis.size(0)
            # plt.figure(figsize=(15, 5))
            # plt.suptitle(f"(VAE o Diffusion) Reconstructions @ step {self.global_step}")
            # for i in range(Bv):
            #     plt.subplot(2, Bv, i + 1)
            #     plt.imshow(x_vis[i].float().cpu().numpy())
            #     plt.title(f"Original {i}")
            #     plt.axis("off")

            #     min_value = 0
            #     max_value = torch.max(gen_vis[i]).item()

            #     colorbar_ticks = [min_value, max_value / 2, max_value]
            #     cbar = plt.colorbar(ticks=colorbar_ticks)
            #     cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in colorbar_ticks])

            #     plt.subplot(2, Bv, Bv + i + 1)
            #     plt.imshow(gen_vis[i].float().cpu().numpy())
            #     plt.title(f"Recon {i}")
            #     plt.axis("off")
            #     cbar = plt.colorbar(ticks=colorbar_ticks)
            #     cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in colorbar_ticks])

            # Global min/max from ground-truth distances (masked)
            fig = plt.figure(figsize=(15, 5))
            plt.suptitle(f"(VAE ∘ Diffusion) Reconstructions @ step {self.global_step}")

            im = None
            for i in range(Bv):
                gt_vals = x[i][pair_mask[i]]

                vmin = gt_vals.min().item()
                vmax = gt_vals.max().item()

                # Ground-truth distances
                ax = plt.subplot(3, Bv, i + 1)
                im = ax.imshow(
                    x_vis[i].float().cpu().numpy(),
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"Original {i}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Diffusion + VAE reconstructions
                ax = plt.subplot(3, Bv, Bv + i + 1)
                ax.imshow(
                    gen_vis[i].float().cpu().numpy(),
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"Recon {i}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # VAE reconstructions with ground-truth latents
                ax = plt.subplot(3, Bv, 2 * Bv + i + 1)
                ax.imshow(
                    x_recon_vae[i].float().cpu().numpy(),
                    vmin=vmin,
                    vmax=vmax,
                )

                ax.set_title(f"VAE Recon {i}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.savefig(
                f"{self.args.save_path}/{self.args.exp_name}/diff_outputs/step_{self.global_step}.png"
            )
            plt.close()


def main(args):
    dm = PPIDataModule(args)

    # Compute dataset statistics
    # mean, std = get_mean_std_distance(
    #     dm.train_dataloader(),
    #     which_dists=args.which_dists,
    #     use_logscale=args.use_logscale,
    # )
    mean, std = 34.9249697902463, 17.601150743644386
    vae = LitContactVAE.load_from_checkpoint(args.ckpt, args=args, mean=mean, std=std)

    dm.setup()
    latent_mean, latent_std = get_latent_mean_std(dm.train_dataloader(), vae)
    # latent_mean, latent_std = torch.load("latent_mean_std.pt").values()

    model = LitContactDiff(
        args,
        mean=mean,
        std=std,
        vae=vae,
        latent_mean=latent_mean,
        latent_std=latent_std,
        lr=args.diffusion_lr,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        model_dim=args.model_dim,
        embedding_size=args.embedding_size,
        heads=args.heads,
        conditioning_depth=args.conditioning_depth,
        noise_emb_dim=args.noise_emb_dim,
        protein_mixer_depth=args.protein_mixer_depth,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.save_path}/{args.exp_name}/diff_ckpts",
            filename="epoch-{epoch}",
            save_top_k=-1,
            every_n_epochs=3,
            save_last=True,
        ),
        EMACallback(decay=0.999, every_n_steps=1),
    ]

    logger = (
        WandbLogger(project="contact_latent_diff_diffusion", name=args.exp_name)
        if args.use_wandb
        else None
    )

    trainer = Trainer(
        accelerator="cuda",
        devices=args.devices,
        max_epochs=args.vae_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=10.0,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        precision="bf16-mixed",  # AMP
    )

    if args.tune_lr:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=1e-5, max_lr=4e-3)

        fig = lr_finder.plot(suggest=True)
        os.makedirs(f"{args.save_path}/{args.exp_name}/diff_outputs", exist_ok=True)
        fig.savefig(f"{args.save_path}/{args.exp_name}/diff_outputs/lr_finder.png")
        new_lr = lr_finder.suggestion()
        print(f"Setting learning rate to {new_lr}")
        model.lr = new_lr

    if args.validate:
        model = LitContactDiff.load_from_checkpoint(
            args.diffusion_ckpt,
            args=args,
            mean=mean,
            std=std,
            vae=vae,
            latent_mean=latent_mean,
            latent_std=latent_std,
            lr=args.diffusion_lr,
            hidden_size=args.hidden_size,
            latent_dim=args.latent_dim,
            model_dim=args.model_dim,
            embedding_size=args.embedding_size,
            heads=args.heads,
            conditioning_depth=args.conditioning_depth,
            noise_emb_dim=args.noise_emb_dim,
            protein_mixer_depth=args.protein_mixer_depth,
        )
        # trainer.validate(model, datamodule=dm)
        dm.setup(stage="test")
        trainer.test(model, dataloaders=dm.test_dataloader())
        return

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="GPU devices to use for training",
    )
    parser.add_argument("--train_path", default="data/DIPS-Het/data.pkl")
    parser.add_argument("--save_path", default="diff_results/")
    parser.add_argument(
        "--patch_size", type=int, default=50, help="Patch size for cropping"
    )
    parser.add_argument("--resolution", type=str, default="backbone")
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit number of samples for debugging, -1 for no limit",
    )
    parser.add_argument("--num_val_sample", type=int, default=30)
    parser.add_argument("--embedding", default="dips_het_embedding_E1_600M.pt")
    parser.add_argument(
        "--embedding_size", type=int, default=1280, help="Residue embedding size"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Whether to use wandb logging"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="contact_latent_diffusion",
        help="Experiment name",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to model checkpoint"
    )

    # Training parameters
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Latent diffusion dimension"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=8, help="Latent diffusion dimension"
    )
    parser.add_argument(
        "--model_dim", type=int, default=128, help="Diffusion model dimension"
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=7,
        help="Number of distance bins for contact map",
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--protein_encoder_depth",
        type=int,
        default=5,
        help="Depth of protein encoder network",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--vae_epochs",
        type=int,
        default=100,
        help="Number of training epochs for the VAE",
    )
    parser.add_argument(
        "--tune_lr",
        action="store_true",
        help="Whether to use LR finder for VAE training",
    )

    # VAE parameters
    parser.add_argument(
        "--vae_lr", type=float, default=1e-5, help="Learning rate for VAE"
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.001, help="Weight for KL divergence loss"
    )
    parser.add_argument("--vae_batch_size", type=int, default=16, help="VAE batch size")
    parser.add_argument(
        "--channel_depths",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Channel depths for VAE",
    )
    parser.add_argument(
        "--use_symmetry",
        action="store_true",
        help="Whether to use norms for distance symmetry or predict distances directly",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["distances", "contact"],
        default="distances",
        help="Target mode for VAE",
    )
    parser.add_argument(
        "--pred_type",
        type=str,
        choices=["direct", "ipa"],
        default="direct",
        help="Prediction type for the distance/contact map. 'direct' predicts distances/contacts directly, 'ipa' uses invariant point attention for coordinates.",
    )
    parser.add_argument(
        "--use_log_distances",
        action="store_true",
        help="Whether to use log-transformed distances",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Whether to run validation only",
    )

    # Diffusion model parameters
    parser.add_argument(
        "--diffusion_lr",
        type=float,
        default=1e-4,
        help="Learning rate for diffusion model",
    )
    parser.add_argument(
        "--diffusion_batch_size", type=int, default=16, help="Diffusion batch size"
    )
    parser.add_argument(
        "--conditioning_depth",
        type=int,
        default=6,
        help="Number of layers in diffusion model",
    )
    parser.add_argument(
        "--protein_mixer_depth",
        type=int,
        default=4,
        help="Depth of protein mixer before cross-attention",
    )
    parser.add_argument(
        "--mixer_depths",
        type=int,
        nargs="+",
        default=[2, 2],
        help="Mixer depths before and after cross-attention",
    )
    parser.add_argument(
        "--noise_emb_dim",
        type=int,
        default=64,
        help="Dimension of noise embedding in diffusion model",
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--diffusion_ckpt",
        type=str,
        default=None,
        help="Path to diffusion model checkpoint",
    )
    parser.add_argument(
        "--diffusion_target",
        type=str,
        choices=["noise", "x0"],
        default="noise",
        help="Diffusion target prediction",
    )
    parser.add_argument(
        "--which_dists",
        type=str,
        choices=["all", "between"],
        default="all",
        help="Which distances to use for computing latent statistics",
    )
    parser.add_argument(
        "--use_logscale",
        action="store_true",
        help="Whether to use log scale distances for computing distance statistics",
    )
    parser.add_argument(
        "--use_nll",
        action="store_true",
        help="Whether to use NLL for distance prediction",
    )

    args = parser.parse_args()
    main(args)
