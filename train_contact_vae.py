import pdb
import os
import wandb
import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from models.distance_vae import (
    NonConvProteinContactMapVAE,
    TriangularDistanceMapVAE,
    DistanceMapVAE,
    TokenDistanceVAE,
    CVAE,
)
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.weight_averaging import EMAWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from datasets import (
    build_dataset,
    make_protein_collate_fn,
    build_dataloader,
)
from latent_utils import (
    pad_coords,
    pad_masks,
    compute_class_weights,
    compute_contact_metrics,
    get_mean_std_distance,
)
from utils import recursive_to


def get_latent_mean_std(dataloader, vae):
    all_latents = []
    print("Computing latent mean and std over training set...")
    try:
        stats = torch.load("latent_mean_std.pt")
        print(
            f"Loaded precomputed latent mean and std from latent_mean_std.pt"
        )
        return stats["mean"], stats["std"]
    except:
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
                z = vae.vae.get_latents(x, binder, target)
                z = (
                    z * torch.cat([bind_mask, tgt_mask], dim=1).unsqueeze(-1).float()
                )  # Mask out padded residues

                non_zero_latents = z[z.abs().sum(dim=-1) > 0]  # [num_residues, latent_dim]
                all_latents.append(non_zero_latents)

        all_latents = torch.cat(all_latents, dim=0)  # [total_residues, latent_dim]

        per_dim_mean = all_latents.mean(dim=0)
        per_dim_std = all_latents.std(dim=0)

        torch.save(
            {"mean": per_dim_mean, "std": per_dim_std},
            "latent_mean_std.pt",
        )

    return per_dim_mean, per_dim_std


def mse_loss(pred, target, mask):
    loss = ((pred - target) * mask) ** 2  # B x N x M
    loss = loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2))  # B
    loss = loss.mean()

    return loss


class PPIDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds, self.val_ds, _, self.test_ds = build_dataset(self.args)

        if self.args.embedding:
            self.embedding = torch.load(self.args.embedding, map_location="cpu")
        else:
            self.embedding = None

        self.collate = make_protein_collate_fn(
            embedding=self.embedding, args=self.args, device=None
        )

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return build_dataloader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collate,
            pad_last_to_full=False,
            infinite=False,
            num_workers=4,
            train_mode=False,
        )

    def val_dataloader(self):
        return build_dataloader(
            self.val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate,
            pad_last_to_full=False,
            infinite=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return build_dataloader(
            self.test_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate,
            pad_last_to_full=False,
            infinite=False,
            num_workers=4,
        )


class LitContactVAE(pl.LightningModule):
    def __init__(
        self,
        kl_weight,
        hidden_size,
        latent_dim,
        heads,
        emb_size,
        pred_type="direct",
        use_nll=False,
        dropout=0.0,
        lr=1e-4,
        epochs=100,
        target_mode="distances",
        mean=34.9249697902463,
        std=17.601150743644386,
        bin_edges=None,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.epochs = epochs
        self.target_mode = target_mode

        self.max_rows = 500
        self.max_cols = 500
        self.vae: CVAE = CVAE(
            kl_weight=kl_weight,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            heads=heads,
            emb_size=emb_size,
            mean=mean,
            std=std,
            pred_type=pred_type,
            use_nll=use_nll,
            dropout=dropout,
        )
        # self.vae: tokendistancevae = tokendistancevae(
        #     kl_weight=kl_weight,
        #     hidden_size=hidden_size,
        #     latent_dim=latent_dim,
        #     heads=heads,
        #     emb_size=emb_size,
        #     mean=mean,
        #     std=std,
        #     pred_type=pred_type,
        #     use_nll=use_nll,
        #     dropout=dropout,
        # )
        # self.vae: NonConvProteinContactMapVAE = NonConvProteinContactMapVAE(
        #     rows=self.max_rows,
        #     cols=self.max_cols,
        #     kl_weight=args.kl_weight,
        #     hidden_size=args.hidden_size,
        #     latent_dim=args.latent_dim,
        #     mean=mean,
        #     std=std,
        #     emb_size=args.embedding_size,
        #     protein_encoder_depth=args.protein_encoder_depth,
        #     use_symmetry=args.use_symmetry,
        #     pred_type=args.pred_type,
        #     dropout=args.dropout,
        # )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.vae.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-7
        )
        return [optimizer], [scheduler]

    # -------------------------------
    # TRAINING STEP
    # -------------------------------
    def training_step(self, batch, batch_idx):
        binder, target = batch

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        # Pad coords and masks
        # binder_ca, target_ca = pad_coords(
        #     binder_ca, target_ca, self.max_rows, self.max_cols
        # )
        # bind_A, target_A = pad_masks(binder.A, target.A, self.max_rows, self.max_cols)

        # Masks
        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)

        # Distances (Å)
        x = torch.cdist(binder_ca, target_ca)

        # Forward pass through VAE (always return reconstruction so we can compute metrics)
        # loss, recon_loss, aa_recon_loss, kl_loss, tri_loss, x_recon_phys = (

        losses, x_recon_phys, _ = self.vae.forward(
            x, binder, target, pair_mask, self.global_step, return_recon=True
        )

        for loss in losses:
            self.log(f"train/{loss}", losses[loss])

        loss = losses["loss"]

        # Logging base VAE losses
        # self.log("train/total_loss", loss, prog_bar=True)
        # self.log("train/recon_loss", recon_loss)
        # self.log("train/aa_recon_loss", aa_recon_loss)
        # self.log("train/kl_loss", kl_loss)
        # self.log("train/tri_loss", tri_loss)

        # 1. Log-MSE: The actual optimization objective performance
        x_log = x
        x_recon_log = x_recon_phys
        # mse_log = F.mse_loss(x_recon_log[pair_mask], x_log[pair_mask])
        mse_log = mse_loss(x_recon_log, x_log, pair_mask)
        self.log("train/mse_log", mse_log, prog_bar=True)

        # 2. Bounded Physical MSE: Focus on relevant interaction range (0-20A)
        x_bounded = x.clamp(max=20.0)
        x_recon_bounded = x_recon_phys.clamp(max=20.0)
        # mse_bounded = F.mse_loss(x_recon_bounded[pair_mask], x_bounded[pair_mask])
        mse_bounded = mse_loss(x_recon_bounded, x_bounded, pair_mask)
        self.log("train/mse_bounded_20A", mse_bounded, prog_bar=True)

        if self.target_mode == "distances":
            # Continuous distance regression metric
            with torch.no_grad():
                # mse = F.mse_loss(x_recon_phys[pair_mask], x[pair_mask])
                mse = mse_loss(x_recon_phys, x, pair_mask)
            self.log(
                "train/mse",
                mse,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

        elif self.target_mode == "contact":
            # Contact classification metrics
            with torch.no_grad():
                true_bins = self.vae.bin_distances(x)
                pred_bins = x_recon_phys
                acc, rec, f1 = compute_contact_metrics(pred_bins, true_bins, pair_mask)

            self.log(
                "train/contact_acc",
                acc,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/contact_recall",
                rec,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train/contact_f1",
                f1,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        binder, target = batch

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        # binder_ca, target_ca = pad_coords(
        #     binder_ca, target_ca, self.max_rows, self.max_cols
        # )
        # bind_A, target_A = pad_masks(binder.A, target.A, self.max_rows, self.max_cols)

        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)

        # Ground-truth distances (Å)
        x = torch.cdist(binder_ca, target_ca)

        # Always ask VAE to return reconstruction for validation
        # loss, _, _, _, _, x_recon_phys = self.vae(
        with torch.no_grad():
            losses, x_recon_phys, uncertainty = self.vae.forward(
                x, binder, target, pair_mask, self.global_step, return_recon=True
            )
            M, N = binder_ca.size(1), target_ca.size(1)
            sample_latent = torch.randn(x.size(0), M + N, self.vae.latent_dim).to(
                x.device
            )
            # if hasattr(self, "latent_mean"):
            #     sample_latent = (sample_latent - self.latent_mean) / self.latent_std

            sample_latent = sample_latent - sample_latent.mean(dim=1, keepdim=True)
            x_recon_phys_sampled = self.vae.decode(
                sample_latent,
                binder,
                target,
                upscale=True,
            )

        self.log(
            "val/mse_loss", losses["loss"], prog_bar=True, on_step=False, on_epoch=True
        )
        self.log(
            "val/aa_recon_loss",
            losses["aa_recon_loss"],
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/aa_f1", losses["aa_f1"], prog_bar=False, on_step=False, on_epoch=True
        )

        if self.target_mode == "distances":
            with torch.no_grad():
                # mse = F.mse_loss(x_recon_phys[pair_mask], x[pair_mask])
                mse = mse_loss(x_recon_phys, x, pair_mask)
                sample_mse = mse_loss(x_recon_phys_sampled, x, pair_mask)

            self.log(
                "val/mse",
                mse,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val/sample_mse",
                sample_mse,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        elif self.target_mode == "contact":
            with torch.no_grad():
                true_bins = self.vae.bin_distances(x)
                pred_bins = x_recon_phys
                acc, rec, f1 = compute_contact_metrics(pred_bins, true_bins, pair_mask)

            self.log(
                "val/contact_acc",
                acc,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val/contact_recall",
                rec,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val/contact_f1",
                f1,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        # Save visualization for batch 0
        if batch_idx == 0:
            os.makedirs(f"{args.save_path}/{args.exp_name}/vae_outputs", exist_ok=True)
            generated = x_recon_phys.cpu()

            if self.target_mode == "contact":
                x = self.vae.bin_distances(x)

            B = x.size(0)
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"VAE Reconstructions @ step {self.global_step}")

            for i in range(B):
                bind_mask_i = bind_mask[i].cpu()
                tgt_mask_i = tgt_mask[i].cpu()

                x_i = x[i].cpu()[bind_mask_i][:, tgt_mask_i]
                gen_i = generated[i].cpu()[bind_mask_i][:, tgt_mask_i]

                plt.subplot(2, B, i + 1)
                # plt.imshow(x[i].float().cpu().numpy())
                plt.imshow(x_i.float().cpu().numpy())
                plt.title(f"Original {i}")
                plt.axis("off")
                min_value = 0
                max_value = torch.max(generated[i]).item()

                colorbar_ticks = [min_value, max_value / 2, max_value]
                cbar = plt.colorbar(ticks=colorbar_ticks)
                cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in colorbar_ticks])

                plt.subplot(2, B, B + i + 1)
                # plt.imshow(generated[i].float().cpu().numpy())
                plt.imshow(gen_i.float().cpu().numpy())
                plt.title(f"Recon {i}")
                plt.axis("off")
                cbar = plt.colorbar(ticks=colorbar_ticks)
                cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in colorbar_ticks])

            plt.savefig(
                f"{args.save_path}/{args.exp_name}/vae_outputs/step_{self.global_step}.png"
            )
            plt.close()

    def test_step(self, batch, batch_idx):
        binder, target = batch

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)

        x = torch.cdist(binder_ca, target_ca)

        # losses, x_recon_phys = self.vae.decode(
        #     x, binder, target, pair_mask, self.global_step, return_recon=True
        # )
        losses, x_recon_phys = self.vae.forward(
            x, binder, target, pair_mask, self.global_step, return_recon=True
        )

        # self.log(
        #     "test/loss", losses["loss"], prog_bar=True, on_step=False, on_epoch=True
        # )
        # self.log(
        #     "test/aa_recon_loss",
        #     losses["aa_recon_loss"],
        #     prog_bar=False,
        #     on_step=False,
        #     on_epoch=True,
        # )
        # self.log(
        #     "test/aa_f1", losses["aa_f1"], prog_bar=False, on_step=False, on_epoch=True
        # )

        if self.target_mode == "distances":
            with torch.no_grad():
                # mse = F.mse_loss(x_recon_phys[pair_mask], x[pair_mask])
                mse = mse_loss(x_recon_phys, x, pair_mask)
            self.log(
                "val/mse",
                mse,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        elif self.target_mode == "contact":
            with torch.no_grad():
                true_bins = self.vae.bin_distances(x)
                pred_bins = x_recon_phys
                acc, rec, f1 = compute_contact_metrics(pred_bins, true_bins, pair_mask)

            self.log(
                "test/contact_acc",
                acc,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "test/contact_recall",
                rec,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "test/contact_f1",
                f1,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        # Save visualization for batch 0
        if batch_idx == 0:
            os.makedirs(f"{args.save_path}/{args.exp_name}/vae_outputs", exist_ok=True)
            generated = x_recon_phys.cpu()

            if self.target_mode == "contact":
                x = self.vae.bin_distances(x)

            B = x.size(0)
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"VAE Reconstructions @ step {self.global_step}")

            for i in range(B):
                bind_mask_i = bind_mask[i].cpu()
                tgt_mask_i = tgt_mask[i].cpu()

                x_i = x[i].cpu()[bind_mask_i][:, tgt_mask_i]
                gen_i = generated[i].cpu()[bind_mask_i][:, tgt_mask_i]

                plt.subplot(2, B, i + 1)
                # plt.imshow(x[i].float().cpu().numpy())
                plt.imshow(x_i.float().cpu().numpy())
                plt.title(f"Original {i}")
                plt.axis("off")
                min_value = 0
                max_value = torch.max(generated[i]).item()

                colorbar_ticks = [min_value, max_value / 2, max_value]
                cbar = plt.colorbar(ticks=colorbar_ticks)
                cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in colorbar_ticks])

                plt.subplot(2, B, B + i + 1)
                # plt.imshow(generated[i].float().cpu().numpy())
                plt.imshow(gen_i.float().cpu().numpy())
                plt.title(f"Recon {i}")
                plt.axis("off")
                cbar = plt.colorbar(ticks=colorbar_ticks)
                cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in colorbar_ticks])

            plt.savefig(
                f"{args.save_path}/{args.exp_name}/vae_outputs/step_{self.global_step}.png"
            )
            plt.close()


def test_for_posterior_collapse(args, datamodule, mean, std):
    model = LitContactVAE.load_from_checkpoint(args.ckpt, args=args, mean=mean, std=std)
    model.eval().cuda()
    dataloader = datamodule.val_dataloader()

    # z_means = torch.zeros((0, model.vae.latent_dim))
    # z_stds = torch.zeros((0, model.vae.latent_dim))
    all_latents = []
    latents = torch.zeros((0, model.vae.latent_dim))

    with torch.no_grad():
        for batch in tqdm(dataloader):
            binder, target = batch

            binder_ca = binder.X[:, :, 1]
            target_ca = target.X[:, :, 1]

            # binder_ca, target_ca = pad_coords(
            #     binder_ca, target_ca, model.max_rows, model.max_cols
            # )
            # bind_A, target_A = pad_masks(
            #     binder.A, target.A, model.max_rows, model.max_cols
            # )

            bind_mask = binder.A[:, :, 1].clamp(max=1).bool()
            tgt_mask = target.A[:, :, 1].clamp(max=1).bool()

            pair_mask = (bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)).cuda()

            x = torch.cdist(binder_ca, target_ca).cuda()

            binder, target = (
                binder.to(torch.device("cuda")),
                target.to(torch.device("cuda")),
            )

            latent = model.vae(
                x, binder, target, pair_mask, 0, return_recon=False, return_latent=True
            )
            all_latents.append(latent)

            # z_means = torch.cat([z_means, mu.mean(dim=1).detach().cpu()], dim=0)
            # z_stds = torch.cat([z_stds, logvar.mean(dim=1).detach().cpu()], dim=0)
            latents = torch.cat([latents, latent.mean(dim=1).detach().cpu()], dim=0)

    # z_mean_per_dim = z_means.mean(dim=0)
    # z_std_per_dim = z_stds.mean(dim=0)

    latent_mean = latents.mean(dim=0)
    latent_std = latents.std(dim=0)

    latents = (latents - latent_mean.unsqueeze(0)) / latent_std.unsqueeze(0)

    latent_mean = latents.mean(dim=0)
    latent_std = latents.std(dim=0)

    z_mean_per_dim = latent_mean
    z_std_per_dim = latent_std

    # Plot histograms
    os.makedirs(f"{args.save_path}/{args.exp_name}/vae_outputs", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(z_mean_per_dim.numpy(), bins=30, color="blue", alpha=0.7)
    plt.title("Histogram of Latent Dimension Means")
    plt.xlabel("Mean Value")
    plt.ylabel("Frequency")
    plt.subplot(1, 2, 2)
    plt.hist(z_std_per_dim.numpy(), bins=30, color="green", alpha=0.7)
    plt.title("Histogram of Latent Dimension STDs")
    plt.xlabel("STD Value")
    plt.ylabel("Frequency")
    plt.savefig(
        f"{args.save_path}/{args.exp_name}/vae_outputs/posterior_collapse_check.png"
    )

    torch.save(all_latents, f"{args.save_path}/{args.exp_name}/vae_outputs/latents.pt")


def main(args):
    dm = PPIDataModule(args)

    # Compute dataset statistics
    # mean, std = get_mean_std_distance(dm.train_dataloader())
    mean, std = 34.9249697902463, 17.601150743644386

    # bin_edges = torch.tensor([2, 4, 8, 10, 14, 18, 24, 32, 40, 50, 65, 80]).float()
    # class_weights = compute_class_weights(dm.train_dataloader(), bin_edges)

    if args.check_posterior_collapse:
        test_for_posterior_collapse(args, dm, mean, std)
        return

    model = LitContactVAE(
        kl_weight=args.kl_weight,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        heads=args.heads,
        emb_size=args.embedding_size,
        pred_type=args.pred_type,
        use_nll=args.use_nll,
        dropout=args.dropout,
        lr=args.vae_lr,
        epochs=args.vae_epochs,
        target_mode=args.target_mode,
        mean=mean,
        std=std,
        bin_edges=None,
        class_weights=None,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.save_path}/{args.exp_name}/vae_ckpts",
            filename="epoch-{epoch}",
            save_top_k=-1,
            every_n_epochs=5,
        )
    ]

    logger = (
        WandbLogger(project="contact_latent_diff_vae", name=args.exp_name)
        if args.use_wandb
        else None
    )

    trainer = Trainer(
        accelerator="cuda",
        devices=args.devices,
        max_epochs=args.vae_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision="bf16-mixed",  # AMP
    )

    if args.tune_lr:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=1e-6, max_lr=5e-4)

        fig = lr_finder.plot(suggest=True)
        os.makedirs(f"{args.save_path}/{args.exp_name}/vae_outputs", exist_ok=True)
        fig.savefig(f"{args.save_path}/{args.exp_name}/vae_outputs/lr_finder.png")
        new_lr = lr_finder.suggestion()
        print(f"Setting learning rate to {new_lr}")
        model.lr = new_lr

    # load checkpoint
    if args.validate:
        model = LitContactVAE.load_from_checkpoint(
            args.ckpt,
            kl_weight=args.kl_weight,
            hidden_size=args.hidden_size,
            latent_dim=args.latent_dim,
            heads=args.heads,
            emb_size=args.embedding_size,
            pred_type=args.pred_type,
            use_nll=args.use_nll,
            dropout=args.dropout,
            lr=args.vae_lr,
            epochs=args.vae_epochs,
            target_mode=args.target_mode,
            mean=mean,
            std=std,
            bin_edges=None,
            class_weights=None,
        )
        # trainer.validate(model, datamodule=dm)
        latent_mean, latent_std = get_latent_mean_std(dm.train_dataloader(), model)
        model.latent_mean = latent_mean
        model.latent_std = latent_std

        trainer.validate(model, dataloaders=dm.val_dataloader())
        return

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        type=int,
        help="Number of GPU devices to use",
        default=1,
    )
    parser.add_argument(
        "--check_posterior_collapse",
        action="store_true",
        help="Test for posterior collapse",
    )
    parser.add_argument("--train_path", default="data/DIPS-Het/data.pkl")
    parser.add_argument("--save_path", default="vae_results/")
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
        default="ipa",
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
    parser.add_argument(
        "--use_nll",
        action="store_true",
        help="Whether to use NLL for distance prediction",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()
    main(args)
