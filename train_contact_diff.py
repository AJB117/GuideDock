import argparse
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
from torch import Tensor

from datasets.datamodule import PPIDataModule
from models.distance_fm import LatentFlowMatching
from train_contact_vae import LitContactVAE
from utils import (
    EMACallback,
    Features,
    Meter_Unbound_Bound,
    get_latent_mean_std,
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

        self.diffusion_model: LatentFlowMatching = LatentFlowMatching(
            hidden_dim=hidden_size,
            model_dim=model_dim,
            latent_dim=latent_dim,
            emb_size=embedding_size,
            heads=heads,
            conditioning_depth=conditioning_depth,
            noise_emb_dim=noise_emb_dim,
            device=torch.device("cuda"),
            protein_mixer_depth=protein_mixer_depth,
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
        z: Tensor,
        bind_H: Tensor,
        target_H: Tensor,
        step: int,
        res_mask: Tensor = None,
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

    def training_step(self, batch, batch_idx):
        binder, target = batch

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()

        x = torch.cdist(binder_ca, target_ca)

        with torch.no_grad():
            z = self.vae.vae.get_latents(x, binder, target)
            binder_cond, target_cond = self.vae.vae.get_diff_conditioning(
                binder, target
            )

        res_mask = torch.cat([bind_mask, tgt_mask], dim=1).unsqueeze(-1).float()

        loss = self.diffusion_model(
            binder,
            target,
            z,
            binder_cond,
            target_cond,
            res_mask=res_mask,
            guidance_scale=5.0,
        )

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        return loss

    @torch.inference_mode(False)
    def test_step(self, batch, batch_idx):
        binder, target = batch

        binder_ca = binder.X[:, :, 1]
        target_ca = target.X[:, :, 1]

        B = binder_ca.size(0)

        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)

        x = torch.cdist(binder_ca, target_ca)

        with torch.no_grad(), torch.inference_mode():
            z = self.vae.vae.get_latents(x, binder, target)
            binder_cond, target_cond = self.vae.vae.get_diff_conditioning(
                binder, target
            )

        res_mask = torch.cat([bind_mask, tgt_mask], dim=1).unsqueeze(-1).float()

        with torch.no_grad(), torch.inference_mode():
            val_loss = self.diffusion_model(
                binder,
                target,
                z,
                binder_cond,
                target_cond,
                res_mask=res_mask,
            )
        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

        with torch.no_grad(), torch.inference_mode():
            samples = self.diffusion_model.sample(
                binder,
                target,
                binder_cond,
                target_cond,
                B,
                steps=args.num_steps,
                guidance_scale=5.0,
            )

            x_recon_phys = self.vae.vae.decode(samples, binder, target, upscale=True)

            x_recon_vae = self.vae.vae.decode(z, binder, target, upscale=True)

        mse_gt = F.mse_loss(
            x_recon_phys[pair_mask],
            x[pair_mask],
        )
        mse_vae = F.mse_loss(
            x_recon_vae[pair_mask],
            x[pair_mask],
        )

        within_8_mask = x < 8.0
        mse_gt_8A = F.mse_loss(
            x_recon_phys[pair_mask & within_8_mask],
            x[pair_mask & within_8_mask],
        )

        top_k = 50
        ndcg_scores = []
        for i in range(B):
            true_dists = x[i][pair_mask[i]]
            pred_dists = x_recon_phys[i][pair_mask[i]]

            if true_dists.size(0) < top_k:
                continue

            true_sorted, true_indices = torch.topk(-true_dists, k=top_k)
            pred_sorted = pred_dists[true_indices]

            gains = 1.0 / torch.log2(
                torch.arange(2, top_k + 2, device=true_dists.device).float()
            )
            ideal_gains = gains

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

        x_vis = x_vis.clone()
        gen_vis = generated.clone()
        x_vis[~pair_mask] = 0.0
        gen_vis[~pair_mask] = 0.0

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

        bind_mask = binder.A[:, :, 1].clamp(max=1).bool()
        tgt_mask = target.A[:, :, 1].clamp(max=1).bool()
        pair_mask = bind_mask.unsqueeze(2) & tgt_mask.unsqueeze(1)

        x = torch.cdist(binder_ca, target_ca)

        z = self.vae.vae.get_latents(x, binder, target)
        binder_cond, target_cond = self.vae.vae.get_diff_conditioning(binder, target)

        res_mask = torch.cat([bind_mask, tgt_mask], dim=1).unsqueeze(-1).float()

        val_loss = self.diffusion_model(
            binder,
            target,
            z,
            binder_cond,
            target_cond,
            res_mask=res_mask,
        )
        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

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

            x_recon_phys = self.vae.vae.decode(samples, binder, target, upscale=True)

            x_recon_vae = self.vae.vae.decode(z, binder, target, upscale=True)

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

            x_vis = x_vis.clone()
            gen_vis = generated.clone()
            x_vis[~pair_mask] = 0.0
            gen_vis[~pair_mask] = 0.0

            Bv = x_vis.size(0)

            plt.figure(figsize=(15, 5))
            plt.suptitle(f"(VAE ∘ Diffusion) Reconstructions @ step {self.global_step}")

            im = None
            for i in range(Bv):
                gt_vals = x[i][pair_mask[i]]

                vmin = gt_vals.min().item()
                vmax = gt_vals.max().item()

                ax = plt.subplot(3, Bv, i + 1)
                im = ax.imshow(
                    x_vis[i].float().cpu().numpy(),
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"Original {i}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax = plt.subplot(3, Bv, Bv + i + 1)
                ax.imshow(
                    gen_vis[i].float().cpu().numpy(),
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"Recon {i}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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
    dm = PPIDataModule(args, mode="latent")

    mean, std = 34.9249697902463, 17.601150743644386
    vae = LitContactVAE.load_from_checkpoint(args.ckpt, args=args, mean=mean, std=std)

    dm.setup()
    latent_mean, latent_std = get_latent_mean_std(
        dm.train_dataloader(),
        vae,
        cache_path="latent_mean_std_all_esm2.pt",
        recompute=True,
    )

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
        precision="bf16-mixed",
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
        choices=["distances"],
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
