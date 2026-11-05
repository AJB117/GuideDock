import os
import random
from typing import Any, Dict

import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from args import parse_args
from datasets.datamodule import PPIDataModule
from models import AllAtomEnergyModel
from sample import evaluate_samples
from train_contact_diff import LitContactDiff
from train_contact_vae import LitContactVAE
from utils import EMACallback, NoiseSchedule

torch.manual_seed(1212010)
np.random.seed(1212010)
random.seed(1212010)
torch.set_float32_matmul_precision("high")


class LitPoseDiffusionModel(L.LightningModule):
    """PyTorch Lightning module for protein-protein docking diffusion model."""

    def __init__(self, args, dist_model=None):
        super().__init__()
        self.save_hyperparameters(args)

        model = AllAtomEnergyModel(
            args,
            device=torch.device("cuda"),
            resolution=args.resolution,
            dist_model=dist_model,
        )

        self.model = model

        self.noise_schedule = NoiseSchedule(
            args.tr_s_min, args.tr_s_max, args.rot_s_min, args.rot_s_max
        )

        self.best_val_loss = float("inf")
        self.best_val_crmsd = float("inf")

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"Initialized model with resolution {args.resolution} and {num_params} parameters"
        )

    def configure_optimizers(self):
        """Configure optimizer and optional scheduler."""
        all_parameters = list(self.model.parameters())

        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                all_parameters, lr=self.hparams.lr, weight_decay=1e-5
            )
        elif self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                all_parameters, lr=self.hparams.lr, weight_decay=0.0
            )
        elif self.hparams.optimizer == "radam":
            optimizer = torch.optim.RAdam(
                all_parameters, lr=self.hparams.lr, weight_decay=0.0
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                all_parameters,
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=0.0,
                fused=True,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.hparams.optimizer}")

        return optimizer

    def forward(self, binder, target, tr_s, rot_s, time_steps):
        """Forward pass through the model."""
        return self.model(
            binder, target, tr_s, rot_s, self.hparams.resolution, time_steps
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        binder, target = batch
        batch_size = binder.X.shape[0]

        rot_s = torch.zeros(batch_size, 1, device=self.device)
        tr_s = torch.zeros(batch_size, 1, device=self.device)
        time_steps = torch.zeros(batch_size, 1, device=self.device)

        fixed_t = None
        for idx in range(batch_size):
            t = fixed_t if fixed_t is not None else np.random.uniform()
            time_steps[idx] = t
            t_tr, t_rot = t, t
            tr_s_, rot_s_ = self.noise_schedule(t_tr, t_rot)
            rot_s[idx] = rot_s_
            tr_s[idx] = tr_s_

        loss, loss_dict = self.model(
            binder,
            target,
            tr_s,
            rot_s,
            self.hparams.resolution,
            time_steps,
            get_loss_dict=True,
        )

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True)

        if batch_idx % 20 == 0:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    self.log(f"norms/{name}", grad_norm, on_step=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step - handled in on_test_epoch_end."""
        return None

    @torch.inference_mode(False)
    def on_test_epoch_end(self):
        """Test step - run sampling-based evaluation on test set."""
        test_loader = self.trainer.test_dataloaders

        test_lrmsd, test_irmsd, test_crmsd, test_dockq = evaluate_samples(
            self.hparams,
            self.device,
            test_loader,
            self.model,
            self.noise_schedule,
            self.hparams.save_dir,
            self.global_step,
            vae=None,
            use_ca=True,
        )

        print(
            f"\n\033[94mTest Results - Step {self.global_step}: "
            f"test_lrmsd={test_lrmsd:.4f}, test_irmsd={test_irmsd:.4f}, "
            f"test_crmsd={test_crmsd:.4f}, test_dockq={test_dockq:.4f}\033[0m"
        )

        self.log("test/lrmsd", test_lrmsd, prog_bar=True, sync_dist=True)
        self.log("test/irmsd", test_irmsd, prog_bar=True, sync_dist=True)
        self.log("test/crmsd", test_crmsd, prog_bar=True, sync_dist=True)
        self.log("test/dockq", test_dockq, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step - compute loss on validation set."""
        if dataloader_idx == 0:
            binder, target = batch
            batch_size = binder.X.shape[0]

            rot_s = torch.zeros(batch_size, 1, device=self.device)
            tr_s = torch.zeros(batch_size, 1, device=self.device)
            time_steps = torch.zeros(batch_size, 1, device=self.device)

            for idx in range(batch_size):
                time_steps[idx] = t = np.random.uniform()
                t_tr, t_rot = t, t
                tr_s_, rot_s_ = self.noise_schedule(t_tr, t_rot)
                rot_s[idx] = rot_s_
                tr_s[idx] = tr_s_

            with torch.no_grad():
                loss = self.model(
                    binder, target, tr_s, rot_s, self.hparams.resolution, time_steps
                )

            self.log(
                "val/loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return {"val_loss": loss}

        return None

    def on_validation_epoch_end(self):
        """Run sampling-based evaluation at the end of validation epoch."""

        sample_loader = self.trainer.val_dataloaders[1]

        with torch.no_grad():
            val_lrmsd, val_irmsd, val_crmsd, val_dockq = evaluate_samples(
                self.hparams,
                self.device,
                sample_loader,
                self.model,
                self.noise_schedule,
                self.hparams.save_dir,
                self.global_step,
                vae=None,
            )

        if val_dockq == 0.0:
            return

        self.log("val/lrmsd", val_lrmsd, prog_bar=True, sync_dist=True)
        self.log("val/irmsd", val_irmsd, prog_bar=True, sync_dist=True)
        self.log("val/crmsd", val_crmsd, prog_bar=True, sync_dist=True)
        self.log("val/dockq", val_dockq, prog_bar=True, sync_dist=True)

        print(
            f"\n\033[92mEpoch {self.current_epoch}, Step {self.global_step}: "
            f"val_lrmsd={val_lrmsd:.4f}, val_irmsd={val_irmsd:.4f}, "
            f"val_crmsd={val_crmsd:.4f}, val_dockq={val_dockq:.4f}\033[0m"
        )

        if val_crmsd < self.best_val_crmsd:
            self.best_val_crmsd = val_crmsd
            print(f"New best validation CRMSD: {val_crmsd:.4f}")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save additional state including EMA and best metrics."""
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["best_val_crmsd"] = self.best_val_crmsd

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load best metrics."""
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]
        if "best_val_crmsd" in checkpoint:
            self.best_val_crmsd = checkpoint["best_val_crmsd"]


def main(args=None):
    if args is None:
        args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.linalg.inv(torch.eye(1, device="cuda"))

    datamodule = PPIDataModule(args)

    dist_model = None

    if args.use_guidedock:
        mean, std = 34.9249697902463, 17.601150743644386

        vae = LitContactVAE(
            kl_weight=1e-4,
            hidden_size=96,
            latent_dim=8,
            heads=8,
            emb_size=1280,
        )
        vae.load_state_dict(
            torch.load(args.vae_ckpt, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
        )
        vae.eval()

        per_dim_mean, per_dim_std = torch.load("latent_mean_std_all_esm2.pt").values()

        dist_model = LitContactDiff(
            args,
            mean=mean,
            std=std,
            vae=vae,
            hidden_size=96,
            latent_mean=per_dim_mean,
            latent_std=per_dim_std,
            conditioning_depth=7,
            protein_mixer_depth=5,
        )
        dist_model.load_state_dict(
            torch.load(args.diff_ckpt, map_location="cpu", weights_only=False)[
                "state_dict"
            ],
        )
        dist_model.eval()

    if args.ckpt:
        model = LitPoseDiffusionModel(args)
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.model.dist_model = dist_model
        print(f"Loaded model from checkpoint: {args.ckpt}")
    else:
        model = LitPoseDiffusionModel(args, dist_model=dist_model)

    checkpoint_last = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="last",
        save_last=True,
        save_top_k=0,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
    )

    checkpoint_crmsd = ModelCheckpoint(
        dirpath=f"{args.save_dir}/bests_crmsd",
        filename="best_step_{step}",
        monitor="val/crmsd",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
    )

    checkpoint_lrsmd = ModelCheckpoint(
        dirpath=f"{args.save_dir}/bests_lrsmd",
        filename="best_step_{step}",
        monitor="val/lrmsd",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
    )

    checkpoint_dockq = ModelCheckpoint(
        dirpath=f"{args.save_dir}/bests_dockq",
        filename="best_step_{step}",
        monitor="val/dockq",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
    )

    callbacks = [
        EMACallback(decay=args.ema_decay),
        checkpoint_last,
        checkpoint_crmsd,
        checkpoint_lrsmd,
        checkpoint_dockq,
        EarlyStopping(
            monitor="val/crmsd",
            patience=args.patience,
            mode="min",
            verbose=True,
        ),
    ]

    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project="ppi_docking",
            name=args.exp_name,
            save_dir=args.save_dir,
        )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=args.num_steps,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.clip_norm if args.clip_norm != -1 else None,
        precision="16-mixed" if getattr(args, "use_amp", True) else "32-true",
        accumulate_grad_batches=getattr(args, "gradient_accumulate_every", 1),
        val_check_interval=args.eval_every,
        log_every_n_steps=1,
        enable_progress_bar=args.verbose,
        enable_model_summary=True,
        deterministic=False,
    )

    ckpt_path = getattr(args, "ckpt", None)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")

    if args.test:
        print("Running in test mode...")

        datamodule.setup(stage="test")

        trainer.test(
            model=model,
            dataloaders=datamodule.test_dataloader(),
        )
        return

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    print("Training completed!")
    print(f"Best validation loss: {model.best_val_loss:.4f}")
    print(f"Best validation CRMSD: {model.best_val_crmsd:.4f}")


if __name__ == "__main__":
    main()
