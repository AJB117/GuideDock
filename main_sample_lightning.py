import pdb
import os
import random
import argparse
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger

from datasets import (
    build_dataset,
    make_protein_collate_fn,
    build_dataloader,
)
from utils import NoiseSchedule, freeze_module
from args import parse_args
from models import AllAtomEnergyModel
from train_contact_vae import LitContactVAE
from train_contact_diff import LitContactDiff
from main_lightning import LitPoseDiffusionModel
from sample import evaluate_samples
from copy import deepcopy

# Set random seeds
torch.manual_seed(1212010)
np.random.seed(1212010)
random.seed(1212010)
torch.set_float32_matmul_precision("high")


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

        self.collate = make_protein_collate_fn(
            embedding=self.embedding,
            args=self.args,
            device=None,  # Lightning handles device placement
        )

    def train_dataloader(self):
        return build_dataloader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.collate,
            pad_last_to_full=True,
            infinite=True,
            num_workers=8,
            train_mode=True,
        )

    def val_dataloader(self):
        """Return both validation loaders for loss and sampling."""
        val_loader = build_dataloader(
            self.val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.collate,
            pad_last_to_full=True,
            infinite=False,
            num_workers=8,
        )

        sample_loader = build_dataloader(
            self.sample_val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate,
            pad_last_to_full=True,
            infinite=False,
            num_workers=0,
        )

        return [val_loader, sample_loader]

    def test_dataloader(self):
        return build_dataloader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate,
            pad_last_to_full=True,
            infinite=False,
            num_workers=8,
        )


class LitGuideDock(L.LightningModule):
    """PyTorch Lightning module for protein-protein docking diffusion model."""

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        #     mean = 46.17057889442541
        #     std = 18.528250269522413
        vae = LitContactVAE.load_from_checkpoint(
            args.vae_ckpt,
            kl_weight=1e-4,
            hidden_size=128,
            latent_dim=8,
            heads=8,
            emb_size=1280,
            use_nll=False,
            dropout=0.12,
            weights_only=False,
            pred_type="direct",
        )
        self.vae = vae

        per_dim_mean, per_dim_std = torch.load("latent_mean_std.pt").values()
        distance_diffusion = LitContactDiff.load_from_checkpoint(
            args.diff_ckpt,
            vae=vae,
            mean=34.9249697902463,
            std=17.601150743644386,
            latent_mean=per_dim_mean,
            latent_std=per_dim_std,
            latent_dim=8,
            hidden_size=128,
            model_dim=128,
            embedding_size=1280,
            heads=8,
            conditioning_depth=4,
            protein_mixer_depth=2,
            noise_emb_dim=64,
        )
        self.distance_diffusion = distance_diffusion

        # Initialize models
        pose_diffusion = LitPoseDiffusionModel.load_from_checkpoint(
            args.ckpt,
            vae=self.vae,
            distance_diffusion=self.distance_diffusion,
            resolution=args.resolution,
            model_dim=args.model_dim,
            hidden_size=args.hidden_size,
            heads=args.heads,
            depth=args.depth,
            noise_emb_dim=args.noise_emb_dim,
            use_contact_guidance=args.use_contact_guidance,
        )
        self.pose_diffusion = pose_diffusion

        # Noise schedule
        self.noise_schedule = NoiseSchedule(
            args.tr_s_min, args.tr_s_max, args.rot_s_min, args.rot_s_max
        )

        # Best validation metrics
        self.best_val_loss = float("inf")
        self.best_val_crmsd = float("inf")

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"Initialized model with resolution {args.resolution} and {num_params} parameters"
        )

    def forward(self, binder, target, tr_s, rot_s, time_steps):
        """Forward pass through the model."""
        return self.model(
            binder, target, tr_s, rot_s, self.hparams.resolution, time_steps
        )

    def test_step(self, batch, batch_idx):
        """Test step - handled in on_test_epoch_end."""
        return None

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
            # vae=self.vae.vae
            vae=None,
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
        if dataloader_idx == 0:  # Loss validation
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

        return None  # Sample loader handled in validation_epoch_end

    def on_validation_epoch_end(self):
        """Run sampling-based evaluation at the end of validation epoch."""

        # Get sample validation dataloader
        sample_loader = self.trainer.val_dataloaders[1]

        with torch.no_grad():
            # Run sampling evaluation
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

        # Skip if evaluation failed
        if val_dockq == 0.0:
            return

        # Log metrics
        self.log("val/lrmsd", val_lrmsd, prog_bar=True, sync_dist=True)
        self.log("val/irmsd", val_irmsd, prog_bar=True, sync_dist=True)
        self.log("val/crmsd", val_crmsd, prog_bar=True, sync_dist=True)
        self.log("val/dockq", val_dockq, prog_bar=True, sync_dist=True)

        print(
            f"\n\033[92mEpoch {self.current_epoch}, Step {self.global_step}: "
            f"val_lrmsd={val_lrmsd:.4f}, val_irmsd={val_irmsd:.4f}, "
            f"val_crmsd={val_crmsd:.4f}, val_dockq={val_dockq:.4f}\033[0m"
        )

        # Update best metrics
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


def main():
    """Main training function."""
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Keep linalg kernels loaded (performance warm-up)
    if torch.cuda.is_available():
        torch.linalg.inv(torch.eye(1, device="cuda"))

    # Initialize data module
    datamodule = PPIDataModule(args)

    # Initialize model
    model = LitGuideDock(args)

    # Setup callbacks
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
        # EMA update callback
        EMACallback(decay=args.ema_decay),
        # Checkpoint callbacks
        checkpoint_last,
        checkpoint_crmsd,
        checkpoint_lrsmd,
        checkpoint_dockq,
        # Early stopping
        EarlyStopping(
            monitor="val/crmsd",
            patience=args.patience,
            mode="min",
            verbose=True,
        ),
    ]

    # Setup logger
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project="ppi_docking",
            name=args.exp_name,
            save_dir=args.save_dir,
        )

    # Setup trainer
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
        deterministic=False,  # For performance
    )

    # Resume from checkpoint if provided
    ckpt_path = getattr(args, "ckpt", None)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")

    if args.test:
        print("Running in test mode...")
        # trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        datamodule.setup(stage="test")
        # trainer.validate(
        #     model=model,
        #     dataloaders=datamodule.val_dataloader(),
        #     weights_only=False,
        #     ckpt_path=ckpt_path,
        # )
        trainer.test(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=ckpt_path,
            weights_only=False,
        )
        return

    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    print("Training completed!")
    print(f"Best validation loss: {model.best_val_loss:.4f}")
    print(f"Best validation CRMSD: {model.best_val_crmsd:.4f}")


if __name__ == "__main__":
    main()
