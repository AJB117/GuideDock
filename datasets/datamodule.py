import pytorch_lightning as L
import torch

from .dataset import build_dataloader, build_dataset, make_protein_collate_fn


class PPIDataModule(L.LightningDataModule):
    def __init__(self, args, mode="docking"):
        super().__init__()
        if mode not in {"docking", "vae", "latent"}:
            raise ValueError(f"Unknown data mode: {mode}")
        self.args = args
        self.mode = mode
        self.embedding = None
        self.embedding_esm2 = None
        self.collate = None
        self.batch_size = (
            args.diffusion_batch_size if mode == "latent" else args.batch_size
        )
        self.num_workers = 4 if mode == "vae" else 8

    def setup(self, stage=None):
        if self.collate is not None:
            return
        self.train_ds, self.val_ds, self.sample_val_ds, self.test_ds = build_dataset(
            self.args
        )
        if self.args.embedding:
            self.embedding = torch.load(self.args.embedding, map_location="cpu")
        esm2_path = getattr(self.args, "embedding_esm2", None)
        if self.mode == "docking" and esm2_path:
            self.embedding_esm2 = torch.load(esm2_path, map_location="cpu")
        self.collate = make_protein_collate_fn(
            embedding=self.embedding,
            args=self.args,
            device=None,
            embedding_esm2=self.embedding_esm2,
        )

    def _loader(self, dataset, shuffle=False, sampling=False, test=False):
        docking = self.mode == "docking"
        return build_dataloader(
            dataset,
            batch_size=1 if sampling or (test and docking) else self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate,
            pad_last_to_full=docking and not test,
            infinite=docking and shuffle,
            num_workers=0 if sampling or (test and docking) else self.num_workers,
            train_mode=docking and shuffle,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        loader = self._loader(self.val_ds)
        if self.mode == "docking":
            return [loader, self._loader(self.sample_val_ds, sampling=True)]
        return loader

    def test_dataloader(self):
        return self._loader(self.test_ds, test=True)
