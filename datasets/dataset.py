from __future__ import annotations

import pickle
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    Subset,
)
from tqdm import tqdm

from utils import (
    ALPHABET,
    ATOM_TYPES,
    RES_ATOM14,
    Features,
    compute_dihedrals,
    compute_spline_curvature,
    compute_spline_torsion,
    get_resolution_indices,
    parametrize_backbone_spline,
)

from .loader import featurize


class ProteinDockingDataset(Dataset):
    """
    A PyTorch Dataset for rigid protein-protein docking pairs.

    Each item is a dict with keys:
      - 'pdb'
      - 'binder_seq', 'binder_coords' [L_b, sel, 3], 'binder_dihedrals', 'binder_atypes'
      - 'target_seq', 'target_coords' [L_t, sel, 3], 'target_dihedrals', 'target_atypes'
      - plus mirror 'make_site' fields (e.g., 'binder_full', 'target_full', etc.)

    Notes:
      * Coordinates are centered by the mean of *target* coordinates (flattened).
      * Entries with any-zero atom triplets per residue are skipped (binder or target).
    """

    def __init__(
        self,
        data_path: str,
        patch_size: int,
        patch_strategy: str = "dist_com",
        resolution: str = "all_atom",
        limit: Optional[int] = None,
        strict: bool = True,
    ):
        super().__init__()
        self.data_path = data_path
        self.patch_size = patch_size
        self.patch_strategy = patch_strategy
        self.resolution = resolution

        sel, bb_idx = self._resolution_to_sel(resolution, strict=strict)
        self.sel = sel
        self.bb_idx = bb_idx

        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)
            all_data = []
            for d in raw_data:
                if "db55" in data_path or isinstance(d, dict):
                    all_data.append(
                        (
                            d["pdb"],
                            d["aseq"],
                            d["acoords"]["bound"],
                            d["bseq"],
                            d["bcoords"]["bound"],
                        )
                    )
                else:
                    all_data.append(d)

        self._data: List[Dict[str, Any]] = []
        for idx, (pdb, aseq, acoords, bseq, bcoords) in tqdm(enumerate(all_data)):
            if limit is not None and limit > 0 and idx >= limit:
                break

            if len(aseq) != len(acoords) or len(bseq) != len(bcoords):
                continue

            entry: Dict[str, Any] = {"pdb": pdb}

            entry["binder_seq"] = bseq
            entry["binder_coords"] = torch.as_tensor(bcoords, dtype=torch.float32)[
                :, :sel
            ]

            if self._has_any_zero_atom(entry["binder_coords"]):
                print(f"Skipping {pdb}: zero coordinates in binder.")
                continue

            entry["binder_dihedrals"] = compute_dihedrals(torch.as_tensor(bcoords))
            entry["binder_atypes"] = torch.as_tensor(
                [
                    [ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]]
                    for s in entry["binder_seq"]
                ],
                dtype=torch.long,
            )[:, :sel]

            entry["target_seq"] = aseq
            entry["target_coords"] = torch.as_tensor(acoords, dtype=torch.float32)[
                :, :sel
            ]

            target_com = (
                entry["target_coords"]
                .reshape(-1, 3)
                .mean(dim=0, keepdim=True)[:, None, ...]
            )

            entry["target_coords"] = entry["target_coords"] - target_com
            entry["binder_coords"] = entry["binder_coords"] - target_com

            if self._has_any_zero_atom(entry["target_coords"]):
                continue

            entry["target_dihedrals"] = compute_dihedrals(torch.as_tensor(acoords))
            entry["target_atypes"] = torch.as_tensor(
                [
                    [ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]]
                    for s in entry["target_seq"]
                ],
                dtype=torch.long,
            )[:, :sel]

            tck_bind, u_bind = parametrize_backbone_spline(
                entry["binder_coords"], use_all_atoms=True
            )
            tck_tgt, u_tgt = parametrize_backbone_spline(
                entry["target_coords"], use_all_atoms=True
            )

            M = len(entry["binder_coords"])
            N = len(entry["target_coords"])

            torsion_bind = compute_spline_torsion(tck_bind, u_bind)
            curvature_bind = compute_spline_curvature(tck_bind, u_bind)

            entry["binder_spline_curvature"] = (
                torch.tensor(curvature_bind).float().view(M, sel, 1)
            )
            entry["binder_spline_torsion"] = (
                torch.tensor(torsion_bind).float().view(M, sel, 1)
            )

            torsion_tgt = compute_spline_torsion(tck_tgt, u_tgt)
            curvature_tgt = compute_spline_curvature(tck_tgt, u_tgt)

            entry["target_spline_curvature"] = (
                torch.tensor(curvature_tgt).float().view(N, sel, 1)
            )
            entry["target_spline_torsion"] = (
                torch.tensor(torsion_tgt).float().view(N, sel, 1)
            )

            self.make_site(entry, "target", resolution=resolution)
            self.make_site(entry, "binder", resolution=resolution)

            self._data.append(entry)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]

    @staticmethod
    def _resolution_to_sel(resolution: str, strict: bool = True) -> Tuple[int, int]:
        try:
            return get_resolution_indices(resolution)
        except ValueError:
            if strict:
                raise
            return 14, 1

    @staticmethod
    def _has_any_zero_atom(coords: torch.Tensor) -> bool:
        """
        coords: [L, sel, 3]
        Returns True if any atom triplet is exactly zero somewhere.
        """
        return (coords == 0).all(dim=2).any(dim=1).any().item()

    @staticmethod
    def make_site(
        entry: Dict[str, Any], field: str, resolution: str = "all_atom"
    ) -> None:

        if resolution not in {"all_atom", "backbone", "CA"}:
            raise ValueError(f"Unknown resolution: {resolution}")
        entry[f"{field}_full"] = entry[f"{field}_seq"]

        entry[f"{field}_coords"] = entry[f"{field}_coords"]
        entry[f"{field}_atypes"] = entry[f"{field}_atypes"]
        entry[f"{field}_dihedrals"] = entry[f"{field}_dihedrals"]

    def shuffle_(self) -> None:
        random.shuffle(self._data)

    def welford_std_coords(
        self, idx: Optional[int] = None, field: str = "binder"
    ) -> float:
        """
        Compute std of flattened coordinates via Welford's algorithm.
        If idx is None, computes over the whole dataset for the given field.
        """
        mean = torch.tensor(0.0, dtype=torch.float64)
        m2 = torch.tensor(0.0, dtype=torch.float64)
        n = 0

        entries = [self._data[idx]] if idx is not None else self._data
        for e in entries:
            flat = e[f"{field}_coords"].reshape(-1).to(dtype=torch.float64)
            if flat.numel() == 0:
                continue
            for x in flat:
                n += 1
                delta = x - mean
                mean = mean + delta / n
                m2 = m2 + delta * (x - mean)

        if n < 2:
            return 0.0
        return torch.sqrt(m2 / (n - 1)).item()


def make_protein_collate_fn(
    embedding: Optional[Dict[str, torch.Tensor]],
    args,
    binder: str = "binder",
    target: str = "target",
    device: Optional[torch.device] = None,
    resolution: Optional[str] = None,
    patch_size: Optional[int] = None,
    embedding_esm2: Optional[Dict[str, torch.Tensor]] = None,
) -> Callable[
    [List[Dict[str, Any]]], Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]
]:
    """
    Returns a collate_fn that replicates your make_local_batch behavior.

    Result:
        ((bind_X, bind_S, bind_A, bind_V, bind_O, bind_T, bind_C, bind_Si, pdbs),
        (tgt_X, tgt_S, tgt_A, tgt_V, tgt_O, tgt_T, tgt_C, tgt_Si, pdbs))
    All tensors are returned on CPU unless device is provided.
    """

    if resolution is None:
        resolution = getattr(args, "resolution", "all_atom")
    if patch_size is None:
        patch_size = getattr(args, "patch_size", 50)

    def _collate(batch: List[Dict[str, Any]]):
        pdbs = [b["pdb"] for b in batch]

        bind_X, bind_S, bind_A, bind_V, bind_O, bind_T, bind_Si, bind_C = featurize(
            batch,
            binder,
            device=torch.device("cpu"),
            patch_size=patch_size,
            resolution=resolution,
        )
        tgt_X, tgt_S, tgt_A, tgt_V, tgt_O, tgt_T, tgt_Si, tgt_C = featurize(
            batch,
            target,
            device=torch.device("cpu"),
            patch_size=patch_size,
            resolution=resolution,
        )

        if embedding is not None:
            bind_S = torch.zeros(
                bind_S.size(0), bind_S.size(1), args.embedding_size, dtype=torch.float32
            )
            tgt_S = torch.zeros(
                tgt_S.size(0), tgt_S.size(1), args.embedding_size, dtype=torch.float32
            )
        else:
            bind_S = torch.zeros(bind_S.size(0), bind_S.size(1), 1, dtype=torch.long)
            tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), 1, dtype=torch.long)

        if embedding_esm2 is not None:
            bind_S_ESM2 = torch.zeros(
                bind_S.size(0), bind_S.size(1), args.embedding_size, dtype=torch.float32
            )
            tgt_S_ESM2 = torch.zeros(
                tgt_S.size(0), tgt_S.size(1), args.embedding_size, dtype=torch.float32
            )
        else:
            bind_S_ESM2 = None
            tgt_S_ESM2 = None

        missing = []

        for i, b in enumerate(batch):
            L_tgt = len(b[f"{target}_full"])
            L_bind = len(b[f"{binder}_full"])

            tgt_indices = torch.tensor(
                [ALPHABET.index(a) for a in b[f"{target}_full"]], dtype=torch.long
            )
            bnd_indices = torch.tensor(
                [ALPHABET.index(a) for a in b[f"{binder}_full"]], dtype=torch.long
            )
            if embedding is not None:
                try:
                    tgt_S[i, :L_tgt] = embedding[b[f"{target}_full"]]
                except:
                    missing.append(b[f"{target}_full"])
                try:
                    bind_S[i, :L_bind] = embedding[b[f"{binder}_full"]]
                except:
                    missing.append(b[f"{binder}_full"])
            else:
                tgt_S[i, :L_tgt] = tgt_indices.unsqueeze(-1)
                bind_S[i, :L_bind] = bnd_indices.unsqueeze(-1)
            if embedding_esm2 is not None:
                tgt_S_ESM2[i, :L_tgt] = embedding_esm2[b[f"{target}_full"]]
                bind_S_ESM2[i, :L_bind] = embedding_esm2[b[f"{binder}_full"]]

            bind_Si[i, :L_bind] = bnd_indices
            tgt_Si[i, :L_tgt] = tgt_indices

            tgt_O[i, :L_tgt] = torch.tensor(
                [ALPHABET.index(a) for a in b[f"{target}_full"]], dtype=torch.long
            )
            bind_O[i, :L_bind] = torch.tensor(
                [ALPHABET.index(a) for a in b[f"{binder}_full"]], dtype=torch.long
            )

        with open("missing.txt", "w+") as file:
            for miss in missing:
                file.writelines(miss + "\n")

        return Features(
            bind_X,
            bind_S,
            bind_A,
            bind_V,
            bind_O,
            bind_T,
            bind_C,
            bind_Si,
            pdbs,
        ), Features(tgt_X, tgt_S, tgt_A, tgt_V, tgt_O, tgt_T, tgt_C, tgt_Si, pdbs)

    return _collate


class RepeatSingleIndexBatchSampler(BatchSampler):
    """
    For diffusion training: each batch consists of B clones of a **single** dataset index.
    The index is sampled at random if `sampler` is RandomSampler.
    """

    def __init__(self, sampler, batch_size, shuffle):
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for idx in self.sampler:
            if self.shuffle:
                idx = random.randint(0, len(self.sampler) - 1)

            yield [idx] * self.batch_size

    def __len__(self):
        return len(self.sampler)


class PadToFullBatchBatchSampler(BatchSampler):
    """
    Wraps another sampler and ensures every yielded batch has exactly batch_size elements.
    If the last batch is short, it pads with random choices (with replacement) from the dataset indices.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
        drop_last: bool = False,
        dataset_len: Optional[int] = None,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self._sampler = sampler
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._dataset_len = dataset_len

    def __iter__(self):
        batch = []
        all_indices = list(self._sampler)
        n = len(all_indices)

        for idx in all_indices:
            batch.append(idx)
            if len(batch) == self._batch_size:
                yield batch
                batch = []

        if batch and not self._drop_last:
            need = self._batch_size - len(batch)
            population = all_indices if n > 0 else list(range(self._dataset_len or 0))
            if not population:
                yield batch
            else:
                pad = random.choices(population, k=need)
                batch.extend(pad)
                yield batch

    def __len__(self) -> int:

        all_indices = list(self._sampler)
        n = len(all_indices)
        if self._drop_last:
            return n // self._batch_size
        return (n + self._batch_size - 1) // self._batch_size


def get_dips_train_val_test_keys():
    with open("data/DIPS-Het/train.txt") as f:
        train_keys = set([x for x in f.read().splitlines()])
    with open("data/DIPS-Het/val.txt") as f:
        val_keys = set([x for x in f.read().splitlines()])
    with open("data/DIPS-Het/test.txt") as f:
        test_keys = set([x for x in f.read().splitlines()])

    return train_keys, val_keys, test_keys


def get_db5_train_val_test_keys():
    with open("data/db55/train.txt") as f:
        train_keys = set([x for x in f.read().splitlines()])
    with open("data/db55/val.txt") as f:
        val_keys = set([x for x in f.read().splitlines()])
    with open("data/db55/test.txt") as f:
        test_keys = set([x for x in f.read().splitlines()])

    return train_keys, val_keys, test_keys


def get_cutoff_train_val_test_keys():
    with open("data/cutoff/test.txt") as f:
        test_keys = set([x for x in f.read().splitlines()])

    return None, None, test_keys


def get_cutoff_full_train_val_test_keys():
    with open("data/cutoff_full/test.txt") as f:
        test_keys = set([x for x in f.read().splitlines()])

    return None, None, test_keys


def get_sabdab_full_train_val_test_keys():
    with open("data/sabdab/test.txt") as f:
        test_keys = set([x for x in f.read().splitlines()])

    return None, None, test_keys


def build_dataset(args):
    """
    Load full dataset once, then use Subset wrappers for train/val/test.
    """

    full_ds = ProteinDockingDataset(
        data_path=args.train_path,
        patch_size=args.patch_size,
        patch_strategy="com",
        resolution=args.resolution,
        limit=args.limit,
    )

    if "DIPS" in args.train_path:
        train_keys, val_keys, test_keys = get_dips_train_val_test_keys()

        pdb_to_indices = {}
        for idx, entry in enumerate(full_ds._data):
            pdb = entry["pdb"]
            pdb_to_indices.setdefault(pdb, []).append(idx)

        def collect_indices(key_set):
            idxs = []
            for k in key_set:
                idxs.extend(pdb_to_indices.get(k, []))
            return idxs

        train_idx = collect_indices(train_keys)
        val_idx = collect_indices(val_keys)
        test_idx = collect_indices(test_keys)
    elif "db55" in args.train_path:
        train_keys, val_keys, test_keys = get_db5_train_val_test_keys()

        pdb_to_indices = {}
        for idx, entry in enumerate(full_ds._data):
            pdb = entry["pdb"].split("_")[0]
            pdb_to_indices.setdefault(pdb, []).append(idx)

        def collect_indices(key_set):
            idxs = []
            for k in sorted(key_set):
                if k in pdb_to_indices:
                    idxs.extend(pdb_to_indices[k])
            return idxs

        train_idx = collect_indices(train_keys)
        val_idx = collect_indices(val_keys)
        test_idx = collect_indices(test_keys)
    elif "cutoff_full" in args.train_path:
        train_keys, val_keys, test_keys = get_cutoff_full_train_val_test_keys()
        pdb_to_indices = {}
        for idx, entry in enumerate(full_ds._data):
            pdb_to_indices.setdefault(entry["pdb"], []).append(idx)

        def collect_indices(key_set):
            idxs = []
            for k in sorted(key_set):
                if k in pdb_to_indices:
                    idxs.extend(pdb_to_indices[k])
            return idxs

        train_idx = collect_indices(test_keys)
        val_idx = collect_indices(test_keys)
        test_idx = collect_indices(test_keys)
    elif "sabdab" in args.train_path:
        train_keys, val_keys, test_keys = get_sabdab_full_train_val_test_keys()
        pdb_to_indices = {}
        for idx, entry in enumerate(full_ds._data):
            pdb_to_indices.setdefault(entry["pdb"], []).append(idx)

        def collect_indices(key_set):
            idxs = []
            for k in sorted(key_set):
                if k in pdb_to_indices:
                    idxs.extend(pdb_to_indices[k])
            return idxs

        train_idx = collect_indices(test_keys)
        val_idx = collect_indices(test_keys)
        test_idx = collect_indices(test_keys)
    elif "cutoff" in args.train_path:
        train_keys, val_keys, test_keys = get_cutoff_train_val_test_keys()
        pdb_to_indices = {}
        for idx, entry in enumerate(full_ds._data):
            pdb_to_indices.setdefault(entry["pdb"], []).append(idx)

        def collect_indices(key_set):
            idxs = []
            for k in sorted(key_set):
                if k in pdb_to_indices:
                    idxs.extend(pdb_to_indices[k])
            return idxs

        train_idx = collect_indices(test_keys)
        val_idx = collect_indices(test_keys)
        test_idx = collect_indices(test_keys)
    elif "all" in args.train_path:
        dips_train_keys, dips_val_keys, dips_test_keys = get_dips_train_val_test_keys()
        db5_train_keys, db5_val_keys, db5_test_keys = get_db5_train_val_test_keys()

        train_keys = dips_train_keys | db5_train_keys
        val_keys = dips_val_keys | db5_val_keys
        test_keys = dips_test_keys | db5_test_keys

        pdb_to_indices = {}
        for idx, entry in enumerate(full_ds._data):
            pdb = entry["pdb"]

            if "l_b" in pdb or "r_b" in pdb:
                pdb = pdb.split("_")[0]

            pdb_to_indices.setdefault(pdb, []).append(idx)

        def collect_indices(key_set):
            idxs = []
            for k in key_set:
                idxs.extend(pdb_to_indices.get(k, []))
            return idxs

        val_idx = collect_indices(val_keys)
        test_idx = collect_indices(test_keys)
        train_idx = collect_indices(train_keys)
    else:
        all_idx = list(range(len(full_ds)))
        random.shuffle(all_idx)
        split = int(0.9 * len(all_idx))
        train_idx, val_idx = all_idx[:split], all_idx[split:]
        test_idx = []

    train = Subset(full_ds, train_idx)
    val = Subset(full_ds, val_idx)
    sample_val = Subset(full_ds, test_idx[: args.num_val_sample])
    test = Subset(full_ds, test_idx)

    if getattr(args, "include_skempi", False):
        skempi_ds = ProteinDockingDataset(
            data_path=args.test_path,
            patch_size=args.patch_size,
            patch_strategy="com",
            resolution=args.resolution,
            limit=args.limit,
        )

        base_len = len(full_ds)
        full_ds._data.extend(skempi_ds._data)
        train.indices.extend(range(base_len, base_len + len(skempi_ds)))

    return train, val, sample_val, test


def build_dataloader(
    dataset: ProteinDockingDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pad_last_to_full: bool = False,
    infinite: bool = False,
    generator: Optional[torch.Generator] = None,
    train_mode: bool = False,
) -> DataLoader:
    """
    Returns a DataLoader; if pad_last_to_full=True, the last batch is padded (with replacement) to full size.
    If infinite=True, wraps the loader in an endless generator.
    """
    base_loader = partial(
        DataLoader, dataset, num_workers=num_workers, collate_fn=collate_fn
    )

    sampler = (
        RandomSampler(dataset, generator=generator)
        if shuffle
        else SequentialSampler(dataset)
    )
    if train_mode:
        batch_sampler = RepeatSingleIndexBatchSampler(
            sampler, batch_size=batch_size, shuffle=shuffle
        )
        loader = base_loader(batch_sampler=batch_sampler)
    elif pad_last_to_full:
        batch_sampler = PadToFullBatchBatchSampler(
            sampler, batch_size=batch_size, drop_last=False, dataset_len=len(dataset)
        )
        loader = base_loader(batch_sampler=batch_sampler, shuffle=shuffle)
    else:
        loader = base_loader(batch_size=batch_size, shuffle=shuffle)

    if not infinite:
        return loader

    class _InfiniteLoader:
        def __init__(self, loader):
            self.loader = loader
            self._it = iter(loader)

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self._it)
            except StopIteration:
                self._it = iter(self.loader)
                return next(self._it)

    return _InfiniteLoader(loader)
