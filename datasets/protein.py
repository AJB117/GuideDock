import random
import torch
import pickle
import pdb as debugger

from tqdm import tqdm
from .loader import *
from utils import *


class DockingDataset:
    def __init__(self, data_path, patch_size, patch_strategy="com", resolution="all_atom"):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        bb_idx = 1
        if resolution == "all_atom":
            sel = 14
        elif resolution == "backbone":
            sel = 4
        elif resolution == "CA":
            sel = 1
            bb_idx = 0
        else:
            raise ValueError(f"Unknown resolution: {resolution}")

        self.data = []
        # for pdb, aseq, acoords, bseq, bcoords in tqdm(data):
        for entry in tqdm(data):
            pdb, aseq, acoords_bnd, bseq, bcoords_bnd, acoords, bcoords = (
                entry["pdb"],
                entry["aseq"],
                # entry['acoords']['unbound'],
                entry["acoords"]["bound"][:, :sel],  # bound coords used for evaluation
                entry["bseq"],
                # entry['bcoords']['unbound'],
                entry["bcoords"]["bound"],
                entry["acoords"]["bound"],  # bound coords used for evaluation
                entry["bcoords"]["bound"],
            )
            # if len(aseq) == len(acoords) and len(bseq) == len(bcoords):
            entry = {"pdb": pdb}
            entry["binder_seq"] = bseq
            entry["binder_coords"] = torch.tensor(bcoords).float()[:, :sel]
            entry["binder_coords_bnd"] = torch.tensor(bcoords_bnd).float()[:, :sel]
            # entry["binder_dihedrals"] = torch.zeros(len(bseq), 6)
            entry["binder_dihedrals"] = compute_dihedrals(entry["binder_coords"])
        

            entry["binder_atypes"] = []
            for s in entry["binder_seq"]:
                if s not in ALPHABET:
                    s = "#"
                a = RES_ATOM14[ALPHABET.index(s)]
                entry["binder_atypes"].append([ATOM_TYPES.index(c) for c in a])
            entry["binder_atypes"] = torch.tensor(entry["binder_atypes"])

            entry["target_seq"] = aseq
            entry["target_coords"] = torch.tensor(acoords).float()[:, :sel]
            mean = entry["target_coords"].reshape(-1, 3).mean(dim=0, keepdim=True)[:, None, ...]
            entry["binder_coords"] = entry["binder_coords"] - mean
            entry["target_coords"] = entry["target_coords"] - mean

            entry["target_coords_bnd"] = torch.tensor(acoords_bnd).float()[:, :sel]
            # entry["target_dihedrals"] = torch.zeros(len(aseq), 6)
            entry["target_dihedrals"] = compute_dihedrals(entry["target_coords"])

            entry["target_atypes"] = []
            for s in entry["target_seq"]:
                if s not in ALPHABET:
                    s = "#"
                a = RES_ATOM14[ALPHABET.index(s)]
                entry["target_atypes"].append([ATOM_TYPES.index(c) for c in a])
            entry["target_atypes"] = torch.tensor(entry["target_atypes"])

            tck_bind, u_bind = parametrize_backbone_spline(
                torch.tensor(bcoords), use_all_atoms=True
            )
            tck_tgt, u_tgt = parametrize_backbone_spline(
                torch.tensor(acoords), use_all_atoms=True
            )

            torsion_bind = compute_spline_torsion(tck_bind, u_bind)
            curvature_bind = compute_spline_curvature(tck_bind, u_bind)

            entry["binder_spline_curvature"] = torch.tensor(curvature_bind).float()
            entry["binder_spline_torsion"] = torch.tensor(torsion_bind).float()

            torsion_tgt = compute_spline_torsion(tck_tgt, u_tgt)
            curvature_tgt = compute_spline_curvature(tck_tgt, u_tgt)

            entry["target_spline_curvature"] = torch.tensor(curvature_tgt).float()
            entry["target_spline_torsion"] = torch.tensor(torsion_tgt).float()

            DockingDataset.make_site(entry, "target", resolution=resolution)
            DockingDataset.make_site(entry, "binder", resolution=resolution)
            self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def make_site(entry, field, resolution="all_atom"):
        if resolution == "all_atom":
            sel = 14
        elif resolution == "backbone":
            sel = 4
        elif resolution == "CA":
            sel = 1
        else:
            raise ValueError(f"Unknown resolution: {resolution}")

        entry[f"{field}_full"] = entry[f"{field}_seq"]
        entry[f"{field}_full_coords"] = entry[f"{field}_coords"]
        entry[f"{field}_coords"] = entry[f"{field}_coords"][:, :sel]
        entry[f"{field}_atypes"] = entry[f"{field}_atypes"][:, :sel]
        entry[f"{field}_dihedrals"] = entry[f"{field}_dihedrals"]

    @staticmethod
    def make_local_batch(
        batch,
        embedding,
        args,
        binder,
        target,
        device,
        one_hot_embedding=None,
        patch_size=50,
        dry=False,
        resolution="all_atom",
    ):
        bind_X, bind_S, bind_A, bind_V, bind_O, bind_X_bnd, bind_T, bind_C = featurize_docking(
            batch, binder, device=device, patch_size=patch_size, resolution=resolution
        )
        tgt_X, tgt_S, tgt_A, tgt_V, tgt_O, tgt_X_bnd, tgt_T, tgt_C = featurize_docking(
            batch, target, device=device, patch_size=patch_size, resolution=resolution
        )
        if dry:
            return bind_X_bnd, tgt_X_bnd

        if embedding is not None:
            bind_S = torch.zeros(bind_S.size(0), bind_S.size(1), args.embedding_size)
            tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), args.embedding_size)
        else:
            bind_S = torch.zeros(bind_S.size(0), bind_S.size(1), 1, dtype=torch.long)
            tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), 1, dtype=torch.long)

        for i, b in enumerate(batch):
            L_tgt = len(b[f"{target}_full"])
            L_bind = len(b[f"{binder}_full"])

            # tgt_temp_seq = b[f"{target}_full"].replace("X", "#")
            # bnd_temp_seq = b[f"{binder}_full"].replace("X", "#")
            b[f"{target}_full"] = b[f"{target}_full"].replace("X", "#")
            b[f"{binder}_full"] = b[f"{binder}_full"].replace("X", "#")

            if embedding is not None:
                tgt_S[i, : L_tgt] = embedding[b[f"{target}_full"]]
                bind_S[i, : L_bind] = embedding[b[f"{binder}_full"]]
            else:
                tgt_indices = torch.tensor(
                    [ALPHABET.index(a) for a in b[f"{target}_full"]], device=device
                )
                tgt_S[i, : L_tgt] = tgt_indices.unsqueeze(-1)
                bnd_indices = torch.tensor(
                    [ALPHABET.index(a) for a in b[f"{binder}_full"]], device=device
                )
                bind_S[i, : L_bind] = bnd_indices.unsqueeze(-1)
            tgt_O[i, : L_tgt] = torch.tensor(
                [ALPHABET.index(a) for a in b[f"{target}_full"]], device=device
            )
            bind_O[i, : L_bind] = torch.tensor(
                [ALPHABET.index(a) for a in b[f"{binder}_full"]], device=device
            )

        bind_features = Features(
            X=bind_X,
            S=bind_S,
            A=bind_A,
            V=bind_V,
            O=bind_O,
            T=bind_T,
            C=bind_C,
            Si=bnd_indices,
            pdbs=[b['pdb'] for b in batch],
        )
        tgt_features = Features(
            X=tgt_X,
            S=tgt_S,
            A=tgt_A,
            V=tgt_V,
            O=tgt_O,
            T=tgt_T,
            C=tgt_C,
            Si=tgt_indices,
            pdbs=[b['pdb'] for b in batch],
        )

        return (
            bind_features,
            tgt_features,
            bind_X_bnd,
            tgt_X_bnd,
        )


def get_indices(binder_coords, target_coords, patch_size, strategy="dist_com", resolution="all_atom"):
    sel = 0 if resolution == "CA" else 1
    if strategy == "dist_com":  # residues closest to each other's centers of mass
        binder_coords_centered = binder_coords - binder_coords[:, sel].mean(
            dim=0, keepdims=True
        )
        target_coords_centered = target_coords - target_coords[:, sel].mean(
            dim=0, keepdims=True
        )

        # Find target residues closest to binder's center of mass (origin after centering)
        dist = target_coords_centered[:, sel] - binder_coords_centered[:, sel].mean(
            dim=0, keepdims=True
        )
        patch_size = min(patch_size, len(dist))
        target_idx = dist.norm(dim=-1).sort().indices[:patch_size].sort().values

        # Find binder residues closest to target's center of mass (origin after centering)
        dist = binder_coords_centered[:, sel] - target_coords_centered[:, sel].mean(
            dim=0, keepdims=True
        )
        binder_idx = dist.norm(dim=-1).sort().indices[:patch_size].sort().values

    elif (
        strategy == "com"
    ):  # residues around their centers of mass independent of distance between them
        binder_com = binder_coords[:, sel].mean(dim=0, keepdims=True)
        target_com = target_coords[:, sel].mean(dim=0, keepdims=True)

        binder_dist = (binder_coords[:, sel] - binder_com).norm(dim=-1)
        target_dist = (target_coords[:, sel] - target_com).norm(dim=-1)

        binder_idx = binder_dist.sort().indices[:patch_size].sort().values
        target_idx = target_dist.sort().indices[:patch_size].sort().values

    elif strategy == "random":
        target_idx = torch.randperm(len(target_coords))[:patch_size]
        binder_idx = torch.randperm(len(binder_coords))[:patch_size]

    else:
        raise ValueError(f"Unknown patch strategy: {strategy}")

    return target_idx, binder_idx


class ProteinDataset:
    def __init__(
        self, data_path, patch_size, patch_strategy="dist_com", resolution="all_atom"
    ):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.data = []
        bb_idx = 1
        if resolution == "all_atom":
            sel = 14
        elif resolution == "backbone":
            sel = 4
        elif resolution == "CA":
            sel = 1
            bb_idx = 0
        else:
            raise ValueError(f"Unknown resolution: {resolution}")
        self.sel = sel
        self.all_translations = {}
        # self.outlier_idx = set(torch.load("outlier_indices.pt"))

        for idx, (pdb, aseq, acoords, bseq, bcoords) in tqdm(enumerate(data)):
            # if idx in self.outlier_idx:
            #     print(f"Skipping outlier entry {pdb} at index {idx}.")
            #     continue
            # elif idx > 1: break

            # if len(aseq) == 3119 or len(bseq) == 3119:
            #     continue

            if len(aseq) == len(acoords) and len(bseq) == len(bcoords):
                entry = {"pdb": pdb}
                entry["binder_seq"] = bseq
                entry["binder_coords"] = torch.tensor(bcoords).float()[:, :sel]

                # if torch.all(entry["binder_coords"] == 0) or torch.all(entry["binder_coords"][0] == 0):
                if (entry['binder_coords'] == 0).all(dim=2).any(dim=1).any():
                    print(f"Skipping entry {pdb} due to zero coordinates in binder.")
                    continue

                # entry["binder_dihedrals"] = torch.zeros(len(bseq), 6)
                entry["binder_dihedrals"] = compute_dihedrals(
                    torch.tensor(bcoords)
                )
                entry["binder_atypes"] = torch.tensor(
                    [
                        [ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]]
                        for s in entry["binder_seq"]
                    ]
                )
                entry["target_seq"] = aseq
                entry["target_coords"] = torch.tensor(acoords).float()[:, :sel]

                # mean = entry["target_coords"][:,bb_idx,:].mean(dim=0)
                # mean =  entry["target_coords"].mean(dim=1).mean(dim=0)
                target_com = entry["target_coords"].reshape(-1, 3).mean(dim=0, keepdim=True)[:, None, ...]
                entry["target_coords"] = entry["target_coords"] - target_com
                entry["binder_coords"] = entry["binder_coords"] - target_com


                binder_com = entry["binder_coords"].reshape(-1, 3).mean(dim=0, keepdim=True)[:, None, ...]

                # translation_diff = binder_com - target_com
                translation_diff = target_com - binder_com
                self.all_translations[pdb] = translation_diff.reshape(-1)


                # if torch.all(entry["target_coords"] == 0) or torch.all(entry["target_coords"][0][0] == 0):
                if (entry["target_coords"] == 0).all(dim=2).any(dim=1).any():
                    print(f"Skipping entry {pdb} due to zero coordinates in target.")
                    continue

                # entry["target_dihedrals"] = torch.zeros(len(aseq), 6)
                entry["target_dihedrals"] = compute_dihedrals(
                    torch.tensor(acoords)
                )
                entry["target_atypes"] = torch.tensor(
                    [
                        [ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]]
                        for s in entry["target_seq"]
                    ]
                )

                # entry["target_idx"], entry["binder_idx"] = get_indices(
                #     entry["binder_coords"],
                #     entry["target_coords"],
                #     max(len(entry['binder_coords']), len(entry['target_coords'])),
                #     # patch_size,
                #     strategy=patch_strategy,
                #     resolution=resolution,
                # )

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

                entry["binder_spline_curvature"] = torch.tensor(curvature_bind).float().view(M, sel, 1)
                entry["binder_spline_torsion"] = torch.tensor(torsion_bind).float().view(M, sel, 1)

                torsion_tgt = compute_spline_torsion(tck_tgt, u_tgt)
                curvature_tgt = compute_spline_curvature(tck_tgt, u_tgt)

                entry["target_spline_curvature"] = torch.tensor(curvature_tgt).float().view(N, sel, 1)
                entry["target_spline_torsion"] = torch.tensor(torsion_tgt).float().view(N, sel, 1)

                ProteinDataset.make_site(entry, "target", resolution=resolution)
                ProteinDataset.make_site(entry, "binder", resolution=resolution)
                self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def welford_std_coords(self, idx):
        mean = torch.tensor(0.0, dtype=torch.float64)
        m2 = torch.tensor(0.0, dtype=torch.float64)
        n = 0

        for entry in self.data[idx]:
            flat = entry["binder_coords"].reshape(-1)
            k = flat.numel()
            if k == 0:
                continue

            n += k

            delta = flat - mean
            mean += delta.sum() / n
            m2 += (delta * (flat - mean)).sum()

        return torch.sqrt(m2 / (n - 1)).item()
    

    @staticmethod
    def make_site(entry, field, resolution="all_atom"):
        if resolution == "all_atom":
            sel = 14
        elif resolution == "backbone":
            sel = 4
        elif resolution == "CA":
            sel = 1
        else:
            raise ValueError(f"Unknown resolution: {resolution}")

        entry[f"{field}_full"] = entry[f"{field}_seq"]
        entry[f"{field}_coords"] = entry[f"{field}_coords"][:, :sel]
        entry[f"{field}_atypes"] = entry[f"{field}_atypes"][:, :sel]
        entry[f"{field}_dihedrals"] = entry[f"{field}_dihedrals"]

    @staticmethod
    def make_local_batch(
        batch,
        embedding,
        args,
        binder,
        target,
        device,
        one_hot_embedding=None,
        patch_size=50,
        resolution="all_atom",
        supernodes=10
    ):
        bind_X, bind_S, bind_A, bind_V, bind_O, bind_T, bind_C = featurize(
            batch, binder, device=device, patch_size=patch_size, resolution=resolution
        )
        tgt_X, tgt_S, tgt_A, tgt_V, tgt_O, tgt_T, tgt_C = featurize(
            batch, target, device=device, patch_size=patch_size, resolution=resolution
        )

        if embedding is not None:
            bind_S = torch.zeros(bind_S.size(0), bind_S.size(1), args.embedding_size)
            tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), args.embedding_size)
        else:
            bind_S = torch.zeros(bind_S.size(0), bind_S.size(1), 1, dtype=torch.long)
            tgt_S = torch.zeros(tgt_S.size(0), tgt_S.size(1), 1, dtype=torch.long)

        for i, b in enumerate(batch):
            L_tgt = len(b[f"{target}_full"])
            L_bind = len(b[f"{binder}_full"])

            if embedding is not None:
                tgt_S[i, : L_tgt] = embedding[b[f"{target}_full"]]
                bind_S[i, : L_bind] = embedding[b[f"{binder}_full"]]
            else:
                tgt_indices = torch.tensor(
                    [ALPHABET.index(a) for a in b[f"{target}_full"]], device=device
                )
                tgt_S[i, : L_tgt] = tgt_indices.unsqueeze(-1)
                bnd_indices = torch.tensor(
                    [ALPHABET.index(a) for a in b[f"{binder}_full"]], device=device
                )
                bind_S[i, : L_bind] = bnd_indices.unsqueeze(-1)

            tgt_O[i, : L_tgt] = torch.tensor(
                [ALPHABET.index(a) for a in b[f"{target}_full"]], device=device
            )
            bind_O[i, : L_bind] = torch.tensor(
                [ALPHABET.index(a) for a in b[f"{binder}_full"]], device=device
            )

        bind_features = Features(
            bind_X,
            bind_S,
            bind_A,
            bind_V,
            bind_O,
            bind_T,
            bind_C,
        )
        tgt_features = Features(
            tgt_X,
            tgt_S,
            tgt_A,
            tgt_V,
            tgt_O,
            tgt_T,
            tgt_C,
        )

        return bind_features, tgt_features

    def __shuffle__(self):
        random.shuffle(self.data)
    
# class ProteinDataset:
#     # --------------------------------------------------------------------- #
#     # NEW HELPERS
#     # --------------------------------------------------------------------- #
#     @staticmethod
#     def _pad_to_multiple(t, k, pad_value=0):
#         """Pad length‑dim (dim=1) of `t` so it is a multiple of k."""
#         if k <= 1:
#             return t
#         B, L, *rest = t.shape
#         pad = (-L) % k
#         if pad:
#             pad_shape = (B, pad, *rest)
#             pad_tensor = torch.full(pad_shape,
#                                     pad_value,
#                                     dtype=t.dtype,
#                                     device=t.device)
#             t = torch.cat([t, pad_tensor], dim=1)
#         return t

#     @staticmethod
#     def _coarsen_numeric(t, supernodes, reduction="mean"):
#         """Average (or max) over `supernodes`‑sized windows on dim‑1."""
#         if supernodes <= 1:
#             return t
#         t = ProteinDataset._pad_to_multiple(t, supernodes, pad_value=0)
#         B, L, *rest = t.shape
#         t = t.reshape(B, L // supernodes, supernodes, *rest)
#         if reduction == "mean":
#             return t.mean(dim=2)
#         elif reduction == "max":
#             return t.max(dim=2).values
#         else:
#             raise ValueError(f"Unknown numeric reduction: {reduction}")

#     @staticmethod
#     def _coarsen_categorical(t, supernodes):
#         """Keep the FIRST residue in every window (mode/majority can be added later)."""
#         if supernodes <= 1:
#             return t
#         t = ProteinDataset._pad_to_multiple(t, supernodes, pad_value=0)
#         B, L, *rest = t.shape
#         t = t.reshape(B, L // supernodes, supernodes, *rest)
#         return t[:, :, 0]  # first element in each window

#     # --------------------------------------------------------------------- #
#     # MODIFIED make_local_batch
#     # --------------------------------------------------------------------- #
#     @staticmethod
#     def make_local_batch(
#         batch,
#         embedding,
#         args,
#         binder,
#         target,
#         device,
#         one_hot_embedding=None,   # unused, kept for API stability
#         patch_size=50,
#         resolution="all_atom",
#         n_supernodes=10,            # <‑‑ NEW
#     ):
#         # -------- original featurisation -------- #
#         bind_X, bind_S, bind_A, bind_V, bind_D, bind_O = featurize(
#             batch, binder, device=device, patch_size=patch_size, resolution=resolution
#         )
#         tgt_X,  tgt_S,  tgt_A,  tgt_V,  tgt_D,  tgt_O = featurize(
#             batch, target, device=device, patch_size=patch_size, resolution=resolution
#         )

#         # allocate S according to embedding availability
#         if embedding is not None:
#             bind_S = torch.zeros(bind_S.size(0), bind_S.size(1), args.esm_size,
#                                  device=bind_S.device)
#             tgt_S  = torch.zeros(tgt_S.size(0),  tgt_S.size(1),  args.esm_size,
#                                  device=tgt_S.device)
#         else:
#             bind_S = torch.zeros(bind_S.size(0), bind_S.size(1), 1,
#                                  dtype=torch.long, device=bind_S.device)
#             tgt_S  = torch.zeros(tgt_S.size(0),  tgt_S.size(1),  1,
#                                  dtype=torch.long, device=tgt_S.device)

#         # -------- fill per‑residue embeddings / indices -------- #
#         for i, b in enumerate(batch):
#             L_tgt  = len(b[f"{target}_full"])
#             L_bind = len(b[f"{binder}_full"])

#             if embedding is not None:
#                 tgt_S[i,  :L_tgt]  = embedding[b[f"{target}_full"]]
#                 bind_S[i, :L_bind] = embedding[b[f"{binder}_full"]]
#             else:
#                 tgt_S[i,  :L_tgt, 0]  = torch.tensor(
#                     [ALPHABET.index(a) for a in b[f"{target}_full"]], device=device)
#                 bind_S[i, :L_bind, 0] = torch.tensor(
#                     [ALPHABET.index(a) for a in b[f"{binder}_full"]], device=device)

#             tgt_O[i,  :L_tgt]  = torch.tensor(
#                 [ALPHABET.index(a) for a in b[f"{target}_full"]], device=device)
#             bind_O[i, :L_bind] = torch.tensor(
#                 [ALPHABET.index(a) for a in b[f"{binder}_full"]], device=device)

#         # ------------------------------------------------------------------ #
#         # SUPER‑NODE COARSENING (only if n_supernodes > 1)
#         # ------------------------------------------------------------------ #
#         if n_supernodes and n_supernodes > 1:
#             # numeric tensors – take MEAN
#             bind_X = ProteinDataset._coarsen_numeric(bind_X, n_supernodes, reduction="mean")
#             tgt_X  = ProteinDataset._coarsen_numeric(tgt_X,  n_supernodes, reduction="mean")
#             bind_V = ProteinDataset._coarsen_numeric(bind_V, n_supernodes, reduction="max")
#             tgt_V  = ProteinDataset._coarsen_numeric(tgt_V,  n_supernodes, reduction="max")

#             # sequence embeddings
#             if embedding is not None:                # float – mean
#                 bind_S = ProteinDataset._coarsen_numeric(bind_S, n_supernodes, reduction="mean")
#                 tgt_S  = ProteinDataset._coarsen_numeric(tgt_S,  n_supernodes, reduction="mean")
#             else:                                    # categorical – first residue
#                 bind_S = ProteinDataset._coarsen_categorical(bind_S, n_supernodes)
#                 tgt_S  = ProteinDataset._coarsen_categorical(tgt_S,  n_supernodes)

#             # categorical add‑ons need the SAME length after coarsening
#             bind_A = ProteinDataset._coarsen_categorical(bind_A, n_supernodes)
#             tgt_A  = ProteinDataset._coarsen_categorical(tgt_A,  n_supernodes)
#             bind_O = ProteinDataset._coarsen_categorical(bind_O, n_supernodes)
#             tgt_O  = ProteinDataset._coarsen_categorical(tgt_O,  n_supernodes)

#         # ------------------------------------------------------------------ #
#         return (bind_X, bind_S, bind_A, bind_O, bind_V), \
#                (tgt_X,  tgt_S,  tgt_A,  tgt_O, tgt_V)

