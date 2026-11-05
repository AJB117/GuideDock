import argparse
import gzip
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.constants import ALPHABET, RES_ATOM14, RESTYPE_1to3

_STRUCTURE_INDEX = {}


def normalize_key(key):
    pdb, separator, rest = key.strip().partition("_")
    return pdb.upper() + separator + rest


def initialize_worker(index):
    global _STRUCTURE_INDEX
    _STRUCTURE_INDEX = index


def build_indexed_entry(key):
    return build_entry((key, _STRUCTURE_INDEX))


@lru_cache(maxsize=8)
def load_structure(path):
    path = Path(path)
    parser = (
        MMCIFParser(QUIET=True)
        if ".cif" in [suffix.lower() for suffix in path.suffixes]
        else PDBParser(QUIET=True)
    )
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt") as handle:
        return next(parser.get_structure(path.name, handle).get_models())


def extract_chain_group(model, chain_ids=None):
    chain_ids = list(model.child_dict) if chain_ids is None else chain_ids
    restype_3to1 = {value: key for key, value in RESTYPE_1to3.items()}
    sequence, coordinates = [], []
    for chain_id in chain_ids:
        if chain_id not in model:
            raise ValueError(f"Missing chain {chain_id!r}")
        for residue in model[chain_id]:
            residue_type = restype_3to1.get(residue.resname)
            if (
                residue_type not in ALPHABET
                or residue_type == "#"
                or not all(atom in residue for atom in ("N", "CA", "C"))
            ):
                continue
            template = RES_ATOM14[ALPHABET.index(residue_type)]
            coords = np.zeros((14, 3), dtype=np.float32)
            for index, atom in enumerate(template):
                if atom and atom in residue:
                    coords[index] = residue[atom].coord
            if not np.isfinite(coords).all():
                raise ValueError(f"Nonfinite coordinates in chain {chain_id!r}")
            sequence.append(residue_type)
            coordinates.append(coords)
    if not sequence:
        raise ValueError("No standard residues with complete N/CA/C atoms")
    return "".join(sequence), np.stack(coordinates)


def index_structures(directory):
    extensions = (".pdb", ".cif", ".pdb.gz", ".cif.gz")
    index = {}
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        extension = next(
            (suffix for suffix in extensions if path.name.lower().endswith(suffix)),
            None,
        )
        if extension is None:
            continue
        key = normalize_key(path.name[: -len(extension)])
        if key.lower().endswith(("_r_b", "_l_b")):
            key = key[:-4] + key[-4:].upper()
        if key in index:
            raise ValueError(f"Ambiguous structure {key}: {index[key]} and {path}")
        index[key] = str(path)
    if not index:
        raise FileNotFoundError(f"No PDB or mmCIF structures found in {directory}")
    return index


def build_entry(item):
    key, index = item
    try:
        pdb, binder_group, receptor_group = key.split("_")
        receptor_path, binder_path = index.get(key + "_R_B"), index.get(key + "_L_B")
        if receptor_path and binder_path:
            aseq, acoords = extract_chain_group(load_structure(receptor_path))
            bseq, bcoords = extract_chain_group(load_structure(binder_path))
        else:
            path = index.get(key) or index.get(pdb)
            if path is None:
                raise FileNotFoundError(f"No structure found for {key}")
            binder_ids = (
                binder_group.split(",") if "," in binder_group else list(binder_group)
            )
            receptor_ids = (
                receptor_group.split(",")
                if "," in receptor_group
                else list(receptor_group)
            )
            if set(binder_ids) & set(receptor_ids):
                raise ValueError("Binder and receptor chains overlap")
            model = load_structure(path)
            aseq, acoords = extract_chain_group(model, receptor_ids)
            bseq, bcoords = extract_chain_group(model, binder_ids)
        return (key, aseq, acoords, bseq, bcoords), None
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        StopIteration,
        PDBConstructionException,
    ) as error:
        return None, f"{key}: {error}"


def build_dipshet(structures, keys, output, workers=1, skip_invalid=False):
    if workers < 1:
        raise ValueError("workers must be positive")
    index = index_structures(structures)
    keys = list(dict.fromkeys(normalize_key(key) for key in keys if key.strip()))
    if not keys:
        raise ValueError("No complex identifiers supplied")
    items = ((key, index) for key in keys)
    if workers == 1:
        results = list(map(build_entry, items))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=initialize_worker, initargs=(index,)
        ) as pool:
            results = list(pool.map(build_indexed_entry, keys, chunksize=16))
    entries = [entry for entry, error in results if error is None]
    errors = [error for entry, error in results if error is not None]
    if errors and not skip_invalid:
        raise ValueError(
            f"Failed to build {len(errors)} complexes:\n" + "\n".join(errors[:20])
        )
    if not entries:
        raise ValueError("No valid complexes were built")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return len(entries), errors


def main():
    split_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build GuideDock atom14 tuples from DIPS-Het structures"
    )
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument(
        "--keys",
        type=Path,
        nargs="+",
        help="Complex IDs in PDB_BINDER_RECEPTOR format; defaults to bundled train/val/test splits",
    )
    parser.add_argument("--output", type=Path, default=split_dir / "data.pkl")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args()
    key_files = args.keys or [
        split_dir / f"{split}.txt" for split in ("train", "val", "test")
    ]
    keys = list(
        dict.fromkeys(
            key
            for path in key_files
            for key in path.read_text().splitlines()
            if key.strip()
        )
    )
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        keys = keys[: args.limit]
    count, errors = build_dipshet(
        args.structures, keys, args.output, args.workers, args.skip_invalid
    )
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Wrote {count} complexes to {args.output}; skipped {len(errors)}")


if __name__ == "__main__":
    main()
