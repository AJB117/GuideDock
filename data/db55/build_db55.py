import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from biopandas.pdb import PandasPdb
from tqdm import tqdm

from utils.constants import ALPHABET, RES_ATOM14, RESTYPE_1to3


def filter_residues(residues):
    residues_filtered = []
    for residue in residues:
        df = residue[1]
        Natom = df[df["atom_name"] == "N"]
        alphaCatom = df[df["atom_name"] == "CA"]
        Catom = df[df["atom_name"] == "C"]
        if Natom.shape[0] == 1 and alphaCatom.shape[0] == 1 and Catom.shape[0] == 1:
            residues_filtered.append(residue)

    return residues_filtered


def extract_sequence_and_coords(pdb_filename):
    """Extract atom coordinates from PDB file using biopandas and reshape to [N, 14, 3]."""
    ppdb = PandasPdb().read_pdb(pdb_filename)
    df = ppdb.df["ATOM"]

    df.rename(
        columns={
            "chain_id": "chain",
            "residue_number": "residue",
            "residue_name": "resname",
            "x_coord": "x",
            "y_coord": "y",
            "z_coord": "z",
            "element_symbol": "element",
        },
        inplace=True,
    )

    restype_3to1 = {v: k for k, v in RESTYPE_1to3.items()}

    coordinates = []
    seq = []

    residues = list(df.groupby(["chain", "residue", "resname"]))
    residues = filter_residues(residues)

    for (chain_id, res_num, res_name), res_df in residues:
        res_1letter = restype_3to1.get(res_name, "#")
        if res_1letter == "#":
            res_1letter = "X"
        seq.append(res_1letter)

        res_idx = ALPHABET.index(res_1letter) if res_1letter in ALPHABET else 0

        expected_atoms = RES_ATOM14[res_idx]

        res_coords = np.zeros((14, 3), dtype=np.float32)

        for _, row in res_df.iterrows():
            atom_name = row["atom_name"].strip()

            if atom_name in expected_atoms:
                atom_idx = expected_atoms.index(atom_name)
                res_coords[atom_idx] = [row["x"], row["y"], row["z"]]

        coordinates.append(res_coords)

    coordinates = np.array(coordinates, dtype=np.float32)
    seq = "".join(seq)

    assert len(seq) == len(coordinates)

    return seq, coordinates


def process(tup):
    receptor_bnd = tup["receptor_bnd"]
    ligand_bnd = tup["ligand_bnd"]
    receptor_unbnd = tup["receptor_unbnd"]
    ligand_unbnd = tup["ligand_unbnd"]

    receptor_file_bnd = f"benchmark5.5/structures/{receptor_bnd}.pdb"
    ligand_file_bnd = f"benchmark5.5/structures/{ligand_bnd}.pdb"

    receptor_file_unbnd = f"benchmark5.5/structures/{receptor_unbnd}.pdb"
    ligand_file_unbnd = f"benchmark5.5/structures/{ligand_unbnd}.pdb"

    aseq_bnd, acoords_bnd = extract_sequence_and_coords(receptor_file_bnd)
    bseq_bnd, bcoords_bnd = extract_sequence_and_coords(ligand_file_bnd)

    aseq_unbnd, acoords_unbnd = extract_sequence_and_coords(receptor_file_unbnd)
    bseq_unbnd, bcoords_unbnd = extract_sequence_and_coords(ligand_file_unbnd)

    assert len(aseq_bnd) == len(acoords_bnd)
    assert len(bseq_bnd) == len(bcoords_bnd)

    return {
        "pdb": receptor_bnd + ligand_bnd,
        "receptor": receptor_bnd,
        "ligand": ligand_bnd,
        "aseq": aseq_bnd,
        "acoords": {
            "bound": acoords_bnd,
            "unbound": acoords_unbnd,
        },
        "bseq": bseq_bnd,
        "bcoords": {
            "bound": bcoords_bnd,
            "unbound": bcoords_unbnd,
        },
    }


if __name__ == "__main__":
    data = []
    visited = set()
    ab_map = defaultdict(list)
    with open("Table_BM5.5.csv", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            if not row["Complex"]:
                continue
            try:
                cplx, chains = row["Complex"].split("_")
            except:
                continue
            chains = chains.split(":")

            receptor_bnd = f"{cplx}_r_b"
            ligand_bnd = f"{cplx}_l_b"

            receptor_unbnd = f"{cplx}_r_u"
            ligand_unbnd = f"{cplx}_l_u"

            ab_map[cplx].append(
                {
                    "receptor_bnd": receptor_bnd,
                    "ligand_bnd": ligand_bnd,
                    "receptor_unbnd": receptor_unbnd,
                    "ligand_unbnd": ligand_unbnd,
                    "receptor_chains": chains[0],
                    "ligand_chains": chains[1],
                }
            )

    data = []
    for cplx in tqdm(ab_map):
        for entry in ab_map[cplx]:
            data.append(entry)

    with Pool(120) as pool:
        processed_data = pool.map(process, data)

    print(f"Total processed complexes: {len(processed_data)}")
    with open("data.pkl", "wb") as f:
        pickle.dump(processed_data, f)
