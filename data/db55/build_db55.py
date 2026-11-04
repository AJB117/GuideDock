import csv
import numpy as np
import pickle
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm
from biopandas.pdb import PandasPdb
RESTYPE_1to3 = {
     "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN","E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    #  "U": "SEC"
}
ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 
# 'U'
]
ATOM_TYPES = [
    '', 'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
    # 'SE'
]
RES_ATOM14 = [
    [''] * 14,
    ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    # ['N', 'CA', 'C', 'O', 'CB', 'SE', '', '', '', '', '', '', '', '']
]
AA_WEIGHTS = {"G":57, "A":71, "S":87, "P":97, "V":99, "T":101, "C":103, "I":113, "L":113, "N":114, "D":115, "Q":128, "K":128, "E":129, "M":131, "H":137, "F":147, "R":156, "Y":163, "W":186}
ATOM_WEIGHTS = {"C": 12, "N": 14, "O": 16, "S": 32, "Cl": 35, "Si": 28, "F": 19, "Ca": 40}

def filter_residues(residues):
    residues_filtered = []
    for residue in residues:
        df = residue[1]
        Natom = df[df['atom_name'] == 'N']
        alphaCatom = df[df['atom_name'] == 'CA']
        Catom = df[df['atom_name'] == 'C']
        if Natom.shape[0] == 1 and alphaCatom.shape[0] == 1 and Catom.shape[0] == 1:
            residues_filtered.append(residue)

    return residues_filtered

def extract_sequence_and_coords(pdb_filename):
    """Extract atom coordinates from PDB file using biopandas and reshape to [N, 14, 3]."""
    ppdb = PandasPdb().read_pdb(pdb_filename)
    df = ppdb.df['ATOM']

    # Rename columns for consistency
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 
                      'residue_name': 'resname', 'x_coord': 'x', 'y_coord': 'y', 
                      'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    
    # Create a mapping from 3-letter to 1-letter residue codes
    restype_3to1 = {v: k for k, v in RESTYPE_1to3.items()}
    
    # Extract sequence (for alignment if needed)
    # seq = []
    # for _, residue_group in df.groupby(['chain', 'residue', 'resname']):
    #     seq.append(restype_3to1.get(residue_group['resname'].iloc[0], '#'))
    # seq = "".join(seq)
    
    # Structure coordinates by residue and atom type
    coordinates = []
    seq = []
    
    # Group by residue
    residues = list(df.groupby(['chain', 'residue', 'resname']))
    residues = filter_residues(residues)
    # for (chain_id, res_num, res_name), res_df in df.groupby(['chain', 'residue', 'resname']):
    for (chain_id, res_num, res_name), res_df in residues:
        # Convert 3-letter code to 1-letter code
        res_1letter = restype_3to1.get(res_name, '#')
        if res_1letter == '#':
            res_1letter = 'X'  # Use 'X' for unknown residues
        seq.append(res_1letter)

        # Get residue type index in ALPHABET
        res_idx = ALPHABET.index(res_1letter) if res_1letter in ALPHABET else 0
        
        # Get expected atom names for this residue type
        expected_atoms = RES_ATOM14[res_idx]

        # Initialize empty coordinates for this residue
        res_coords = np.zeros((14, 3), dtype=np.float32)
        
        # Fill in coordinates for existing atoms
        for _, row in res_df.iterrows():
            atom_name = row['atom_name'].strip()
            # Find position in the standard 14-atom representation
            if atom_name in expected_atoms:
                atom_idx = expected_atoms.index(atom_name)
                res_coords[atom_idx] = [row['x'], row['y'], row['z']]

            # else:
            #     print(atom_name)
            #     missing += 1
        
        coordinates.append(res_coords)
    
    # Convert to numpy array
    coordinates = np.array(coordinates, dtype=np.float32)
    seq = ''.join(seq)

    assert len(seq) == len(coordinates)
    
    return seq, coordinates

def process(tup):
    receptor_bnd = tup["receptor_bnd"]
    ligand_bnd = tup["ligand_bnd"]
    receptor_unbnd = tup["receptor_unbnd"]
    ligand_unbnd = tup["ligand_unbnd"]
    receptor_chains = tup["receptor_chains"]
    ligand_chains = tup["ligand_chains"]
    
    # Extract bound structures
    receptor_file_bnd = f"benchmark5.5/structures/{receptor_bnd}.pdb"
    ligand_file_bnd = f"benchmark5.5/structures/{ligand_bnd}.pdb"
    
    # Extract unbound structures
    receptor_file_unbnd = f"benchmark5.5/structures/{receptor_unbnd}.pdb"
    ligand_file_unbnd = f"benchmark5.5/structures/{ligand_unbnd}.pdb"

    # Get sequences and coordinates using BioPandas with reshaping
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