import os

import torch

from .constants import ALPHABET, RES_ATOM14, RESTYPE_1to3


def save_trajectory_to_pdb(
    coords_trajectory,
    sequence,
    filename,
    chain_id="A",
    time_points=None,
    model_labels=None,
):
    """
    Save protein trajectory (multiple time steps) to a multi-model PDB file.

    Args:
        coords_trajectory: torch.Tensor of shape [T, N, 14, 3] - T time steps, N residues, 14 atoms each
                          or list of tensors with shape [N, 14, 3]
        sequence: str - amino acid sequence
        filename: str - output PDB filename
        chain_id: str - PDB chain identifier
        time_points: list of float - optional time points for each frame (for REMARK)
        model_labels: list of str - optional labels for each model
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if isinstance(coords_trajectory, list):
        num_models = len(coords_trajectory)
    else:
        if len(coords_trajectory.shape) == 3:
            coords_trajectory = coords_trajectory.unsqueeze(0)
        num_models = coords_trajectory.shape[0]

    with open(filename, "w") as f:
        f.write("HEADER    PROTEIN TRAJECTORY\n")
        f.write(f"REMARK    Trajectory with {num_models} models\n")
        if time_points is not None:
            f.write(f"REMARK    Time points: {', '.join(map(str, time_points))}\n")

        for model_idx in range(num_models):
            if isinstance(coords_trajectory, list):
                coords = coords_trajectory[model_idx]
            else:
                coords = coords_trajectory[model_idx]

            if hasattr(coords, "numpy"):
                coords = coords.numpy()
            elif hasattr(coords, "cpu"):
                coords = coords.cpu().numpy()

            f.write(f"MODEL     {model_idx + 1:4d}\n")
            if model_labels is not None and model_idx < len(model_labels):
                f.write(f"REMARK    {model_labels[model_idx]}\n")
            if time_points is not None and model_idx < len(time_points):
                f.write(f"REMARK    Time: {time_points[model_idx]:.3f}\n")

            atom_counter = 1

            for res_idx, (res_coords, aa) in enumerate(zip(coords, sequence)):
                if aa not in ALPHABET:
                    aa = "#"

                aa_idx = ALPHABET.index(aa)
                atom_template = RES_ATOM14[aa_idx]

                aa_3letter = RESTYPE_1to3.get(aa, "UNK")

                for atom_idx, (atom_coords, atom_name) in enumerate(
                    zip(res_coords, atom_template)
                ):
                    if atom_name == "":
                        continue

                    if torch.allclose(
                        torch.tensor(atom_coords), torch.zeros(3), atol=1e-6
                    ):
                        continue

                    f.write(
                        f"ATOM  {atom_counter:5d}  {atom_name:<4s}{aa_3letter} {chain_id}{res_idx + 1:4d}    "
                        f"{atom_coords[0]:8.3f}{atom_coords[1]:8.3f}{atom_coords[2]:8.3f}"
                        f"  1.00 20.00           {atom_name[0]:>2s}\n"
                    )

                    atom_counter += 1

            f.write("ENDMDL\n")

        f.write("END\n")
