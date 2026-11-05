# GuideDock

GuideDock performs protein–protein docking with continuous distance guidance. A conditional VAE encodes inter-chain distances, and latent flow matching generates distance maps for pose sampling.

Use the existing PyTorch and Lightning environment with the dataset, embedding, and checkpoint paths supplied through the command line.

| Entry point | Purpose |
| --- | --- |
| `train_contact_vae.py` | Train the continuous-distance VAE |
| `train_contact_diff.py` | Train latent flow matching |
| `main_lightning.py` | Train or evaluate the docking model |
| `main_sample_lightning.py` | Run the same docking implementation in test mode |
| `capri_eval.py` | Summarize CAPRI metrics |

Each entry point supports `--help`. Neural building blocks live in `models/common.py`, general helpers in `utils/utils.py`, and shared data loading in `datasets/datamodule.py`.

## Build DIPS-Het

The builder accepts a directory containing PDB or mmCIF structures, including gzip files. It supports full complexes named by PDB code or complex ID, and prepared receptor/ligand pairs named `<ID>_r_b.pdb` and `<ID>_l_b.pdb`.

Complex IDs use `PDB_BINDER_RECEPTOR` order. For example, `5VLL_A_YZ` selects binder chain A and receptor chains Y/Z. Multiple one-character chains are concatenated; comma-separated groups support longer chain IDs.

```bash
python data/DIPS-Het/build_dipshet.py --structures /path/to/structures --workers 8
```

By default, IDs come from the bundled `train.txt`, `val.txt`, and `test.txt` files, and output goes to `data/DIPS-Het/data.pkl`. Use `--keys` to supply different ID files, `--limit` for a small build, or `--output` to select another destination. Invalid complexes fail the build unless `--skip-invalid` is specified. The builder requires NumPy and Biopython.

Each output record is `(id, receptor_sequence, receptor_atom14, binder_sequence, binder_atom14)`. Coordinates are float32 arrays with shape `(residues, 14, 3)` and use the shared residue templates.

## Checks

```bash
python -m unittest discover -s tests -v
```
