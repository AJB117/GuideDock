# GuideDock

GuideDock predicts the relative pose of a binder and receptor from their individual structures. It generates continuous inter-residue distance maps with latent flow matching, then uses them as soft geometric constraints to steer pose diffusion through particle resampling and rigid-body gradient updates. The goal is fast, sample-efficient rigid docking without inference-time MSAs.

## Usage

| Script | Purpose |
| --- | --- |
| `train_contact_vae.py` | Train the distance-map VAE |
| `train_contact_diff.py` | Train latent flow matching |
| `main.py` | Train or evaluate the docking model |
| `main_sample.py` | Run docking inference |
| `capri_eval.py` | Summarize CAPRI metrics |

Each script supports `--help`. Use `--resample_distance_per_pose` for independent distance guides and `--save_all_samples` to save every candidate.

Download DIPS-Het structures and build the dataset:

```bash
bash data/DIPS-Het/download.sh
python data/DIPS-Het/build_dipshet.py --structures data/DIPS-Het/structures --workers 8
```

The builder uses the bundled splits and writes `data/DIPS-Het/data.pkl`.

## Citation

Our work is in submission.

```bibtex
@misc{guidedock,
  title = {{GuideDock}: Geometry-Guided Diffusion for Rigid Protein-Protein Docking},
  author = {Soga, Patrick and Lei, Zhenyu and He, Yinhan and Bilodeau, Camille and Li, Jundong},
  year = {2026},
  note = {In submission},
}
```
