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

## Checks

```bash
python -m unittest discover -s tests -v
```
