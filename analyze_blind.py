# import re
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # ============================
# # User-configurable parameters
# # ============================
# ROOT_DIR = "docking_results/top_k_ablation_noguide"
# OUTPUT_PDF = "topk_success_fraction_with_ci.pdf"

# THRESHOLDS = {
#     "cRMSD": 5.0,     # Å (lower is better)
#     "iRMSD": 4.0,     # Å (lower is better)
#     "lRMSD": 5.0,     # Å (lower is better)
#     "fNat": 0.3,      # higher is better
#     "DockQ": 0.23,    # higher is better
# }

# # Explicit legend order (IMPORTANT)
# LEGEND_ORDER = ["cRMSD", "iRMSD", "lRMSD", "fNat", "DockQ"]

# N_BOOTSTRAP = 1000
# CI_ALPHA = 0.05  # 95% CI

# # ============================
# # Matplotlib configuration
# # ============================
# plt.rcParams.update({
#     "font.family": "serif",
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

# # ============================
# # Helper functions
# # ============================
# def extract_k(folder_name):
#     match = re.search(r"top-(\d+)-noguide", folder_name)
#     if match is None:
#         raise ValueError(f"Could not parse k from folder name: {folder_name}")
#     return int(match.group(1))


# def load_meters(val_pt_path):
#     obj = torch.load(val_pt_path, map_location="cpu", weights_only=False)

#     if isinstance(obj, dict) and "meter" in obj:
#         return obj["meter"]
#     elif isinstance(obj, (list, tuple)):
#         return obj
#     else:
#         raise ValueError(
#             f"Unsupported format in {val_pt_path}. "
#             f"Expected dict with key 'meter' or list/tuple."
#         )


# def collect_metric_values(meters, metric_name):
#     values = []

#     for meter in meters:
#         if metric_name == "cRMSD":
#             values.extend(meter.complex_rmsd_list_all)
#         elif metric_name == "iRMSD":
#             values.extend(meter.interface_rmsd_list_all)
#         elif metric_name == "lRMSD":
#             values.extend(meter.ligand_rmsd_list_all)
#         elif metric_name == "fNat":
#             values.extend(meter.fnat_list_all)
#         elif metric_name == "DockQ":
#             values.extend(meter.dockQ_list_all)
#         else:
#             raise ValueError(f"Unknown metric: {metric_name}")

#     return np.asarray(values)


# def success_mask(values, metric_name, threshold):
#     if metric_name in {"DockQ", "fNat"}:
#         return values > threshold
#     else:
#         return values < threshold


# def bootstrap_ci(successes, n_bootstrap, alpha):
#     n = len(successes)
#     if n == 0:
#         return 0.0, 0.0, 0.0

#     boot_means = []
#     for _ in range(n_bootstrap):
#         sample = np.random.choice(successes, size=n, replace=True)
#         boot_means.append(sample.mean())

#     boot_means = np.asarray(boot_means)

#     mean = successes.mean()
#     lo = np.percentile(boot_means, 100 * alpha / 2)
#     hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

#     return mean, lo, hi


# # ============================
# # Main analysis
# # ============================
# def main():
#     root = Path(ROOT_DIR)

#     ks_all = []
#     results = {
#         metric: {"mean": [], "lo": [], "hi": []}
#         for metric in THRESHOLDS
#     }

#     for subdir in sorted(root.iterdir()):
#         if not subdir.is_dir():
#             continue
#         if not subdir.name.startswith("top-"):
#             continue

#         k = extract_k(subdir.name)
#         if k > 20:
#             continue
#         val_pt = subdir / "val_0.pt"

#         if not val_pt.exists():
#             print(f"[WARNING] Missing {val_pt}")
#             continue

#         meters = load_meters(val_pt)
#         ks_all.append(k)

#         for metric, threshold in THRESHOLDS.items():
#             values = collect_metric_values(meters, metric)
#             successes = success_mask(values, metric, threshold).astype(float)

#             mean, lo, hi = bootstrap_ci(
#                 successes,
#                 n_bootstrap=N_BOOTSTRAP,
#                 alpha=CI_ALPHA,
#             )

#             results[metric]["mean"].append(mean)
#             results[metric]["lo"].append(lo)
#             results[metric]["hi"].append(hi)

#     if len(ks_all) == 0:
#         raise RuntimeError("No valid top-k folders found.")

#     # Sort by k
#     order = np.argsort(ks_all)
#     ks = np.array(ks_all)[order]

#     for metric in results:
#         for key in results[metric]:
#             results[metric][key] = np.array(results[metric][key])[order]

#     # ============================
#     # Plot
#     # ============================
#     fig, ax = plt.subplots(figsize=(6.5, 4.8))

#     line_handles = {}

#     for metric in LEGEND_ORDER:
#         data = results[metric]

#         label = (
#             f"{metric} "
#             f"{'>' if metric in {'DockQ', 'fNat'} else '<'} "
#             f"{THRESHOLDS[metric]}"
#         )

#         line, = ax.plot(
#             ks,
#             data["mean"],
#             marker="o",
#             linewidth=2,
#             label=label,
#         )

#         ax.fill_between(
#             ks,
#             data["lo"],
#             data["hi"],
#             alpha=0.2,
#         )

#         line_handles[metric] = line

#     ax.set_xlabel("Number of Generated Samples (k)", fontsize=12)
#     ax.set_ylabel("Success Fraction", fontsize=12)

#     # ax.set_ylim(0.0, 1.0)
#     ax.set_ylim(0.0, 0.8)
#     ax.grid(True, linestyle="--", alpha=0.5)

#     # Legend below plot: 3 RMSD metrics on top row, 2 fNat-based below
#     fig.subplots_adjust(bottom=0.32)

#     handles = [line_handles[m] for m in LEGEND_ORDER]
#     labels = [h.get_label() for h in handles]

#     ax.legend(
#         handles,
#         labels,
#         loc="upper center",
#         bbox_to_anchor=(0.5, -0.22),
#         ncol=3,               # forces 3 + 2 layout
#         frameon=False,
#         fontsize=9,
#         columnspacing=1.4,
#         handlelength=2.6,
#     )

#     fig.savefig(
#         OUTPUT_PDF,
#         dpi=300,
#         format="pdf",
#     )

#     print(f"[DONE] Saved figure to {OUTPUT_PDF}")


# if __name__ == "__main__":
#     main()

import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def set_style():
    sns.set_theme(
        context="talk",
        style="whitegrid",
        font="serif",
        rc={
            # ---- Typography / export ----
            "pdf.fonttype": 42,
            "ps.fonttype": 42,

            # ---- Axes background ONLY ----
            "axes.facecolor": "#EEF0F2",

            # ---- Grid ----
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.color": "#C3C4C5",

            # ---- Axes ----
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
        },
    )

# ============================
# User-configurable parameters
# ============================
ROOT_DIR = "docking_results/top_k_ablation_noguide"
OUTPUT_PDF = "topk_oracle_success_fraction_with_ci.pdf"

THRESHOLDS = {
    "C-RMSD": 5.0,     # Å (lower is better)
    "I-RMSD": 4.0,     # Å (lower is better)
    "L-RMSD": 5.0,     # Å (lower is better)
    # "DockQ": 0.20,    # higher is better
}

# LEGEND_ORDER = ["cRMSD", "iRMSD", "lRMSD", "fNat", "DockQ"]
LEGEND_ORDER = ["C-RMSD", "I-RMSD", "L-RMSD"]
MAX_K = 30
N_BOOTSTRAP = 1000
CI_ALPHA = 0.05  # 95% CI

# ============================
# Matplotlib configuration
# ============================
# plt.rcParams.update({
#     "font.family": "serif",
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })
set_style()

# ============================
# Helper functions
# ============================
def extract_k(folder_name):
    match = re.search(r"top-(\d+)-noguide", folder_name)
    if match is None:
        raise ValueError(f"Could not parse k from folder name: {folder_name}")
    return int(match.group(1))


def load_meters(val_pt_path):
    obj = torch.load(val_pt_path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "meter" in obj:
        return obj["meter"]
    elif isinstance(obj, (list, tuple)):
        return obj
    else:
        raise ValueError(
            f"Unsupported format in {val_pt_path}. "
            f"Expected dict with key 'meter' or list/tuple."
        )


def get_metric_array(meter, metric):
    if metric == "C-RMSD":
        return np.asarray(meter.complex_rmsd_list_all)
    elif metric == "I-RMSD":
        return np.asarray(meter.interface_rmsd_list_all)
    elif metric == "L-RMSD":
        return np.asarray(meter.ligand_rmsd_list_all)
    elif metric == "fNat":
        return np.asarray(meter.fnat_list_all)
    elif metric == "DockQ":
        return np.asarray(meter.dockQ_list_all)
    else:
        raise ValueError(metric)


def oracle_best_per_target(meters, metric):
    """
    meters: list of Meter_Unbound_Bound (length k)
    returns: array [n_targets] of oracle-best values
    """
    per_meter = [get_metric_array(m, metric) for m in meters]
    values = np.stack(per_meter, axis=0)  # [k, n_targets]

    if metric in {"fNat", "DockQ"}:
        return values.max(axis=0)
    else:
        return values.min(axis=0)


def success_mask(values, metric, threshold):
    if metric in {"fNat", "DockQ"}:
        return values > threshold
    else:
        return values < threshold


def bootstrap_ci(successes, n_bootstrap, alpha):
    n = len(successes)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(successes, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.asarray(boot_means)
    mean = successes.mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return mean, lo, hi


# ============================
# Main analysis
# ============================
def main():
    root = Path(ROOT_DIR)

    ks_all = []
    results = {
        metric: {"mean": [], "lo": [], "hi": []}
        for metric in THRESHOLDS
    }

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        if not subdir.name.startswith("top-"):
            continue

        k = extract_k(subdir.name)
        if k > MAX_K:
            continue

        val_pt = subdir / "val_0.pt"
        if not val_pt.exists():
            continue

        meters = load_meters(val_pt)
        ks_all.append(k)

        for metric, threshold in THRESHOLDS.items():
            oracle_values = oracle_best_per_target(meters, metric)
            successes = success_mask(
                oracle_values, metric, threshold
            ).astype(float)

            mean, lo, hi = bootstrap_ci(
                successes,
                n_bootstrap=N_BOOTSTRAP,
                alpha=CI_ALPHA,
            )

            results[metric]["mean"].append(mean)
            results[metric]["lo"].append(lo)
            results[metric]["hi"].append(hi)

    # Sort by k
    order = np.argsort(ks_all)
    ks = np.array(ks_all)[order]

    for metric in results:
        for key in results[metric]:
            results[metric][key] = np.array(results[metric][key])[order]

    # ============================
    # Plot
    # ============================
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.set_facecolor("#EEF0F2")

    # Horizontal grid only (line-plot appropriate)
    ax.grid(
        True,
        axis="y",
        linestyle="--",
        color="#C3C4C5",
    )
    ax.grid(False, axis="x")

    for metric in LEGEND_ORDER:
        data = results[metric]

        label = (
            f"{metric} "
            f"{'>' if metric in {'fNat', 'DockQ'} else '<'} "
            f"{THRESHOLDS[metric]}"
        )

        ax.plot(
            ks,
            data["mean"],
            marker="o",
            linewidth=6,
            label=label,
            markersize=12,
        )

        ax.fill_between(
            ks,
            data["lo"],
            data["hi"],
            alpha=0.2,
        )

    ax.set_xlabel("Number of Generated Samples (k)", fontsize=18)
    ax.set_ylabel("Oracle Success Fraction", fontsize=18)
    ax.set_ylim(0.0, 1.0)
    # ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    fig.subplots_adjust(bottom=0.32)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=13,
    )
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")

    # ---- De-emphasize the others ----
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.tight_layout()

    fig.savefig(OUTPUT_PDF, dpi=300, format="pdf")
    print(f"[DONE] Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
