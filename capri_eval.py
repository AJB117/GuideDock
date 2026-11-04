#!/usr/bin/env python3

import argparse
import pickle
import torch
from collections import Counter


def load_file(path):
    """Load a dictionary from .pt or .pkl"""
    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, map_location="cpu", weights_only=False)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use .pt/.pth or .pkl")


def classify_capri(fnat, lrmsd, irmsd):
    # Incorrect
    if (fnat < 0.1) or (lrmsd > 10.0 and irmsd > 4.0):
        return "Incorrect"

    # High
    if (fnat >= 0.5) and (lrmsd <= 1.0 or irmsd <= 1.0):
        return "High"

    # Medium
    if ((fnat >= 0.3 and fnat < 0.5) and (lrmsd <= 5.0 or irmsd <= 2.0)) or \
       ((fnat >= 0.5) and (lrmsd > 1.0 and irmsd > 1.0)):
        return "Medium"

    # Acceptable
    if ((fnat >= 0.1 and fnat < 0.3) and (lrmsd <= 10.0 or irmsd <= 4.0)) or \
       ((fnat >= 0.3) and (lrmsd > 5.0 and irmsd > 2.0)):
        return "Acceptable"

    return "Incorrect"


def compute_capri(meter, include_bad=False):
    fnats = meter.fnat_list
    lrmsds = meter.ligand_rmsd_list
    irmsds = meter.interface_rmsd_list

    assert len(fnats) == len(lrmsds) == len(irmsds), "Metric list lengths mismatch"

    labels = [
        classify_capri(f, l, i)
        for f, l, i in zip(fnats, lrmsds, irmsds)
    ]

    counts = Counter(labels)

    # Optionally include filtered bad cases
    if include_bad and hasattr(meter, "bad_case"):
        counts["Incorrect"] += meter.bad_case

    total = sum(counts.values())

    results = {
        k: {
            "count": counts.get(k, 0),
            "percent": 100.0 * counts.get(k, 0) / total if total > 0 else 0.0
        }
        for k in ["High", "Medium", "Acceptable", "Incorrect"]
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute CAPRI metrics from saved meter object")
    parser.add_argument("path", help="Path to .pt/.pth/.pkl file containing {'meter': meter}")
    parser.add_argument("--include_bad", action="store_true",
                        help="Include filtered bad cases as Incorrect")
    args = parser.parse_args()

    data = load_file(args.path)

    if "meter" not in data:
        raise KeyError("Input file must contain key 'meter'")

    meter = data["meter"]

    results = compute_capri(meter, include_bad=args.include_bad)

    print("\nCAPRI Evaluation")
    print("=" * 30)
    total = sum(v["count"] for v in results.values())

    for k in ["High", "Medium", "Acceptable", "Incorrect"]:
        print(f"{k:12s}: {results[k]['count']:6d}  ({results[k]['percent']:6.2f}%)")

    print("-" * 30)
    print(f"{'Total':12s}: {total:6d}")


if __name__ == "__main__":
    main()
