#!/usr/bin/env python3
"""Keep only meter data in recursively discovered val_0.pt files."""

import argparse
import os
import tempfile
from pathlib import Path

import torch


def strip_file(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "meter" not in checkpoint:
        raise ValueError("expected a checkpoint dictionary containing a 'meter' key")

    # Save through a temporary file in the same directory, then replace the
    # original so an interrupted write does not leave a partial checkpoint.
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save({"meter": checkpoint["meter"]}, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively replace val_0.pt files with meter-only checkpoints."
    )
    parser.add_argument("folder", type=Path, help="Folder to search recursively")
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"not a directory: {args.folder}")

    files = sorted(args.folder.rglob("val_0.pt"))
    for path in files:
        try:
            strip_file(path)
        except Exception as exc:
            print(f"ERROR {path}: {exc}")
        else:
            print(f"STRIPPED {path}")

    print(f"Processed {len(files)} file(s).")


if __name__ == "__main__":
    main()
