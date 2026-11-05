#!/usr/bin/env python3
# Credit for base code to: https://github.com/wuhuaijin/EBMDock
# Adapted from data/DIPS-Het/step3_pairs_to_pdb.py at 6726358e118b1966d57877392684220d43271df0.
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
import argparse
import gzip
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def atomic_write(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_assembly(pdb, cache):
    path = cache / f"{pdb.lower()}.pdb1.gz"
    if path.is_file():
        try:
            return gzip.decompress(path.read_bytes()).decode()
        except (OSError, EOFError, UnicodeError):
            pass
    url = f"https://files.rcsb.org/download/{pdb}.pdb1.gz"
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GuideDock dataset downloader"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            text = gzip.decompress(data).decode()
            if not any(line.startswith("ATOM") for line in text.splitlines()):
                raise ValueError(f"No protein ATOM records in {url}")
            atomic_write(path, data)
            return text
        except (OSError, EOFError, ValueError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def convert_to_pdb(lines, pair_name, output):
    lig_chain_id, rec_chain_ids = pair_name.split("_")[1:]
    lig_pdb_lines, rec_pdb_lines = [], []
    for line in lines:
        if line[:4] == "ATOM":
            chain_id = line[21]
            if chain_id == lig_chain_id:
                lig_pdb_lines.append(line.strip())
            elif chain_id in rec_chain_ids:
                rec_pdb_lines.append(line.strip())
    if min(len(lig_pdb_lines), len(rec_pdb_lines)) < 100 or max(len(lig_pdb_lines), len(rec_pdb_lines)) > 12000:
        raise ValueError(f"{pair_name}: partners must each have 100–12000 ATOM records, following EBMDock")
    for role, records in [("l", lig_pdb_lines), ("r", rec_pdb_lines)]:
        atomic_write(output / f"{pair_name}_{role}_b.pdb", ("\n".join(records) + "\n").encode())


def process(pdb, pairs, output):
    pending = [pair for pair in pairs if not all((output / f"{pair}_{role}_b.pdb").is_file() and (output / f"{pair}_{role}_b.pdb").stat().st_size for role in ("l", "r"))]
    if not pending:
        return len(pairs), []
    lines = fetch_assembly(pdb, output / ".assemblies").splitlines()
    failures = []
    for pair in pending:
        try:
            convert_to_pdb(lines, pair, output)
        except (OSError, ValueError, IndexError) as error:
            failures.append(str(error))
    return len(pairs) - len(failures), failures


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(prog="download.py", description="Download DIPS-Het biological assemblies and extract EBMDock-style ligand/receptor pairs for GuideDock's existing splits")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "structures")
    parser.add_argument("--keys", type=Path, nargs="+", help="Complex-ID files; defaults to the bundled train/val/test splits")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Download only the first N complexes")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    key_files = args.keys or [script_dir / f"{split}.txt" for split in ("train", "val", "test")]
    keys = list(dict.fromkeys(line.strip() for path in key_files for line in path.read_text().splitlines() if line.strip()))
    if args.limit is not None:
        keys = keys[:args.limit]
    if not keys:
        parser.error("No complex IDs found")
    groups = defaultdict(list)
    for key in keys:
        match = re.fullmatch(r"([A-Za-z0-9]{4})_([^_\s])_([^_\s]+)", key)
        if match is None or match[2] in match[3]:
            parser.error(f"Invalid PDB_BINDER_RECEPTOR complex ID: {key}")
        groups[match[1].upper()].append(key)
    output = args.output_dir.resolve()
    (output / ".assemblies").mkdir(parents=True, exist_ok=True)

    ready, failures = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, pdb, pairs, output): pdb for pdb, pairs in groups.items()}
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                count, errors = future.result()
                ready += count
                failures.extend(errors)
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
            print(f"[{completed}/{len(groups)} assemblies] {ready}/{len(keys)} complexes ready", flush=True)
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"Prepared {ready}/{len(keys)} complexes in {output}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
