#!/usr/bin/env bash
# Credit for base code to: https://github.com/wuhuaijin/EBMDock
# Dataset source: https://github.com/wuhuaijin/EBMDock/tree/6726358e118b1966d57877392684220d43271df0/data/DB5/structures
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir="$script_dir/benchmark5.5/structures"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || { printf '%s\n' 'Missing value for --output-dir' >&2; exit 2; }
            output_dir=$2
            shift 2
            ;;
        -h|--help)
            printf '%s\n' 'Usage: download.sh [--output-dir DIR]' 'Download EBMDock DB5.5 bound/unbound PDB structures.' 'Default: data/db55/benchmark5.5/structures'
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

command -v curl >/dev/null
command -v tar >/dev/null
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/guidedock-db55.XXXXXX")
trap 'rm -rf -- "$work_dir"' EXIT
revision=6726358e118b1966d57877392684220d43271df0
archive_root="EBMDock-$revision"
curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 20 \
    --output "$work_dir/ebmdock.tar.gz" \
    "https://codeload.github.com/wuhuaijin/EBMDock/tar.gz/$revision"
tar -xzf "$work_dir/ebmdock.tar.gz" -C "$work_dir" "$archive_root/data/DB5/structures"
source_dir="$work_dir/$archive_root/data/DB5/structures"
mkdir -p -- "$output_dir"
copied=0
skipped=0
for structure in "$source_dir"/*.pdb; do
    [[ -f "$structure" ]] || continue
    filename=${structure##*/}
    if [[ -s "$output_dir/$filename" ]]; then
        skipped=$((skipped + 1))
    else
        cp -- "$structure" "$output_dir/$filename"
        copied=$((copied + 1))
    fi
done
[[ $((copied + skipped)) -gt 0 ]] || { printf '%s\n' 'No DB5.5 structures found in the archive' >&2; exit 1; }
printf 'Prepared %s PDB files in %s (%s downloaded, %s already present)\n' "$((copied + skipped))" "$output_dir" "$copied" "$skipped"
