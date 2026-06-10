"""
merge_exp1_shards.py  —  Merge parallel Exp 1 shard outputs into one combined result.

After running launch_exp1_parallel.sh you end up with:
    <shards_root>/shard_0/attention_per_layer.csv
    <shards_root>/shard_0/summary_stats.csv
    <shards_root>/shard_0/metadata.json
    <shards_root>/shard_1/...
    ...

This script merges them into a single output directory that is compatible with
plot_exp1_attention_pattern.py.

Usage:
    python -m QA_analysis.experiments.Exp_Net.merge_exp1_shards \\
        --shards_root /path/to/full_run_3b_parallel \\
        --output_dir  /path/to/full_run_3b_merged

    # Or auto-discover shards (looks for shard_* subdirectories):
    python -m QA_analysis.experiments.Exp_Net.merge_exp1_shards \\
        --shards_root /path/to/full_run_3b_parallel

The output_dir defaults to <shards_root>/../<basename>_merged if not specified.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime
from typing import List, Optional


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Exp 1 parallel shard outputs into one combined result directory."
    )
    parser.add_argument(
        "--shards_root",
        required=True,
        help="Root directory containing shard_0/, shard_1/, ... subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Destination directory for merged output. "
            "Defaults to <shards_root>/../<basename>_merged."
        ),
    )
    parser.add_argument(
        "--shard_pattern",
        default="shard_*",
        help="Glob pattern for shard subdirectories (default: shard_*).",
    )
    parser.add_argument(
        "--sort_by_sample_idx",
        action="store_true",
        default=True,
        help="Re-sort merged CSVs by sample_idx before writing (default: True).",
    )
    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def discover_shards(shards_root: str, pattern: str) -> List[str]:
    """Return sorted list of shard directories that contain attention_per_layer.csv."""
    candidates = sorted(glob.glob(os.path.join(shards_root, pattern)))
    valid = []
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "attention_per_layer.csv")):
            valid.append(c)
        else:
            print(f"   ⚠ Skipping {c!r}: no attention_per_layer.csv found.")
    return valid


def merge_csv(shard_dirs: List[str], filename: str, output_path: str, sort_by: Optional[str] = "sample_idx"):
    """Merge a CSV file from all shards, writing header once."""
    rows = []
    fieldnames = None

    for shard in shard_dirs:
        path = os.path.join(shard, filename)
        if not os.path.exists(path):
            print(f"   ⚠ {filename} missing in {shard!r} — skipping.")
            continue
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)

    if not rows:
        print(f"   ⚠ No rows found for {filename} — skipping.")
        return 0

    if sort_by and sort_by in (fieldnames or []):
        try:
            rows.sort(key=lambda r: int(r[sort_by]))
        except (ValueError, KeyError):
            pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"   ✅ {output_path}  ({len(rows)} rows)")
    return len(rows)


def merge_metadata(shard_dirs: List[str], n_layer_rows: int, n_summary_rows: int, output_path: str):
    """Merge metadata.json files from all shards."""
    first_meta: dict = {}
    total_correct = 0
    total_processed = 0
    all_errors: list = []
    shard_metas: list = []

    for shard in shard_dirs:
        path = os.path.join(shard, "metadata.json")
        if not os.path.exists(path):
            print(f"   ⚠ metadata.json missing in {shard!r} — skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not first_meta:
            first_meta = dict(meta)
        total_correct += int(meta.get("n_correct", 0))
        total_processed += int(meta.get("n_processed", 0))
        all_errors.extend(meta.get("errors", []))
        shard_metas.append({"shard": os.path.basename(shard), **meta})

    merged = dict(first_meta)
    merged.update(
        {
            "n_processed": total_processed,
            "n_correct": total_correct,
            "accuracy_over_saved_evaluations": (
                round(total_correct / total_processed, 6) if total_processed else None
            ),
            "errors": all_errors,
            "n_errors": len(all_errors),
            "status": "merged",
            "merged_at": datetime.now().isoformat(),
            "n_shards": len(shard_dirs),
            "shard_breakdown": [
                {
                    "shard": os.path.basename(s),
                    "n_processed": m.get("n_processed", 0),
                    "n_correct": m.get("n_correct", 0),
                    "n_errors": len(m.get("errors", [])),
                }
                for s, m in zip(
                    shard_dirs,
                    [json.load(open(os.path.join(s, "metadata.json"))) for s in shard_dirs
                     if os.path.exists(os.path.join(s, "metadata.json"))],
                )
            ],
        }
    )
    # Update file paths
    merged["files"] = {
        "attention_per_layer": output_path.replace("metadata.json", "attention_per_layer.csv"),
        "summary_stats": output_path.replace("metadata.json", "summary_stats.csv"),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"   ✅ {output_path}")
    return merged


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    shards_root = os.path.abspath(args.shards_root)
    if not os.path.isdir(shards_root):
        print(f"Error: shards_root {shards_root!r} does not exist.")
        sys.exit(1)

    if args.output_dir is None:
        parent = os.path.dirname(shards_root)
        base = os.path.basename(shards_root)
        output_dir = os.path.join(parent, f"{base}_merged")
    else:
        output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nMerging Exp 1 shards")
    print(f"  shards_root : {shards_root}")
    print(f"  pattern     : {args.shard_pattern}")
    print(f"  output_dir  : {output_dir}\n")

    shard_dirs = discover_shards(shards_root, args.shard_pattern)
    if not shard_dirs:
        print("No valid shard directories found. Aborting.")
        sys.exit(1)

    print(f"Found {len(shard_dirs)} shard(s): {[os.path.basename(s) for s in shard_dirs]}\n")

    sort_col = "sample_idx" if args.sort_by_sample_idx else None

    n_layer = merge_csv(
        shard_dirs,
        "attention_per_layer.csv",
        os.path.join(output_dir, "attention_per_layer.csv"),
        sort_by=sort_col,
    )
    n_summary = merge_csv(
        shard_dirs,
        "summary_stats.csv",
        os.path.join(output_dir, "summary_stats.csv"),
        sort_by=sort_col,
    )
    merged_meta = merge_metadata(
        shard_dirs,
        n_layer,
        n_summary,
        os.path.join(output_dir, "metadata.json"),
    )

    print(f"\n{'='*60}")
    print(f"Merge complete.")
    print(f"  Shards merged : {len(shard_dirs)}")
    print(f"  n_processed   : {merged_meta['n_processed']}")
    print(f"  n_correct     : {merged_meta['n_correct']}")
    acc = merged_meta.get("accuracy_over_saved_evaluations")
    if acc is not None:
        print(f"  accuracy      : {acc:.4f}")
    print(f"  n_errors      : {merged_meta['n_errors']}")
    print(f"\nNext step — generate plots:")
    print(
        f"  python -m QA_analysis.experiments.Exp_Net.plot_exp1_attention_pattern \\\n"
        f"      --results_dir '{output_dir}' \\\n"
        f"      --output_dir  '{output_dir}/plots'"
    )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
