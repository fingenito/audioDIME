"""
merge_exp2_shards.py  —  Merge parallel Exp 2 shard outputs into one combined result.

After running launch_exp2_parallel.sh you end up with:
    <shards_root>/shard_0/attention_knockout_results.csv
    <shards_root>/shard_0/metadata.json
    <shards_root>/shard_1/...
    ...

This script merges them into a single output directory compatible with
plot_exp2_knockout.py.

Usage:
    python -m QA_analysis.experiments.Exp_Net.merge_exp2_shards \\
        --shards_root /path/to/full_run_3b_decision_logits_parallel \\
        --output_dir  /path/to/full_run_3b_decision_logits_parallel_merged

    # Auto-discover shards (looks for shard_* subdirectories):
    python -m QA_analysis.experiments.Exp_Net.merge_exp2_shards \\
        --shards_root /path/to/full_run_3b_decision_logits_parallel
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
        description="Merge Exp 2 parallel shard outputs into one combined result directory."
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
        help="Re-sort merged CSV by sample_idx (then evaluation_idx) before writing (default: True).",
    )
    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def discover_shards(shards_root: str, pattern: str) -> List[str]:
    candidates = sorted(glob.glob(os.path.join(shards_root, pattern)))
    valid = []
    for c in candidates:
        csv_path = os.path.join(c, "attention_knockout_results.csv")
        if os.path.isdir(c) and os.path.exists(csv_path):
            # Skip empty CSV files (shard wrote 0 rows)
            with open(csv_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                print(f"   ⚠ Skipping {c!r}: attention_knockout_results.csv is empty.")
            else:
                valid.append(c)
        else:
            print(f"   ⚠ Skipping {c!r}: no attention_knockout_results.csv found.")
    return valid


def merge_csv(
    shard_dirs: List[str],
    filename: str,
    output_path: str,
    sort_by: Optional[str] = "sample_idx",
    sort_secondary: Optional[str] = "evaluation_idx",
) -> int:
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
            secondary = sort_secondary if (sort_secondary and sort_secondary in (fieldnames or [])) else None
            if secondary:
                rows.sort(key=lambda r: (int(r[sort_by]), int(r[secondary])))
            else:
                rows.sort(key=lambda r: int(r[sort_by]))
        except (ValueError, KeyError):
            pass

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"   ✅ {output_path}  ({len(rows)} rows)")
    return len(rows)


def merge_metadata(
    shard_dirs: List[str],
    n_rows: int,
    output_path: str,
) -> dict:
    first_meta: dict = {}
    total_errors: list = []
    total_processed = 0
    total_correct = 0
    shard_breakdown: list = []

    for shard in shard_dirs:
        path = os.path.join(shard, "metadata.json")
        if not os.path.exists(path):
            print(f"   ⚠ metadata.json missing in {shard!r} — skipping.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not first_meta:
            first_meta = dict(meta)
        total_errors.extend(meta.get("errors", []))
        total_processed += int(meta.get("n_processed", 0))
        total_correct += int(meta.get("n_correct", 0))
        shard_breakdown.append(
            {
                "shard": os.path.basename(shard),
                "n_rows": meta.get("n_rows", 0),
                "n_processed": meta.get("n_processed", 0),
                "n_correct": meta.get("n_correct", 0),
                "n_errors": len(meta.get("errors", [])),
            }
        )

    merged = dict(first_meta)
    merged.update(
        {
            "status": "merged",
            "n_rows": n_rows,
            "n_processed": total_processed,
            "n_correct": total_correct,
            "baseline_accuracy": (
                round(total_correct / total_processed, 6) if total_processed else None
            ),
            "errors": total_errors,
            "n_errors": len(total_errors),
            "merged_at": datetime.now().isoformat(),
            "n_shards": len(shard_dirs),
            "shard_breakdown": shard_breakdown,
            "files": {
                "attention_knockout_results": output_path.replace(
                    "metadata.json", "attention_knockout_results.csv"
                ),
            },
        }
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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

    print(f"\nMerging Exp 2 shards")
    print(f"  shards_root : {shards_root}")
    print(f"  pattern     : {args.shard_pattern}")
    print(f"  output_dir  : {output_dir}\n")

    shard_dirs = discover_shards(shards_root, args.shard_pattern)
    if not shard_dirs:
        print("No valid shard directories found. Aborting.")
        sys.exit(1)

    print(f"Found {len(shard_dirs)} shard(s): {[os.path.basename(s) for s in shard_dirs]}\n")

    sort_col = "sample_idx" if args.sort_by_sample_idx else None

    n_rows = merge_csv(
        shard_dirs,
        "attention_knockout_results.csv",
        os.path.join(output_dir, "attention_knockout_results.csv"),
        sort_by=sort_col,
        sort_secondary="evaluation_idx",
    )
    merged_meta = merge_metadata(
        shard_dirs,
        n_rows,
        os.path.join(output_dir, "metadata.json"),
    )

    # Count unique samples and baseline accuracy from the CSV
    try:
        import csv as csv_mod

        rows_seen = {}
        baseline_correct_total = 0
        baseline_total = 0
        with open(os.path.join(output_dir, "attention_knockout_results.csv"), "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                key = (row.get("sample_idx"), row.get("option_permutation_idx", "0"))
                if key not in rows_seen:
                    rows_seen[key] = True
                    bc = row.get("baseline_correct")
                    if bc is not None:
                        try:
                            baseline_correct_total += int(bc)
                            baseline_total += 1
                        except ValueError:
                            pass

        n_unique_eval_items = len(rows_seen)
        baseline_acc = baseline_correct_total / baseline_total if baseline_total else None
        print(f"\n  Unique eval items in merged CSV : {n_unique_eval_items}")
        if baseline_acc is not None:
            print(f"  Baseline accuracy (from CSV)    : {baseline_acc:.4f} ({baseline_correct_total}/{baseline_total})")
    except Exception as e:
        print(f"  (Could not compute baseline accuracy: {e})")

    print(f"\n{'=' * 60}")
    print(f"Merge complete.")
    print(f"  Shards merged  : {len(shard_dirs)}")
    print(f"  Total rows     : {n_rows}  (each row = 1 sample × 1 component × 1 layer window)")
    print(f"  n_errors       : {merged_meta['n_errors']}")
    print(f"\nNext step — generate plots:")
    print(
        f"  python -m QA_analysis.experiments.Exp_Net.plot_exp2_knockout \\\n"
        f"      --results_dir '{output_dir}' \\\n"
        f"      --output_dir  '{output_dir}/plots'"
    )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
