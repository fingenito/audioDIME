"""
merge_exp3_shards.py  —  Merge parallel Exp 3 (Logit Lens) shard outputs.

Each shard directory contains:
    shard_N/logit_lens_results.jsonl   (one JSON record per line per eval item)
    shard_N/metadata.json

This script produces a merged directory with:
    logit_lens_results.jsonl   (all records, sorted by sample_idx then permutation_idx)
    aggregate_stats.json       (pre-computed per-layer stats for fast plotting)
    metadata.json

Usage:
    python -m QA_analysis.experiments.Exp_Net.merge_exp3_shards \\
        --shards_root /path/to/full_run_7b_3perm_parallel

    # With explicit output dir:
    python -m QA_analysis.experiments.Exp_Net.merge_exp3_shards \\
        --shards_root /path/to/full_run_7b_3perm_parallel \\
        --output_dir  /path/to/full_run_7b_3perm_merged
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge Exp 3 (Logit Lens) parallel shard outputs."
    )
    p.add_argument("--shards_root", required=True,
                   help="Root directory containing shard_0/, shard_1/, ... subdirectories.")
    p.add_argument("--output_dir", default=None,
                   help="Destination directory. Defaults to <shards_root>/../<basename>_merged.")
    p.add_argument("--shard_pattern", default="shard_*")
    return p.parse_args()


# =============================================================================
# Shard discovery
# =============================================================================

def discover_shards(shards_root: str, pattern: str) -> List[str]:
    candidates = sorted(glob.glob(os.path.join(shards_root, pattern)))
    valid = []
    for c in candidates:
        jsonl = os.path.join(c, "logit_lens_results.jsonl")
        if os.path.isdir(c) and os.path.exists(jsonl):
            valid.append(c)
        else:
            print(f"   ⚠ Skipping {c!r}: no logit_lens_results.jsonl found.")
    return valid


# =============================================================================
# Merge records
# =============================================================================

def read_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"   ⚠ Skipping malformed line in {path}: {e}")
    return records


def merge_records(shard_dirs: List[str]) -> List[Dict]:
    all_records = []
    for shard in shard_dirs:
        path = os.path.join(shard, "logit_lens_results.jsonl")
        recs = read_jsonl(path)
        print(f"   {os.path.basename(shard)}: {len(recs)} records")
        all_records.extend(recs)

    # Sort by (sample_idx, permutation_idx) for reproducible ordering
    all_records.sort(key=lambda r: (int(r.get("sample_idx", 0)), int(r.get("permutation_idx", 0))))
    return all_records


# =============================================================================
# Aggregate statistics computation
# =============================================================================

def compute_aggregate_stats(records: List[Dict], total_layers: int) -> Dict:
    """
    Pre-compute per-layer and per-category statistics for fast plotting.

    Returns aggregate_stats dict with:
      per_layer:            {layer_str: {mean_frac, sem_frac, mean_correct, mean_wrong, n, n_correct, n_wrong}}
      per_category_per_layer: {category: {layer_str: mean_frac}}
      per_difficulty_per_layer: {difficulty: {layer_str: mean_frac}}
      per_phase_top1_counter: {early/mid/late: {token: count}}
    """
    # Layer ranges for phases
    early_layers = set(range(0, total_layers // 3))
    mid_layers = set(range(total_layers // 3, 2 * total_layers // 3))
    late_layers = set(range(2 * total_layers // 3, total_layers))
    phase_map = {
        "early": early_layers,
        "mid": mid_layers,
        "late": late_layers,
    }

    # Accumulators — primary metrics: entropy_bits and top1_dominance
    # (fraction_music_related kept as secondary for completeness)
    def _make_acc():
        return defaultdict(list)

    layer_entropy:    Dict[int, List[float]] = defaultdict(list)
    layer_entropy_c:  Dict[int, List[float]] = defaultdict(list)
    layer_entropy_w:  Dict[int, List[float]] = defaultdict(list)
    layer_dominance:  Dict[int, List[float]] = defaultdict(list)
    layer_dominance_c:Dict[int, List[float]] = defaultdict(list)
    layer_dominance_w:Dict[int, List[float]] = defaultdict(list)
    layer_fracs:      Dict[int, List[float]] = defaultdict(list)
    cat_layer_entropy:Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    diff_layer_entropy:Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    phase_counters: Dict[str, Counter] = {p: Counter() for p in phase_map}

    for rec in records:
        is_correct = bool(rec.get("is_correct", 0))
        category = str(rec.get("main_category", "Unknown") or "Unknown")
        difficulty = str(rec.get("difficulty", "") or "")
        per_layer = rec.get("per_layer", {})

        for li in range(total_layers):
            stats = per_layer.get(str(li), {})
            frac   = stats.get("fraction_music_related", float("nan"))
            dom    = stats.get("top1_dominance", float("nan"))
            ent    = stats.get("entropy_bits", float("nan"))
            counter = stats.get("top1_counter", {})

            if not math.isnan(ent):
                layer_entropy[li].append(ent)
                layer_dominance[li].append(dom) if not math.isnan(dom) else None
                if is_correct:
                    layer_entropy_c[li].append(ent)
                    if not math.isnan(dom): layer_dominance_c[li].append(dom)
                else:
                    layer_entropy_w[li].append(ent)
                    if not math.isnan(dom): layer_dominance_w[li].append(dom)
                cat_layer_entropy[category][li].append(ent)
                if difficulty:
                    diff_layer_entropy[difficulty][li].append(ent)
            if not math.isnan(frac):
                layer_fracs[li].append(frac)

            for phase, layer_set in phase_map.items():
                if li in layer_set:
                    phase_counters[phase].update(counter)

    def _mean_sem(values: List[float]):
        if not values:
            return None, None
        n = len(values)
        mu = sum(values) / n
        if n == 1:
            return mu, 0.0
        var = sum((v - mu) ** 2 for v in values) / (n - 1)
        sem = (var / n) ** 0.5
        return mu, sem

    # Per-layer summary  (primary: entropy ↓ / dominance ↑ ; secondary: frac_music)
    per_layer_stats: Dict[str, Dict] = {}
    for li in range(total_layers):
        # entropy_bits (lower = more convergent)
        mu_ent,   sem_ent   = _mean_sem(layer_entropy.get(li, []))
        mu_ent_c, sem_ent_c = _mean_sem(layer_entropy_c.get(li, []))
        mu_ent_w, sem_ent_w = _mean_sem(layer_entropy_w.get(li, []))
        # top1_dominance (higher = more convergent)
        mu_dom,   sem_dom   = _mean_sem(layer_dominance.get(li, []))
        mu_dom_c, sem_dom_c = _mean_sem(layer_dominance_c.get(li, []))
        mu_dom_w, sem_dom_w = _mean_sem(layer_dominance_w.get(li, []))
        # fraction_music_related (secondary, kept for completeness)
        mu_frac, sem_frac   = _mean_sem(layer_fracs.get(li, []))

        per_layer_stats[str(li)] = {
            # primary — entropy
            "mean_entropy":         mu_ent,
            "sem_entropy":          sem_ent,
            "mean_entropy_correct": mu_ent_c,
            "sem_entropy_correct":  sem_ent_c,
            "mean_entropy_wrong":   mu_ent_w,
            "sem_entropy_wrong":    sem_ent_w,
            # primary — dominance
            "mean_dominance":         mu_dom,
            "sem_dominance":          sem_dom,
            "mean_dominance_correct": mu_dom_c,
            "sem_dominance_correct":  sem_dom_c,
            "mean_dominance_wrong":   mu_dom_w,
            "sem_dominance_wrong":    sem_dom_w,
            # secondary — music fraction
            "mean_frac":    mu_frac,
            "sem_frac":     sem_frac,
            # counts
            "n":         len(layer_entropy.get(li, [])),
            "n_correct": len(layer_entropy_c.get(li, [])),
            "n_wrong":   len(layer_entropy_w.get(li, [])),
        }

    # Per-category per-layer (mean entropy_bits — lower = more convergent)
    per_cat: Dict[str, Dict[str, Optional[float]]] = {}
    for cat, l_dict in cat_layer_entropy.items():
        per_cat[cat] = {}
        for li in range(total_layers):
            vals = l_dict.get(li, [])
            per_cat[cat][str(li)] = sum(vals) / len(vals) if vals else None

    # Per-difficulty per-layer (mean entropy_bits)
    per_diff: Dict[str, Dict[str, Optional[float]]] = {}
    for diff, l_dict in diff_layer_entropy.items():
        per_diff[diff] = {}
        for li in range(total_layers):
            vals = l_dict.get(li, [])
            per_diff[diff][str(li)] = sum(vals) / len(vals) if vals else None

    # Per-phase top-50 tokens
    per_phase_top: Dict[str, List] = {}
    for phase, ctr in phase_counters.items():
        per_phase_top[phase] = ctr.most_common(50)

    return {
        "n_records": len(records),
        "total_layers": total_layers,
        "phase_layer_ranges": {
            "early": [0, total_layers // 3],
            "mid": [total_layers // 3, 2 * total_layers // 3],
            "late": [2 * total_layers // 3, total_layers],
        },
        "per_layer": per_layer_stats,
        "per_category_per_layer": per_cat,    # mean entropy_bits per cat/layer
        "per_difficulty_per_layer": per_diff,  # mean entropy_bits per difficulty/layer
        "per_phase_top50_tokens": per_phase_top,
    }


# =============================================================================
# Metadata merge
# =============================================================================

def merge_metadata(
    shard_dirs: List[str],
    n_records: int,
    n_correct: int,
    output_path: str,
    aggregate_stats_path: str,
) -> Dict:
    first_meta: dict = {}
    all_errors: list = []
    shard_breakdown = []

    for shard in shard_dirs:
        path = os.path.join(shard, "metadata.json")
        if not os.path.exists(path):
            print(f"   ⚠ metadata.json missing in {shard!r}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not first_meta:
            first_meta = dict(meta)
        all_errors.extend(meta.get("errors", []))
        shard_breakdown.append({
            "shard": os.path.basename(shard),
            "n_processed": meta.get("n_processed", 0),
            "n_correct": meta.get("n_correct", 0),
            "n_errors": len(meta.get("errors", [])),
        })

    merged = dict(first_meta)
    merged.update({
        "status": "merged",
        "n_processed": n_records,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_records, 6) if n_records else None,
        "errors": all_errors,
        "n_errors": len(all_errors),
        "merged_at": datetime.now().isoformat(),
        "n_shards": len(shard_dirs),
        "shard_breakdown": shard_breakdown,
        "files": {
            "logit_lens_results": output_path.replace("metadata.json", "logit_lens_results.jsonl"),
            "aggregate_stats": aggregate_stats_path,
        },
    })

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

    print(f"\nMerging Exp 3 (Logit Lens) shards")
    print(f"  shards_root : {shards_root}")
    print(f"  output_dir  : {output_dir}\n")

    shard_dirs = discover_shards(shards_root, args.shard_pattern)
    if not shard_dirs:
        print("No valid shards found. Aborting.")
        sys.exit(1)

    print(f"Found {len(shard_dirs)} shard(s): {[os.path.basename(s) for s in shard_dirs]}\n")

    # ── Merge records ─────────────────────────────────────────────────────────
    print("Merging JSONL records...")
    records = merge_records(shard_dirs)
    print(f"Total records: {len(records)}\n")

    jsonl_out = os.path.join(output_dir, "logit_lens_results.jsonl")
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   ✅ {jsonl_out}  ({len(records)} records)")

    # ── Infer total_layers from first shard metadata ──────────────────────────
    total_layers = 28  # default for 7B
    first_meta_path = os.path.join(shard_dirs[0], "metadata.json")
    if os.path.exists(first_meta_path):
        with open(first_meta_path) as f:
            m = json.load(f)
        total_layers = int(m.get("total_layers", total_layers))
    print(f"Total layers: {total_layers}")

    # ── Aggregate statistics ──────────────────────────────────────────────────
    print("\nComputing aggregate statistics...")
    agg_stats = compute_aggregate_stats(records, total_layers)
    agg_path = os.path.join(output_dir, "aggregate_stats.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg_stats, f, indent=2, ensure_ascii=False)
    print(f"   ✅ {agg_path}")

    # ── Metadata ─────────────────────────────────────────────────────────────
    print("\nMerging metadata...")
    n_correct = sum(int(r.get("is_correct", 0)) for r in records)
    merged_meta = merge_metadata(
        shard_dirs, len(records), n_correct,
        os.path.join(output_dir, "metadata.json"),
        agg_path,
    )

    print(f"\n{'=' * 60}")
    print("Merge complete.")
    print(f"  Shards merged : {len(shard_dirs)}")
    print(f"  n_processed   : {merged_meta['n_processed']}")
    print(f"  n_correct     : {merged_meta['n_correct']}")
    acc = merged_meta.get("accuracy")
    if acc is not None:
        print(f"  accuracy      : {acc:.4f}")
    print(f"  n_errors      : {merged_meta['n_errors']}")
    print(f"\nNext — plot:")
    print(
        f"  python -m QA_analysis.experiments.Exp_Net.plot_exp3_logit_lens \\\n"
        f"      --results_dir '{output_dir}' \\\n"
        f"      --output_dir  '{output_dir}/plots'"
    )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
