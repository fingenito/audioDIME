"""
plot_exp2_knockout.py  —  Visualise Exp 2 attention knockout results.

Reads attention_knockout_results.csv (merged or single-shard) and produces
four publication-quality figures:

  M2_1_knockout_effect_overall.{png,pdf}
        Mean Δprob_correct per component (averaged over all layer windows).
        The most negative bar = component whose removal hurts accuracy most.

  M2_2_layer_window_heatmap.{png,pdf}
        Heatmap: rows = components, columns = layer windows.
        Colour = mean Δprob_correct.  Shows which layers are critical for which
        component.

  M2_3_accuracy_drop.{png,pdf}
        Grouped bar chart: baseline accuracy vs. knockout accuracy per
        component (averaged over all layer windows).

  M2_4_flip_rate.{png,pdf}
        Fraction of eval items where blocking a component causes the predicted
        answer to change, broken down by component × layer window.

Usage:
    python -m QA_analysis.experiments.Exp_Net.plot_exp2_knockout \\
        --results_dir /path/to/full_run_3b_decision_logits_parallel_merged \\
        --output_dir  /path/to/full_run_3b_decision_logits_parallel_merged/plots
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    print("Warning: scipy not available — significance markers will be omitted.")


# =============================================================================
# Colour palette (matches Exp 1 style)
# =============================================================================

COMPONENT_COLOURS = {
    "audio":              "#1f77b4",   # blue
    "instruction":        "#d62728",   # red
    "question":           "#2ca02c",   # green
    "options":            "#ff7f0e",   # orange
    "other_text":         "#9467bd",   # purple
    "all_text_to_audio":  "#8c564b",   # dark brown — compound isolation knockout
}

COMPONENT_LABELS = {
    "audio":              "Audio tokens (direct)",
    "instruction":        "Instructions/scaffold",
    "question":           "Question text",
    "options":            "Answer options",
    "other_text":         "Other text/template",
    "all_text_to_audio":  "All text → audio (isolation)",
}

COMPONENT_ORDER = ["audio", "instruction", "question", "options", "other_text", "all_text_to_audio"]


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Exp 2 attention knockout results."
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing attention_knockout_results.csv (and optional metadata.json).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save figures. Defaults to <results_dir>/plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster outputs (default 150).",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help=(
            "Filter rows by 'mode' column (decision_logits or generation). "
            "If None, uses all rows."
        ),
    )
    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================


def load_results(results_dir: str, mode_filter: Optional[str]) -> List[Dict]:
    path = os.path.join(results_dir, "attention_knockout_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"attention_knockout_results.csv not found in {results_dir}")

    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if mode_filter and row.get("mode") != mode_filter:
                continue
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {path}")
    if mode_filter:
        print(f"  (filtered to mode={mode_filter!r})")
    return rows


def load_metadata(results_dir: str) -> dict:
    path = os.path.join(results_dir, "metadata.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def collect_layer_windows(rows: List[Dict]) -> List[Tuple[int, int]]:
    seen = set()
    for r in rows:
        seen.add((int(r["layer_start"]), int(r["layer_end"])))
    return sorted(seen)


def collect_components(rows: List[Dict]) -> List[str]:
    seen_order = []
    seen_set = set()
    for r in rows:
        c = r.get("component", "")
        if c and c not in seen_set:
            seen_order.append(c)
            seen_set.add(c)
    # Sort by canonical order, with extras appended
    result = [c for c in COMPONENT_ORDER if c in seen_set]
    result += [c for c in seen_order if c not in result]
    return result


# =============================================================================
# Aggregation helpers
# =============================================================================


def aggregate_by_component_and_window(
    rows: List[Dict],
    value_col: str,
) -> Dict[str, Dict[Tuple[int, int], List[float]]]:
    """
    Returns: {component: {(layer_start, layer_end): [values]}}
    """
    out: Dict[str, Dict[Tuple[int, int], List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        comp = r.get("component", "")
        window = (int(r["layer_start"]), int(r["layer_end"]))
        try:
            v = float(r[value_col])
        except (KeyError, ValueError, TypeError):
            continue
        out[comp][window].append(v)
    return out


def mean_by_component(agg: Dict[str, Dict], windows: List[Tuple[int, int]]) -> Dict[str, float]:
    """Mean over all windows for each component."""
    out = {}
    for comp, window_data in agg.items():
        all_vals = []
        for w in windows:
            all_vals.extend(window_data.get(w, []))
        out[comp] = float(np.mean(all_vals)) if all_vals else 0.0
    return out


def mean_matrix(
    agg: Dict[str, Dict],
    components: List[str],
    windows: List[Tuple[int, int]],
) -> np.ndarray:
    """Matrix [len(components), len(windows)] of mean values."""
    mat = np.zeros((len(components), len(windows)))
    for r, comp in enumerate(components):
        for c, w in enumerate(windows):
            vals = agg.get(comp, {}).get(w, [])
            mat[r, c] = float(np.mean(vals)) if vals else 0.0
    return mat


def accuracy_by_component(
    rows: List[Dict],
    components: List[str],
    windows: List[Tuple[int, int]],
) -> Tuple[Dict[str, float], Dict[str, Dict[Tuple[int, int], float]]]:
    """
    Returns (mean_baseline_acc, ko_acc_per_comp_window).
    baseline_acc is averaged across all eval items seen (one row per eval = baseline).
    """
    # Baseline accuracy: deduplicate by (sample_idx, option_permutation_idx)
    baseline_seen = set()
    baseline_vals = []
    ko_vals: Dict[str, Dict[Tuple[int, int], List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        comp = r.get("component", "")
        window = (int(r["layer_start"]), int(r["layer_end"]))
        key = (r.get("sample_idx"), r.get("option_permutation_idx", "0"))

        if key not in baseline_seen:
            baseline_seen.add(key)
            try:
                baseline_vals.append(int(r["baseline_correct"]))
            except (KeyError, ValueError):
                pass

        try:
            ko_vals[comp][window].append(int(r["knockout_correct"]))
        except (KeyError, ValueError):
            pass

    baseline_acc = float(np.mean(baseline_vals)) if baseline_vals else 0.0

    ko_acc_mean: Dict[str, Dict[Tuple[int, int], float]] = {}
    for comp in components:
        ko_acc_mean[comp] = {}
        for w in windows:
            vals = ko_vals.get(comp, {}).get(w, [])
            ko_acc_mean[comp][w] = float(np.mean(vals)) if vals else 0.0

    return baseline_acc, ko_acc_mean


def sem_matrix(
    agg: Dict[str, Dict],
    components: List[str],
    windows: List[Tuple[int, int]],
) -> np.ndarray:
    """Standard Error of Mean matrix [len(components), len(windows)]."""
    mat = np.zeros((len(components), len(windows)))
    for r, comp in enumerate(components):
        for c, w in enumerate(windows):
            vals = agg.get(comp, {}).get(w, [])
            if len(vals) > 1:
                mat[r, c] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return mat


def pvalue_matrix(
    agg: Dict[str, Dict],
    components: List[str],
    windows: List[Tuple[int, int]],
) -> np.ndarray:
    """One-sample t-test p-values against H0: mean==0. Returns NaN when n<2."""
    mat = np.full((len(components), len(windows)), np.nan)
    if not _HAS_SCIPY:
        return mat
    for r, comp in enumerate(components):
        for c, w in enumerate(windows):
            vals = agg.get(comp, {}).get(w, [])
            if len(vals) >= 2:
                _, p = _scipy_stats.ttest_1samp(vals, 0.0)
                mat[r, c] = float(p)
    return mat


def sig_marker(p: float) -> str:
    """Return significance star string for a p-value."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def flip_rate_matrix(
    rows: List[Dict],
    components: List[str],
    windows: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Matrix [len(components), len(windows)] of fraction of eval_items where
    the predicted answer changed after knockout.
    Looks at 'pred_changed' (decision_logits) or 'answer_changed' (generation).
    """
    flip_vals: Dict[str, Dict[Tuple[int, int], List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        comp = r.get("component", "")
        window = (int(r["layer_start"]), int(r["layer_end"]))
        raw = r.get("pred_changed") or r.get("answer_changed")
        try:
            flip_vals[comp][window].append(float(raw))
        except (ValueError, TypeError):
            pass

    mat = np.zeros((len(components), len(windows)))
    for ri, comp in enumerate(components):
        for ci, w in enumerate(windows):
            vals = flip_vals.get(comp, {}).get(w, [])
            mat[ri, ci] = float(np.mean(vals)) if vals else 0.0
    return mat


# =============================================================================
# Figure 1 — Overall knockout effect
# =============================================================================


def plot_overall_effect(
    components: List[str],
    mean_delta: Dict[str, float],
    n_samples: int,
    output_dir: str,
    dpi: int,
    delta_metric_label: str = "Mean Δprob(correct answer)",
    sem_delta: Optional[Dict[str, float]] = None,
    pval_delta: Optional[Dict[str, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    comps_present = [c for c in components if c in mean_delta]
    vals = [mean_delta[c] for c in comps_present]
    sems = [sem_delta.get(c, 0.0) if sem_delta else 0.0 for c in comps_present]
    pvals = [pval_delta.get(c, np.nan) if pval_delta else np.nan for c in comps_present]
    colours = [COMPONENT_COLOURS.get(c, "#7f7f7f") for c in comps_present]
    labels = [COMPONENT_LABELS.get(c, c) for c in comps_present]
    x = np.arange(len(comps_present))

    bars = ax.bar(x, vals, color=colours, width=0.6, edgecolor="white", linewidth=0.8,
                  yerr=sems, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.7})

    # Horizontal zero line
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # Annotate each bar with value and significance
    for i, (bar, v, p) in enumerate(zip(bars, vals, pvals)):
        sem = sems[i]
        ypos_val = v - 0.003 - sem if v < 0 else v + 0.001 + sem
        va = "top" if v < 0 else "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos_val,
            f"{v:+.3f}",
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
        )
        marker = sig_marker(p)
        if marker:
            ypos_sig = v - sem - 0.008 if v < 0 else v + sem + 0.005
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos_sig,
                marker,
                ha="center",
                va="center" if v >= 0 else "top",
                fontsize=11,
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel(delta_metric_label, fontsize=11)
    ax.set_title(
        f"Experiment 2 — Attention Knockout: Effect on Answer\n"
        f"(n={n_samples} evals, averaged over all layer windows; "
        f"error bars=SEM; * p<.05  ** p<.01  *** p<.001)",
        fontsize=10,
    )
    yrange = max(abs(min(vals)), abs(max(vals)), 0.01) + max(sems, default=0) + 0.01
    ax.set_ylim(-yrange * 1.5, yrange * 1.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"M2_1_knockout_effect_overall.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"   Saved {p}")
    plt.close(fig)


# =============================================================================
# Figure 2 — Layer window heatmap
# =============================================================================


def plot_layer_window_heatmap(
    components: List[str],
    windows: List[Tuple[int, int]],
    delta_matrix: np.ndarray,
    output_dir: str,
    dpi: int,
    delta_metric_label: str = "Mean Δprob(correct answer)",
    pvalue_matrix_: Optional[np.ndarray] = None,
    sem_matrix_: Optional[np.ndarray] = None,
    filename_stem: str = "M2_2_layer_window_heatmap",
    title_suffix: str = "",
) -> None:
    labels_comp = [COMPONENT_LABELS.get(c, c) for c in components]
    labels_win = [f"L{s}–{e}" for s, e in windows]

    fig, ax = plt.subplots(figsize=(max(6, len(windows) * 2.0), max(4, len(components) * 1.2)))

    vmax = max(abs(delta_matrix).max(), 0.01)
    im = ax.imshow(
        delta_matrix,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels(labels_win, fontsize=11)
    ax.set_yticks(np.arange(len(components)))
    ax.set_yticklabels(labels_comp, fontsize=11)

    # Annotate cells: value + SEM + significance marker
    for r in range(len(components)):
        for c in range(len(windows)):
            v = delta_matrix[r, c]
            text_colour = "white" if abs(v) > vmax * 0.55 else "black"
            # Build annotation string
            sem_str = ""
            if sem_matrix_ is not None:
                sem_str = f"\n±{sem_matrix_[r, c]:.3f}"
            sig_str = ""
            if pvalue_matrix_ is not None:
                sig_str = sig_marker(pvalue_matrix_[r, c])
            cell_text = f"{v:+.3f}{sem_str}"
            if sig_str:
                cell_text += f"\n{sig_str}"
            ax.text(c, r, cell_text, ha="center", va="center",
                    fontsize=8 if sem_matrix_ is not None else 9,
                    color=text_colour, fontweight="bold",
                    multialignment="center")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(delta_metric_label, fontsize=10)

    sig_note = "  * p<.05  ** p<.01  *** p<.001" if pvalue_matrix_ is not None else ""
    ax.set_title(
        f"Experiment 2 — Knockout Effect by Component × Layer Window{title_suffix}\n"
        f"(red = blocking hurts,  blue = blocking helps;  ±SEM shown{sig_note})",
        fontsize=10,
    )
    ax.set_xlabel("Layer window", fontsize=11)
    ax.set_ylabel("Knocked-out component", fontsize=11)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"{filename_stem}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"   Saved {p}")
    plt.close(fig)


# =============================================================================
# Figure 3 — Accuracy drop
# =============================================================================


def plot_accuracy_drop(
    components: List[str],
    windows: List[Tuple[int, int]],
    baseline_acc: float,
    ko_acc: Dict[str, Dict[Tuple[int, int], float]],
    output_dir: str,
    dpi: int,
) -> None:
    comps_present = [c for c in components if c in ko_acc]
    # Mean knockout accuracy across all windows for each component
    ko_mean_acc = {
        comp: float(np.mean([ko_acc[comp].get(w, 0.0) for w in windows]))
        for comp in comps_present
    }

    x = np.arange(len(comps_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    baseline_bars = ax.bar(
        x - width / 2,
        [baseline_acc] * len(comps_present),
        width,
        label="Baseline (no knockout)",
        color="#aec7e8",
        edgecolor="white",
        linewidth=0.8,
    )
    ko_bars = ax.bar(
        x + width / 2,
        [ko_mean_acc[c] for c in comps_present],
        width,
        label="After knockout (mean over windows)",
        color=[COMPONENT_COLOURS.get(c, "#7f7f7f") for c in comps_present],
        edgecolor="white",
        linewidth=0.8,
    )

    # Annotate bars
    for bar in list(baseline_bars) + list(ko_bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Annotate drop
    for i, comp in enumerate(comps_present):
        drop = ko_mean_acc[comp] - baseline_acc
        ax.annotate(
            f"{drop:+.3f}",
            xy=(x[i] + width / 2, ko_mean_acc[comp]),
            xytext=(x[i] + width / 2, ko_mean_acc[comp] - 0.04),
            ha="center",
            fontsize=8,
            color="darkred" if drop < 0 else "darkgreen",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [COMPONENT_LABELS.get(c, c) for c in comps_present],
        rotation=15,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, min(1.0, max(baseline_acc, max(ko_mean_acc.values())) + 0.12))
    ax.set_title(
        "Experiment 2 — Accuracy Before and After Attention Knockout\n"
        "(knockout: source → component; averaged over all layer windows)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"M2_3_accuracy_drop.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"   Saved {p}")
    plt.close(fig)


# =============================================================================
# Figure 4 — Prediction flip rate heatmap
# =============================================================================


def plot_flip_rate(
    components: List[str],
    windows: List[Tuple[int, int]],
    flip_matrix: np.ndarray,
    output_dir: str,
    dpi: int,
) -> None:
    labels_comp = [COMPONENT_LABELS.get(c, c) for c in components]
    labels_win = [f"L{s}–{e}" for s, e in windows]

    fig, ax = plt.subplots(figsize=(max(6, len(windows) * 1.8), max(4, len(components) * 1.1)))

    im = ax.imshow(
        flip_matrix,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(flip_matrix.max(), 0.01),
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels(labels_win, fontsize=11)
    ax.set_yticks(np.arange(len(components)))
    ax.set_yticklabels(labels_comp, fontsize=11)

    for r in range(len(components)):
        for c in range(len(windows)):
            v = flip_matrix[r, c]
            text_colour = "white" if v > flip_matrix.max() * 0.65 else "black"
            ax.text(c, r, f"{v:.2%}", ha="center", va="center", fontsize=9,
                    color=text_colour, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, format="%.0%%")
    cbar.set_label("Prediction flip rate", fontsize=10)

    ax.set_title(
        "Experiment 2 — Prediction Flip Rate by Component × Layer Window\n"
        "(fraction of eval items where knockout changed the predicted answer)",
        fontsize=10,
    )
    ax.set_xlabel("Layer window", fontsize=11)
    ax.set_ylabel("Knocked-out component", fontsize=11)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"M2_4_flip_rate.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"   Saved {p}")
    plt.close(fig)


# =============================================================================
# Figure 5 — Δprob_correct per layer window, all components overlaid
# =============================================================================


def plot_layer_window_lines(
    components: List[str],
    windows: List[Tuple[int, int]],
    delta_matrix: np.ndarray,
    output_dir: str,
    dpi: int,
    delta_metric_label: str = "Mean Δprob(correct answer)",
    sem_matrix_: Optional[np.ndarray] = None,
    filename_stem: str = "M2_5_layer_window_lines",
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(windows))
    labels_win = [f"L{s}–{e}" for s, e in windows]

    for ri, comp in enumerate(components):
        colour = COMPONENT_COLOURS.get(comp, "#7f7f7f")
        label = COMPONENT_LABELS.get(comp, comp)
        y = delta_matrix[ri]
        ax.plot(x, y, marker="o", color=colour, label=label, linewidth=2, markersize=7)
        if sem_matrix_ is not None:
            sem = sem_matrix_[ri]
            ax.fill_between(x, y - sem, y + sem, color=colour, alpha=0.15)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_win, fontsize=11)
    ax.set_xlabel("Layer window", fontsize=11)
    ax.set_ylabel(delta_metric_label, fontsize=11)
    shade_note = "  (shaded band = ±SEM)" if sem_matrix_ is not None else ""
    ax.set_title(
        f"Experiment 2 — Knockout Effect per Layer Window{title_suffix}\n"
        f"(decision → component; lower = component more important in that window{shade_note})",
        fontsize=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"{filename_stem}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"   Saved {p}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nPlotting Exp 2 knockout results")
    print(f"  results_dir : {args.results_dir}")
    print(f"  output_dir  : {output_dir}")
    if args.mode:
        print(f"  mode filter : {args.mode}")
    print()

    rows = load_results(args.results_dir, args.mode)
    if not rows:
        print("No rows to plot. Exiting.")
        return

    meta = load_metadata(args.results_dir)
    windows = collect_layer_windows(rows)
    components = collect_components(rows)

    print(f"  Layer windows : {windows}")
    print(f"  Components    : {components}")
    print()

    # ── collect unique eval items for n_samples ─────────────────────────────
    unique_items = {(r.get("sample_idx"), r.get("option_permutation_idx", "0")) for r in rows}
    n_eval_items = len(unique_items)
    unique_samples = {r.get("sample_idx") for r in rows}
    n_samples = len(unique_samples)
    print(f"  Unique samples       : {n_samples}")
    print(f"  Unique eval items    : {n_eval_items}")
    print()

    # ── detect mode and prepare delta metric ─────────────────────────────────
    #
    # decision_logits mode → rows have "delta_prob_correct" (continuous)
    # generation mode      → rows have "baseline_correct"/"knockout_correct" (0/1)
    #                         We synthesise delta = knockout_correct - baseline_correct
    #                         so the same aggregation/plot functions work for both.
    #
    row_modes = {r.get("mode") for r in rows}
    is_generation = "generation" in row_modes and "decision_logits" not in row_modes

    has_delta_prob = any(r.get("delta_prob_correct") not in (None, "") for r in rows)
    if not has_delta_prob:
        # generation mode: synthesise delta from per-row accuracy change
        for r in rows:
            try:
                r["delta_prob_correct"] = str(int(r["knockout_correct"]) - int(r["baseline_correct"]))
            except (KeyError, ValueError):
                r["delta_prob_correct"] = "0"

    delta_metric_label = (
        "Mean Δaccuracy (knockout − baseline)"
        if is_generation
        else "Mean Δprob(correct answer)"
    )

    # ── aggregate: delta_prob_correct (primary) ──────────────────────────────
    delta_agg = aggregate_by_component_and_window(rows, "delta_prob_correct")
    mean_delta = mean_by_component(delta_agg, windows)
    delta_mat = mean_matrix(delta_agg, components, windows)
    sem_mat = sem_matrix(delta_agg, components, windows)
    pval_mat = pvalue_matrix(delta_agg, components, windows)

    # SEM averaged over windows (for the overall bar chart)
    sem_overall = {
        comp: float(np.mean([sem_mat[ri, ci] for ci in range(len(windows))]))
        for ri, comp in enumerate(components)
    }
    pval_overall = {}
    if _HAS_SCIPY:
        for ri, comp in enumerate(components):
            all_vals = []
            for w in windows:
                all_vals.extend(delta_agg.get(comp, {}).get(w, []))
            if len(all_vals) >= 2:
                _, p = _scipy_stats.ttest_1samp(all_vals, 0.0)
                pval_overall[comp] = float(p)

    # ── aggregate: delta_prob_baseline_pred (position-bias-immune) ───────────
    # Measures change in P(model's own answer), irrespective of whether that
    # answer is correct. Not affected by which letter the correct answer sits at.
    has_baseline_pred_col = any(r.get("delta_prob_baseline_pred") not in (None, "") for r in rows)
    dbp_agg = aggregate_by_component_and_window(rows, "delta_prob_baseline_pred") \
        if has_baseline_pred_col else {}
    dbp_mat = mean_matrix(dbp_agg, components, windows) if dbp_agg else None
    dbp_sem = sem_matrix(dbp_agg, components, windows) if dbp_agg else None
    dbp_pval = pvalue_matrix(dbp_agg, components, windows) if dbp_agg else None

    baseline_acc, ko_acc = accuracy_by_component(rows, components, windows)
    flip_mat = flip_rate_matrix(rows, components, windows)

    print(f"  Baseline accuracy    : {baseline_acc:.4f}")
    print(f"  Delta metric         : {delta_metric_label}")
    print()

    # ── print summary table ───────────────────────────────────────────────────
    print(f"  === Knockout effect summary ({delta_metric_label}, mean over all windows) ===")
    print(f"  {'Component':14s} {'Δmean':>8}  {'SEM':>7}  {'p-val':>8}  {'sig':>4}  {'acc_drop':>9}")
    print(f"  {'-'*60}")
    for ri, comp in enumerate(components):
        v = mean_delta.get(comp, 0.0)
        ko_mean = float(np.mean([ko_acc.get(comp, {}).get(w, 0.0) for w in windows]))
        sem_v = sem_overall.get(comp, 0.0)
        p_v = pval_overall.get(comp, float("nan"))
        sig = sig_marker(p_v)
        p_str = f"{p_v:.4f}" if not np.isnan(p_v) else "  n/a"
        print(f"  {comp:14s} {v:>+8.4f}  {sem_v:>7.4f}  {p_str:>8}  {sig:>4}  {ko_mean - baseline_acc:>+9.4f}")
    print()

    if has_baseline_pred_col:
        print("  === delta_prob_baseline_pred (position-bias-immune) mean over all windows ===")
        dbp_mean = mean_by_component(dbp_agg, windows)
        for comp in components:
            v = dbp_mean.get(comp, 0.0)
            print(f"  {comp:14s}: {v:+.4f}")
        print()

    # ── generate figures ─────────────────────────────────────────────────────
    print("Generating figures...")

    plot_overall_effect(
        components, mean_delta, n_eval_items, output_dir, args.dpi, delta_metric_label,
        sem_delta=sem_overall, pval_delta=pval_overall,
    )
    plot_layer_window_heatmap(
        components, windows, delta_mat, output_dir, args.dpi, delta_metric_label,
        pvalue_matrix_=pval_mat, sem_matrix_=sem_mat,
    )
    plot_accuracy_drop(components, windows, baseline_acc, ko_acc, output_dir, args.dpi)
    plot_flip_rate(components, windows, flip_mat, output_dir, args.dpi)
    plot_layer_window_lines(
        components, windows, delta_mat, output_dir, args.dpi, delta_metric_label,
        sem_matrix_=sem_mat,
    )

    # M2_6: position-bias-immune heatmap (delta_prob_baseline_pred)
    if dbp_mat is not None:
        plot_layer_window_heatmap(
            components, windows, dbp_mat, output_dir, args.dpi,
            delta_metric_label="Mean Δprob(baseline prediction)",
            pvalue_matrix_=dbp_pval, sem_matrix_=dbp_sem,
            filename_stem="M2_6_heatmap_baseline_pred",
            title_suffix=" — position-bias-immune metric",
        )
        plot_layer_window_lines(
            components, windows, dbp_mat, output_dir, args.dpi,
            delta_metric_label="Mean Δprob(baseline prediction)",
            sem_matrix_=dbp_sem,
            filename_stem="M2_7_lines_baseline_pred",
            title_suffix=" — position-bias-immune",
        )

    print(f"\nAll figures saved to: {output_dir}")
    print(
        "Files:\n"
        "  M2_1_knockout_effect_overall.{png,pdf}  — bar chart, error bars, significance\n"
        "  M2_2_layer_window_heatmap.{png,pdf}      — heatmap delta_prob_correct + SEM + sig\n"
        "  M2_3_accuracy_drop.{png,pdf}             — accuracy before/after\n"
        "  M2_4_flip_rate.{png,pdf}                 — prediction flip rate\n"
        "  M2_5_layer_window_lines.{png,pdf}        — line plot + SEM bands\n"
        "  M2_6_heatmap_baseline_pred.{png,pdf}     — heatmap delta_prob(model's pred) [position-bias-immune]\n"
        "  M2_7_lines_baseline_pred.{png,pdf}       — line plot for same metric\n"
    )


if __name__ == "__main__":
    main()
