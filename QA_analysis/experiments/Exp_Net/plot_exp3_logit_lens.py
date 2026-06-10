"""
plot_exp3_logit_lens.py  —  Thesis-quality plots for Exp 3 (Logit Lens).

KEY FINDING: Audio tokens do NOT semanticise into music vocabulary.
Instead, they converge to non-English "attractor" tokens through the layers —
entropy collapses from ~9 bits (random) to ~1 bit (83 % predict the same token).

Reads:
    <results_dir>/aggregate_stats.json   (pre-computed by merge_exp3_shards.py)
    <results_dir>/logit_lens_results.jsonl  (per-record data, for box plots)

Produces 4 figures:

    M3_1_entropy_collapse.{png,pdf}
        Two-panel: entropy_bits collapse + top1_dominance rise across layers,
        split by correct / wrong.  Primary evidence for the attractor phenomenon.

    M3_2_top_tokens_per_phase.{png,pdf}
        Top-N most frequent top-1 predictions in Early / Mid / Late phases.
        Shows the qualitative shift from multilingual noise → attractor tokens.

    M3_3_entropy_heatmap.{png,pdf}
        Heatmap: category × layer, colour = mean entropy_bits.
        Lower entropy (warmer colour) = higher convergence.

    M3_4_entropy_at_key_layers.{png,pdf}
        Box plots of entropy_bits at key transition layers (L0, L5, L_last),
        split by correct / wrong.  Tests whether convergence rate predicts accuracy.

Usage:
    python -m QA_analysis.experiments.Exp_Net.plot_exp3_logit_lens \\
        --results_dir /path/to/merged \\
        --output_dir  /path/to/merged/plots
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style constants ───────────────────────────────────────────────────────────
_FIG_DPI    = 180
_FONT_SIZE  = 11
_TITLE_SIZE = 13
_LABEL_SIZE = 11
_TICK_SIZE  = 10
_LEGEND_SIZE = 10

_COL_CORRECT = "#2ca02c"   # green
_COL_WRONG   = "#d62728"   # red
_COL_ALL     = "#1f77b4"   # blue

_COL_BG_EARLY = "#e8f4fd"
_COL_BG_MID   = "#fef9e7"
_COL_BG_LATE  = "#fdf2f8"

plt.rcParams.update({
    "font.size":           _FONT_SIZE,
    "axes.titlesize":      _TITLE_SIZE,
    "axes.labelsize":      _LABEL_SIZE,
    "xtick.labelsize":     _TICK_SIZE,
    "ytick.labelsize":     _TICK_SIZE,
    "legend.fontsize":     _LEGEND_SIZE,
    "figure.dpi":          _FIG_DPI,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
})


# =============================================================================
# Helpers
# =============================================================================

def _load_aggregate(results_dir: str) -> Dict:
    path = os.path.join(results_dir, "aggregate_stats.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"aggregate_stats.json not found in {results_dir}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_per_record_entropy(
    results_dir: str,
    key_layers: List[int],
) -> Dict[int, Dict[str, List[float]]]:
    """
    Read entropy_bits at specific layers from per-record JSONL.
    Returns {layer_idx: {"correct": [...], "wrong": [...]}}
    """
    out: Dict[int, Dict[str, List[float]]] = {
        li: {"correct": [], "wrong": []} for li in key_layers
    }
    jsonl = os.path.join(results_dir, "logit_lens_results.jsonl")
    if not os.path.exists(jsonl):
        print("   ⚠ logit_lens_results.jsonl not found — box plots will be empty.")
        return out

    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            group = "correct" if bool(rec.get("is_correct", 0)) else "wrong"
            per_layer = rec.get("per_layer", {})
            for li in key_layers:
                v = per_layer.get(str(li), {}).get("entropy_bits", float("nan"))
                if not math.isnan(v):
                    out[li][group].append(v)
    return out


def _phase_bands(ax, phase_ranges: Dict, labels: bool = True) -> None:
    """Draw coloured background bands for early/mid/late phases."""
    colors  = {"early": _COL_BG_EARLY, "mid": _COL_BG_MID, "late": _COL_BG_LATE}
    lbl_map = {"early": "Early", "mid": "Mid", "late": "Late"}
    ylim = ax.get_ylim()
    for phase, (s, e) in phase_ranges.items():
        ax.axvspan(s, e - 0.5, color=colors[phase], alpha=0.40, zorder=0)
        if labels:
            mid = (s + e) / 2
            ax.text(mid, ylim[1] * 0.97, lbl_map[phase],
                    ha="center", va="top", fontsize=8,
                    color="gray", style="italic", zorder=1)


def _clean_token(tok: str) -> str:
    return tok.replace("▁", " ").strip() or "∅"


def _save(fig: plt.Figure, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"),
                    dpi=_FIG_DPI, bbox_inches="tight")
    print(f"   ✅ {name}.png / .pdf")
    plt.close(fig)


def _get(per_layer: Dict, li: int, key: str) -> float:
    v = per_layer.get(str(li), {}).get(key)
    return v if (v is not None and not math.isnan(v)) else float("nan")


# =============================================================================
# M3_1 — Entropy collapse  (primary result figure)
# =============================================================================

def plot_entropy_collapse(agg: Dict, output_dir: str) -> None:
    """
    Two-panel figure showing the audio-token representation collapse across layers:
      (a) Entropy (bits) — drops from ~9 (uniform) to ~1 (attractor)
      (b) Top-1 dominance — rises from ~5 % to ~85 %
    Both split by correct / wrong predictions.
    """
    total_layers = int(agg["total_layers"])
    phase_ranges = agg.get("phase_layer_ranges", {
        "early": [0, total_layers // 3],
        "mid":   [total_layers // 3, 2 * total_layers // 3],
        "late":  [2 * total_layers // 3, total_layers],
    })
    layers = list(range(total_layers))
    pl = agg["per_layer"]

    # ── collect arrays ────────────────────────────────────────────────────────
    ent_all = np.array([_get(pl, li, "mean_entropy")         for li in layers])
    ent_sem = np.array([_get(pl, li, "sem_entropy")          for li in layers])
    ent_c   = np.array([_get(pl, li, "mean_entropy_correct") for li in layers])
    sem_c   = np.array([_get(pl, li, "sem_entropy_correct")  for li in layers])
    ent_w   = np.array([_get(pl, li, "mean_entropy_wrong")   for li in layers])
    sem_w   = np.array([_get(pl, li, "sem_entropy_wrong")    for li in layers])

    dom_all = np.array([_get(pl, li, "mean_dominance")         for li in layers])
    dom_sem = np.array([_get(pl, li, "sem_dominance")          for li in layers])
    dom_c   = np.array([_get(pl, li, "mean_dominance_correct") for li in layers])
    dsem_c  = np.array([_get(pl, li, "sem_dominance_correct")  for li in layers])
    dom_w   = np.array([_get(pl, li, "mean_dominance_wrong")   for li in layers])
    dsem_w  = np.array([_get(pl, li, "sem_dominance_wrong")    for li in layers])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Exp 3 — Logit Lens: Audio-Token Representation Collapse Across Layers",
        fontsize=_TITLE_SIZE + 1, y=1.02,
    )

    # ── shared helpers ────────────────────────────────────────────────────────
    def _plot_split(ax, y_all, y_sem, y_c, s_c, y_w, s_w, ylabel, title):
        ax.plot(layers, y_all, color=_COL_ALL,     lw=2.2, label="All",     zorder=4)
        ax.fill_between(layers, y_all - y_sem, y_all + y_sem,
                        color=_COL_ALL, alpha=0.15, zorder=3)
        ax.plot(layers, y_c, color=_COL_CORRECT, lw=1.8, ls="--", label="Correct", zorder=4)
        ax.fill_between(layers, y_c - s_c, y_c + s_c,
                        color=_COL_CORRECT, alpha=0.12, zorder=3)
        ax.plot(layers, y_w, color=_COL_WRONG,   lw=1.8, ls=":",  label="Wrong",   zorder=4)
        ax.fill_between(layers, y_w - s_w, y_w + s_w,
                        color=_COL_WRONG, alpha=0.12, zorder=3)
        ax.set_xlabel("Layer", fontsize=_LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=_LABEL_SIZE)
        ax.set_xlim(-0.5, total_layers - 0.5)
        ax.legend(loc="best", framealpha=0.85, fontsize=_LEGEND_SIZE)
        ax.set_title(title, fontsize=_TITLE_SIZE)

    # ── (a) Entropy ───────────────────────────────────────────────────────────
    ax = axes[0]
    _plot_split(ax, ent_all, ent_sem, ent_c, sem_c, ent_w, sem_w,
                ylabel="Shannon entropy of top-1 predictions (bits)",
                title="(a) Entropy collapse")
    ax.set_ylim(bottom=0)

    # Phase bands + transition markers
    _phase_bands(ax, phase_ranges)
    # Typical transition layers discovered in test: L5 and L24
    for li, lbl, col in [(5,  "L5\n(phase\ntransition)", "#8B0000"),
                          (24, "L24\n(attractor\nshift)",  "#FF8C00")]:
        if li < total_layers:
            ax.axvline(li, color=col, ls="--", lw=1.1, alpha=0.75, zorder=5)
            ax.text(li + 0.3, ax.get_ylim()[1] * 0.65, lbl,
                    color=col, fontsize=7.5, va="top", zorder=6)

    # ── (b) Dominance ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    _plot_split(ax2, dom_all, dom_sem, dom_c, dsem_c, dom_w, dsem_w,
                ylabel="Top-1 dominance (fraction of audio tokens\npredicting the same single token)",
                title="(b) Attractor convergence")
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    _phase_bands(ax2, phase_ranges)
    for li, lbl, col in [(5,  "L5", "#8B0000"), (24, "L24", "#FF8C00")]:
        if li < total_layers:
            ax2.axvline(li, color=col, ls="--", lw=1.1, alpha=0.75, zorder=5)

    fig.tight_layout()
    _save(fig, output_dir, "M3_1_entropy_collapse")


# =============================================================================
# M3_2 — Top tokens per phase  (unchanged — already shows attractor tokens)
# =============================================================================

def plot_top_tokens_per_phase(agg: Dict, output_dir: str, top_n: int = 15) -> None:
    """
    Three horizontal bar charts: top-N most frequent top-1 predictions
    for audio tokens in Early / Mid / Late layers.

    The attractor tokens (Chinese / Vietnamese) will dominate Mid and Late bars,
    making this a qualitative illustration of the convergence finding.
    """
    phase_data   = agg.get("per_phase_top50_tokens", {})
    phase_ranges = agg.get("phase_layer_ranges", {})
    phase_order  = ["early", "mid", "late"]
    phase_titles = {
        "early": f"Early  (L{phase_ranges.get('early',[0,0])[0]}–{phase_ranges.get('early',[0,9])[1]-1})",
        "mid":   f"Mid    (L{phase_ranges.get('mid',[0,0])[0]}–{phase_ranges.get('mid',[0,18])[1]-1})",
        "late":  f"Late   (L{phase_ranges.get('late',[0,0])[0]}–{phase_ranges.get('late',[0,27])[1]-1})",
    }
    bar_colors = {"early": "#5b9bd5", "mid": "#ed7d31", "late": "#a9d18e"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Exp 3 — Top-1 Token Predictions for Audio Tokens by Processing Phase\n"
        "(Mid/Late dominated by non-English attractor tokens)",
        fontsize=_TITLE_SIZE + 1,
    )

    for ax, phase in zip(axes, phase_order):
        tc_list = phase_data.get(phase, [])[:top_n]
        if not tc_list:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(phase_titles.get(phase, phase), fontsize=_TITLE_SIZE)
            continue

        tokens = [_clean_token(tc[0]) for tc in tc_list][::-1]
        counts = [tc[1] for tc in tc_list][::-1]
        total  = sum(c for _, c in phase_data.get(phase, [])) or 1  # all top-50 total

        bars = ax.barh(range(len(tokens)), [c / total * 100 for c in counts],
                       color=bar_colors[phase], alpha=0.85, edgecolor="white", lw=0.5)

        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel("% of audio-token predictions", fontsize=_LABEL_SIZE)
        ax.set_title(phase_titles.get(phase, phase), fontsize=_TITLE_SIZE)
        ax.set_xlim(0, None)
        ax.set_ylim(-0.5, len(tokens) - 0.5)

        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{cnt:,}", va="center", ha="left", fontsize=8, color="gray")

    fig.tight_layout()
    _save(fig, output_dir, "M3_2_top_tokens_per_phase")


# =============================================================================
# M3_3 — Category × layer entropy heatmap
# =============================================================================

def plot_entropy_heatmap(agg: Dict, output_dir: str) -> None:
    """
    Heatmap: rows = HumMusQA categories, columns = layers.
    Colour = mean entropy_bits.  Lower entropy (warmer) = higher convergence.

    If all categories collapse at the same rate → audio processing is category-agnostic.
    If categories differ → richer semantic content influences convergence speed.
    """
    per_cat     = agg.get("per_category_per_layer", {})
    total_layers = int(agg["total_layers"])

    if not per_cat:
        print("   ⚠ No category data — skipping M3_3.")
        return

    categories = sorted(per_cat.keys())
    layers     = list(range(total_layers))

    matrix = np.full((len(categories), total_layers), fill_value=np.nan)
    for r, cat in enumerate(categories):
        for li in layers:
            v = per_cat[cat].get(str(li))
            if v is not None and not math.isnan(v):
                matrix[r, li] = v

    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 9.0
    vmin = 0.0

    fig, ax = plt.subplots(
        figsize=(max(10, total_layers * 0.38), max(4, len(categories) * 0.6))
    )
    fig.suptitle(
        "Exp 3 — Logit Lens: Entropy of Audio-Token Predictions by Category and Layer\n"
        "(lower = more convergent; warmer colour = stronger attractor effect)",
        fontsize=_TITLE_SIZE + 1,
    )

    # Use reversed RdYlBu: low entropy → warm (red/orange), high entropy → cool (blue)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("Mean entropy_bits  (lower = more convergent)", fontsize=_LABEL_SIZE)

    step = max(1, total_layers // 14)
    tick_locs = list(range(0, total_layers, step))
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"L{i}" for i in tick_locs], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=_FONT_SIZE)
    ax.set_xlabel("Layer", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Category", fontsize=_LABEL_SIZE)

    # Phase boundary lines
    phase_ranges = agg.get("phase_layer_ranges", {})
    for phase, (s, e) in phase_ranges.items():
        if s > 0:
            ax.axvline(s - 0.5, color="white", lw=1.8, ls="--", alpha=0.9)
        lbl = {"early": "Early", "mid": "Mid", "late": "Late"}.get(phase, phase)
        mid = (s + e) / 2
        ax.text(mid, -0.8, lbl, ha="center", va="top",
                fontsize=8, color="gray", style="italic")

    fig.tight_layout()
    _save(fig, output_dir, "M3_3_entropy_heatmap")


# =============================================================================
# M3_4 — Entropy at key transition layers (correct vs wrong)
# =============================================================================

def plot_entropy_at_key_layers(
    results_dir: str,
    agg: Dict,
    output_dir: str,
) -> None:
    """
    Box plots of entropy_bits at key layers:
      L0   — initial state (nearly random)
      L5   — first-transition layer (entropy drops sharply here)
      L_last — final layer (maximum convergence)
    Split by correct / wrong.  Tests whether deeper convergence predicts accuracy.
    """
    total_layers = int(agg["total_layers"])
    # Key layers: L0, the L5 transition, and last
    transition = 5 if total_layers > 5 else total_layers // 5
    key_layers = sorted({0, transition, total_layers - 1})
    key_labels = {
        0:              "L0\n(initial)",
        transition:     f"L{transition}\n(transition)",
        total_layers-1: f"L{total_layers-1}\n(final)",
    }

    data = _load_per_record_entropy(results_dir, key_layers)

    fig, axes = plt.subplots(
        1, len(key_layers), figsize=(4.5 * len(key_layers), 5.5), sharey=False
    )
    if len(key_layers) == 1:
        axes = [axes]

    fig.suptitle(
        "Exp 3 — Logit Lens: Entropy at Key Layers (Correct vs Wrong)\n"
        "Lower entropy = stronger attractor convergence",
        fontsize=_TITLE_SIZE + 1,
    )

    for ax, li in zip(axes, key_layers):
        vals_c = data[li]["correct"]
        vals_w = data[li]["wrong"]

        groups = [vals_c, vals_w]
        labels = [f"Correct\n(n={len(vals_c)})", f"Wrong\n(n={len(vals_w)})"]
        box_colors = [_COL_CORRECT, _COL_WRONG]

        bp = ax.boxplot(
            groups,
            labels=labels,
            patch_artist=True,
            medianprops={"color": "black", "lw": 2.0},
            whiskerprops={"lw": 1.3},
            capprops={"lw": 1.3},
            flierprops={"marker": "o", "ms": 3, "alpha": 0.35},
            notch=False,
            widths=0.45,
        )
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)

        # Jittered individual points
        np.random.seed(42)
        for xi, (vals, col) in enumerate(zip(groups, box_colors), start=1):
            if vals:
                jitter = np.random.uniform(-0.18, 0.18, size=len(vals))
                ax.scatter(xi + jitter, vals, color=col,
                           alpha=0.22, s=10, zorder=6)

        # Mean marker
        for xi, (vals, col) in enumerate(zip(groups, box_colors), start=1):
            if vals:
                ax.scatter([xi], [np.mean(vals)], marker="D", s=40,
                           color=col, zorder=8, edgecolors="white", lw=0.8)

        ax.set_title(key_labels.get(li, f"L{li}"), fontsize=_TITLE_SIZE)
        ax.set_ylabel("Entropy (bits)" if li == key_layers[0] else "")
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, output_dir, "M3_4_entropy_at_key_layers")


# =============================================================================
# CLI & main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Exp 3 Logit Lens results (entropy collapse / attractor tokens)"
    )
    p.add_argument("--results_dir", required=True,
                   help="Merged results directory (containing aggregate_stats.json).")
    p.add_argument("--output_dir", default=None,
                   help="Output directory for plots. Defaults to <results_dir>/plots.")
    p.add_argument("--top_n_tokens", type=int, default=15,
                   help="Top-N tokens to show in M3_2 bar charts (default: 15).")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    results = os.path.abspath(args.results_dir)
    outdir  = os.path.abspath(
        args.output_dir if args.output_dir else os.path.join(results, "plots")
    )
    os.makedirs(outdir, exist_ok=True)

    print(f"\nPlotting Exp 3 — Logit Lens (entropy collapse / attractor tokens)")
    print(f"  results_dir : {results}")
    print(f"  output_dir  : {outdir}\n")

    agg          = _load_aggregate(results)
    total_layers = int(agg["total_layers"])
    n_records    = int(agg.get("n_records", 0))
    print(f"aggregate_stats loaded: {n_records} records, {total_layers} layers\n")

    print("Generating M3_1 — entropy collapse...")
    plot_entropy_collapse(agg, outdir)

    print("Generating M3_2 — top tokens per phase...")
    plot_top_tokens_per_phase(agg, outdir, top_n=args.top_n_tokens)

    print("Generating M3_3 — entropy heatmap by category...")
    plot_entropy_heatmap(agg, outdir)

    print("Generating M3_4 — entropy at key transition layers...")
    plot_entropy_at_key_layers(results, agg, outdir)

    print(f"\n✅  All plots saved to {outdir}")
    print(f"\nKey finding narrative:")
    print(f"  Audio tokens do NOT develop music-related semantics.")
    print(f"  Instead, they converge to non-English attractor tokens:")
    print(f"  entropy drops from ~9 bits (uniform over vocab) to ~1 bit (83% → one token).")
    print(f"  This is consistent with UC_audio≈3% (ExpE), T-SHAP>A-SHAP (ExpA),")
    print(f"  and the weak causal audio effect in Exp2.")


if __name__ == "__main__":
    main()
