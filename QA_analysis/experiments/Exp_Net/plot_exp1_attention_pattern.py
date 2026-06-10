"""
Plot utilities for Experiment 1 — Attention Pattern Analysis.

Input:
    attention_per_layer.csv
    summary_stats.csv
    metadata.json

Output:
    PNG/PDF plots:
      - M1_layer_attention_components.png
      - M1_layer_attention_correct_wrong.png
      - M1_audio_attention_by_category_heatmap.png
      - M1_attention_by_difficulty.png
      - M1_late_component_audio_ratio_by_category.png

Uso:
    python plot_exp1_attention_pattern.py \
        --results_dir results/attention_patterns_hummusqa \
        --output_dir results/attention_patterns_hummusqa/plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--min_category_n", type=int, default=1)
    return parser.parse_args()


def infer_n_layers(df: pd.DataFrame, prefix: str = "audio_layer_") -> int:
    cols = [c for c in df.columns if c.startswith(prefix)]
    ids = []
    for c in cols:
        try:
            ids.append(int(c.split("_")[-1]))
        except Exception:
            pass
    return max(ids) + 1 if ids else 0


def _cols(prefix: str, n_layers: int) -> List[str]:
    return [f"{prefix}_layer_{i}" for i in range(n_layers)]


def load_results(results_dir: str, n_layers: int | None) -> Tuple[pd.DataFrame, pd.DataFrame, dict, int, dict]:
    layer_path = os.path.join(results_dir, "attention_per_layer.csv")
    summary_path = os.path.join(results_dir, "summary_stats.csv")
    metadata_path = os.path.join(results_dir, "metadata.json")

    df = pd.read_csv(layer_path)
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if n_layers is None:
        n_layers = infer_n_layers(df)
    component_cols = {
        "audio": _cols("audio", n_layers),
        "instruction": _cols("instruction", n_layers),
        "question": _cols("question", n_layers),
        "options": _cols("options", n_layers),
        "other_text": _cols("other_text", n_layers),
        "query_text": _cols("query_text", n_layers),
    }

    for col in [c for cols in component_cols.values() for c in cols]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_correct"] = pd.to_numeric(df.get("is_correct", 0), errors="coerce").fillna(0).astype(int)

    return df, summary, metadata, n_layers, component_cols


def mean_sem(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data = data.astype(float)
    mean = np.nanmean(data, axis=0)
    n = np.sum(np.isfinite(data), axis=0)
    std = np.nanstd(data, axis=0)
    sem = np.divide(std, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(std), where=n > 0)
    return mean, sem


def save_fig(fig, output_dir: str, name: str):
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, f"{name}.png")
    pdf = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {png}")


def _available(cols: List[str], df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in cols)


def plot_main_components(df, component_cols, n_layers, output_dir):
    layers = np.arange(n_layers)
    series = [
        ("audio", "Audio tokens", 2.4),
        ("question", "Question text", 2.1),
        ("options", "Answer options", 2.1),
        ("instruction", "Instructions/scaffold", 1.8),
        ("other_text", "Other text/template", 1.7),
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    for key, label, lw in series:
        cols = component_cols.get(key, [])
        if not _available(cols, df):
            continue
        mean, sem = mean_sem(df[cols].to_numpy())
        ax.plot(layers, mean, lw=lw, label=label)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.14)

    ax.set_title("Experiment 1 — Attention from generated answer tokens to HumMusQA components")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Attention fraction normalized over original input tokens")
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    save_fig(fig, output_dir, "M1_layer_attention_components")


def plot_correct_wrong(df, component_cols, n_layers, output_dir):
    layers = np.arange(n_layers)
    fig, ax = plt.subplots(figsize=(11, 5))

    for is_correct, linestyle, label_suffix in [(1, "-", "correct"), (0, "--", "wrong")]:
        sub = df[df["is_correct"] == is_correct]
        if len(sub) == 0:
            continue
        for key, label in [
            ("audio", "Audio"),
            ("question", "Question"),
            ("options", "Options"),
            ("instruction", "Instruction"),
            ("other_text", "Other text"),
        ]:
            cols = component_cols.get(key, [])
            if not _available(cols, sub):
                continue
            mean, _ = mean_sem(sub[cols].to_numpy())
            ax.plot(layers, mean, lw=2.0, linestyle=linestyle, label=f"{label} — {label_suffix} (n={len(sub)})")

    ax.set_title("Experiment 1 — Attention pattern split by correctness")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Attention fraction")
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    save_fig(fig, output_dir, "M1_layer_attention_correct_wrong")


def plot_category_heatmap(df, audio_cols, n_layers, output_dir, min_category_n):
    cats = []
    rows = []
    counts = []
    for cat, sub in df.groupby("main_category"):
        if not isinstance(cat, str) or len(cat.strip()) == 0:
            continue
        if len(sub) < min_category_n:
            continue
        vals = sub[audio_cols].to_numpy(dtype=float)
        cats.append(cat)
        rows.append(np.nanmean(vals, axis=0))
        counts.append(len(sub))

    if not rows:
        print("⚠️ no category heatmap generated")
        return

    order = np.argsort(cats)
    mat = np.vstack([rows[i] for i in order])
    cats = [cats[i] for i in order]
    counts = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.48 * len(cats))))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=np.nanmax(mat) if np.isfinite(mat).any() else 1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean audio attention")
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(cats, counts)], fontsize=8)
    ax.set_xticks(np.arange(0, n_layers, max(1, n_layers // 8)))
    ax.set_xlabel("Transformer layer")
    ax.set_title("Experiment 1 — Mean audio attention by HumMusQA category")
    save_fig(fig, output_dir, "M1_audio_attention_by_category_heatmap")


def plot_difficulty(df, component_cols, n_layers, output_dir):
    layers = np.arange(n_layers)
    diffs = [d for d in ["Low", "Medium", "High"] if d in set(df["difficulty"].astype(str))]
    if not diffs:
        diffs = sorted(df["difficulty"].dropna().astype(str).unique().tolist())

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    panels = [
        ("audio", "Audio attention"),
        ("question", "Question attention"),
        ("options", "Options attention"),
        ("instruction", "Instruction attention"),
        ("other_text", "Other text attention"),
    ]
    for diff in diffs:
        sub = df[df["difficulty"].astype(str) == diff]
        if len(sub) == 0:
            continue
        for ax, (key, _title) in zip(axes, panels):
            cols = component_cols.get(key, [])
            if not _available(cols, sub):
                continue
            mean, _ = mean_sem(sub[cols].to_numpy())
            ax.plot(layers, mean, lw=2, label=f"{diff} (n={len(sub)})")

    for ax, (_key, title) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Attention fraction")
        ax.set_xlim(0, n_layers - 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Experiment 1 — Attention pattern by difficulty")
    save_fig(fig, output_dir, "M1_attention_by_difficulty")


def plot_late_ratio_by_category(summary, output_dir, min_category_n):
    ratio_cols = [
        ("late_question_audio_ratio", "Question/audio"),
        ("late_options_audio_ratio", "Options/audio"),
        ("late_text_audio_ratio", "All text/audio"),
    ]
    ratio_cols = [(col, label) for col, label in ratio_cols if col in summary.columns]
    if summary.empty or not ratio_cols:
        return
    for col, _label in ratio_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    rows = []
    for cat, sub in summary.groupby("main_category"):
        if not isinstance(cat, str) or len(cat.strip()) == 0 or len(sub) < min_category_n:
            continue
        medians = []
        for col, _label in ratio_cols:
            vals = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
            medians.append(float(vals.median()) if len(vals) else np.nan)
        if not np.isfinite(medians).any():
            continue
        rows.append((cat, len(sub), medians))
    if not rows:
        return
    rows.sort(key=lambda x: np.nan_to_num(x[2][0], nan=-1.0))

    labels = [f"{cat} (n={n})" for cat, n, _ in rows]
    y = np.arange(len(rows))
    width = 0.8 / max(1, len(ratio_cols))
    fig, ax = plt.subplots(figsize=(11, max(4, 0.46 * len(rows))))
    for j, (_col, label) in enumerate(ratio_cols):
        vals = [ms[j] for _, _, ms in rows]
        ax.barh(y + (j - (len(ratio_cols) - 1) / 2) * width, vals, height=width, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Median late-layer component/audio attention ratio")
    ax.set_title("Experiment 1 — Late text-component dominance by category")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=9)
    save_fig(fig, output_dir, "M1_late_component_audio_ratio_by_category")


def print_console_summary(df, summary, component_cols, metadata=None):
    print("\n" + "=" * 70)
    print("Experiment 1 summary")
    print("=" * 70)
    print(f"Samples: {len(df)}")
    if len(df):
        n_permutations = int((metadata or {}).get("option_permutations", 1) or 1)
        label = "Permuted-option accuracy" if n_permutations > 1 else "Fixed-order accuracy diagnostic"
        print(f"{label}: {df['is_correct'].mean():.4f}")
        if n_permutations > 1 and (metadata or {}).get("semantic_consistency_rate") is not None:
            print(f"Option-order semantic consistency: {metadata['semantic_consistency_rate']:.4f}")
    n_layers = len(component_cols["audio"])
    thirds = [(0, n_layers // 3), (n_layers // 3, 2 * n_layers // 3), (2 * n_layers // 3, n_layers)]
    names = ["early", "middle", "late"]
    for name, (s, e) in zip(names, thirds):
        pieces = []
        for key in ["audio", "question", "options", "instruction", "other_text"]:
            cols = component_cols.get(key, [])
            if _available(cols, df):
                arr = df[cols].to_numpy(dtype=float)
                pieces.append(f"{key}: {np.nanmean(arr[:, s:e]):.6f}")
        print(f"{name:>6} " + " | ".join(pieces))
    if not summary.empty and "main_category" in summary.columns:
        print("\nSamples by category:")
        print(summary["main_category"].value_counts().to_string())
    print()


# =============================================================================
# Thesis-quality figures (F1, F2, F3)
# These complement the existing M1_* plots with better visual design for
# publication: other_text excluded (attention-sink noise), per-token
# normalization, and explicit bridge to Exp2 knockout results.
# =============================================================================

_THESIS_SERIES = [
    # (component_key, display_label, hex_color, linewidth)
    ("audio",       "Audio tokens",         "#1f77b4", 2.5),
    ("question",    "Question text",         "#2ca02c", 2.0),
    ("options",     "Answer options",        "#ff7f0e", 2.5),
    ("instruction", "Instructions/scaffold", "#d62728", 2.0),
]

_N_TOKEN_COLS = {
    "audio":       "n_audio_tokens",
    "question":    "n_question_tokens",
    "options":     "n_options_tokens",
    "instruction": "n_instruction_tokens",
}


def _per_token_matrix(df: pd.DataFrame, cols: List[str], n_col: str) -> np.ndarray:
    """Return per-row, per-layer per-token attention × 10³ (readable units)."""
    if n_col not in df.columns:
        return np.full((len(df), len(cols)), np.nan)
    n_tokens = df[n_col].to_numpy(dtype=float)
    attn = df[cols].to_numpy(dtype=float)
    out = np.divide(
        attn, n_tokens[:, np.newaxis],
        out=np.zeros_like(attn),
        where=n_tokens[:, np.newaxis] > 0,
    )
    return out * 1e3  # units: ×10⁻³ attention per token


def plot_thesis_main(df: pd.DataFrame, component_cols: dict, n_layers: int, output_dir: str):
    """F1 — 2-panel: raw attention fraction (excl. other_text) + per-token log-scale.

    Left panel  matches AVLLM paper style for direct comparison.
    Right panel reveals the per-token magnitude gap (audio ~200× less than options).
    """
    layers = np.arange(n_layers)
    n_eval = len(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for key, label, color, lw in _THESIS_SERIES:
        cols = component_cols.get(key, [])
        if not _available(cols, df):
            continue
        mean, sem = mean_sem(df[cols].to_numpy())

        # Panel 1: raw fraction
        ax1.plot(layers, mean, lw=lw, label=label, color=color)
        ax1.fill_between(layers, mean - sem, mean + sem, alpha=0.15, color=color)

        # Panel 2: per-token ×10⁻³ (log scale)
        n_col = _N_TOKEN_COLS.get(key, "")
        pt = _per_token_matrix(df, cols, n_col)
        pt_mean, pt_sem = mean_sem(pt)
        valid = pt_mean > 0
        ax2.plot(layers[valid], pt_mean[valid], lw=lw, label=label, color=color)
        lo = np.maximum(pt_mean - pt_sem, 1e-6)
        ax2.fill_between(layers[valid], lo[valid], (pt_mean + pt_sem)[valid],
                         alpha=0.15, color=color)

    for ax, vline, vtext, title, ylabel in [
        (ax1, [(2, "L2\naudio peak"), (14, "L14\noptions peak")],
         None,
         "Attention fraction per component (excl. other_text)",
         "Mean attention fraction (answer token → component)"),
        (ax2, [(2, "L2"), (14, "L14")],
         None,
         "Per-token attention ×10⁻³ — log scale",
         "Mean attention per token  (×10⁻³, log scale)"),
    ]:
        for lx, ltxt in vline:
            ax.axvline(lx, color="gray", lw=0.9, ls="--", alpha=0.55)
            ylim = ax.get_ylim()
            ax.text(lx + 0.3, ylim[1] * 0.96 if ylim[1] > 0 else 0.9, ltxt,
                    fontsize=7.5, color="gray", va="top")
        ax.set_xlabel("Transformer layer", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, n_layers - 1)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=9)

    ax2.set_yscale("log")
    fig.suptitle(
        f"Experiment 1 — Attention Pattern  |  Qwen2.5-Omni-7B  |  HumMusQA N={n_eval} evals",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "M1_thesis_main")


def plot_thesis_correct_wrong(df: pd.DataFrame, component_cols: dict, n_layers: int, output_dir: str):
    """F2 — 2-panel: audio correct/wrong + options correct/wrong.

    Highlights:
    - Audio attention is nearly identical between correct and wrong answers
      (attention to audio does not discriminate correctness).
    - Options attention at L14 is slightly *higher* for wrong answers:
      the model re-reads options more when confused, yet still fails.
    """
    layers = np.arange(n_layers)
    correct = df[df["is_correct"] == 1]
    wrong   = df[df["is_correct"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    panels = [
        (axes[0], "audio",   "Audio tokens",  "#1f77b4", 2,  "L2\naudio peak"),
        (axes[1], "options", "Answer options", "#ff7f0e", 2.5, "L14\noptions peak"),
    ]

    for ax, key, label, color, lw, peak_label in panels:
        cols = component_cols.get(key, [])
        if not _available(cols, df):
            continue

        mc, sc = mean_sem(correct[cols].to_numpy())
        mw, sw = mean_sem(wrong[cols].to_numpy())

        ax.plot(layers, mc, lw=lw,      color=color, label=f"Correct (n={len(correct)})")
        ax.fill_between(layers, mc - sc, mc + sc, alpha=0.18, color=color)
        ax.plot(layers, mw, lw=lw, ls="--", color=color, label=f"Wrong (n={len(wrong)})")
        ax.fill_between(layers, mw - sw, mw + sw, alpha=0.10, color=color)

        # Shade the difference: wrong > correct → red tint; correct > wrong → green tint
        diff = mw - mc
        ax.fill_between(layers, mc, mc + np.maximum(diff, 0),
                        where=diff > 0, alpha=0.18, color="#d62728", label="Wrong > Correct")
        ax.fill_between(layers, mc + np.minimum(diff, 0), mc,
                        where=diff < 0, alpha=0.18, color="#2ca02c", label="Correct > Wrong")

        # Vertical guide at key layer
        peak_layer = int(np.argmax(np.maximum(mc, mw)))
        ax.axvline(peak_layer, color="gray", lw=0.9, ls=":", alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(peak_layer + 0.3, ylim[1] * 0.97, peak_label,
                fontsize=7.5, color="gray", va="top")

        ax.set_title(f"{label} — Correct vs Wrong answers", fontsize=11)
        ax.set_xlabel("Transformer layer", fontsize=11)
        ax.set_ylabel("Mean attention fraction", fontsize=10)
        ax.set_xlim(0, n_layers - 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Experiment 1 — Attention by correctness  |  Qwen2.5-Omni-7B  |  HumMusQA",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "M1_thesis_correct_wrong")


def plot_thesis_phase_summary(df: pd.DataFrame, component_cols: dict, n_layers: int, output_dir: str):
    """F3 — Grouped bar chart: Early / Mid / Late phases × components (per-token ×10⁻³).

    Bridges Exp1 → Exp2 knockout:
    - Audio is concentrated in Early layers → consistent with isolation-knockout
      effect in Exp2 L0-16.
    - Options dominate in Mid layers (L9-18) → matches Exp2 knockout peak at L12-16.
    """
    n_third = n_layers // 3
    phases = [
        (f"Early\n(L0–{n_third - 1})",        list(range(0, n_third))),
        (f"Mid\n(L{n_third}–{2*n_third - 1})", list(range(n_third, 2 * n_third))),
        (f"Late\n(L{2*n_third}–{n_layers-1})", list(range(2 * n_third, n_layers))),
    ]

    comp_info = [
        ("audio",       "Audio",        "#1f77b4", "n_audio_tokens"),
        ("instruction", "Instruction",  "#d62728", "n_instruction_tokens"),
        ("question",    "Question",     "#2ca02c", "n_question_tokens"),
        ("options",     "Options",      "#ff7f0e", "n_options_tokens"),
    ]

    n_comp  = len(comp_info)
    n_phase = len(phases)
    means   = np.zeros((n_comp, n_phase))
    sems_   = np.zeros((n_comp, n_phase))

    for ci, (key, _label, _color, n_col) in enumerate(comp_info):
        cols = component_cols.get(key, [])
        if not _available(cols, df):
            continue
        pt = _per_token_matrix(df, cols, n_col)
        for pi, (_pname, layer_ids) in enumerate(phases):
            valid_ids = [l for l in layer_ids if l < pt.shape[1]]
            phase_data = pt[:, valid_ids]
            row_means = np.nanmean(phase_data, axis=1)
            means[ci, pi]  = np.nanmean(row_means)
            n_ok = np.sum(np.isfinite(row_means))
            sems_[ci, pi]  = np.nanstd(row_means) / np.sqrt(max(n_ok, 1))

    x     = np.arange(n_phase)
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))

    for ci, (_key, label, color, _) in enumerate(comp_info):
        offset = (ci - (n_comp - 1) / 2) * width
        ax.bar(x + offset, means[ci], width=width * 0.92,
               color=color, label=label, alpha=0.85)
        ax.errorbar(x + offset, means[ci], yerr=sems_[ci],
                    fmt="none", color="black", capsize=3, lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in phases], fontsize=11)
    ax.set_ylabel("Mean per-token attention  (×10⁻³)", fontsize=11)
    ax.set_title(
        "Experiment 1 — Per-token attention by processing phase\n"
        "(excl. other_text; Qwen2.5-Omni-7B, HumMusQA)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    # Annotate the options/mid bar — bridge to Exp2
    opt_idx   = [k for k, (key, *_) in enumerate(comp_info) if key == "options"][0]
    mid_idx   = 1  # Mid phase is always index 1
    opt_val   = means[opt_idx, mid_idx]
    opt_x     = x[mid_idx] + (opt_idx - (n_comp - 1) / 2) * width
    ann_y     = opt_val + means.max() * 0.25
    ax.annotate(
        "Options peak in Mid phase\n→ matches Exp2 knockout L12–16",
        xy=(opt_x, opt_val + sems_[opt_idx, mid_idx] * 1.1),
        xytext=(opt_x + 0.35, ann_y),
        fontsize=8.5, color="#ff7f0e",
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff7f0e", alpha=0.85),
    )

    # Annotate audio/early
    aud_idx = [k for k, (key, *_) in enumerate(comp_info) if key == "audio"][0]
    aud_val = means[aud_idx, 0]
    aud_x   = x[0] + (aud_idx - (n_comp - 1) / 2) * width
    ax.annotate(
        "Audio integr. in Early layers\n→ Exp2 all_text→audio effect\n   peaks in L0–16",
        xy=(aud_x, aud_val + sems_[aud_idx, 0] * 1.1),
        xytext=(aud_x - 0.65, ann_y),
        fontsize=8.5, color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.85),
    )

    fig.tight_layout()
    save_fig(fig, output_dir, "M1_thesis_phase_summary")


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    df, summary, metadata, n_layers, component_cols = load_results(args.results_dir, args.n_layers)
    audio_cols = component_cols["audio"]
    print(f"Loaded {len(df)} samples, n_layers={n_layers}")
    print_console_summary(df, summary, component_cols, metadata)

    # ── existing figures ──────────────────────────────────────────────────────
    plot_main_components(df, component_cols, n_layers, output_dir)
    plot_correct_wrong(df, component_cols, n_layers, output_dir)
    plot_category_heatmap(df, audio_cols, n_layers, output_dir, args.min_category_n)
    plot_difficulty(df, component_cols, n_layers, output_dir)
    plot_late_ratio_by_category(summary, output_dir, args.min_category_n)

    # ── thesis-quality figures (F1, F2, F3) ──────────────────────────────────
    print("\nGenerating thesis figures …")
    plot_thesis_main(df, component_cols, n_layers, output_dir)
    plot_thesis_correct_wrong(df, component_cols, n_layers, output_dir)
    plot_thesis_phase_summary(df, component_cols, n_layers, output_dir)

    print(f"\n✅ All plots saved in: {output_dir}")


if __name__ == "__main__":
    main()
