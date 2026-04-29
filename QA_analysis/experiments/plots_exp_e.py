"""
Esperimento E — Plot
====================
Genera i 5 grafici di Exp E dal parquet aggregato.

Grafici:
    G-E1  scatter sufficiency × necessity per k=5 (4 quadranti)
    G-E2  barplot suf/nec per feature_source × k
    G-E3  heatmap sufficiency per categoria × feature_source
    G-E4  curve di sufficiency al variare di k (correct vs wrong)
    G-E5  distribuzione verdetti causali per macrofamiglia

Utilizzo:
    python plots_exp_e.py --batch-dir Results_QA/experiments/exp_E/batch_run_00
"""

import os
import argparse
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PALETTE = {
    "UC_audio":   "#1D9E75",
    "UC_text":    "#378ADD",
    "MI":         "#D85A30",
    "correct":    "#1D9E75",
    "wrong":      "#E24B4A",
    "percettiva": "#1D9E75",
    "analitica":  "#534AB7",
    "knowledge":  "#D85A30",
    "unknown":    "#888780",
    # verdetti
    "causal_strong":           "#1D9E75",
    "redundant":               "#EF9F27",
    "critical_not_sufficient": "#378ADD",
    "decorative":              "#E24B4A",
}

CATEGORY_ORDER = [
    "Instrumentation", "Sound Texture", "Metre and Rhythm", "Musical Texture",
    "Harmony", "Melody", "Structure", "Performance",
    "Genre and Style", "Mood and Expression",
]
MACROFAMILY_ORDER = ["percettiva", "analitica", "knowledge", "unknown"]
VERDICT_ORDER = ["causal_strong", "critical_not_sufficient", "redundant", "decorative"]

# Default soglie (devono matchare quelle usate in batch_exp_e.py)
SUF_THRESHOLD = 0.7
NEC_THRESHOLD = 0.4


# =============================================================================
def load_exp_e_data(batch_dir: str):
    import pandas as pd
    parquet = os.path.join(batch_dir, "aggregated", "exp_e_long.parquet")
    csv = os.path.join(batch_dir, "aggregated", "exp_e_long.csv")
    if os.path.exists(parquet):
        df = pd.read_parquet(parquet)
    elif os.path.exists(csv):
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"Aggregato non trovato in {batch_dir}/aggregated/")
    print(f"Caricato: {len(df)} righe ({df['sample_id'].nunique()} sample)")
    return df


def _plots_dir(batch_dir: str) -> str:
    d = os.path.join(batch_dir, "plots")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, path: str, show: bool):
    import matplotlib.pyplot as plt
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  → {path}")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# G-E1
# =============================================================================
def plot_scatter_suf_nec(df, plots_dir: str, show: bool = False, k_target: int = 5):
    import matplotlib.pyplot as plt
    sub = df[df["k"] == k_target].copy()
    if sub.empty:
        print(f"  G-E1 skip: nessun dato per k={k_target}")
        return

    marker_map = {"percettiva": "o", "analitica": "s", "knowledge": "^", "unknown": "D"}

    fig, ax = plt.subplots(figsize=(9, 7))

    for fs in ["UC_audio", "UC_text", "MI"]:
        for macro in MACROFAMILY_ORDER:
            s = sub[(sub["feature_source"] == fs) & (sub["macro_family"] == macro)]
            if s.empty:
                continue
            ax.scatter(
                s["sufficiency_score"], s["necessity_score"],
                c=PALETTE.get(fs, "#888780"),
                marker=marker_map.get(macro, "o"),
                alpha=0.55, s=50, edgecolors="none",
                label=f"{fs}/{macro}",
            )

    # Soglie e quadranti
    ax.axvline(SUF_THRESHOLD, color="#888780", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(NEC_THRESHOLD, color="#888780", lw=0.8, ls="--", alpha=0.7)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmax * 0.98, ymin + 0.02, "causal_strong\n(suf alta + nec bassa)",
            ha="right", va="bottom", fontsize=8, color="#0F6E56", alpha=0.7)
    ax.text(xmax * 0.98, ymax * 0.98, "redundant\n(suf alta + nec alta)",
            ha="right", va="top", fontsize=8, color="#854F0B", alpha=0.7)
    ax.text(0.02, ymin + 0.02, "critical_not_sufficient\n(suf bassa + nec bassa)",
            ha="left", va="bottom", fontsize=8, color="#185FA5", alpha=0.7)
    ax.text(0.02, ymax * 0.98, "decorative\n(suf bassa + nec alta)",
            ha="left", va="top", fontsize=8, color="#A32D2D", alpha=0.7)

    ax.set_xlabel(f"sufficiency_score (≥{SUF_THRESHOLD} = sufficiente)", fontsize=10)
    ax.set_ylabel(f"necessity_score (≤{NEC_THRESHOLD} = necessaria)", fontsize=10)
    ax.set_title(f"G-E1 — Sufficiency × Necessity (k={k_target})", fontsize=11)
    ax.legend(fontsize=7.5, ncol=2, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(plots_dir, f"G-E1_scatter_suf_nec_k{k_target}.pdf")
    _save(fig, out, show)


# =============================================================================
# G-E2
# =============================================================================
def plot_barplot_source_k(df, plots_dir: str, show: bool = False):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, metric, title in [
        (axes[0], "sufficiency_score", "Sufficiency"),
        (axes[1], "necessity_score", "Necessity"),
    ]:
        agg = df.groupby(["feature_source", "k"])[metric].agg(["mean", "std"]).reset_index()
        sources = ["UC_audio", "UC_text", "MI"]
        ks = sorted(df["k"].unique())
        x = np.arange(len(sources))
        width = 0.8 / len(ks)
        colors = ["#9FE1CB", "#5DCAA5", "#0F6E56"]

        for k_idx, k in enumerate(ks):
            sub = agg[agg["k"] == k].set_index("feature_source").reindex(sources)
            offset = (k_idx - len(ks)/2 + 0.5) * width
            ax.bar(x + offset, sub["mean"], width * 0.9,
                   yerr=sub["std"], color=colors[k_idx % len(colors)],
                   label=f"k={k}", capsize=2, error_kw={"linewidth": 0.6, "ecolor": "#444441"})

        ax.set_xticks(x)
        ax.set_xticklabels(sources, fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"G-E2 — {title} score per feature_source × k", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if metric == "sufficiency_score":
            ax.axhline(SUF_THRESHOLD, color="#888780", lw=0.8, ls="--", alpha=0.7)
        else:
            ax.axhline(NEC_THRESHOLD, color="#888780", lw=0.8, ls="--", alpha=0.7)

    out = os.path.join(plots_dir, "G-E2_barplot_source_k.pdf")
    _save(fig, out, show)


# =============================================================================
# G-E3
# =============================================================================
def plot_heatmap_categories(df, plots_dir: str, show: bool = False, k_target: int = 5):
    import matplotlib.pyplot as plt
    sub = df[df["k"] == k_target]
    if sub.empty:
        print(f"  G-E3 skip: nessun dato per k={k_target}")
        return

    sources = ["UC_audio", "UC_text", "MI"]
    cats = [c for c in CATEGORY_ORDER if c in sub["category"].unique()]
    cats += [c for c in sub["category"].unique() if c not in cats and c]

    n_cats, n_sources = len(cats), len(sources)
    mean_mat = np.full((n_cats, n_sources), np.nan)
    count_mat = np.zeros((n_cats, n_sources), dtype=int)

    for i, cat in enumerate(cats):
        for j, src in enumerate(sources):
            s = sub[(sub["category"] == cat) & (sub["feature_source"] == src)]
            if len(s) > 0:
                mean_mat[i, j] = s["sufficiency_score"].mean()
                count_mat[i, j] = len(s)

    fig, ax = plt.subplots(figsize=(8, max(5, n_cats * 0.55)))
    im = ax.imshow(mean_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    for i in range(n_cats):
        for j in range(n_sources):
            v = mean_mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}\nn={count_mat[i,j]}",
                    ha="center", va="center", fontsize=8,
                    color="white" if (v < 0.35 or v > 0.75) else "black")

    ax.set_xticks(range(n_sources))
    ax.set_xticklabels(sources, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(cats, fontsize=9)
    plt.colorbar(im, ax=ax, label="sufficiency_score (media)", shrink=0.6, pad=0.05)
    ax.set_title(f"G-E3 — Sufficiency per categoria × feature_source (k={k_target})",
                 fontsize=11, pad=10)

    out = os.path.join(plots_dir, f"G-E3_heatmap_categories_k{k_target}.pdf")
    _save(fig, out, show)


# =============================================================================
# G-E4
# =============================================================================
def plot_curves_k(df, plots_dir: str, show: bool = False):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    sources_to_plot = [("UC_audio", axes[0]), ("UC_text", axes[1])]

    for src, ax in sources_to_plot:
        sub = df[df["feature_source"] == src]
        if sub.empty:
            continue
        for correct_val, color, label in [
            (True, PALETTE["correct"], "correct"),
            (False, PALETTE["wrong"], "wrong"),
        ]:
            agg = (sub[sub["correct_baseline"] == correct_val]
                   .groupby("k")["sufficiency_score"]
                   .agg(["mean", "std", "count"])
                   .reset_index())
            if agg.empty:
                continue
            ax.errorbar(agg["k"], agg["mean"],
                        yerr=agg["std"] / np.sqrt(agg["count"].clip(lower=1)),
                        marker="o", markersize=6, lw=1.5,
                        color=color, label=label, capsize=3)

        ax.axhline(SUF_THRESHOLD, color="#888780", lw=0.8, ls="--", alpha=0.7,
                   label=f"soglia ({SUF_THRESHOLD})")
        ax.set_xlabel("k (numero top-feature)", fontsize=10)
        ax.set_ylabel("sufficiency_score (media ± SE)", fontsize=10)
        ax.set_title(f"G-E4 — {src}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Curva di causalità al variare di k", fontsize=12, y=1.02)
    out = os.path.join(plots_dir, "G-E4_curves_k.pdf")
    _save(fig, out, show)


# =============================================================================
# G-E5
# =============================================================================
def plot_verdicts_distribution(df, plots_dir: str, show: bool = False, k_target: int = 5):
    import matplotlib.pyplot as plt
    sub = df[df["k"] == k_target]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    # Per macrofamiglia × feature_source: percentuali di verdetto
    families = [f for f in MACROFAMILY_ORDER if f in sub["macro_family"].unique()]
    sources = ["UC_audio", "UC_text", "MI"]

    bar_x_labels = []
    counts_per_bar = []
    for fam in families:
        for src in sources:
            s = sub[(sub["macro_family"] == fam) & (sub["feature_source"] == src)]
            if len(s) == 0:
                counts_per_bar.append({v: 0 for v in VERDICT_ORDER})
            else:
                counts_per_bar.append(
                    {v: int((s["causal_verdict"] == v).sum()) for v in VERDICT_ORDER}
                )
            bar_x_labels.append(f"{fam}\n{src}")

    n_bars = len(bar_x_labels)
    x = np.arange(n_bars)
    bottom = np.zeros(n_bars)
    for v in VERDICT_ORDER:
        heights = np.array([d[v] for d in counts_per_bar], dtype=float)
        totals = np.array([sum(d.values()) for d in counts_per_bar], dtype=float)
        pct = np.divide(heights, totals,
                        out=np.zeros_like(heights), where=totals > 0) * 100
        ax.bar(x, pct, bottom=bottom, color=PALETTE.get(v, "#888780"),
               label=v, edgecolor="white", linewidth=0.5)
        bottom += pct

    ax.set_xticks(x)
    ax.set_xticklabels(bar_x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% sample", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(f"G-E5 — Distribuzione verdetti causali (k={k_target})", fontsize=11)
    ax.legend(fontsize=8.5, ncol=2, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(plots_dir, f"G-E5_verdicts_k{k_target}.pdf")
    _save(fig, out, show)


# =============================================================================
def print_summary(df) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY — EXPERIMENT E")
    print("=" * 60)
    print(f"  Sample totali           : {df['sample_id'].nunique()}")
    print(f"  Righe (sample × fs × k) : {len(df)}")
    print()
    print(f"  % sufficient (suf>={SUF_THRESHOLD}):")
    for fs in ["UC_audio", "UC_text", "MI"]:
        s = df[df["feature_source"] == fs]
        if len(s) == 0:
            continue
        pct = (s["is_sufficient"]).mean() * 100
        print(f"    {fs:10s}: {pct:.1f}%")
    print(f"  % necessary (nec<={NEC_THRESHOLD}):")
    for fs in ["UC_audio", "UC_text", "MI"]:
        s = df[df["feature_source"] == fs]
        if len(s) == 0:
            continue
        pct = (s["is_necessary"]).mean() * 100
        print(f"    {fs:10s}: {pct:.1f}%")
    print()
    print("  Distribuzione verdetti:")
    for v in VERDICT_ORDER:
        pct = (df["causal_verdict"] == v).mean() * 100
        print(f"    {v:25s}: {pct:.1f}%")
    print("=" * 60)


def run_all_plots(batch_dir: str, show: bool = False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.dpi": 150, "savefig.dpi": 150})
    warnings.filterwarnings("ignore", category=UserWarning)

    df = load_exp_e_data(batch_dir)
    print_summary(df)
    pdir = _plots_dir(batch_dir)
    print(f"\nSalvataggio plot in: {pdir}\n")

    print("G-E1 — Scatter suf × nec...")
    plot_scatter_suf_nec(df, pdir, show, k_target=5)

    print("G-E2 — Barplot source × k...")
    plot_barplot_source_k(df, pdir, show)

    print("G-E3 — Heatmap categorie...")
    plot_heatmap_categories(df, pdir, show, k_target=5)

    print("G-E4 — Curve al variare di k...")
    plot_curves_k(df, pdir, show)

    print("G-E5 — Verdetti...")
    plot_verdicts_distribution(df, pdir, show, k_target=5)

    print(f"\nFatto. PDF in: {pdir}")


def main():
    parser = argparse.ArgumentParser(description="Plot Esperimento E")
    parser.add_argument("--batch-dir", required=True, type=str)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run_all_plots(args.batch_dir, show=args.show)


if __name__ == "__main__":
    main()