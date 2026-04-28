"""
Experiment A — Analisi e grafici
=================================

Grafici prodotti:
    G-A1  heatmap categorica categoria × metriche XAI
    G-A2  profilo XAI per difficoltà
    G-A3  scatter A-SHAP × MI-ratio
          colore = correct/wrong
          marker = difficoltà
    G-A4  profilo XAI correct vs wrong

Utilizzo:
    python -m QA_analysis.experiments.plots_exp_a --batch-dir /path/to/batch_run_XX
"""

import os
import argparse
import warnings
from typing import Dict, List

import numpy as np


# =============================================================================
# Palette e stile
# =============================================================================

PALETTE = {
    "audio":   "#1D9E75",
    "text":    "#378ADD",
    "uc":      "#534AB7",
    "mi":      "#D85A30",
    "correct": "#1D9E75",
    "wrong":   "#E24B4A",
    "neutral": "#888780",
}

CATEGORY_ORDER = [
    "Instrumentation",
    "Sound Texture",
    "Metre and Rhythm",
    "Musical Texture",
    "Harmony",
    "Melody",
    "Structure",
    "Performance",
    "Genre and Style",
    "Mood and Expression",
    "Historical and Cultural Context",
    "Lyrics",
    "Functional Context",
]

DIFFICULTY_ORDER = ["low", "medium", "high", "unknown"]

DIFFICULTY_MARKERS = {
    "low": "s",       # quadrato
    "medium": "o",    # cerchio
    "mid": "o",       # alias
    "high": "D",      # rombo/rettangolo orientato
    "unknown": "X",
}


# =============================================================================
# Helpers
# =============================================================================

def _ensure_derived_columns(df):
    """
    Aggiunge colonne derivate utili ai plot, senza modificare i risultati originali.
    """
    df = df.copy()

    for c in [
        "a_shap", "t_shap",
        "uc_audio_l1", "uc_text_l1",
        "mi_audio_l1", "mi_text_l1",
    ]:
        if c not in df.columns:
            df[c] = 0.0

    df["uc_total_l1"] = df["uc_audio_l1"] + df["uc_text_l1"]
    df["mi_total_l1"] = df["mi_audio_l1"] + df["mi_text_l1"]
    denom = df["uc_total_l1"] + df["mi_total_l1"]
    df["mi_ratio"] = np.where(denom > 0, df["mi_total_l1"] / denom, 0.0)

    df["audio_text_balance"] = df["a_shap"] - df["t_shap"]

    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(bool)
    else:
        df["correct"] = False

    if "difficulty" not in df.columns:
        df["difficulty"] = "unknown"

    df["difficulty_norm"] = (
        df["difficulty"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"med": "medium", "mid": "medium", "easy": "low", "hard": "high"})
    )

    if "category" not in df.columns:
        df["category"] = "unknown"

    return df


def load_exp_a_data(batch_dir: str):
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas richiesto. pip install pandas")

    parquet_path = os.path.join(batch_dir, "aggregated", "exp_a_results.parquet")
    csv_path = os.path.join(batch_dir, "aggregated", "exp_a_results.csv")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"Caricato parquet: {len(df)} sample da {parquet_path}")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Caricato CSV: {len(df)} sample da {csv_path}")
    else:
        raise FileNotFoundError(
            f"Nessun file aggregato trovato in {batch_dir}/aggregated/"
        )

    return _ensure_derived_columns(df)


def _plots_dir(batch_dir: str) -> str:
    d = os.path.join(batch_dir, "plots")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, path: str, show: bool) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(path, bbox_inches="tight", dpi=170)
    print(f"  → salvato: {path}")
    if show:
        plt.show()
    plt.close(fig)


def _ordered_present_categories(df) -> List[str]:
    present = df["category"].dropna().unique().tolist()
    ordered = [c for c in CATEGORY_ORDER if c in present]
    ordered += [c for c in present if c not in ordered]
    return ordered


def _ordered_present_difficulties(df) -> List[str]:
    present = df["difficulty_norm"].dropna().unique().tolist()
    ordered = [d for d in DIFFICULTY_ORDER if d in present]
    ordered += [d for d in present if d not in ordered]
    return ordered


# =============================================================================
# G-A1 — Heatmap categorica
# NON TOCCATA NELLA LOGICA
# =============================================================================

def plot_heatmap_categories(df, plots_dir: str, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("a_shap",       "A-SHAP"),
        ("t_shap",       "T-SHAP"),
        ("uc_audio_l1",  "UC audio"),
        ("uc_text_l1",   "UC text"),
        ("mi_audio_l1",  "MI audio"),
        ("mi_text_l1",   "MI text"),
    ]

    ordered_cats = _ordered_present_categories(df)

    n_cats = len(ordered_cats)
    n_metrics = len(metrics)

    mean_mat = np.full((n_cats, n_metrics), np.nan)
    std_mat = np.full((n_cats, n_metrics), np.nan)
    count_vec = np.full(n_cats, 0, dtype=int)

    grp = df.groupby("category")
    for i, cat in enumerate(ordered_cats):
        if cat not in grp.groups:
            continue
        sub = grp.get_group(cat)
        count_vec[i] = len(sub)
        for j, (col, _) in enumerate(metrics):
            if col in sub.columns:
                mean_mat[i, j] = sub[col].mean()
                std_mat[i, j] = sub[col].std()

    col_labels = [lbl for _, lbl in metrics]

    fig, ax = plt.subplots(
        figsize=(max(10, n_metrics * 1.6), max(6, n_cats * 0.55))
    )

    norm_mat = np.zeros_like(mean_mat)
    for j in range(n_metrics):
        col = mean_mat[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        cmin, cmax = valid.min(), valid.max()
        if cmax > cmin:
            norm_mat[:, j] = (col - cmin) / (cmax - cmin)
        else:
            norm_mat[:, j] = 0.5

    im = ax.imshow(norm_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    for i in range(n_cats):
        for j in range(n_metrics):
            if np.isnan(mean_mat[i, j]):
                continue
            val_str = f"{mean_mat[i, j]:.2f}"
            std_str = f"±{std_mat[i, j]:.2f}" if not np.isnan(std_mat[i, j]) else ""
            bg = norm_mat[i, j]
            txt_color = "white" if bg > 0.6 else "black"
            ax.text(
                j, i, f"{val_str}\n{std_str}",
                ha="center", va="center",
                fontsize=7.5, color=txt_color,
            )

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(ordered_cats, fontsize=9)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n_cats))
    ax2.set_yticklabels(
        [f"n={count_vec[i]}" for i in range(n_cats)],
        fontsize=8,
        color=PALETTE["neutral"],
    )
    ax2.tick_params(right=False)

    plt.colorbar(
        im,
        ax=ax,
        label="Valore normalizzato per colonna",
        shrink=0.6,
        pad=0.12,
    )
    ax.set_title(
        "G-A1 — Mappa categorica: metriche XAI per categoria musicale",
        fontsize=11,
        pad=12,
    )

    out = os.path.join(plots_dir, "G-A1_heatmap_categories.png")
    _save(fig, out, show)


# =============================================================================
# G-A2 — Profilo XAI per difficoltà
# =============================================================================

def plot_difficulty_profile(df, plots_dir: str, show: bool = False) -> None:
    """
    Mostra come cambia il profilo modale al variare della difficoltà.

    Pannello 1:
        A-SHAP vs T-SHAP

    Pannello 2:
        UC audio / UC text / MI audio / MI text / MI ratio
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    difficulties = _ordered_present_difficulties(df)

    if not difficulties:
        print("  ! G-A2 saltato: nessuna difficoltà disponibile.")
        return

    metrics_left = [
        ("a_shap", "A-SHAP", PALETTE["audio"]),
        ("t_shap", "T-SHAP", PALETTE["text"]),
    ]

    metrics_right = [
        ("uc_audio_l1", "UC audio", PALETTE["audio"]),
        ("uc_text_l1", "UC text", PALETTE["text"]),
        ("mi_audio_l1", "MI audio", PALETTE["mi"]),
        ("mi_text_l1", "MI text", "#F09A73"),
        ("mi_ratio", "MI ratio", PALETTE["uc"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    ax1, ax2 = axes

    x = np.arange(len(difficulties))

    # ---- pannello 1: A-SHAP / T-SHAP
    width1 = 0.34
    for k, (col, label, color) in enumerate(metrics_left):
        vals = [
            df[df["difficulty_norm"] == d][col].mean()
            if len(df[df["difficulty_norm"] == d]) > 0 else np.nan
            for d in difficulties
        ]
        offset = (k - 0.5) * width1
        ax1.bar(
            x + offset,
            vals,
            width1 * 0.9,
            color=color,
            alpha=0.82,
            label=label,
        )

    ax1.axhline(0.5, color=PALETTE["neutral"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in difficulties])
    ax1.set_ylabel("Contributo normalizzato")
    ax1.set_title("Dipendenza modale MM-SHAP")
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    # ---- pannello 2: DIME
    width2 = 0.15
    for k, (col, label, color) in enumerate(metrics_right):
        vals = [
            df[df["difficulty_norm"] == d][col].mean()
            if len(df[df["difficulty_norm"] == d]) > 0 else np.nan
            for d in difficulties
        ]
        offset = (k - (len(metrics_right) - 1) / 2) * width2
        ax2.bar(
            x + offset,
            vals,
            width2 * 0.9,
            color=color,
            alpha=0.82,
            label=label,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.capitalize() for d in difficulties])
    ax2.set_ylabel("Valore medio")
    ax2.set_title("Scomposizione DIME per difficoltà")
    ax2.legend(frameon=True, fontsize=8, ncol=2)
    ax2.grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for i, d in enumerate(difficulties):
            n = len(df[df["difficulty_norm"] == d])
            ax.text(
                i,
                ax.get_ylim()[1] * 0.96,
                f"n={n}",
                ha="center",
                va="top",
                fontsize=8,
                color=PALETTE["neutral"],
            )

    fig.suptitle(
        "G-A2 — Profilo XAI per difficoltà della domanda",
        fontsize=12,
        y=1.03,
    )

    out = os.path.join(plots_dir, "G-A2_difficulty_profile.png")
    _save(fig, out, show)


# =============================================================================
# G-A3 — Scatter A-SHAP × MI-ratio
# =============================================================================

def plot_scatter_ashap_miratio(df, plots_dir: str, show: bool = False) -> None:
    """
    Scatter interpretativo.

    X = A-SHAP
    Y = MI-ratio = MI_total / (UC_total + MI_total)

    Colore:
        verde = correct
        rosso = wrong

    Marker:
        low    = quadrato
        medium = cerchio
        high   = rombo/rettangolo
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(9.2, 7.2))

    difficulties = _ordered_present_difficulties(df)

    for diff in difficulties:
        sub_d = df[df["difficulty_norm"] == diff]
        if len(sub_d) == 0:
            continue

        marker = DIFFICULTY_MARKERS.get(diff, "X")

        for corr_value, color, label_corr in [
            (True, PALETTE["correct"], "correct"),
            (False, PALETTE["wrong"], "wrong"),
        ]:
            sub = sub_d[sub_d["correct"] == corr_value]
            if len(sub) == 0:
                continue

            ax.scatter(
                sub["a_shap"],
                sub["mi_ratio"],
                marker=marker,
                s=85,
                c=color,
                alpha=0.78,
                edgecolors="black",
                linewidths=0.45,
                label=f"{diff} / {label_corr}",
                zorder=3,
            )

    # soglie interpretative
    x_thr = 0.5
    y_thr = float(df["mi_ratio"].median()) if len(df) else 0.5

    ax.axvline(x_thr, color=PALETTE["neutral"], linestyle="--", linewidth=1.0, alpha=0.65)
    ax.axhline(y_thr, color=PALETTE["neutral"], linestyle="--", linewidth=1.0, alpha=0.65)

    ax.set_xlim(-0.02, 1.02)
    y_max = max(1.0, float(df["mi_ratio"].max()) * 1.08 if len(df) else 1.0)
    ax.set_ylim(-0.02, y_max)

    # etichette quadranti
    ax.text(
        0.98,
        y_max * 0.96,
        "vero ascolto\n(audio + interazione alti)",
        ha="right",
        va="top",
        fontsize=8.5,
        color=PALETTE["audio"],
    )
    ax.text(
        0.98,
        0.03,
        "audio non interattivo\n(audio alto, MI basso)",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=PALETTE["mi"],
    )
    ax.text(
        0.02,
        y_max * 0.96,
        "cross-modal anomalo\n(audio basso, MI alto)",
        ha="left",
        va="top",
        fontsize=8.5,
        color=PALETTE["uc"],
    )
    ax.text(
        0.02,
        0.03,
        "possibile bias testuale\n(audio basso, MI basso)",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=PALETTE["neutral"],
    )

    # annota outlier utili
    annotate_cols = ["sample_id", "idx"]
    id_col = next((c for c in annotate_cols if c in df.columns), None)

    if id_col is not None and len(df) <= 80:
        for _, r in df.iterrows():
            if (
                r["mi_ratio"] >= df["mi_ratio"].quantile(0.90)
                or r["a_shap"] >= df["a_shap"].quantile(0.90)
                or r["a_shap"] <= df["a_shap"].quantile(0.10)
                or not bool(r["correct"])
            ):
                ax.annotate(
                    str(r[id_col]),
                    (r["a_shap"], r["mi_ratio"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=6.5,
                    alpha=0.7,
                )

    ax.set_xlabel("A-SHAP — contributo audio normalizzato")
    ax.set_ylabel("MI-ratio — quota di interazione multimodale")
    ax.set_title("G-A3 — A-SHAP × MI-ratio: correttezza e difficoltà")
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legenda colore
    color_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["correct"],
            markeredgecolor="black",
            markersize=9,
            label="correct",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["wrong"],
            markeredgecolor="black",
            markersize=9,
            label="wrong",
        ),
    ]

    # legenda marker difficoltà
    marker_handles = []
    for d in difficulties:
        marker_handles.append(
            Line2D(
                [0], [0],
                marker=DIFFICULTY_MARKERS.get(d, "X"),
                color="w",
                markerfacecolor="#BBBBBB",
                markeredgecolor="black",
                markersize=9,
                label=d,
            )
        )

    leg1 = ax.legend(
        handles=color_handles,
        title="Outcome",
        loc="upper left",
        frameon=True,
        fontsize=8,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=marker_handles,
        title="Difficulty",
        loc="lower right",
        frameon=True,
        fontsize=8,
    )

    out = os.path.join(plots_dir, "G-A3_scatter_ashap_miratio.png")
    _save(fig, out, show)


# =============================================================================
# G-A4 — Profilo XAI correct vs wrong
# =============================================================================

def plot_correctness_profile(df, plots_dir: str, show: bool = False) -> None:
    """
    Confronta il profilo medio delle metriche XAI tra risposte corrette e sbagliate.
    Funziona anche se sono tutte corrette o tutte sbagliate.
    """
    import matplotlib.pyplot as plt

    metrics = [
        ("a_shap", "A-SHAP"),
        ("t_shap", "T-SHAP"),
        ("uc_audio_l1", "UC audio"),
        ("uc_text_l1", "UC text"),
        ("mi_audio_l1", "MI audio"),
        ("mi_text_l1", "MI text"),
        ("mi_ratio", "MI ratio"),
    ]

    x = np.arange(len(metrics))
    width = 0.36

    correct_df = df[df["correct"] == True]
    wrong_df = df[df["correct"] == False]

    correct_vals = [
        correct_df[col].mean() if len(correct_df) > 0 else np.nan
        for col, _ in metrics
    ]
    wrong_vals = [
        wrong_df[col].mean() if len(wrong_df) > 0 else np.nan
        for col, _ in metrics
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))

    if len(correct_df) > 0:
        ax.bar(
            x - width / 2,
            correct_vals,
            width,
            color=PALETTE["correct"],
            alpha=0.78,
            label=f"correct (n={len(correct_df)})",
        )

    if len(wrong_df) > 0:
        ax.bar(
            x + width / 2,
            wrong_vals,
            width,
            color=PALETTE["wrong"],
            alpha=0.78,
            label=f"wrong (n={len(wrong_df)})",
        )

    # se manca una classe, avvisa nel plot
    if len(correct_df) == 0 or len(wrong_df) == 0:
        missing = "wrong" if len(wrong_df) == 0 else "correct"
        ax.text(
            0.5,
            0.94,
            f"Nota: nel batch corrente non ci sono risposte {missing}.",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color=PALETTE["neutral"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], rotation=25, ha="right")
    ax.set_ylabel("Valore medio")
    ax.set_title("G-A4 — Profilo XAI per risposte corrette vs sbagliate")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(plots_dir, "G-A4_correctness_profile.png")
    _save(fig, out, show)


# =============================================================================
# Summary
# =============================================================================

def print_summary(df) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICO — EXPERIMENT A")
    print("=" * 60)
    print(f"  Totale sample   : {len(df)}")
    print(f"  Corretti        : {df['correct'].sum()} ({df['correct'].mean()*100:.1f}%)")
    print(f"  A-SHAP medio    : {df['a_shap'].mean():.3f} ± {df['a_shap'].std():.3f}")
    print(f"  T-SHAP medio    : {df['t_shap'].mean():.3f} ± {df['t_shap'].std():.3f}")
    print(f"  UC audio L1 med : {df['uc_audio_l1'].mean():.3f}")
    print(f"  MI audio L1 med : {df['mi_audio_l1'].mean():.3f}")
    print(f"  UC text L1 med  : {df['uc_text_l1'].mean():.3f}")
    print(f"  MI text L1 med  : {df['mi_text_l1'].mean():.3f}")
    print(f"  MI-ratio med    : {df['mi_ratio'].mean():.3f}")
    print()

    print("  Per difficoltà:")
    for diff in _ordered_present_difficulties(df):
        sub = df[df["difficulty_norm"] == diff]
        print(
            f"    {diff:10s}: "
            f"A-SHAP={sub['a_shap'].mean():.3f} | "
            f"MI-ratio={sub['mi_ratio'].mean():.3f} | "
            f"acc={sub['correct'].mean()*100:.1f}% "
            f"(n={len(sub)})"
        )

    print()
    print("  Per categoria:")
    for cat in _ordered_present_categories(df):
        sub = df[df["category"] == cat]
        print(
            f"    {cat:32s}: "
            f"A-SHAP={sub['a_shap'].mean():.3f} | "
            f"MI-ratio={sub['mi_ratio'].mean():.3f} | "
            f"acc={sub['correct'].mean()*100:.1f}% "
            f"(n={len(sub)})"
        )

    print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================

def run_all_plots(batch_dir: str, show: bool = False) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 170,
    })

    warnings.filterwarnings("ignore", category=UserWarning)

    df = load_exp_a_data(batch_dir)
    print_summary(df)

    pdir = _plots_dir(batch_dir)
    print(f"\nSalvataggio plot in: {pdir}\n")

    print("G-A1 — Heatmap categorica...")
    plot_heatmap_categories(df, pdir, show)

    print("G-A2 — Profilo per difficoltà...")
    plot_difficulty_profile(df, pdir, show)

    print("G-A3 — Scatter A-SHAP × MI-ratio...")
    plot_scatter_ashap_miratio(df, pdir, show)

    print("G-A4 — Profilo correct vs wrong...")
    plot_correctness_profile(df, pdir, show)

    print(f"\nTutti i plot salvati in: {pdir}")


def main():
    parser = argparse.ArgumentParser(description="Plot Esperimento A")
    parser.add_argument(
        "--batch-dir",
        required=True,
        metavar="PATH",
        help="Path della batch dir, es. Results_QA/experiments/exp_A/batch_run_00",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra finestre interattive oltre al salvataggio PNG",
    )
    args = parser.parse_args()

    run_all_plots(args.batch_dir, show=args.show)


if __name__ == "__main__":
    main()