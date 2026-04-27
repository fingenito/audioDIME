"""
Experiment A — Analisi e grafici
=================================
Legge i risultati aggregati di exp_A (parquet/CSV) e produce
tutti i grafici dell'esperimento A.

Grafici prodotti:
    G-A1  heatmap categorica (categoria × metriche XAI)
    G-A2  radar chart per macrofamiglia
    G-A3  stacked bar UC vs MI per difficoltà × categoria
    G-A4  scatter A-SHAP vs MI_audio (colore = correct/wrong)
    G-A5  violin plot distribuzione a_shap per categoria

Output:
    batch_run_XX/plots/
        G-A1_heatmap_categories.pdf
        G-A2_radar_macrofamily.pdf
        G-A3_stacked_bar_uc_mi.pdf
        G-A4_scatter_ashap_mi.pdf
        G-A5_violin_ashap_category.pdf

Utilizzo:
    python plots_exp_a.py --batch-dir Results_QA/exp_A/batch_run_00
    python plots_exp_a.py --batch-dir ...  --show    # apri finestre interattive
"""

import os
import sys
import json
import argparse
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Palette e stile coerenti
# =============================================================================

PALETTE = {
    "audio":      "#1D9E75",
    "text":       "#378ADD",
    "uc":         "#534AB7",
    "mi":         "#D85A30",
    "correct":    "#1D9E75",
    "wrong":      "#E24B4A",
    "percettiva": "#1D9E75",
    "analitica":  "#534AB7",
    "knowledge":  "#D85A30",
    "unknown":    "#888780",
}

MACROFAMILY_ORDER = ["percettiva", "analitica", "knowledge", "unknown"]

# Ordine categorie per i plot (raggruppa per macrofamiglia)
CATEGORY_ORDER = [
    # percettive
    "Instrumentation", "Sound Texture", "Metre", "Rhythm",
    "Tempo", "Dynamics",
    # analitiche
    "Harmony", "Melody", "Form", "Performance",
    # knowledge
    "Genre", "Style", "Historical", "Cultural", "Lyrics",
    "Emotion", "Mood",
]

MACRO_FAMILY_MAP: Dict[str, str] = {
    "Instrumentation": "percettiva",
    "Sound Texture": "percettiva",
    "Metre": "percettiva",
    "Rhythm": "percettiva",
    "Tempo": "percettiva",
    "Dynamics": "percettiva",
    "Harmony": "analitica",
    "Melody": "analitica",
    "Form": "analitica",
    "Performance": "analitica",
    "Genre": "knowledge",
    "Style": "knowledge",
    "Historical": "knowledge",
    "Cultural": "knowledge",
    "Lyrics": "knowledge",
    "Emotion": "knowledge",
    "Mood": "knowledge",
}

# =============================================================================
# 1) Caricamento dati
# =============================================================================

def load_exp_a_data(batch_dir: str):
    """
    Carica il parquet aggregato di Exp A.
    Fallback su CSV se pandas/pyarrow non supportano il parquet.
    """
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
            f"Nessun file aggregato trovato in {batch_dir}/aggregated/\n"
            f"Esegui prima la fase di aggregazione."
        )

    # Aggiunge macro_family se mancante
    if "macro_family" not in df.columns:
        def _infer_macro(cat):
            if not isinstance(cat, str):
                return "unknown"
            for keyword, fam in MACRO_FAMILY_MAP.items():
                if keyword.lower() in cat.lower():
                    return fam
            return "unknown"
        df["macro_family"] = df["category"].apply(_infer_macro)

    return df


def _plots_dir(batch_dir: str) -> str:
    d = os.path.join(batch_dir, "plots")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, path: str, show: bool) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  → salvato: {path}")
    if show:
        plt.show()
    plt.close(fig)

# =============================================================================
# G-A1 — Heatmap categorica
# =============================================================================

def plot_heatmap_categories(df, plots_dir: str, show: bool = False) -> None:
    """
    Matrice categorie × {a_shap, t_shap, uc_audio_l1, uc_text_l1,
                          mi_audio_l1, mi_text_l1}
    Valore = media, annotata con ±std.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import pandas as pd

    metrics = [
        ("a_shap",       "A-SHAP"),
        ("t_shap",       "T-SHAP"),
        ("uc_audio_l1",  "UC audio"),
        ("uc_text_l1",   "UC text"),
        ("mi_audio_l1",  "MI audio"),
        ("mi_text_l1",   "MI text"),
    ]

    # Ordina le categorie presenti nel dataset
    present_cats = df["category"].dropna().unique().tolist()
    ordered_cats = [c for c in CATEGORY_ORDER if c in present_cats]
    ordered_cats += [c for c in present_cats if c not in ordered_cats]

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

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.6), max(6, n_cats * 0.55)))

    # Normalizzazione per colonna (0-1) per la colorazione
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
            ax.text(j, i, f"{val_str}\n{std_str}", ha="center", va="center",
                    fontsize=7.5, color=txt_color)

    # Y labels con macrofamiglia
    y_labels = []
    for cat in ordered_cats:
        macro = MACRO_FAMILY_MAP.get(cat, "unknown")
        y_labels.append(f"{cat}  [{macro}]")

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(y_labels, fontsize=9)

    # Colonna n_samples a destra
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n_cats))
    ax2.set_yticklabels([f"n={count_vec[i]}" for i in range(n_cats)],
                        fontsize=8, color="#888780")
    ax2.tick_params(right=False)

    # Separatori macrofamiglie
    families = [MACRO_FAMILY_MAP.get(c, "unknown") for c in ordered_cats]
    for i in range(1, n_cats):
        if families[i] != families[i - 1]:
            ax.axhline(i - 0.5, color="#444441", linewidth=1.0, linestyle="--")

    plt.colorbar(im, ax=ax, label="Valore normalizzato per colonna", shrink=0.6, pad=0.12)
    ax.set_title("G-A1 — Mappa categorica: metriche XAI per categoria musicale",
                 fontsize=11, pad=12)

    out = os.path.join(plots_dir, "G-A1_heatmap_categories.pdf")
    _save(fig, out, show)

# =============================================================================
# G-A2 — Radar chart per macrofamiglia
# =============================================================================

def plot_radar_macrofamily(df, plots_dir: str, show: bool = False) -> None:
    """
    Un radar per macrofamiglia con 6 assi:
    A-SHAP, T-SHAP, UC_audio, UC_text, MI_audio, MI_text.
    Tutte e tre le macrofamiglie sovrapposte su un singolo plot.
    """
    import matplotlib.pyplot as plt

    axes = ["A-SHAP", "T-SHAP", "UC audio", "UC text", "MI audio", "MI text"]
    cols = ["a_shap", "t_shap", "uc_audio_l1", "uc_text_l1", "mi_audio_l1", "mi_text_l1"]
    n_axes = len(axes)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # chiudi il poligono

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_facecolor("#f8f8f5")

    families_present = [f for f in MACROFAMILY_ORDER
                        if f in df["macro_family"].unique()]

    for fam in families_present:
        sub = df[df["macro_family"] == fam]
        vals = [sub[c].mean() if c in sub.columns else 0.0 for c in cols]

        # Normalizza 0-1 rispetto al massimo globale per colonna
        max_vals = [df[c].max() if c in df.columns else 1.0 for c in cols]
        vals_norm = [v / m if m > 0 else 0.0 for v, m in zip(vals, max_vals)]
        vals_norm += vals_norm[:1]

        color = PALETTE.get(fam, "#888780")
        ax.plot(angles, vals_norm, "-o", color=color, linewidth=2, markersize=5,
                label=fam, zorder=3)
        ax.fill(angles, vals_norm, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7, color="#888780")
    ax.grid(color="#cccccc", linewidth=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title("G-A2 — Radar per macrofamiglia (valori normalizzati)",
                 fontsize=11, pad=20)

    out = os.path.join(plots_dir, "G-A2_radar_macrofamily.pdf")
    _save(fig, out, show)

# =============================================================================
# G-A3 — Stacked bar UC vs MI per difficoltà × categoria
# =============================================================================

def plot_stacked_bar_uc_mi(df, plots_dir: str, show: bool = False) -> None:
    """
    Barre impilate UC_total vs MI_total.
    X = categoria, tre gruppi affiancati per difficoltà.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    diff_levels = ["low", "easy", "medium", "high", "hard", "unknown"]
    present_diffs = [d for d in diff_levels if d in df["difficulty"].unique()]
    if not present_diffs:
        present_diffs = df["difficulty"].dropna().unique().tolist()

    present_cats = [c for c in CATEGORY_ORDER if c in df["category"].unique()]
    if not present_cats:
        present_cats = df["category"].dropna().unique().tolist()

    n_cats = len(present_cats)
    n_diffs = len(present_diffs)
    x = np.arange(n_cats)
    width = 0.8 / max(n_diffs, 1)

    diff_colors_uc = ["#9FE1CB", "#5DCAA5", "#0F6E56"]
    diff_colors_mi = ["#F5C4B3", "#F0997B", "#993C1D"]
    diff_colors_uc = (diff_colors_uc * 3)[:n_diffs]
    diff_colors_mi = (diff_colors_mi * 3)[:n_diffs]

    fig, ax = plt.subplots(figsize=(max(12, n_cats * 1.2), 6))

    for d_idx, diff in enumerate(present_diffs):
        sub_diff = df[df["difficulty"] == diff]
        offset = (d_idx - n_diffs / 2 + 0.5) * width

        uc_vals, mi_vals = [], []
        for cat in present_cats:
            sub = sub_diff[sub_diff["category"] == cat]
            uc_vals.append(sub["uc_audio_l1"].mean() + sub["uc_text_l1"].mean()
                           if len(sub) > 0 else 0.0)
            mi_vals.append(sub["mi_audio_l1"].mean() + sub["mi_text_l1"].mean()
                           if len(sub) > 0 else 0.0)

        uc_vals = np.array(uc_vals, dtype=float)
        mi_vals = np.array(mi_vals, dtype=float)

        ax.bar(x + offset, uc_vals, width * 0.9,
               color=diff_colors_uc[d_idx], label=f"UC [{diff}]")
        ax.bar(x + offset, mi_vals, width * 0.9,
               bottom=uc_vals, color=diff_colors_mi[d_idx],
               label=f"MI [{diff}]", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(present_cats, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Media contributo (L1)", fontsize=10)
    ax.set_title("G-A3 — UC vs MI per categoria e difficoltà", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(plots_dir, "G-A3_stacked_bar_uc_mi.pdf")
    _save(fig, out, show)

# =============================================================================
# G-A4 — Scatter A-SHAP vs MI_audio (colore = correct/wrong)
# =============================================================================

def plot_scatter_ashap_mi(df, plots_dir: str, show: bool = False) -> None:
    """
    Scatter A-SHAP × MI_audio_l1.
    Colore = correct/wrong. Marker = macrofamiglia.
    Quadranti con etichette interpretative.
    """
    import matplotlib.pyplot as plt

    marker_map = {
        "percettiva": "o",
        "analitica": "s",
        "knowledge": "^",
        "unknown": "D",
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    for macro in MACROFAMILY_ORDER:
        sub = df[df["macro_family"] == macro]
        if len(sub) == 0:
            continue
        correct_sub = sub[sub["correct"] == True]
        wrong_sub = sub[sub["correct"] == False]
        mkr = marker_map.get(macro, "o")

        ax.scatter(correct_sub["a_shap"], correct_sub["mi_audio_l1"],
                   c=PALETTE["correct"], marker=mkr, alpha=0.65, s=55,
                   edgecolors="none",
                   label=f"{macro} / correct" if len(correct_sub) > 0 else None)
        ax.scatter(wrong_sub["a_shap"], wrong_sub["mi_audio_l1"],
                   c=PALETTE["wrong"], marker=mkr, alpha=0.65, s=55,
                   edgecolors="none",
                   label=f"{macro} / wrong" if len(wrong_sub) > 0 else None)

    # Linee di soglia (mediane del dataset)
    ax_med = float(df["a_shap"].median())
    mi_med = float(df["mi_audio_l1"].median())
    ax.axvline(ax_med, color="#888780", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(mi_med, color="#888780", linewidth=0.8, linestyle="--", alpha=0.7)

    # Etichette quadranti
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmax * 0.98, ymax * 0.98, "vero ascolto\n(audio + MI alti)",
            ha="right", va="top", fontsize=7.5, color="#0F6E56", alpha=0.7)
    ax.text(xmax * 0.98, ymin + 0.01, "ascolto non interattivo\n(audio alto, MI basso)",
            ha="right", va="bottom", fontsize=7.5, color="#D85A30", alpha=0.7)
    ax.text(0.01, ymax * 0.98, "cross-modal anomalo\n(audio basso, MI alto)",
            ha="left", va="top", fontsize=7.5, color="#534AB7", alpha=0.7)
    ax.text(0.01, ymin + 0.01, "UC testo puro\n(audio basso, MI basso)",
            ha="left", va="bottom", fontsize=7.5, color="#888780", alpha=0.7)

    ax.set_xlabel("A-SHAP (contributo audio)", fontsize=10)
    ax.set_ylabel("MI audio (L1)", fontsize=10)
    ax.set_title("G-A4 — A-SHAP vs MI audio: correct (verde) / wrong (rosso)",
                 fontsize=11)
    ax.legend(fontsize=7.5, ncol=2, loc="center right",
              bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(plots_dir, "G-A4_scatter_ashap_mi.pdf")
    _save(fig, out, show)

# =============================================================================
# G-A5 — Violin plot A-SHAP per categoria (correct vs wrong split)
# =============================================================================

def plot_violin_ashap_category(df, plots_dir: str, show: bool = False) -> None:
    """
    Un violin per categoria, split correct/wrong.
    Linea tratteggiata a y=0.5 (parità modale).
    """
    import matplotlib.pyplot as plt

    present_cats = [c for c in CATEGORY_ORDER if c in df["category"].unique()]
    if not present_cats:
        present_cats = df["category"].dropna().unique().tolist()

    n = len(present_cats)
    fig, ax = plt.subplots(figsize=(max(14, n * 1.4), 6))

    positions_correct = np.arange(n) * 3.0
    positions_wrong = positions_correct + 1.0
    tick_positions = (positions_correct + positions_wrong) / 2

    def _violin(data_list, positions, color, label):
        if not any(len(d) > 1 for d in data_list):
            return
        parts = ax.violinplot(
            [d if len(d) > 1 else [d[0], d[0]] for d in data_list],
            positions=positions,
            showmedians=True,
            showextrema=True,
            widths=0.8,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
        for key in ["cmedians", "cmins", "cmaxes", "cbars"]:
            if key in parts:
                parts[key].set_color(color)
                parts[key].set_linewidth(1.2)

    data_correct = []
    data_wrong = []
    for cat in present_cats:
        sub = df[df["category"] == cat]
        c_vals = sub[sub["correct"] == True]["a_shap"].dropna().values
        w_vals = sub[sub["correct"] == False]["a_shap"].dropna().values
        data_correct.append(c_vals if len(c_vals) > 0 else np.array([0.5]))
        data_wrong.append(w_vals if len(w_vals) > 0 else np.array([0.5]))

    _violin(data_correct, positions_correct, PALETTE["correct"], "correct")
    _violin(data_wrong, positions_wrong, PALETTE["wrong"], "wrong")

    ax.axhline(0.5, color="#888780", linewidth=0.8, linestyle="--", alpha=0.7,
               label="parità audio/testo")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(present_cats, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("A-SHAP", fontsize=10)
    ax.set_title("G-A5 — Distribuzione A-SHAP per categoria (correct vs wrong)",
                 fontsize=11)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["correct"], alpha=0.55, label="correct"),
        Patch(facecolor=PALETTE["wrong"], alpha=0.55, label="wrong"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(plots_dir, "G-A5_violin_ashap_category.pdf")
    _save(fig, out, show)

# =============================================================================
# Summary stats (stampate a schermo)
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
    print()
    print("  Per macrofamiglia (A-SHAP medio):")
    for macro in MACROFAMILY_ORDER:
        sub = df[df["macro_family"] == macro]
        if len(sub) == 0:
            continue
        print(f"    {macro:14s}: {sub['a_shap'].mean():.3f} (n={len(sub)})")
    print()
    print("  Per difficoltà (A-SHAP medio / accuratezza):")
    for diff in df["difficulty"].dropna().unique():
        sub = df[df["difficulty"] == diff]
        print(f"    {diff:10s}: A-SHAP={sub['a_shap'].mean():.3f} | "
              f"acc={sub['correct'].mean()*100:.1f}% (n={len(sub)})")
    print()
    print("  Categorie con category='unknown':", (df["category"] == "unknown").sum())
    print("  Difficoltà con difficulty='unknown':", (df["difficulty"] == "unknown").sum())
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
        "figure.dpi": 150,
        "savefig.dpi": 150,
    })
    warnings.filterwarnings("ignore", category=UserWarning)

    df = load_exp_a_data(batch_dir)
    print_summary(df)

    pdir = _plots_dir(batch_dir)
    print(f"\nSalvataggio plot in: {pdir}\n")

    print("G-A1 — Heatmap categorica...")
    plot_heatmap_categories(df, pdir, show)

    print("G-A2 — Radar macrofamiglia...")
    plot_radar_macrofamily(df, pdir, show)

    print("G-A3 — Stacked bar UC vs MI...")
    plot_stacked_bar_uc_mi(df, pdir, show)

    print("G-A4 — Scatter A-SHAP vs MI...")
    plot_scatter_ashap_mi(df, pdir, show)

    print("G-A5 — Violin A-SHAP per categoria...")
    plot_violin_ashap_category(df, pdir, show)

    print(f"\nTutti i plot salvati in: {pdir}")


def main():
    parser = argparse.ArgumentParser(description="Plot Esperimento A")
    parser.add_argument(
        "--batch-dir",
        required=True,
        metavar="PATH",
        help="Path della batch dir (es. Results_QA/exp_A/batch_run_00)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Apri finestre interattive oltre al salvataggio PDF",
    )
    args = parser.parse_args()
    run_all_plots(args.batch_dir, show=args.show)


if __name__ == "__main__":
    main()
