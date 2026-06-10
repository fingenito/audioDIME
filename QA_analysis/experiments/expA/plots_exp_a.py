"""
Experiment A — Analisi e grafici
=================================
Utilizzo:
    python -m QA_analysis.experiments.expA.plots_exp_a \
        --batch-dir .../exp_A/batch_run_00
"""

import os
import argparse
import warnings
from typing import List

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

# Soglia robustezza statistica
N_ROBUST = 20


# =============================================================================
# Helpers
# =============================================================================

def _ensure_derived_columns(df):
    df = df.copy()
    for c in ["a_shap","t_shap","uc_audio_l1","uc_text_l1","mi_audio_l1","mi_text_l1"]:
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
        df["difficulty"].astype(str).str.lower().str.strip()
        .replace({"med":"medium","mid":"medium","easy":"low","hard":"high"})
    )
    if "category" not in df.columns:
        df["category"] = "unknown"
    return df


def load_exp_a_data(batch_dir: str):
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas richiesto.")
    parquet_path = os.path.join(batch_dir, "aggregated", "exp_a_results.parquet")
    csv_path     = os.path.join(batch_dir, "aggregated", "exp_a_results.csv")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"Caricato parquet: {len(df)} sample")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Caricato CSV: {len(df)} sample")
    else:
        raise FileNotFoundError(f"Nessun aggregato in {batch_dir}/aggregated/")
    return _ensure_derived_columns(df)


def _plots_dir(batch_dir: str) -> str:
    d = os.path.join(batch_dir, "plots")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, path: str, show: bool) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(path, bbox_inches="tight", dpi=170)
    print(f"  -> saved: {path}")
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
# G-A1 — Due PNG separati, layout con fig.add_axes (niente twinx/divider)
# =============================================================================

def plot_heatmap_categories(df, plots_dir: str, show: bool = False) -> None:
    """
    Produce un unico PNG:
        G-A1_combined_mmshap_dime.png

    La figura contiene:
        - pannello sinistro: MM-SHAP A-SHAP / T-SHAP
        - pannello destro: audioDIME UC/MI audio+text
        - categorie condivise a sinistra
        - numerosità n= condivise a destra
        - colorbar separate perché le scale sono diverse
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    metrics_left = [
        ("a_shap", "A-SHAP"),
        ("t_shap", "T-SHAP"),
    ]

    metrics_right = [
        ("uc_audio_l1", "UC audio"),
        ("uc_text_l1",  "UC text"),
        ("mi_audio_l1", "MI audio"),
        ("mi_text_l1",  "MI text"),
    ]

    ordered_cats = _ordered_present_categories(df)
    n_cats = len(ordered_cats)

    mean_L = np.full((n_cats, 2), np.nan)
    std_L  = np.full((n_cats, 2), np.nan)

    mean_R = np.full((n_cats, 4), np.nan)
    std_R  = np.full((n_cats, 4), np.nan)

    counts = np.zeros(n_cats, dtype=int)

    grp = df.groupby("category")
    for i, cat in enumerate(ordered_cats):
        if cat not in grp.groups:
            continue

        sub = grp.get_group(cat)
        counts[i] = len(sub)

        for j, (col, _) in enumerate(metrics_left):
            if col in sub.columns:
                mean_L[i, j] = sub[col].mean()
                std_L[i, j] = sub[col].std()

        for j, (col, _) in enumerate(metrics_right):
            if col in sub.columns:
                mean_R[i, j] = sub[col].mean()
                std_R[i, j] = sub[col].std()

    low_n = [i for i, n in enumerate(counts) if n < N_ROBUST]

    # Scale reali dei due pannelli
    mm_vmin, mm_vmax = 0.0, 1.0
    dime_vmin = float(np.nanmin(mean_R))
    dime_vmax = float(np.nanmax(mean_R))

    # -------------------------------------------------------------------------
    # Layout globale in pollici
    # -------------------------------------------------------------------------
    CELL_H = 0.82

    CELL_W_MM = 2.15
    CELL_W_DIME = 1.65

    YLAB_W = 4.10

    GAP_PANEL = 0.75

    GAP_CB_MM = 0.28
    CB_W_MM = 0.22
    GAP_AFTER_CB_MM = 0.75

    GAP_CB_DIME = 0.28
    CB_W_DIME = 0.22

    GAP_N = 0.95
    NLAB_W = 0.90
    RIGHT_M = 0.35

    MARG_T = 1.25
    MARG_B = 1.25

    mm_w = mean_L.shape[1] * CELL_W_MM
    dime_w = mean_R.shape[1] * CELL_W_DIME
    img_h = n_cats * CELL_H

    total_w = (
        YLAB_W
        + mm_w
        + GAP_CB_MM
        + CB_W_MM
        + GAP_AFTER_CB_MM
        + GAP_PANEL
        + dime_w
        + GAP_CB_DIME
        + CB_W_DIME
        + GAP_N
        + NLAB_W
        + RIGHT_M
    )

    total_h = MARG_T + img_h + MARG_B

    fig = plt.figure(figsize=(total_w, total_h))

    def px(x):
        return x / total_w

    def py(y):
        return y / total_h

    img_bottom = py(MARG_B)
    img_height = py(img_h)

    mm_left = YLAB_W
    mm_cb_left = mm_left + mm_w + GAP_CB_MM

    dime_left = (
        mm_left
        + mm_w
        + GAP_CB_MM
        + CB_W_MM
        + GAP_AFTER_CB_MM
        + GAP_PANEL
    )
    dime_cb_left = dime_left + dime_w + GAP_CB_DIME

    n_left = dime_cb_left + CB_W_DIME + GAP_N

    # -------------------------------------------------------------------------
    # Helper per disegnare una heatmap
    # -------------------------------------------------------------------------
    def _draw_heatmap(
        ax,
        mean_mat,
        std_mat,
        vmin,
        vmax,
        col_labels,
        title,
        show_ylabels=False,
    ):
        im = ax.imshow(
            mean_mat,
            aspect="auto",
            cmap=ax._custom_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest"
        )

        n_r, n_c = mean_mat.shape

        for i in range(n_r):
            for j in range(n_c):
                if np.isnan(mean_mat[i, j]):
                    continue

                raw = float(mean_mat[i, j])
                if vmax > vmin:
                    bg_norm = (raw - vmin) / (vmax - vmin)
                else:
                    bg_norm = 0.5

                tc = "white" if bg_norm > 0.58 else "#111111"

                ax.text(
                    j,
                    i - 0.14,
                    f"{mean_mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11.0,
                    color=tc,
                    fontweight="semibold"
                )

                std_val = std_mat[i, j]
                std_txt = f"\u00b1{std_val:.2f}" if not np.isnan(std_val) else "\u00b1--"

                ax.text(
                    j,
                    i + 0.23,
                    std_txt,
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    color=tc,
                    alpha=0.86
                )

        # hatching condiviso sulle righe poco robuste
        for i in low_n:
            for j in range(n_c):
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    hatch="////",
                    edgecolor="#BDBDBD",
                    linewidth=0,
                    alpha=0.60
                )
                ax.add_patch(rect)

        ax.set_xticks(range(n_c))
        ax.set_xticklabels(col_labels, fontsize=13.0, fontweight="bold")
        ax.xaxis.set_tick_params(length=0, pad=11)
        ax.xaxis.set_ticks_position("bottom")

        ax.set_yticks(range(n_r))

        if show_ylabels:
            ax.set_yticklabels(ordered_cats, fontsize=12.0)
            ax.yaxis.set_tick_params(length=0, pad=10)
        else:
            ax.set_yticklabels([])
            ax.yaxis.set_tick_params(length=0)

        ax.tick_params(top=False, right=False)

        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.set_title(title, fontsize=13.5, fontweight="bold", pad=18)

        return im

    # -------------------------------------------------------------------------
    # Pannello MM-SHAP
    # -------------------------------------------------------------------------
    ax_mm = fig.add_axes([
        px(mm_left),
        img_bottom,
        px(mm_w),
        img_height
    ])
    ax_mm._custom_cmap = "Blues"

    im_mm = _draw_heatmap(
        ax=ax_mm,
        mean_mat=mean_L,
        std_mat=std_L,
        vmin=mm_vmin,
        vmax=mm_vmax,
        col_labels=["A-SHAP", "T-SHAP"],
        title="G-A1a — MM-SHAP",
        show_ylabels=True,
    )

    cb_mm_ax = fig.add_axes([
        px(mm_cb_left),
        img_bottom + 0.03 * img_height,
        px(CB_W_MM),
        0.94 * img_height
    ])
    cb_mm = fig.colorbar(im_mm, cax=cb_mm_ax)
    cb_mm.set_label(
        "MM-SHAP contribution",
        fontsize=10.8,
        fontweight="bold",
        labelpad=20,
        rotation=270,
        va="bottom"
    )
    cb_mm.ax.tick_params(labelsize=9.5, pad=4)
    cb_mm.outline.set_visible(False)

    # -------------------------------------------------------------------------
    # Pannello audioDIME
    # -------------------------------------------------------------------------
    ax_dime = fig.add_axes([
        px(dime_left),
        img_bottom,
        px(dime_w),
        img_height
    ])
    ax_dime._custom_cmap = "YlOrRd"

    im_dime = _draw_heatmap(
        ax=ax_dime,
        mean_mat=mean_R,
        std_mat=std_R,
        vmin=dime_vmin,
        vmax=dime_vmax,
        col_labels=["UC audio", "UC text", "MI audio", "MI text"],
        title="G-A1b — audioDIME L\u2081",
        show_ylabels=False,
    )

    cb_dime_ax = fig.add_axes([
        px(dime_cb_left),
        img_bottom + 0.03 * img_height,
        px(CB_W_DIME),
        0.94 * img_height
    ])
    cb_dime = fig.colorbar(im_dime, cax=cb_dime_ax)
    cb_dime.set_label(
        "DIME L\u2081 value",
        fontsize=10.8,
        fontweight="bold",
        labelpad=20,
        rotation=270,
        va="bottom"
    )
    cb_dime.ax.tick_params(labelsize=9.5, pad=4)
    cb_dime.outline.set_visible(False)

    # -------------------------------------------------------------------------
    # Colonna n= condivisa
    # -------------------------------------------------------------------------
    n_ax = fig.add_axes([
        px(n_left),
        img_bottom,
        px(NLAB_W),
        img_height
    ])

    n_ax.set_xlim(0, 1)
    n_ax.set_ylim(n_cats - 0.5, -0.5)
    n_ax.axis("off")

    for i, cnt in enumerate(counts):
        n_ax.text(
            0.0,
            i,
            f"n={cnt}",
            ha="left",
            va="center",
            fontsize=10.2,
            color=PALETTE["neutral"]
        )

    # -------------------------------------------------------------------------
    # Titolo globale
    # -------------------------------------------------------------------------
    fig.text(
        0.5,
        py(MARG_B + img_h + MARG_T * 0.72),
        "G-A1 — XAI contribution profiles by musical category",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Legenda condivisa
    # -------------------------------------------------------------------------
    if low_n:
        hp = mpatches.Patch(
            facecolor="white",
            edgecolor="#AAAAAA",
            hatch="////",
            label=f"n < {N_ROBUST}  —  low statistical robustness"
        )

        fig.legend(
            handles=[hp],
            loc="lower left",
            fontsize=10,
            frameon=True,
            framealpha=0.95,
            bbox_to_anchor=(px(YLAB_W), py(0.11))
        )

    _save(
        fig,
        os.path.join(plots_dir, "G-A1_combined_mmshap_dime.png"),
        show
    )



# =============================================================================
# G-A2 — Profilo XAI per difficoltà
# =============================================================================

def plot_difficulty_profile(df, plots_dir: str, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    difficulties = _ordered_present_difficulties(df)
    if not difficulties:
        print("  ! G-A2 saltato: nessuna difficolta disponibile.")
        return

    metrics_left = [
        ("a_shap", "A-SHAP", PALETTE["audio"]),
        ("t_shap", "T-SHAP", PALETTE["text"]),
    ]
    metrics_right = [
        ("uc_audio_l1", "UC audio", PALETTE["audio"]),
        ("uc_text_l1",  "UC text",  PALETTE["text"]),
        ("mi_audio_l1", "MI audio", PALETTE["mi"]),
        ("mi_text_l1",  "MI text",  "#F09A73"),
        ("mi_ratio",    "MI ratio", PALETTE["uc"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    ax1, ax2 = axes
    x = np.arange(len(difficulties))

    width1 = 0.34
    for k, (col, label, color) in enumerate(metrics_left):
        vals = [
            df[df["difficulty_norm"] == d][col].mean()
            if len(df[df["difficulty_norm"] == d]) > 0 else np.nan
            for d in difficulties
        ]
        ax1.bar(x + (k - 0.5) * width1, vals, width1 * 0.9,
                color=color, alpha=0.82, label=label)

    ax1.axhline(0.5, color=PALETTE["neutral"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in difficulties])
    ax1.set_ylabel("Normalized contribution")
    ax1.set_title("MM-SHAP modality dependence")
    ax1.legend(frameon=True, fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    width2 = 0.15
    for k, (col, label, color) in enumerate(metrics_right):
        vals = [
            df[df["difficulty_norm"] == d][col].mean()
            if len(df[df["difficulty_norm"] == d]) > 0 else np.nan
            for d in difficulties
        ]
        offset = (k - (len(metrics_right) - 1) / 2) * width2
        ax2.bar(x + offset, vals, width2 * 0.9,
                color=color, alpha=0.82, label=label)

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.capitalize() for d in difficulties])
    ax2.set_ylabel("Mean value")
    ax2.set_title("DIME decomposition by difficulty")
    ax2.legend(frameon=True, fontsize=8, ncol=2)
    ax2.grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, d in enumerate(difficulties):
            n = len(df[df["difficulty_norm"] == d])
            ax.text(i, ax.get_ylim()[1] * 0.96, f"n={n}",
                    ha="center", va="top", fontsize=8, color=PALETTE["neutral"])

    fig.suptitle("G-A2 \u2014 XAI profile by question difficulty", fontsize=12, y=1.03)
    _save(fig, os.path.join(plots_dir, "G-A2_difficulty_profile.png"), show)


# =============================================================================
# G-A3 — Heatmap 2D A-SHAP x MI-ratio
# =============================================================================

def plot_heatmaps_ashap_miratio(df, plots_dir: str, show: bool = False) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    def _single_heatmap(sub_df, title, filename):
        fig, ax = plt.subplots(figsize=(9.2, 7.2))
        if len(sub_df) == 0:
            ax.text(0.5, 0.5, "No samples available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color=PALETTE["neutral"])
        else:
            sns.histplot(data=sub_df, x="a_shap", y="mi_ratio",
                         bins=30, binrange=((0.0, 1.0), (0.0, 1.0)),
                         cbar=True, cbar_kws=dict(shrink=.75, label="Number of samples"),
                         ax=ax)
            cx, cy = float(sub_df["a_shap"].mean()), float(sub_df["mi_ratio"].mean())
            ax.scatter(cx, cy, s=180, c="red", marker="X",
                       edgecolors="white", linewidths=1.2, zorder=10,
                       label=f"centroid ({cx:.2f}, {cy:.2f})")
            ax.axvline(cx, color="red", linestyle=":", linewidth=1.2, alpha=0.85, zorder=9)
            ax.axhline(cy, color="red", linestyle=":", linewidth=1.2, alpha=0.85, zorder=9)
            ax.legend(loc="upper left", fontsize=8, frameon=True)

        x_thr = 0.5
        y_thr = float(df["mi_ratio"].median()) if len(df) else 0.5
        ax.axvline(x_thr, color=PALETTE["neutral"], linestyle="--", linewidth=1.0, alpha=0.65)
        ax.axhline(y_thr, color=PALETTE["neutral"], linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.text(0.98, 0.96, "true listening\n(high audio + interaction)",
                ha="right", va="top", fontsize=8.5, color=PALETTE["audio"], transform=ax.transAxes)
        ax.text(0.98, 0.04, "non-interactive audio\n(high audio, low MI)",
                ha="right", va="bottom", fontsize=8.5, color=PALETTE["mi"], transform=ax.transAxes)
        ax.text(0.02, 0.96, "anomalous cross-modal\n(low audio, high MI)",
                ha="left", va="top", fontsize=8.5, color=PALETTE["uc"], transform=ax.transAxes)
        ax.text(0.02, 0.04, "possible textual bias\n(low audio, low MI)",
                ha="left", va="bottom", fontsize=8.5, color=PALETTE["neutral"], transform=ax.transAxes)
        ax.set_xlabel("A-SHAP \u2014 normalized audio contribution")
        ax.set_ylabel("MI-ratio \u2014 share of multimodal interaction")
        ax.set_title(title)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _save(fig, os.path.join(plots_dir, filename), show)

    _single_heatmap(df[df["correct"] == True],
                    "G-A3a \u2014 A-SHAP \u00d7 MI-ratio heatmap: correct answers",
                    "G-A3a_heatmap_correct.png")
    _single_heatmap(df[df["correct"] == False],
                    "G-A3b \u2014 A-SHAP \u00d7 MI-ratio heatmap: wrong answers",
                    "G-A3b_heatmap_wrong.png")
    _single_heatmap(df[df["difficulty_norm"] == "low"],
                    "G-A3c \u2014 A-SHAP \u00d7 MI-ratio heatmap: low difficulty",
                    "G-A3c_heatmap_difficulty_low.png")
    _single_heatmap(df[df["difficulty_norm"] == "medium"],
                    "G-A3d \u2014 A-SHAP \u00d7 MI-ratio heatmap: medium difficulty",
                    "G-A3d_heatmap_difficulty_medium.png")
    _single_heatmap(df[df["difficulty_norm"] == "high"],
                    "G-A3e \u2014 A-SHAP \u00d7 MI-ratio heatmap: high difficulty",
                    "G-A3e_heatmap_difficulty_high.png")


# =============================================================================
# G-A4 — Profilo XAI correct vs wrong
# =============================================================================

def plot_correctness_profile(df, plots_dir: str, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("a_shap",      "A-SHAP"),
        ("t_shap",      "T-SHAP"),
        ("uc_audio_l1", "UC audio"),
        ("uc_text_l1",  "UC text"),
        ("mi_audio_l1", "MI audio"),
        ("mi_text_l1",  "MI text"),
        ("mi_ratio",    "MI ratio"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    correct_df = df[df["correct"] == True]
    wrong_df   = df[df["correct"] == False]
    correct_vals = [correct_df[col].mean() if len(correct_df) > 0 else np.nan for col, _ in metrics]
    wrong_vals   = [wrong_df[col].mean()   if len(wrong_df)   > 0 else np.nan for col, _ in metrics]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    if len(correct_df) > 0:
        ax.bar(x - width / 2, correct_vals, width, color=PALETTE["correct"], alpha=0.78,
               label=f"correct (n={len(correct_df)})")
    if len(wrong_df) > 0:
        ax.bar(x + width / 2, wrong_vals, width, color=PALETTE["wrong"], alpha=0.78,
               label=f"wrong (n={len(wrong_df)})")
    if len(correct_df) == 0 or len(wrong_df) == 0:
        missing = "wrong" if len(wrong_df) == 0 else "correct"
        ax.text(0.5, 0.94, f"Note: the current batch contains no {missing} answers.",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=PALETTE["neutral"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], rotation=25, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("G-A4 \u2014 XAI profile for correct vs wrong answers")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(plots_dir, "G-A4_correctness_profile.png"), show)


# =============================================================================
# Summary
# =============================================================================

def print_summary(df) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICO — EXPERIMENT A")
    print("=" * 60)
    print(f"  Totale sample   : {len(df)}")
    print(f"  Corretti        : {df['correct'].sum()} ({df['correct'].mean()*100:.1f}%)")
    print(f"  A-SHAP medio    : {df['a_shap'].mean():.3f} +/- {df['a_shap'].std():.3f}")
    print(f"  T-SHAP medio    : {df['t_shap'].mean():.3f} +/- {df['t_shap'].std():.3f}")
    print(f"  UC audio L1 med : {df['uc_audio_l1'].mean():.3f}")
    print(f"  MI audio L1 med : {df['mi_audio_l1'].mean():.3f}")
    print(f"  UC text L1 med  : {df['uc_text_l1'].mean():.3f}")
    print(f"  MI text L1 med  : {df['mi_text_l1'].mean():.3f}")
    print(f"  MI-ratio med    : {df['mi_ratio'].mean():.3f}")
    print()
    print("  Per difficolta:")
    for diff in _ordered_present_difficulties(df):
        sub = df[df["difficulty_norm"] == diff]
        print(f"    {diff:10s}: A-SHAP={sub['a_shap'].mean():.3f} | "
              f"MI-ratio={sub['mi_ratio'].mean():.3f} | "
              f"acc={sub['correct'].mean()*100:.1f}% (n={len(sub)})")
    print()
    print("  Per categoria:")
    for cat in _ordered_present_categories(df):
        sub = df[df["category"] == cat]
        print(f"    {cat:32s}: A-SHAP={sub['a_shap'].mean():.3f} | "
              f"MI-ratio={sub['mi_ratio'].mean():.3f} | "
              f"acc={sub['correct'].mean()*100:.1f}% (n={len(sub)})")
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
        "font.family":   "DejaVu Sans",
        "font.size":      10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi":    150,
        "savefig.dpi":   170,
    })
    warnings.filterwarnings("ignore", category=UserWarning)

    df   = load_exp_a_data(batch_dir)
    print_summary(df)
    pdir = _plots_dir(batch_dir)
    print(f"\nSalvataggio plot in: {pdir}\n")

    print("G-A1 — Heatmap categorica (due PNG separati)...")
    plot_heatmap_categories(df, pdir, show)

    print("G-A2 — Profilo per difficolta...")
    plot_difficulty_profile(df, pdir, show)

    print("G-A3 — Heatmap 2D A-SHAP x MI-ratio...")
    plot_heatmaps_ashap_miratio(df, pdir, show)

    print("G-A4 — Profilo correct vs wrong...")
    plot_correctness_profile(df, pdir, show)

    print(f"\nTutti i plot salvati in: {pdir}")


def main():
    parser = argparse.ArgumentParser(description="Plot Esperimento A")
    parser.add_argument("--batch-dir", required=True, metavar="PATH")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run_all_plots(args.batch_dir, show=args.show)


if __name__ == "__main__":
    main()