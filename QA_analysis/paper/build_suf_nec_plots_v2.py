"""
Variante v2: stessa identica figura di build_suf_nec_plots.py (stessi colori,
stessa scala 0-1.1, stesso font, stesso layout, nessun titolo/caption, stesse
barre d'errore mean +/- SE), ma disegnata chiamando davvero sns.lineplot()
(err_style='bars') invece del ax.errorbar() manuale di v1. Output in una
cartella separata per confronto diretto.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_suf_nec_v2")
os.makedirs(OUT_DIR, exist_ok=True)

CONDITIONS = {
    ("qwen", "complete"):   f"{BASE}/Results_paper/experiments/exp_E/qwen_complete/aggregated/exp_e_curves.csv",
    ("qwen", "audio_only"): f"{BASE}/Results_paper/experiments/exp_E/qwen_audio_only/aggregated/exp_e_curves.csv",
    ("qwen", "text_only"):  f"{BASE}/Results_paper/experiments/exp_E/qwen_text_only/aggregated/exp_e_curves.csv",
    ("af3", "complete"):    f"{BASE}/Results_paper_af3/experiments/exp_E/af3_complete/aggregated/exp_e_curves.csv",
    ("af3", "audio_only"):  f"{BASE}/Results_paper_af3/experiments/exp_E/af3_audio_only/aggregated/exp_e_curves.csv",
    ("af3", "text_only"):   f"{BASE}/Results_paper_af3/experiments/exp_E/af3_text_only/aggregated/exp_e_curves.csv",
}

# ---------------------------------------------------------------------------
# Stile: identico a build_suf_nec_plots.py
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.edgecolor": "#4A4A4A",
    "axes.linewidth": 0.8,
    "grid.color": "#DADADA",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})

# Stessa identita' cromatica di v1.
COLOR_CORRECT = "#4C8C6B"   # verde spento
COLOR_WRONG = "#B23A2E"     # rosso mattone spento
COLOR_MI = "#B23A2E"        # rosso mattone spento
COLOR_UC_AUDIO = "#4C8C6B"  # verde spento
COLOR_UC_TEXT = "#2C6E8C"   # blu acciaio spento

K_MAX = 16
LINE_LW = 1.8
MARKER_SIZE = 4.5
CAPSIZE = 4
ERR_LW = 1.1


P_ORIG_MIN = 0.40


def load_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["k"] = df["k"].astype(int)
    df["correct_baseline"] = df["correct_baseline"].astype(str) == "True"
    df["sufficiency_score"] = pd.to_numeric(df["sufficiency_score"], errors="coerce")
    df["necessity_score"] = pd.to_numeric(df["necessity_score"], errors="coerce")
    df["prob_original_target"] = pd.to_numeric(df["prob_original_target"], errors="coerce")
    df = df[df["prob_original_target"] >= P_ORIG_MIN]
    return df


def style_axis(ax):
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlim(0.5, K_MAX + 0.5)
    ax.set_xticks(range(2, K_MAX + 1, 2))
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sns.despine(ax=ax)


def draw_band(ax, data: pd.DataFrame, ycol: str, group_col: str, groups: list, colors: dict, offsets: list):
    """Disegna media +/- SEM come barre d'errore puntuali, usando davvero
    sns.lineplot() (errorbar='se' -> stessa statistica di v1, err_style='bars').
    lineplot non supporta il dodge nativo tra gruppi hue, quindi ogni gruppo
    viene disegnato con una chiamata separata su una colonna k spostata di poco,
    cosi' le barre d'errore non si sovrappongono (stesso offset usato in v1)."""
    for g, off in zip(groups, offsets):
        gsub = data[data[group_col] == g].copy()
        gsub["k_off"] = gsub["k"] + off
        sns.lineplot(
            data=gsub, x="k_off", y=ycol, color=colors[g],
            errorbar="se", err_style="bars",
            linewidth=LINE_LW, marker="o", markersize=MARKER_SIZE,
            markeredgecolor="white", markeredgewidth=0.5,
            err_kws={"capsize": CAPSIZE, "capthick": ERR_LW, "elinewidth": ERR_LW},
            ax=ax, legend=False,
        )


def plot_mi_by_correctness(df: pd.DataFrame, out_path: str):
    sub = df[df["feature_source"] == "MI"].dropna(subset=["sufficiency_score", "necessity_score"])
    sub = sub.assign(group=sub["correct_baseline"].map({True: "Correct", False: "Incorrect"}))
    groups = ["Correct", "Incorrect"]
    colors = {"Correct": COLOR_CORRECT, "Incorrect": COLOR_WRONG}
    offsets = [-0.10, 0.10]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    draw_band(axes[0], sub, "sufficiency_score", "group", groups, colors, offsets)
    axes[0].set_xlabel("k (top-ranked MI features)")
    axes[0].set_ylabel("Sufficiency")
    style_axis(axes[0])

    draw_band(axes[1], sub, "necessity_score", "group", groups, colors, offsets)
    axes[1].set_xlabel("k (top-ranked MI features)")
    axes[1].set_ylabel("Necessity")
    style_axis(axes[1])

    handles = [
        plt.Line2D([0], [0], color=COLOR_CORRECT, lw=1.8, marker="o", markersize=4, label="Correct"),
        plt.Line2D([0], [0], color=COLOR_WRONG, lw=1.8, marker="o", markersize=4, label="Incorrect"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=2, frameon=False, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)


def plot_modality_diagnosis(df: pd.DataFrame, out_path: str):
    sub = df[df["correct_baseline"] == True].dropna(subset=["sufficiency_score", "necessity_score"])
    sub = sub[sub["feature_source"].isin(["MI", "UC_audio", "UC_text"])]
    order = ["UC_text", "UC_audio", "MI"]
    labels = {"UC_text": "UC$_{text}$", "UC_audio": "UC$_{audio}$", "MI": "MI"}
    palette = {"UC_text": COLOR_UC_TEXT, "UC_audio": COLOR_UC_AUDIO, "MI": COLOR_MI}
    sub = sub.assign(group=sub["feature_source"].map(labels))
    order_labels = [labels[o] for o in order]
    palette_labels = {labels[k]: v for k, v in palette.items()}
    offsets = [-0.13, 0.0, 0.13]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    draw_band(axes[0], sub, "sufficiency_score", "group", order_labels, palette_labels, offsets)
    axes[0].set_xlabel("k (top-ranked features)")
    axes[0].set_ylabel("Sufficiency")
    style_axis(axes[0])

    draw_band(axes[1], sub, "necessity_score", "group", order_labels, palette_labels, offsets)
    axes[1].set_xlabel("k (top-ranked features)")
    axes[1].set_ylabel("Necessity")
    style_axis(axes[1])

    handles = [
        plt.Line2D([0], [0], color=palette_labels[l], lw=1.8, marker="o", markersize=5, label=l)
        for l in order_labels
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=3, frameon=False, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)


def main():
    for (model, condition), path in CONDITIONS.items():
        print(f"Carico {model} / {condition} ...")
        df = load_long(path)

        out_a = os.path.join(OUT_DIR, f"{model}_{condition}_mi_by_correctness.pdf")
        plot_mi_by_correctness(df, out_a)
        print(f"  salvato: {out_a}")

        out_b = os.path.join(OUT_DIR, f"{model}_{condition}_modality_diagnosis.pdf")
        plot_modality_diagnosis(df, out_b)
        print(f"  salvato: {out_b}")

    print(f"\nFatto. Tutte le figure in: {OUT_DIR}")


if __name__ == "__main__":
    main()
