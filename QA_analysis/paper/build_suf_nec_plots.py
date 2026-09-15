"""
Rigenera le curve di sufficienza/necessita' (stile Fig.6/8 della tesi) con
seaborn/matplotlib, in stile pulito e formale per un paper ICASSP.

Per ogni modello (qwen, af3) e ogni condizione (complete, audio_only, text_only)
produce due figure a doppio pannello:
  - <model>_<condition>_mi_by_correctness.pdf   (MI: sufficienza+necessita', corretti vs sbagliati)
  - <model>_<condition>_modality_diagnosis.pdf  (MI/UC_audio/UC_text, solo corretti)

Nessun titolo/caption incorporato nell'immagine.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_suf_nec")
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
# Stile: pulito, formale, sfondo bianco, nessun colore acceso/viola.
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

# Palette: stessa identita' cromatica dei grafici originali di tesi
# (verde/rosso per corretti/sbagliati, rosso/verde/blu per MI/UC_audio/UC_text),
# ma in tonalita' spente invece dei colori accesi di default.
COLOR_CORRECT = "#4C8C6B"   # verde spento
COLOR_WRONG = "#B23A2E"     # rosso mattone spento
COLOR_MI = "#B23A2E"        # rosso mattone spento
COLOR_UC_AUDIO = "#4C8C6B"  # verde spento
COLOR_UC_TEXT = "#2C6E8C"   # blu acciaio spento

K_MAX = 16
CAPSIZE = 3
ERR_LW = 1.1
MARKER_SIZE = 4.5


P_ORIG_MIN = 0.40


def load_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["k"] = df["k"].astype(int)
    df["correct_baseline"] = df["correct_baseline"].astype(str) == "True"
    df["sufficiency_score"] = pd.to_numeric(df["sufficiency_score"], errors="coerce")
    df["necessity_score"] = pd.to_numeric(df["necessity_score"], errors="coerce")
    df["prob_original_target"] = pd.to_numeric(df["prob_original_target"], errors="coerce")
    # Esclude i campioni con p_orig troppo vicino al chance level (0.25), per cui
    # la normalizzazione (p - 0.25)/(p_orig - 0.25) sarebbe instabile (vedi Method).
    df = df[df["prob_original_target"] >= P_ORIG_MIN]
    return df


def style_axis(ax):
    ax.axhline(0.5, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlim(0.5, K_MAX + 0.5)
    ax.set_xticks(range(2, K_MAX + 1, 2))
    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sns.despine(ax=ax)


def mean_se_by_group(sub: pd.DataFrame, ycol: str, group_col: str, groups: list) -> dict:
    """Per ogni gruppo, ritorna (k_values, mean, sem) ordinati per k."""
    out = {}
    for g in groups:
        gsub = sub[sub[group_col] == g]
        stats = gsub.groupby("k")[ycol].agg(["mean", "sem"]).reset_index().sort_values("k")
        out[g] = (stats["k"].to_numpy(), stats["mean"].to_numpy(), stats["sem"].to_numpy())
    return out


def draw_errorbars(ax, stats_by_group: dict, groups: list, colors: dict, offsets: list):
    for g, off in zip(groups, offsets):
        k_vals, means, sems = stats_by_group[g]
        ax.errorbar(
            k_vals + off, means, yerr=sems,
            fmt="o-", color=colors[g], ecolor=colors[g],
            linewidth=ERR_LW, markersize=MARKER_SIZE,
            capsize=CAPSIZE, capthick=ERR_LW, elinewidth=ERR_LW,
            markeredgecolor="white", markeredgewidth=0.5,
            zorder=3,
        )


def plot_mi_by_correctness(df: pd.DataFrame, out_path: str):
    sub = df[df["feature_source"] == "MI"].dropna(subset=["sufficiency_score", "necessity_score"])
    sub = sub.assign(group=sub["correct_baseline"].map({True: "Correct", False: "Incorrect"}))
    groups = ["Correct", "Incorrect"]
    colors = {"Correct": COLOR_CORRECT, "Incorrect": COLOR_WRONG}
    offsets = [-0.10, 0.10]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    stats_suf = mean_se_by_group(sub, "sufficiency_score", "group", groups)
    draw_errorbars(axes[0], stats_suf, groups, colors, offsets)
    axes[0].set_xlabel("k (top-ranked MI features)")
    axes[0].set_ylabel("Sufficiency")
    style_axis(axes[0])

    stats_nec = mean_se_by_group(sub, "necessity_score", "group", groups)
    draw_errorbars(axes[1], stats_nec, groups, colors, offsets)
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

    stats_suf = mean_se_by_group(sub, "sufficiency_score", "group", order_labels)
    draw_errorbars(axes[0], stats_suf, order_labels, palette_labels, offsets)
    axes[0].set_xlabel("k (top-ranked features)")
    axes[0].set_ylabel("Sufficiency")
    style_axis(axes[0])

    stats_nec = mean_se_by_group(sub, "necessity_score", "group", order_labels)
    draw_errorbars(axes[1], stats_nec, order_labels, palette_labels, offsets)
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
