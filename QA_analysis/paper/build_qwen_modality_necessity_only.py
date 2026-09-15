"""
Estrae, per le sole condizioni Qwen, il solo pannello di necessita' della
modality diagnosis (UC_text/UC_audio/MI) come figura singola, identica in
stile/scala/colori al pannello destro di build_suf_nec_plots.py, ma senza
il pannello di sufficienza accanto. Output nella stessa cartella
figures_suf_nec, con nome distinto (suffisso "_necessity").
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_suf_nec_plots import (
    CONDITIONS, OUT_DIR,
    load_long, style_axis, mean_se_by_group, draw_errorbars,
    COLOR_MI, COLOR_UC_AUDIO, COLOR_UC_TEXT,
)


def plot_necessity_only(df, out_path):
    sub = df[df["correct_baseline"] == True].dropna(subset=["sufficiency_score", "necessity_score"])
    sub = sub[sub["feature_source"].isin(["MI", "UC_audio", "UC_text"])]
    order = ["UC_text", "UC_audio", "MI"]
    labels = {"UC_text": "UC$_{text}$", "UC_audio": "UC$_{audio}$", "MI": "MI"}
    palette = {"UC_text": COLOR_UC_TEXT, "UC_audio": COLOR_UC_AUDIO, "MI": COLOR_MI}
    sub = sub.assign(group=sub["feature_source"].map(labels))
    order_labels = [labels[o] for o in order]
    palette_labels = {labels[k]: v for k, v in palette.items()}
    offsets = [-0.13, 0.0, 0.13]

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.3))

    stats_nec = mean_se_by_group(sub, "necessity_score", "group", order_labels)
    draw_errorbars(ax, stats_nec, order_labels, palette_labels, offsets)
    ax.set_xlabel("k (top-ranked features)")
    ax.set_ylabel("Necessity")
    style_axis(ax)

    handles = [
        plt.Line2D([0], [0], color=palette_labels[l], lw=1.8, marker="o", markersize=5, label=l)
        for l in order_labels
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=3, frameon=False, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)


def main():
    qwen_conditions = {k: v for k, v in CONDITIONS.items() if k[0] == "qwen"}
    for (model, condition), path in qwen_conditions.items():
        print(f"Carico {model} / {condition} ...")
        df = load_long(path)
        out = os.path.join(OUT_DIR, f"{model}_{condition}_modality_diagnosis_necessity.pdf")
        plot_necessity_only(df, out)
        print(f"  salvato: {out}")

    print(f"\nFatto. Figure necessity-only in: {OUT_DIR}")


if __name__ == "__main__":
    main()
