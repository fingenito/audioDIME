"""
Esperimento E — Plot v4
========================
Plot allineati con batch_exp_e.py v4.

Metriche primarie (chance-normalized, MCQA 4-way con p_chance=0.25):
    sufficiency_score = (p_suf - 0.25) / (p_orig - 0.25)
    necessity_score   = (p_orig - p_nec) / (p_orig - 0.25)

Output principali:
    G-M1   Sufficiency curve MI (correct vs wrong)
    G-M2   Necessity curve MI   (correct vs wrong)
    G-M3   Distribuzione minimal_sufficient_k e minimal_necessary_k
    G-M4   Curve MI per macro_family
    G-M5   Curve MI per categoria (heatmap di minimal-k)
    G-M6   Composizione media top-k MI (audio/testo)
    G-M7   Diagnostico UC_audio vs UC_text vs MI (sufficiency curve sovrapposte)

Filtro standard: usiamo solo sample con baseline_above_chance=True per le
medie. Sample con baseline_above_chance=False sono comunque conservati in
parquet ma esclusi dai plot principali.

Utilizzo:
    python -m QA_analysis.experiments.expE.plots_exp_e \
        --batch-dir <BATCH_E>
"""

import os
import argparse
import warnings
from typing import Any, Dict, List, Optional

import numpy as np


# =============================================================================
# Palette e ordinamenti
# =============================================================================
PALETTE = {
    "MI":       "#D85A30",
    "UC_audio": "#1D9E75",
    "UC_text":  "#378ADD",
    "MI_balanced": "#7A3E9D",
    "correct": "#1D9E75",
    "wrong":   "#E24B4A",

    "percettiva": "#1D9E75",
    "analitica":  "#534AB7",
    "knowledge":  "#D85A30",
    "unknown":    "#888780",
}

CATEGORY_ORDER = [
    "Instrumentation", "Sound Texture", "Metre and Rhythm", "Musical Texture",
    "Harmony", "Melody", "Structure", "Performance",
    "Genre and Style", "Mood and Expression",
    "Functional Context", "Historical and Cultural Context", "Lyrics",
]
MACROFAMILY_ORDER = ["percettiva", "analitica", "knowledge", "unknown"]


# Soglie operative per le linee guida sui plot
ALPHA_SUFFICIENT = 0.5
BETA_NECESSARY   = 0.5
# Filtro di stabilità per evitare metriche esplosive quando
# p_orig è troppo vicino al random.
# Mantiene comunque il confronto correct vs wrong.
MIN_PROB_ORIGINAL_TARGET_FOR_MAIN_PLOTS = 0.40


# =============================================================================
# I/O
# =============================================================================
def load_exp_e_data(batch_dir: str):
    import pandas as pd
    curves_pq = os.path.join(batch_dir, "aggregated", "exp_e_curves.parquet")
    summary_pq = os.path.join(batch_dir, "aggregated", "exp_e_summary.parquet")

    if not os.path.exists(curves_pq):
        raise FileNotFoundError(
            f"File mancante: {curves_pq}. Lancia il batch v4 e riprova."
        )

    df_curves = pd.read_parquet(curves_pq)
    df_summary = pd.read_parquet(summary_pq) if os.path.exists(summary_pq) else None

    print(f"Caricato curves: {len(df_curves)} righe "
          f"({df_curves['sample_id'].nunique()} sample × "
          f"{df_curves['feature_source'].nunique()} source × "
          f"{df_curves['k'].nunique()} k)")
    if df_summary is not None:
        print(f"Caricato summary: {len(df_summary)} righe")

    return df_curves, df_summary


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


def _filter_valid(
    df,
    only_above_chance: bool = True,
    min_prob_original_target: Optional[float] = MIN_PROB_ORIGINAL_TARGET_FOR_MAIN_PLOTS,
):
    """
    Filtra:
      - sample above chance
      - opzionalmente sample con p_orig troppo vicino al random

    Non rimuove i wrong: mantiene correct vs wrong.
    """
    sub = df.copy()

    if only_above_chance and "baseline_above_chance" in sub.columns:
        sub = sub[sub["baseline_above_chance"] == True]

    if (
        min_prob_original_target is not None
        and "prob_original_target" in sub.columns
    ):
        sub = sub[sub["prob_original_target"] >= float(min_prob_original_target)]

    return sub

# =============================================================================
# G-M1 / G-M2 — Curve sufficiency e necessity MI (correct vs wrong)
# =============================================================================
def _plot_curve_correct_wrong(
    df, plots_dir, show,
    feature_source: str,
    metric: str,
    title: str,
    ylabel: str,
    threshold: float,
    threshold_label: str,
    out_name: str,
):
    import matplotlib.pyplot as plt

    sub = _filter_valid(df)
    sub = sub[sub["feature_source"] == feature_source]
    sub = sub[sub[metric].notnull()]
    if sub.empty:
        print(f"  skip {out_name}: no data")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for correct_val, color, label in [
        (True,  PALETTE["correct"], "correct"),
        (False, PALETTE["wrong"],   "wrong"),
    ]:
        s = sub[sub["correct_baseline"] == correct_val]
        if s.empty:
            continue
        agg = s.groupby("k")[metric].agg(["mean", "std", "count"]).reset_index()
        agg = agg.sort_values("k")
        se = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
        ax.errorbar(
            agg["k"], agg["mean"], yerr=se,
            marker="o", markersize=7, lw=1.8,
            color=color, label=f"{label} (n={int(agg['count'].iloc[0])})",
            capsize=3,
        )

    ax.axhline(threshold, color="#888780", lw=0.8, ls="--", alpha=0.7,
               label=f"threshold ({threshold_label})")
    ax.axhline(0.0, color="#444441", lw=0.5, alpha=0.4)

    ks_present = sorted(sub["k"].unique())
    ax.set_xticks(ks_present)
    ax.set_xlabel("k (number of MI features in the rationale)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

    _save(fig, os.path.join(plots_dir, out_name), show)


def plot_mi_sufficiency_curve(df, plots_dir, show):
    _plot_curve_correct_wrong(
        df, plots_dir, show,
        feature_source="MI",
        metric="sufficiency_score",
        title="G-M1 — MI Sufficiency curve (correct vs wrong, chance-normalized)",
        ylabel="sufficiency = (p_suf - 0.25) / (p_orig - 0.25) — mean ± SE",
        threshold=ALPHA_SUFFICIENT,
        threshold_label=f"α = {ALPHA_SUFFICIENT}",
        out_name="G-M1_mi_sufficiency_curve.png",
    )


def plot_mi_necessity_curve(df, plots_dir, show):
    _plot_curve_correct_wrong(
        df, plots_dir, show,
        feature_source="MI",
        metric="necessity_score",
        title="G-M2 — MI Necessity curve (correct vs wrong, chance-normalized)",
        ylabel="necessity = (p_orig - p_nec) / (p_orig - 0.25)  — mean ± SE",
        threshold=BETA_NECESSARY,
        threshold_label=f"β = {BETA_NECESSARY}",
        out_name="G-M2_mi_necessity_curve.png",
    )

def plot_mi_balanced_sufficiency_curve(df, plots_dir, show):
    _plot_curve_correct_wrong(
        df, plots_dir, show,
        feature_source="MI_balanced",
        metric="sufficiency_score",
        title="G-M8 — MI Balanced Sufficiency curve (correct vs wrong)",
        ylabel="sufficiency = (p_suf - 0.25) / (p_orig - 0.25) — mean ± SE",
        threshold=ALPHA_SUFFICIENT,
        threshold_label=f"α = {ALPHA_SUFFICIENT}",
        out_name="G-M8_mi_balanced_sufficiency_curve.png",
    )


def plot_mi_balanced_necessity_curve(df, plots_dir, show):
    _plot_curve_correct_wrong(
        df, plots_dir, show,
        feature_source="MI_balanced",
        metric="necessity_score",
        title="G-M9 — MI Balanced Necessity curve (correct vs wrong)",
        ylabel="necessity = (p_orig - p_nec) / (p_orig - 0.25) — mean ± SE",
        threshold=BETA_NECESSARY,
        threshold_label=f"β = {BETA_NECESSARY}",
        out_name="G-M9_mi_balanced_necessity_curve.png",
    )

def plot_mi_vs_balanced_comparison(df, plots_dir, show):
    """
    Confronta MI globale e MI_balanced sui sample correct e wrong.
    Serve a vedere se la sufficiency migliora quando il rationale contiene
    sia audio sia testo.
    """
    import matplotlib.pyplot as plt

    sub = _filter_valid(df)
    sub = sub[sub["feature_source"].isin(["MI", "MI_balanced"])]

    if sub.empty:
        print("  skip G-M10: no MI / MI_balanced data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)

    for ax, metric, title, threshold in [
        (
            axes[0],
            "sufficiency_score",
            "G-M10a — Sufficiency: Global MI vs MI balanced",
            ALPHA_SUFFICIENT,
        ),
        (
            axes[1],
            "necessity_score",
            "G-M10b — Necessity: Global MI vs MI balanced",
            BETA_NECESSARY,
        ),
    ]:
        for src in ["MI", "MI_balanced"]:
            for correct_val, linestyle, label_suffix in [
                (True, "-", "correct"),
                (False, "--", "wrong"),
            ]:
                s = sub[
                    (sub["feature_source"] == src)
                    & (sub["correct_baseline"] == correct_val)
                    & (sub[metric].notnull())
                ]
                if s.empty:
                    continue

                agg = s.groupby("k")[metric].agg(["mean", "std", "count"]).reset_index()
                agg = agg.sort_values("k")
                se = agg["std"] / np.sqrt(agg["count"].clip(lower=1))

                ax.errorbar(
                    agg["k"],
                    agg["mean"],
                    yerr=se,
                    marker="o",
                    markersize=5,
                    lw=1.5,
                    ls=linestyle,
                    color=PALETTE.get(src, "#888780"),
                    label=f"{src} — {label_suffix} (n={int(agg['count'].iloc[0])})",
                    capsize=3,
                    alpha=0.9,
                )

        ax.axhline(threshold, color="#888780", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0.0, color="#444441", lw=0.5, alpha=0.4)
        ax.set_xticks(sorted(sub["k"].unique()))
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

    _save(fig, os.path.join(plots_dir, "G-M10_mi_vs_balanced_comparison.png"), show)


# =============================================================================
# G-M3 — Distribuzione minimal_k
# =============================================================================
def plot_minimal_k_distribution(df_summary, plots_dir, show):
    """
    Distribuzione di minimal_sufficient_k e minimal_necessary_k per MI,
    correct vs wrong.

    Mostra: per ogni k=1..K_max, % di sample che hanno minimal_k = quel valore,
    + barra "never" per sample che non raggiungono mai la soglia entro K_max.
    """
    import matplotlib.pyplot as plt

    if df_summary is None or df_summary.empty:
        print("  skip G-M3: no summary")
        return

    sub = df_summary[df_summary["feature_source"] == "MI"].copy()
    if "baseline_above_chance" in sub.columns:
        sub = sub[sub["baseline_above_chance"] == True]
    if sub.empty:
        print("  skip G-M3: no MI data after filter")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title, color_correct in [
        (axes[0], "minimal_sufficient_k",
         f"G-M3a — MI minimal_sufficient_k distribution (α={ALPHA_SUFFICIENT})",
         PALETTE["correct"]),
        (axes[1], "minimal_necessary_k",
         f"G-M3b — MI minimal_necessary_k distribution (β={BETA_NECESSARY})",
         PALETTE["correct"]),
    ]:
        ks_seen = sorted([int(k) for k in sub[col].dropna().unique().tolist()])
        if not ks_seen:
            ks_seen = [1, 2, 3, 4, 5, 6]
        labels = [str(k) for k in ks_seen] + ["never"]
        x = np.arange(len(labels))
        width = 0.42

        for i, (correct_val, color, label) in enumerate([
            (True,  PALETTE["correct"], "correct"),
            (False, PALETTE["wrong"],   "wrong"),
        ]):
            s = sub[sub["correct_baseline"] == correct_val]
            total = len(s)
            if total == 0:
                continue
            counts = [int((s[col] == k).sum()) for k in ks_seen]
            never = int(s[col].isna().sum())
            heights = [c / total * 100 for c in counts] + [never / total * 100]
            offset = (i - 0.5) * width
            ax.bar(x + offset, heights, width * 0.95,
                   color=color, alpha=0.85,
                   label=f"{label} (n={total})", edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("k")
        ax.set_ylabel("% sample")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _save(fig, os.path.join(plots_dir, "G-M3_mi_minimal_k_distribution.png"), show)


# =============================================================================
# G-M4 — Curve MI per macro_family
# =============================================================================
def plot_mi_curves_by_macrofamily(df, plots_dir, show):
    import matplotlib.pyplot as plt

    sub = _filter_valid(df)
    sub = sub[sub["feature_source"] == "MI"]
    if sub.empty:
        print("  skip G-M4: no MI data")
        return

    fams = [f for f in MACROFAMILY_ORDER if f in sub["macro_family"].unique()]
    if not fams:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)

    for ax, metric, title, threshold in [
        (axes[0], "sufficiency_score", "G-M4a — Sufficiency MI per macro-family",
         ALPHA_SUFFICIENT),
        (axes[1], "necessity_score",   "G-M4b — Necessity MI per macro-family",
         BETA_NECESSARY),
    ]:
        for fam in fams:
            s = sub[(sub["macro_family"] == fam) & (sub[metric].notnull())]
            if s.empty:
                continue
            agg = s.groupby("k")[metric].agg(["mean", "std", "count"]).reset_index()
            agg = agg.sort_values("k")
            se = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            ax.errorbar(
                agg["k"], agg["mean"], yerr=se,
                marker="o", markersize=6, lw=1.5,
                color=PALETTE.get(fam, "#888780"),
                label=f"{fam} (n={int(agg['count'].iloc[0])})",
                capsize=3, alpha=0.9,
            )
        ax.axhline(threshold, color="#888780", lw=0.8, ls="--", alpha=0.7,
                   label=f"soglia {threshold}")
        ax.axhline(0.0, color="#444441", lw=0.5, alpha=0.4)
        ax.set_xticks(sorted(sub["k"].unique()))
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

    _save(fig, os.path.join(plots_dir, "G-M4_mi_curves_by_macrofamily.png"), show)


# =============================================================================
# G-M5 — Heatmap minimal-k per categoria
# =============================================================================
def plot_minimal_k_heatmap_categories(df_summary, plots_dir, show):
    import matplotlib.pyplot as plt

    if df_summary is None or df_summary.empty:
        return
    sub = df_summary[df_summary["feature_source"] == "MI"].copy()
    if "baseline_above_chance" in sub.columns:
        sub = sub[sub["baseline_above_chance"] == True]
    if sub.empty:
        return

    cats = [c for c in CATEGORY_ORDER if c in sub["category"].unique()]
    cats += [c for c in sub["category"].unique()
             if c not in cats and c and c != "unknown"]

    metrics = [
        ("minimal_sufficient_k",
         f"G-M5a — Mediana minimal_sufficient_k per categoria (α={ALPHA_SUFFICIENT})"),
        ("minimal_necessary_k",
         f"G-M5b — Mediana minimal_necessary_k per categoria (β={BETA_NECESSARY})"),
    ]

    n_cats = len(cats)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_cats * 0.4)), sharey=True)

    for ax, (col, title) in zip(axes, metrics):
        mean_vec = []
        count_vec = []
        never_pct = []
        for cat in cats:
            s = sub[sub["category"] == cat]
            count_vec.append(len(s))
            vals = s[col].dropna().tolist()
            if vals:
                mean_vec.append(float(np.median(vals)))
            else:
                mean_vec.append(np.nan)
            never_pct.append(
                100.0 * s[col].isna().sum() / max(1, len(s))
            )

        mat = np.array(mean_vec).reshape(-1, 1)

        finite_vals = [v for v in mean_vec if not np.isnan(v)]
        vmax_dynamic = max(6, int(np.nanmax(finite_vals)) if finite_vals else 6)

        im = ax.imshow(
            mat,
            aspect="auto",
            cmap="RdYlGn_r",
            vmin=1,
            vmax=vmax_dynamic,
        )

        for i, cat in enumerate(cats):
            label = (
                f"{mean_vec[i]:.1f}" if not np.isnan(mean_vec[i]) else "—"
            )
            label += f"\n(never={never_pct[i]:.0f}%, n={count_vec[i]})"
            ax.text(0, i, label, ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(mean_vec[i])
                                       and mean_vec[i] > 4) else "black")

        ax.set_xticks([])
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(cats, fontsize=9)
        ax.set_title(title, fontsize=10)

    plt.colorbar(im, ax=axes, shrink=0.6, pad=0.05,
                 label="mediana k (basso = spiegazione sparsa)")
    _save(fig, os.path.join(plots_dir, "G-M5_mi_minimal_k_heatmap_categories.png"), show)


# =============================================================================
# G-M6 — Composizione media del top-k MI (audio/testo)
# =============================================================================
def plot_mi_topk_composition(df, plots_dir, show):
    import matplotlib.pyplot as plt

    sub = df[df["feature_source"] == "MI"].copy()
    if sub.empty:
        return

    needed = ["topk_audio_fraction", "topk_text_fraction",
              "topk_is_multimodal", "topk_is_unimodal_audio",
              "topk_is_unimodal_text"]
    if any(c not in sub.columns for c in needed):
        print("  skip G-M6: colonne mancanti")
        return

    # unica riga per (sample, k)
    sub_u = sub.sort_values(["sample_id", "k"]).drop_duplicates(["sample_id", "k"])

    ks = sorted(sub_u["k"].dropna().unique().tolist())
    agg = (
        sub_u.groupby("k")[["topk_audio_fraction", "topk_text_fraction"]]
        .mean().reindex(ks)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(ks))
    ax = axes[0]
    audio = agg["topk_audio_fraction"].to_numpy(dtype=float)
    text  = agg["topk_text_fraction"].to_numpy(dtype=float)
    ax.bar(x, audio, label="audio", color=PALETTE["UC_audio"], alpha=0.85)
    ax.bar(x, text,  bottom=audio, label="text", color=PALETTE["UC_text"], alpha=0.85)
    ax.axhline(0.5, color="#888780", lw=0.6, ls=":", alpha=0.6, label="50/50")
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1)
    ax.set_xlabel("k")
    ax.set_ylabel("mean fraction")
    ax.set_title("G-M6a — Mean composition of MI top-k")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # categorizzazione
    ax = axes[1]
    rows = []
    for k in ks:
        s = sub_u[sub_u["k"] == k]
        rows.append({
            "k": k,
            "multimodal": float(s["topk_is_multimodal"].mean() * 100),
            "only_audio": float(s["topk_is_unimodal_audio"].mean() * 100),
            "only_text":  float(s["topk_is_unimodal_text"].mean()  * 100),
        })
    multi = np.array([r["multimodal"] for r in rows], dtype=float)
    only_a = np.array([r["only_audio"] for r in rows], dtype=float)
    only_t = np.array([r["only_text"]  for r in rows], dtype=float)
    bottom = np.zeros(len(ks))
    ax.bar(x, multi, bottom=bottom, label="multimodal", color=PALETTE["MI"], alpha=0.85)
    bottom += multi
    ax.bar(x, only_a, bottom=bottom, label="audio only", color=PALETTE["UC_audio"], alpha=0.85)
    bottom += only_a
    ax.bar(x, only_t, bottom=bottom, label="text only", color=PALETTE["UC_text"], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 100)
    ax.set_xlabel("k")
    ax.set_ylabel("% sample")
    ax.set_title("G-M6b — Top-k MI: multimodal vs unimodal")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, os.path.join(plots_dir, "G-M6_mi_topk_composition.png"), show)


# =============================================================================
# G-M7 — Diagnostico UC_audio vs UC_text vs MI
# =============================================================================
def plot_diagnostic_uc_vs_mi(df, plots_dir, show):
    """
    Diagnostico: curve sufficiency e necessity sovrapposte per MI/UC_audio/UC_text.
    Solo correct, perché altrimenti diventa illeggibile.
    """
    import matplotlib.pyplot as plt

    sub = _filter_valid(df)
    sub = sub[sub["correct_baseline"] == True]
    sources_present = [s for s in ["MI", "UC_audio", "UC_text"]
                       if s in sub["feature_source"].unique()]
    if len(sources_present) < 2:
        print("  skip G-M7: meno di 2 source presenti")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for ax, metric, title, threshold in [
        (axes[0], "sufficiency_score",
         "G-M7a — Sufficiency (correct only)", ALPHA_SUFFICIENT),
        (axes[1], "necessity_score",
         "G-M7b — Necessity (correct only)", BETA_NECESSARY),
    ]:
        for src in sources_present:
            s = sub[(sub["feature_source"] == src) & (sub[metric].notnull())]
            if s.empty:
                continue
            agg = s.groupby("k")[metric].agg(["mean", "std", "count"]).reset_index()
            agg = agg.sort_values("k")
            se = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            ax.errorbar(
                agg["k"], agg["mean"], yerr=se,
                marker="o", markersize=6, lw=1.7,
                color=PALETTE.get(src, "#888780"),
                label=f"{src} (n={int(agg['count'].iloc[0])})",
                capsize=3,
            )
        ax.axhline(threshold, color="#888780", lw=0.8, ls="--", alpha=0.7,
                   label=f"threshold {threshold}")
        ax.axhline(0.0, color="#444441", lw=0.5, alpha=0.4)
        ax.set_xticks(sorted(sub["k"].unique()))
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)

    _save(fig, os.path.join(plots_dir, "G-M7_diagnostic_uc_vs_mi.png"), show)


# =============================================================================
# Summary console
# =============================================================================
def print_summary(df_curves, df_summary):
    print("\n" + "=" * 72)
    print("SUMMARY — EXPERIMENT E v4 (MI-primary, chance-normalized)")
    print("=" * 72)

    n_samples_total = df_curves["sample_id"].nunique()
    if "baseline_above_chance" in df_curves.columns:
        n_above = df_curves[df_curves["baseline_above_chance"] == True]["sample_id"].nunique()
        n_below = n_samples_total - n_above
        print(f"  Sample totali           : {n_samples_total}")
        print(f"  Sample above chance     : {n_above}")
        print(f"  Sample escluse (degeneri): {n_below}")
    else:
        print(f"  Sample totali           : {n_samples_total}")

    print(f"  Soglie:")
    print(f"    α sufficient = {ALPHA_SUFFICIENT}")
    print(f"    β necessary  = {BETA_NECESSARY}")

    sub = _filter_valid(df_curves)

    # Per ogni source, % sample con sufficiency≥α e necessity≥β a k canonical
    print()
    print("  Per source, statistiche a k=3 (canonical):")
    for src in ["MI", "MI_balanced", "UC_audio", "UC_text"]:
        s = sub[(sub["feature_source"] == src) & (sub["k"] == 3)]
        if s.empty:
            continue
        valid_suf = s["sufficiency_score"].dropna()
        valid_nec = s["necessity_score"].dropna()
        if len(valid_suf) == 0 and len(valid_nec) == 0:
            continue
        pct_suf = (valid_suf >= ALPHA_SUFFICIENT).mean() * 100 if len(valid_suf) > 0 else float("nan")
        pct_nec = (valid_nec >= BETA_NECESSARY).mean() * 100   if len(valid_nec) > 0 else float("nan")
        mean_suf = valid_suf.mean() if len(valid_suf) > 0 else float("nan")
        mean_nec = valid_nec.mean() if len(valid_nec) > 0 else float("nan")
        print(f"    {src:10s}: suf≥α = {pct_suf:5.1f}% (mean={mean_suf:+.3f}) | "
              f"nec≥β = {pct_nec:5.1f}% (mean={mean_nec:+.3f})")

    # Minimal k MI
    if df_summary is not None and not df_summary.empty:
        sub_s = df_summary[df_summary["feature_source"] == "MI"]
        if "baseline_above_chance" in sub_s.columns:
            sub_s = sub_s[sub_s["baseline_above_chance"] == True]
        if not sub_s.empty:
            print()
            print("  MI minimal_k (above-chance only):")
            msk = sub_s["minimal_sufficient_k"].dropna()
            mnk = sub_s["minimal_necessary_k"].dropna()
            total = len(sub_s)
            print(f"    minimal_sufficient_k:")
            print(f"      mediana = {msk.median() if len(msk) > 0 else 'n/a'}")
            print(f"      mai raggiunto: {sub_s['minimal_sufficient_k'].isna().sum()}/{total} "
                  f"({100*sub_s['minimal_sufficient_k'].isna().sum()/max(1,total):.1f}%)")
            print(f"    minimal_necessary_k:")
            print(f"      mediana = {mnk.median() if len(mnk) > 0 else 'n/a'}")
            print(f"      mai raggiunto: {sub_s['minimal_necessary_k'].isna().sum()}/{total} "
                  f"({100*sub_s['minimal_necessary_k'].isna().sum()/max(1,total):.1f}%)")

    # Accuracy modello
    acc = df_curves.drop_duplicates("sample_id")["correct_baseline"].mean() * 100
    print(f"\n  Accuracy baseline modello: {acc:.1f}%")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================
def run_all_plots(batch_dir: str, show: bool = False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    })
    warnings.filterwarnings("ignore", category=UserWarning)

    df_curves, df_summary = load_exp_e_data(batch_dir)
    print_summary(df_curves, df_summary)

    pdir = _plots_dir(batch_dir)
    print(f"\nSalvataggio plot in: {pdir}\n")

    print("G-M1 — MI Sufficiency curve...")
    plot_mi_sufficiency_curve(df_curves, pdir, show)

    print("G-M2 — MI Necessity curve...")
    plot_mi_necessity_curve(df_curves, pdir, show)

    print("G-M3 — MI Minimal-k distribution...")
    plot_minimal_k_distribution(df_summary, pdir, show)

    print("G-M4 — MI Curves per macro-family...")
    plot_mi_curves_by_macrofamily(df_curves, pdir, show)

    print("G-M5 — MI Minimal-k heatmap per categoria...")
    plot_minimal_k_heatmap_categories(df_summary, pdir, show)

    print("G-M6 — MI top-k composition...")
    plot_mi_topk_composition(df_curves, pdir, show)

    print("G-M7 — Diagnostico UC_audio vs UC_text vs MI...")
    plot_diagnostic_uc_vs_mi(df_curves, pdir, show)

    print("G-M8 — MI Balanced Sufficiency curve...")
    plot_mi_balanced_sufficiency_curve(df_curves, pdir, show)

    print("G-M9 — MI Balanced Necessity curve...")
    plot_mi_balanced_necessity_curve(df_curves, pdir, show)

    print("G-M10 — MI globale vs MI balanced...")
    plot_mi_vs_balanced_comparison(df_curves, pdir, show)

    print(f"\nFatto. PNG in: {pdir}")


def main():
    parser = argparse.ArgumentParser(description="Plot Esperimento E v4")
    parser.add_argument("--batch-dir", required=True, type=str)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run_all_plots(args.batch_dir, show=args.show)


if __name__ == "__main__":
    main()