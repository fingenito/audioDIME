"""
Esperimento D — Grounding temporale domanda-audio  [v2]
=======================================================
Script offline di analisi e visualizzazione.

Revisione v2:
- Classificazione a 3 livelli (pseudo_text / diffuse / localized)
  per marcatori strutturali e timbrici (non verificabili su estratti)
- Classificazione a 4 livelli (+ correct/wrong) solo per marcatori
  temporali assoluti (beginning, start, end, final, throughout)
- Marker dict costruito dal vocabolario reale delle 320 domande
- Segmenti non-uniformi gestiti tramite boundary reali dal JSON

Utilizzo:
    python exp_d_analysis_v2.py --batch-dir <path/to/batch_run_XX>
    python exp_d_analysis_v2.py --batch-dir <path> --output-dir <path>
"""

import os
import re
import json
import glob
import argparse
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# 1) MARKER DICTIONARY — costruito dal vocabolario reale delle 320 domande
# ==============================================================================
#
# Tre macro-categorie:
#
# A) TEMPORALE_ASSOLUTO: si riferisce a inizio/fine dell'ESTRATTO
#    -> classificazione correct/wrong verificabile
#    -> expected_window definita in posizione normalizzata [0,1] sull'estratto
#
# B) STRUTTURALE: si riferisce a sezioni della canzone (verse, chorus, bridge...)
#    -> NON verificabile senza annotazioni: estratto di 60-90s di posizione ignota
#    -> classificazione: pseudo_text / diffuse / localized
#
# C) TIMBRICO: si riferisce a strumenti/timbro
#    -> Pervasivi (drums, bass): atteso profilo diffuso, nessuna localizzazione
#    -> Semi-localizzati (vocals, sax): potrebbero avere picchi ma non verificabili
#    -> classificazione: pseudo_text / diffuse / localized
#
# Categoria speciale: THROUGHOUT -> per definizione semantica richiede profilo piatto
#                     -> se localized: grounding "wrong" (certamente sbagliato)

MARKER_DICT: Dict[str, Dict[str, Any]] = {

    # ── TEMPORALI ASSOLUTI (correct/wrong verificabile) ────────────────────────
    "beginning":  {"type": "temporal_absolute", "window": (0.00, 0.30)},
    "start":      {"type": "temporal_absolute", "window": (0.00, 0.30)},
    "starts":     {"type": "temporal_absolute", "window": (0.00, 0.30)},
    "begin":      {"type": "temporal_absolute", "window": (0.00, 0.30)},
    "first":      {"type": "temporal_absolute", "window": (0.00, 0.35)},
    "firstly":    {"type": "temporal_absolute", "window": (0.00, 0.35)},
    "intro":      {"type": "temporal_absolute", "window": (0.00, 0.30)},
    "end":        {"type": "temporal_absolute", "window": (0.70, 1.00)},
    "final":      {"type": "temporal_absolute", "window": (0.70, 1.00)},
    "last":       {"type": "temporal_absolute", "window": (0.70, 1.00)},
    "outro":      {"type": "temporal_absolute", "window": (0.70, 1.00)},

    # Speciale: throughout richiede piattezza -> se localized è certamente wrong
    "throughout": {"type": "temporal_diffuse", "window": None},

    # ── TEMPORALI RELATIVI/AMBIGUI (solo 3 livelli) ────────────────────────────
    "during":     {"type": "temporal_relative", "window": None},
    "when":       {"type": "temporal_relative", "window": None},
    "before":     {"type": "temporal_relative", "window": None},
    "after":      {"type": "temporal_relative", "window": None},
    "suddenly":   {"type": "temporal_relative", "window": None},

    # ── STRUTTURALI (solo 3 livelli — posizione non verificabile su estratti) ──
    "introduction":  {"type": "structural", "window": None},
    "introduces":    {"type": "structural", "window": None},
    "introduce":     {"type": "structural", "window": None},
    "introduced":    {"type": "structural", "window": None},
    "verse":         {"type": "structural", "window": None},
    "chorus":        {"type": "structural", "window": None},
    "bridge":        {"type": "structural", "window": None},
    "break":         {"type": "structural", "window": None},
    "section":       {"type": "structural", "window": None},
    "exposition":    {"type": "structural", "window": None},

    # ── TIMBRICI PERVASIVI (attesi diffusi — drums/bass sempre presenti) ────────
    "drums":        {"type": "timbral_pervasive", "window": None},
    "drum":         {"type": "timbral_pervasive", "window": None},
    "drummer":      {"type": "timbral_pervasive", "window": None},
    "kick":         {"type": "timbral_pervasive", "window": None},
    "snare":        {"type": "timbral_pervasive", "window": None},
    "hats":         {"type": "timbral_pervasive", "window": None},
    "cymbal":       {"type": "timbral_pervasive", "window": None},
    "brushes":      {"type": "timbral_pervasive", "window": None},
    "sticks":       {"type": "timbral_pervasive", "window": None},
    "shaker":       {"type": "timbral_pervasive", "window": None},
    "bass":         {"type": "timbral_pervasive", "window": None},
    "pedal":        {"type": "timbral_pervasive", "window": None},
    "percussion":   {"type": "timbral_pervasive", "window": None},

    # ── TIMBRICI SEMI-LOCALIZZATI (spesso in sezioni specifiche) ───────────────
    "guitar":       {"type": "timbral_local", "window": None},
    "guitars":      {"type": "timbral_local", "window": None},
    "guitarist":    {"type": "timbral_local", "window": None},
    "piano":        {"type": "timbral_local", "window": None},
    "keyboard":     {"type": "timbral_local", "window": None},
    "organ":        {"type": "timbral_local", "window": None},
    "vocal":        {"type": "timbral_local", "window": None},
    "vocals":       {"type": "timbral_local", "window": None},
    "voice":        {"type": "timbral_local", "window": None},
    "voices":       {"type": "timbral_local", "window": None},
    "singer":       {"type": "timbral_local", "window": None},
    "vocalist":     {"type": "timbral_local", "window": None},
    "choir":        {"type": "timbral_local", "window": None},
    "saxophone":    {"type": "timbral_local", "window": None},
    "sax":          {"type": "timbral_local", "window": None},
    "trombone":     {"type": "timbral_local", "window": None},
    "brass":        {"type": "timbral_local", "window": None},
    "synth":        {"type": "timbral_local", "window": None},
    "synthesizer":  {"type": "timbral_local", "window": None},
    "synths":       {"type": "timbral_local", "window": None},
    "ukulele":      {"type": "timbral_local", "window": None},
    "flutist":      {"type": "timbral_local", "window": None},
}

# Categorie incluse nell'analisi
TARGET_CATEGORIES = {
    "Instrumentation",
    "Sound Texture",
    "Metre and Rhythm",
    "Musical Texture",
    "Structure",
    "Performance",
}

# Soglie classificazione
THRESH_TEXT_IMPORTANCE    = 0.05   # mi2[w] minimo perché il token conti
THRESH_MI_FLATNESS        = 0.04   # range mi1_time sotto cui è "piatto"
THRESH_PEAK_CONCENTRATION = 0.22   # peak/sum sotto cui è "diffuso"

# ==============================================================================
# 2) PALETTE E STILE
# ==============================================================================

PATTERN_COLORS = {
    "pseudo_text": "#e74c3c",
    "diffuse":     "#f39c12",
    "localized":   "#3498db",
    "correct":     "#2ecc71",
    "wrong":       "#8e44ad",
    "ambiguous":   "#7f8c8d",
}

TYPE_COLORS = {
    "temporal_absolute": "#00bcd4",
    "temporal_diffuse":  "#ff9800",
    "temporal_relative": "#90a4ae",
    "structural":        "#ab47bc",
    "timbral_pervasive": "#ef5350",
    "timbral_local":     "#42a5f5",
}

TYPE_LABELS = {
    "temporal_absolute": "Temporale assoluto (verificabile)",
    "temporal_diffuse":  "Throughout (diffuso atteso)",
    "temporal_relative": "Temporale relativo (ambiguo)",
    "structural":        "Strutturale (non verificabile)",
    "timbral_pervasive": "Timbrico pervasivo",
    "timbral_local":     "Timbrico semi-localizzato",
}

PATTERN_LABELS = {
    "pseudo_text": "Pseudo-text",
    "diffuse":     "Diffuse",
    "localized":   "Localized",
    "correct":     "Correct",
    "wrong":       "Wrong",
    "ambiguous":   "Ambiguous",
}

# Ordine per plot
PATTERN_ORDER_3 = ["pseudo_text", "diffuse", "localized"]
PATTERN_ORDER_5 = ["pseudo_text", "diffuse", "localized", "correct", "wrong"]

BG = "#0f1117"
PANEL = "#1a1d27"
TEXT_C = "#e8eaf0"
GRID_C = "#2a2d3a"

def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    GRID_C,
        "axes.labelcolor":   TEXT_C,
        "xtick.color":       TEXT_C,
        "ytick.color":       TEXT_C,
        "text.color":        TEXT_C,
        "grid.color":        GRID_C,
        "grid.linewidth":    0.5,
        "axes.grid":         True,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "legend.framealpha": 0.3,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  GRID_C,
        "font.family":       "monospace",
        "savefig.facecolor": BG,
        "savefig.dpi":       180,
    })

# ==============================================================================
# 3) LOADING E FILTERING
# ==============================================================================

def load_exp_d_jsons(batch_dir: str) -> List[Dict]:
    per_sample_dir = os.path.join(batch_dir, "per_sample")
    if not os.path.isdir(per_sample_dir):
        raise FileNotFoundError(f"per_sample dir non trovata: {batch_dir}")
    files = sorted(glob.glob(os.path.join(per_sample_dir, "*_exp_d.json")))
    if not files:
        raise FileNotFoundError(f"Nessun *_exp_d.json in {per_sample_dir}")
    records = []
    for f in files:
        try:
            with open(f) as fh:
                records.append(json.load(fh))
        except Exception as e:
            print(f"  [WARN] skip {os.path.basename(f)}: {e}")
    print(f"[Load] {len(records)} campioni")
    return records

def filter_by_category(records: List[Dict], target: Optional[set] = None) -> List[Dict]:
    if target is None:
        target = TARGET_CATEGORIES
    out = [r for r in records if any(t.lower() in str(r.get("category","")).lower() for t in target)]
    print(f"[Filter] {len(out)} campioni nelle categorie target")
    return out

# ==============================================================================
# 4) MARKER DETECTION
# ==============================================================================

def detect_markers(prompt_words: List[str]) -> List[Dict]:
    found = []
    for idx, word in enumerate(prompt_words):
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if clean in MARKER_DICT:
            info = MARKER_DICT[clean]
            found.append({
                "token":      clean,
                "word_orig":  word,
                "word_idx":   idx,
                "marker_type": info["type"],
                "expected_window": info["window"],
            })
    return found

# ==============================================================================
# 5) SEGMENT POSITIONS (non-uniform, dai boundary reali)
# ==============================================================================

def segment_positions(seg_boundaries: List[Dict], n_seg: int) -> np.ndarray:
    """
    Calcola la posizione normalizzata [0,1] del CENTRO di ogni segmento,
    usando i boundary reali del JSON (non-uniformi in onset_guided,
    uniformi ma a durata variabile in fixed_length).
    Fallback a distribuzione uniforme se boundary mancanti.
    """
    if seg_boundaries and len(seg_boundaries) >= n_seg:
        starts, ends = [], []
        for d in seg_boundaries[:n_seg]:
            s = d.get("segment_start_sec")
            e = d.get("segment_end_sec")
            starts.append(float(s) if s is not None else float("nan"))
            ends.append(float(e) if e is not None else float("nan"))
        starts_a = np.array(starts)
        ends_a = np.array(ends)
        if not np.any(np.isnan(starts_a)) and not np.any(np.isnan(ends_a)):
            total = ends_a[-1]
            if total > 0:
                centers = (starts_a + ends_a) / 2.0
                return np.clip(centers / total, 0.0, 1.0)
    return np.linspace(1.0 / (2 * n_seg), 1.0 - 1.0 / (2 * n_seg), n_seg)

# ==============================================================================
# 6) GROUNDING SCORE E CLASSIFICAZIONE
# ==============================================================================

def classify_pattern(
    marker_type: str,
    expected_window: Optional[Tuple[float, float]],
    mi2_w: float,
    mi1: np.ndarray,
    peak_pos: float,
    peak_concentration: float,
) -> str:
    """
    Classificazione a livelli multipli:

    LIVELLO 1 — Pseudo-text (tutti i tipi):
      token conta nel testo MA mi1_time è piatto
      → il modello usa il linguaggio senza ascoltare la posizione

    LIVELLO 2 — Diffuse (tutti i tipi):
      mi1_time non è piatto MA il peak non è concentrato
      → ascolto globale, nessuna localizzazione

    LIVELLO 3 — Localized (strutturali, timbrici, temporali_relativi):
      peak chiaro MA posizione non verificabile
      → il modello localizza qualcosa nell'audio

    LIVELLO 3b — Correct/Wrong (solo temporal_absolute e temporal_diffuse):
      peak chiaro E posizione verificabile
      → grounding corretto o sbagliato rispetto all'estratto

    Nota: 'throughout' (temporal_diffuse) è special case:
      se localized → wrong (semanticamente richiede diffusione)
      se diffuse   → correct (coerente con la semantica)
    """
    mi1_range = float(np.max(mi1) - np.min(mi1))

    # LIVELLO 1: pseudo-text
    if abs(mi2_w) > THRESH_TEXT_IMPORTANCE and mi1_range < THRESH_MI_FLATNESS:
        return "pseudo_text"

    # LIVELLO 2: diffuse
    if peak_concentration < THRESH_PEAK_CONCENTRATION:
        # Caso speciale: throughout diffuso è CORRECT per definizione semantica
        if marker_type == "temporal_diffuse":
            return "correct"
        return "diffuse"

    # LIVELLO 3: il peak è chiaro
    if marker_type == "temporal_diffuse":
        # throughout localized è WRONG: dovrebbe essere piatto
        return "wrong"

    if marker_type == "temporal_absolute" and expected_window is not None:
        lo, hi = expected_window
        if lo <= peak_pos <= hi:
            return "correct"
        else:
            return "wrong"

    # strutturali, timbrici, temporali_relativi: non verificabile
    return "localized"


def compute_grounding(
    marker: Dict,
    mi2_global_word: List[float],
    mi1_time: List[float],
    seg_positions_arr: np.ndarray,
) -> Dict:
    w_idx = marker["word_idx"]
    mi2 = np.array(mi2_global_word, dtype=float)
    mi1 = np.array(mi1_time, dtype=float)

    mi2_w = float(mi2[w_idx]) if w_idx < len(mi2) else 0.0

    # Grounding vector approssimato (prodotto esterno dei marginali)
    # NOTA METODOLOGICA: non è la vera cross-modale, è proxy dei marginali DIME separati
    g_vec = mi2_w * mi1
    g_sum = float(np.sum(np.abs(g_vec)))
    g_norm = g_vec / (g_sum + 1e-12)

    peak_idx = int(np.argmax(np.abs(g_norm)))
    peak_val = float(g_norm[peak_idx])
    peak_pos = float(seg_positions_arr[peak_idx]) if peak_idx < len(seg_positions_arr) else 0.5
    peak_concentration = float(abs(peak_val))

    pattern = classify_pattern(
        marker_type=marker["marker_type"],
        expected_window=marker["expected_window"],
        mi2_w=mi2_w,
        mi1=mi1,
        peak_pos=peak_pos,
        peak_concentration=peak_concentration,
    )

    return {
        "token":             marker["token"],
        "word_orig":         marker["word_orig"],
        "word_idx":          w_idx,
        "marker_type":       marker["marker_type"],
        "expected_window":   marker["expected_window"],
        "mi2_weight":        mi2_w,
        "mi1_time_vec":      mi1.tolist(),
        "grounding_vec":     g_vec.tolist(),
        "grounding_vec_norm": g_norm.tolist(),
        "grounding_score":   float(np.max(np.abs(g_norm))),
        "peak_segment_idx":  peak_idx,
        "peak_segment_pos":  peak_pos,
        "peak_concentration": peak_concentration,
        "mi1_range":         float(np.max(mi1) - np.min(mi1)),
        "pattern":           pattern,
    }


def process_sample(record: Dict) -> Optional[Dict]:
    prompt_words = record.get("prompt_words", [])
    mi2 = record.get("mi2_global_word", [])
    mi1 = record.get("mi1_time", [])
    seg_bounds = record.get("segment_boundaries_sec", [])

    if not prompt_words or not mi2 or not mi1:
        return None

    markers = detect_markers(prompt_words)
    if not markers:
        return None

    n_seg = len(mi1)
    seg_pos = segment_positions(seg_bounds, n_seg)

    results = [compute_grounding(mk, mi2, mi1, seg_pos) for mk in markers]

    priority = {"pseudo_text": 0, "wrong": 1, "diffuse": 2, "localized": 3, "correct": 4, "ambiguous": 5}
    overall = sorted(results, key=lambda x: priority.get(x["pattern"], 99))[0]["pattern"]

    return {
        "sample_id":      record.get("sample_id", ""),
        "category":       record.get("category", ""),
        "difficulty":     record.get("difficulty", ""),
        "macro_family":   record.get("macro_family", ""),
        "prompt_words":   prompt_words,
        "mi2_global_word": mi2,
        "mi1_time":       mi1,
        "seg_positions":  seg_pos.tolist(),
        "stem_names":     record.get("stem_names", []),
        "mi1_stem_x_seg": record.get("mi1_stem_x_seg", []),
        "marker_results": results,
        "overall_pattern": overall,
        "n_markers":      len(results),
    }


def process_all(records: List[Dict]) -> List[Dict]:
    processed, skipped = [], 0
    for r in records:
        res = process_sample(r)
        if res is None:
            skipped += 1
        else:
            processed.append(res)
    print(f"[Process] {len(processed)} con marker | {skipped} senza marker (scartati)")
    return processed

# ==============================================================================
# 7) GRAFICI
# ==============================================================================

# ─── G-D1: Token × Segment heatmap ──────────────────────────────────────────

def plot_gd1(processed: List[Dict], output_path: str, max_samples: int = 12):
    """
    Heatmap grounding matrix (prodotto esterno dei marginali) per campioni
    selezionati. I token marcatori sono evidenziati.
    Un subplot per campione con barplot marginali.
    """
    _apply_style()

    # Selezione rappresentativa per pattern
    by_pattern: Dict[str, List[Dict]] = {}
    for s in processed:
        for mk in s["marker_results"]:
            p = mk["pattern"]
            if p not in by_pattern:
                by_pattern[p] = []
            if s not in by_pattern[p]:
                by_pattern[p].append(s)

    selected = []
    patterns_present = [p for p in PATTERN_ORDER_5 if p in by_pattern]
    per_pat = max(1, max_samples // max(len(patterns_present), 1))
    for pat in patterns_present:
        selected.extend(by_pattern[pat][:per_pat])
    selected = selected[:max_samples]

    if not selected:
        print("[G-D1] Nessun campione.")
        return

    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 7, nrows * 5.5), facecolor=BG)
    fig.suptitle("G-D1 — Grounding Matrix (prodotto esterno MI_text × MI_audio)",
                 color=TEXT_C, fontsize=13, fontweight="bold", y=0.99)

    for i, sample in enumerate(selected):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        words = sample["prompt_words"]
        mi2 = np.array(sample["mi2_global_word"])
        mi1 = np.array(sample["mi1_time"])
        n_seg = len(mi1)
        max_w = min(18, len(words))
        G = np.outer(mi2[:max_w], mi1)
        words_vis = words[:max_w]
        marker_idxs = {mk["word_idx"] for mk in sample["marker_results"] if mk["word_idx"] < max_w}

        vmax = max(float(np.max(np.abs(G))), 1e-9)
        im = ax.imshow(G, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")

        ax.set_xticks(range(n_seg))
        # Usa boundary reali in secondi per le etichette
        seg_pos = np.array(sample["seg_positions"])
        seg_bounds = sample.get("mi1_time", [])  # fallback
        # Prova a recuperare i secondi dal record originale
        ax.set_xticklabels([f"S{j}\n{seg_pos[j]:.0%}" for j in range(n_seg)], fontsize=6)
        ax.set_yticks(range(len(words_vis)))
        ylabels = [f"▶{w}" if i in marker_idxs else w for i, w in enumerate(words_vis)]
        ax.set_yticklabels(ylabels, fontsize=6.5)
        for wi in marker_idxs:
            ax.axhline(wi, color="#f1c40f", linewidth=1.0, alpha=0.5)

        pat = sample["overall_pattern"]
        ax.set_title(
            f"{sample['category'][:22]} | [{PATTERN_LABELS.get(pat, pat)}]",
            color=PATTERN_COLORS.get(pat, TEXT_C), fontsize=8, pad=3
        )
        plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[G-D1] {output_path}")


# ─── G-D2: Pattern distribution per categoria e tipo marcatore ───────────────

def plot_gd2(processed: List[Dict], output_path: str):
    """
    Stacked barplot 100%:
    - Asse sinistro: per categoria musicale
    - Asse destro: per tipo di marcatore (con distinzione verificabile/non)
    Evidenzia la separazione tra pattern verificabili (correct/wrong)
    e non verificabili (localized).
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle("G-D2 — Distribuzione pattern per categoria e tipo marcatore",
                 color=TEXT_C, fontsize=13, fontweight="bold")

    all_markers = [mk for s in processed for mk in s["marker_results"]]

    # -- Aggregazione per categoria --
    cat_data: Dict[str, Counter] = {}
    for s in processed:
        cat = s["category"]
        if cat not in cat_data:
            cat_data[cat] = Counter()
        for mk in s["marker_results"]:
            cat_data[cat][mk["pattern"]] += 1

    # -- Aggregazione per tipo marcatore --
    type_data: Dict[str, Counter] = {}
    for mk in all_markers:
        t = mk["marker_type"]
        if t not in type_data:
            type_data[t] = Counter()
        type_data[t][mk["pattern"]] += 1

    def _stacked(ax, data_dict, title, rotate=True):
        labels = sorted(data_dict.keys())
        totals = {l: sum(data_dict[l].values()) for l in labels}
        labels = [l for l in labels if totals[l] > 0]
        bottoms = np.zeros(len(labels))

        # Prima i pattern non-verificabili (3 livelli base)
        for pat in PATTERN_ORDER_5:
            vals = np.array([data_dict[l].get(pat, 0) / max(totals[l], 1) * 100
                             for l in labels])
            bars = ax.bar(
                range(len(labels)), vals, bottom=bottoms,
                color=PATTERN_COLORS[pat],
                label=PATTERN_LABELS[pat],
                edgecolor=BG, linewidth=0.4,
            )
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 9:
                    ax.text(xi, b + v / 2, f"{v:.0f}%",
                            ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold")
            bottoms += vals

        ax.set_ylim(0, 108)
        ax.set_ylabel("Proporzione (%)")
        ax.set_title(title, color=TEXT_C)
        ax.set_xticks(range(len(labels)))
        if rotate:
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
        else:
            # Usa TYPE_LABELS per etichette più leggibili
            xlabels = [TYPE_LABELS.get(l, l) for l in labels]
            ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=7)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", alpha=0)
        ax.set_axisbelow(True)

    _stacked(axes[0], cat_data, "Per categoria musicale")
    _stacked(axes[1], type_data, "Per tipo di marcatore", rotate=True)

    fig.text(0.5, 0.01,
             "* Localized = peak chiaro ma posizione non verificabile  |  "
             "Correct/Wrong = solo per marcatori temporali assoluti (beginning, end...)",
             ha="center", fontsize=7, color=TEXT_C, alpha=0.55)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[G-D2] {output_path}")


# ─── G-D3: Profili temporali medi per marcatore ──────────────────────────────

def plot_gd3(processed: List[Dict], output_path: str, min_n: int = 3):
    """
    Per ogni marcatore con abbastanza campioni:
    Profilo medio MI_audio normalizzato, separato per pattern.
    X = segmento (con percentuale della posizione normalizzata),
    Y = MI_audio medio normalizzato.
    Asse X usa le posizioni reali dei segmenti (non uniformi).
    """
    _apply_style()

    # Raggruppa profili per (token, pattern)
    profiles: Dict[str, Dict[str, List]] = {}
    positions_by_token: Dict[str, List[np.ndarray]] = {}

    for s in processed:
        mi1 = np.array(s["mi1_time"], dtype=float)
        m = float(mi1.max())
        mi1_norm = mi1 / m if m > 1e-9 else mi1
        seg_pos = np.array(s["seg_positions"])
        n_seg = len(mi1_norm)

        for mk in s["marker_results"]:
            tok = mk["token"]
            pat = mk["pattern"]
            if tok not in profiles:
                profiles[tok] = {}
                positions_by_token[tok] = []
            if pat not in profiles[tok]:
                profiles[tok][pat] = []
            profiles[tok][pat].append(mi1_norm[:n_seg])
            positions_by_token[tok].append(seg_pos[:n_seg])

    valid = sorted([
        tok for tok, pats in profiles.items()
        if sum(len(v) for v in pats.values()) >= min_n
    ])

    if not valid:
        print("[G-D3] Nessun marker con abbastanza campioni.")
        return

    ncols = min(3, len(valid))
    nrows = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 3.8),
                             facecolor=BG, squeeze=False)
    fig.suptitle("G-D3 — Profilo MI_audio medio per marcatore",
                 color=TEXT_C, fontsize=13, fontweight="bold")

    for mi_idx, tok in enumerate(valid):
        row, col = divmod(mi_idx, ncols)
        ax = axes[row][col]

        # Posizione x: media delle posizioni reali dei segmenti per questo token
        all_pos = positions_by_token[tok]
        n_seg_ref = len(all_pos[0]) if all_pos else 8
        x_mean = np.mean(
            [p[:n_seg_ref] if len(p) >= n_seg_ref
             else np.pad(p, (0, n_seg_ref - len(p)), constant_values=np.nan)
             for p in all_pos], axis=0
        )
        x_labels = [f"S{i}\n{x_mean[i]:.0%}" if np.isfinite(x_mean[i]) else f"S{i}"
                    for i in range(len(x_mean))]

        mk_info = MARKER_DICT.get(tok, {})
        mtype = mk_info.get("type", "")
        mwin = mk_info.get("window", None)
        type_col = TYPE_COLORS.get(mtype, TEXT_C)

        pattern_plot_order = ["pseudo_text", "diffuse", "localized", "correct", "wrong"]
        line_styles = {
            "pseudo_text": ("--", PATTERN_COLORS["pseudo_text"]),
            "diffuse":     (":",  PATTERN_COLORS["diffuse"]),
            "localized":   ("-",  PATTERN_COLORS["localized"]),
            "correct":     ("-",  PATTERN_COLORS["correct"]),
            "wrong":       ("-.",  PATTERN_COLORS["wrong"]),
        }

        for pat in pattern_plot_order:
            vecs = profiles[tok].get(pat, [])
            if not vecs:
                continue
            arr = np.stack([
                v[:n_seg_ref] if len(v) >= n_seg_ref
                else np.pad(v, (0, n_seg_ref - len(v)))
                for v in vecs
            ])
            mean_v = arr.mean(axis=0)
            std_v = arr.std(axis=0)
            ls, col_p = line_styles[pat]
            x = np.arange(len(mean_v))
            ax.plot(x, mean_v, color=col_p, linewidth=1.8, linestyle=ls,
                    label=f"{PATTERN_LABELS[pat]} (n={len(vecs)})")
            ax.fill_between(x, mean_v - std_v, mean_v + std_v,
                            color=col_p, alpha=0.12)

        # Finestra attesa (solo temporal_absolute)
        if mwin and mtype == "temporal_absolute":
            lo, hi = mwin
            ax.axvspan(lo * (n_seg_ref - 1), hi * (n_seg_ref - 1),
                       color="#f1c40f", alpha=0.08,
                       label=f"Atteso [{lo:.0%}–{hi:.0%}]")

        ax.set_title(f'"{tok}"  [{TYPE_LABELS.get(mtype, mtype)}]',
                     color=type_col, fontsize=8)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=6)
        ax.set_ylabel("MI_audio (norm.)", fontsize=7)
        ax.set_xlim(-0.5, n_seg_ref - 0.5)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=6.5, loc="upper right")

    for idx in range(len(valid), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[G-D3] {output_path}")


# ─── G-D4: Scatter MI_text vs grounding score ────────────────────────────────

def plot_gd4(processed: List[Dict], output_path: str):
    """
    Scatter:
    X = mi2_weight (importanza token nel testo)
    Y = grounding_score (peak normalizzato del grounding vector)
    Colore = pattern
    Forma = tipo marcatore
    Annotazione = token

    Le zone interpretative sono delineate dalle soglie di classificazione.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=BG)

    shape_map = {
        "temporal_absolute": "D",
        "temporal_diffuse":  "P",
        "temporal_relative": "v",
        "structural":        "s",
        "timbral_pervasive": "o",
        "timbral_local":     "^",
    }

    # Raccogli punti per pattern e tipo
    data_by_pat: Dict[str, Dict] = {}
    for s in processed:
        for mk in s["marker_results"]:
            p = mk["pattern"]
            t = mk["marker_type"]
            if p not in data_by_pat:
                data_by_pat[p] = {"x": [], "y": [], "tok": [], "shape": []}
            data_by_pat[p]["x"].append(mk["mi2_weight"])
            data_by_pat[p]["y"].append(mk["grounding_score"])
            data_by_pat[p]["tok"].append(mk["token"])
            data_by_pat[p]["shape"].append(shape_map.get(t, "o"))

    all_x, all_y = [], []
    for pat in PATTERN_ORDER_5:
        d = data_by_pat.get(pat, {})
        if not d.get("x"):
            continue
        col = PATTERN_COLORS[pat]
        xs, ys = d["x"], d["y"]
        all_x.extend(xs); all_y.extend(ys)

        for mshape in set(d["shape"]):
            idxs = [i for i, sh in enumerate(d["shape"]) if sh == mshape]
            ax.scatter(
                [xs[i] for i in idxs], [ys[i] for i in idxs],
                c=col, marker=mshape, s=70, alpha=0.78,
                edgecolors=BG, linewidths=0.5,
                label=PATTERN_LABELS[pat] if mshape == shape_map["timbral_pervasive"] else None,
                zorder=3,
            )

        # Annotazioni (evita sovraffollamento)
        seen: List[Tuple] = []
        for i, (xi, yi, lbl) in enumerate(zip(xs, ys, d["tok"])):
            too_close = any(abs(xi-px) < 0.025 and abs(yi-py) < 0.035 for px, py in seen)
            if not too_close:
                ax.annotate(lbl, (xi, yi), xytext=(4, 3), textcoords="offset points",
                            fontsize=6, color=col, alpha=0.85)
                seen.append((xi, yi))

    # Linee soglia
    ax.axvline(THRESH_TEXT_IMPORTANCE, color=GRID_C, lw=1, ls="--", alpha=0.6)
    ax.axhline(THRESH_PEAK_CONCENTRATION, color=GRID_C, lw=1, ls="--", alpha=0.6)

    # Etichette quadranti
    xr = max(all_x) * 1.05 if all_x else 1
    yr = max(all_y) * 1.05 if all_y else 1
    kw = dict(fontsize=7, color=TEXT_C, alpha=0.28, ha="center", va="center")
    ax.text(xr * 0.78, yr * 0.88, "Vero grounding\ninterattivo", **kw)
    ax.text(xr * 0.78, yr * 0.10, "Audio localizzato\nsenza driver testuale", **kw)
    ax.text(xr * 0.12, yr * 0.88, "Pseudo-grounding\ntestuale", **kw)
    ax.text(xr * 0.12, yr * 0.10, "Token irrilevante\nnon localizzato", **kw)

    # Legenda forme
    shape_handles = [
        mpatches.Patch(color=TEXT_C, label=f"◆ temporal_absolute"),
        mpatches.Patch(color=TEXT_C, label=f"★ temporal_diffuse (throughout)"),
        mpatches.Patch(color=TEXT_C, label=f"▼ temporal_relative"),
        mpatches.Patch(color=TEXT_C, label=f"■ structural"),
        mpatches.Patch(color=TEXT_C, label=f"● timbral_pervasive"),
        mpatches.Patch(color=TEXT_C, label=f"▲ timbral_local"),
    ]
    leg1 = ax.legend(loc="upper left", fontsize=7.5, title="Pattern")
    leg2 = ax.legend(handles=shape_handles, loc="lower right", fontsize=6.5, title="Tipo marcatore")
    ax.add_artist(leg1)

    ax.set_xlabel("MI_text weight [mi2_global_word(w)]", fontsize=10)
    ax.set_ylabel("Peak grounding score [max |G[w,s]|]", fontsize=10)
    ax.set_title("G-D4 — MI_text vs Peak Grounding Score",
                 color=TEXT_C, fontsize=12, fontweight="bold")

    fig.text(0.5, 0.005,
             "* G[w,s] ≈ mi2_global_word[w] × mi1_time[s] — prodotto esterno dei marginali DIME (limitazione metodologica)",
             ha="center", fontsize=6.5, color=TEXT_C, alpha=0.5)

    plt.tight_layout(rect=[0, 0.025, 1, 1])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[G-D4] {output_path}")


# ==============================================================================
# 8) REPORT
# ==============================================================================

def print_summary(processed: List[Dict]):
    all_markers = [mk for s in processed for mk in s["marker_results"]]
    print(f"\n{'='*65}")
    print("ESPERIMENTO D — REPORT SOMMARIO")
    print(f"{'='*65}")
    print(f"Campioni con marker : {len(processed)}")
    print(f"Token marcatori     : {len(all_markers)}")

    print("\n--- Distribuzione pattern globale ---")
    pc = Counter(mk["pattern"] for mk in all_markers)
    for p in PATTERN_ORDER_5:
        n = pc.get(p, 0)
        print(f"  {PATTERN_LABELS[p]:<20} {n:4d}  ({n/max(len(all_markers),1)*100:.1f}%)")

    print("\n--- Per tipo marcatore ---")
    by_type: Dict[str, List] = {}
    for mk in all_markers:
        t = mk["marker_type"]
        if t not in by_type: by_type[t] = []
        by_type[t].append(mk["pattern"])
    for t in sorted(by_type):
        pats = by_type[t]
        pc2 = Counter(pats)
        verif = "(verificabile)" if t == "temporal_absolute" else "(non verificabile)"
        print(f"  {TYPE_LABELS.get(t, t)} {verif}  n={len(pats)}")
        for p in PATTERN_ORDER_5:
            n = pc2.get(p, 0)
            if n > 0:
                print(f"    {PATTERN_LABELS[p]:<20} {n:3d}  ({n/len(pats)*100:.1f}%)")

    print("\n--- Per categoria ---")
    for cat in sorted({s["category"] for s in processed}):
        mks = [mk for s in processed if s["category"] == cat for mk in s["marker_results"]]
        if not mks: continue
        pc3 = Counter(mk["pattern"] for mk in mks)
        print(f"  {cat[:32]:<32} n={len(mks):3d}  "
              f"pseudo={pc3.get('pseudo_text',0)/len(mks)*100:.0f}%  "
              f"diffuse={pc3.get('diffuse',0)/len(mks)*100:.0f}%  "
              f"local={pc3.get('localized',0)/len(mks)*100:.0f}%  "
              f"correct={pc3.get('correct',0)/len(mks)*100:.0f}%  "
              f"wrong={pc3.get('wrong',0)/len(mks)*100:.0f}%")
    print(f"{'='*65}\n")

# ==============================================================================
# 9) ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir",  required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--category",   nargs="+", default=None)
    parser.add_argument("--max-gd1-samples", type=int, default=12)
    parser.add_argument("--min-marker-samples", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.batch_dir, "exp_d_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Output] {output_dir}")

    records  = load_exp_d_jsons(args.batch_dir)
    filtered = filter_by_category(records, set(args.category) if args.category else None)
    if not filtered:
        print("[ERROR] Nessun campione dopo il filtro.")
        return

    processed = process_all(filtered)
    if not processed:
        print("[ERROR] Nessun campione con marker.")
        return

    print_summary(processed)

    print("\n[Grafici]...")
    plot_gd1(processed, os.path.join(output_dir, "G-D1_token_segment_matrix.png"), args.max_gd1_samples)
    plot_gd2(processed, os.path.join(output_dir, "G-D2_pattern_distribution.png"))
    plot_gd3(processed, os.path.join(output_dir, "G-D3_temporal_profiles.png"), args.min_marker_samples)
    plot_gd4(processed, os.path.join(output_dir, "G-D4_mi_vs_grounding.png"))
    print(f"\n[Done] {output_dir}")

if __name__ == "__main__":
    main()