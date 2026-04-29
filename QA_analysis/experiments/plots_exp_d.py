"""
Esperimento D — Grounding temporale domanda-audio  [v3]
=======================================================
Analisi focalizzata su due verifiche metodologicamente solide:

  A) TEMPORAL: marker temporali assoluti (beginning, end, start, final...)
     → il peak segment dell'audio cade nella finestra attesa? correct/wrong

  B) TIMBRIC: marker timbrici (drums, bass, vocals, guitar, sax...)
     → lo stem htdemucs corrispondente ha MI più alto degli altri? correct/wrong
     Mappatura: drums/kick/snare/hats → "drums"
                bass                  → "bass"
                vocals/voice/singer   → "vocals"
                guitar/piano/sax/...  → "other"  (catch-all)

Classificazione unificata per entrambi i tipi:
  pseudo_text  : mi2[w] alto MA mi1 piatto (il token guida ma l'audio è sordo)
  diffuse      : mi1 ha varianza ma nessun peak chiaro
  correct      : peak/stem dominante nella zona/stem attesa
  wrong        : peak/stem dominante nella zona/stem sbagliata

Utilizzo:
    python exp_d_analysis_v3.py --batch-dir <path/to/batch_run_XX>
    python exp_d_analysis_v3.py --batch-dir <path> --output-dir <path>
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
# 1) CONFIGURAZIONE MARKER
# ==============================================================================
#
# Due soli tipi analizzati:
#
#  temporal_absolute: verifica posizione del peak segment nell'estratto
#  timbral_*:         verifica dominanza dello stem htdemucs corrispondente
#
# Esclusi (non verificabili su estratti senza annotazioni):
#  structural (chorus, verse, bridge...)
#  temporal_relative (during, when, before...)

MARKER_DICT: Dict[str, Dict[str, Any]] = {

    # ── TEMPORALI ASSOLUTI ────────────────────────────────────────────────────
    # Finestra = posizione normalizzata [0,1] attesa del peak nell'estratto
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

    # Speciale: per definizione semantica richiede profilo piatto
    # → se diffuse = correct (il modello ha capito "ovunque")
    # → se localized = wrong (il modello punta a un posto specifico)
    "throughout": {"type": "temporal_diffuse",  "window": None},

    # ── TIMBRICI → stem htdemucs atteso ───────────────────────────────────────
    # "expected_stem" è il nome dello stem che DEVE essere dominante
    "drums":      {"type": "timbral", "expected_stem": "drums"},
    "drum":       {"type": "timbral", "expected_stem": "drums"},
    "drummer":    {"type": "timbral", "expected_stem": "drums"},
    "kick":       {"type": "timbral", "expected_stem": "drums"},
    "snare":      {"type": "timbral", "expected_stem": "drums"},
    "hats":       {"type": "timbral", "expected_stem": "drums"},
    "cymbal":     {"type": "timbral", "expected_stem": "drums"},
    "brushes":    {"type": "timbral", "expected_stem": "drums"},
    "sticks":     {"type": "timbral", "expected_stem": "drums"},
    "shaker":     {"type": "timbral", "expected_stem": "drums"},
    "percussion": {"type": "timbral", "expected_stem": "drums"},

    "bass":       {"type": "timbral", "expected_stem": "bass"},
    "pedal":      {"type": "timbral", "expected_stem": "bass"},

    "vocal":      {"type": "timbral", "expected_stem": "vocals"},
    "vocals":     {"type": "timbral", "expected_stem": "vocals"},
    "voice":      {"type": "timbral", "expected_stem": "vocals"},
    "voices":     {"type": "timbral", "expected_stem": "vocals"},
    "singer":     {"type": "timbral", "expected_stem": "vocals"},
    "vocalist":   {"type": "timbral", "expected_stem": "vocals"},
    "choir":      {"type": "timbral", "expected_stem": "vocals"},
    "singing":    {"type": "timbral", "expected_stem": "vocals"},

    # Tutti nel catch-all "other" di htdemucs
    "guitar":     {"type": "timbral", "expected_stem": "other"},
    "guitars":    {"type": "timbral", "expected_stem": "other"},
    "guitarist":  {"type": "timbral", "expected_stem": "other"},
    "piano":      {"type": "timbral", "expected_stem": "other"},
    "keyboard":   {"type": "timbral", "expected_stem": "other"},
    "organ":      {"type": "timbral", "expected_stem": "other"},
    "saxophone":  {"type": "timbral", "expected_stem": "other"},
    "sax":        {"type": "timbral", "expected_stem": "other"},
    "trombone":   {"type": "timbral", "expected_stem": "other"},
    "brass":      {"type": "timbral", "expected_stem": "other"},
    "synth":      {"type": "timbral", "expected_stem": "other"},
    "synthesizer":{"type": "timbral", "expected_stem": "other"},
    "synths":     {"type": "timbral", "expected_stem": "other"},
    "ukulele":    {"type": "timbral", "expected_stem": "other"},
    "flutist":    {"type": "timbral", "expected_stem": "other"},
    "strings":    {"type": "timbral", "expected_stem": "other"},
}

# Categorie incluse
TARGET_CATEGORIES = {
    "Instrumentation",
    "Sound Texture",
    "Metre and Rhythm",
    "Musical Texture",
    "Structure",
    "Performance",
}

# Soglie
THRESH_TEXT_IMPORTANCE    = 0.05   # mi2[w] minimo perché il token sia "attivo"
THRESH_MI_FLATNESS        = 0.04   # range mi1_time sotto cui è "piatto" → pseudo_text
THRESH_PEAK_CONCENTRATION = 0.22   # peak/sum sotto cui il grounding temporale è "diffuse"
THRESH_STEM_DOMINANCE     = 0.10   # differenza assoluta tra stem dominante e secondo
                                   # sotto cui il grounding timbrico è "diffuse"

# ==============================================================================
# 2) PALETTE E STILE
# ==============================================================================

PATTERN_COLORS = {
    "pseudo_text": "#e74c3c",
    "diffuse":     "#f39c12",
    "correct":     "#2ecc71",
    "wrong":       "#8e44ad",
}
PATTERN_LABELS = {
    "pseudo_text": "Pseudo-text",
    "diffuse":     "Diffuse",
    "correct":     "Correct",
    "wrong":       "Wrong",
}
PATTERN_ORDER = ["pseudo_text", "diffuse", "correct", "wrong"]

TYPE_COLORS = {
    "temporal_absolute": "#00bcd4",
    "temporal_diffuse":  "#ff9800",
    "timbral":           "#ab47bc",
}
TYPE_LABELS = {
    "temporal_absolute": "Temporale assoluto",
    "temporal_diffuse":  "Throughout",
    "timbral":           "Timbrico",
}
STEM_COLORS = {
    "drums":  "#ef5350",
    "bass":   "#42a5f5",
    "other":  "#ab47bc",
    "vocals": "#2ecc71",
}

BG    = "#0f1117"
PANEL = "#1a1d27"
TEXT_C = "#e8eaf0"
GRID_C = "#2a2d3a"

def _style():
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
        "font.family":       "monospace",
        "savefig.facecolor": BG,
        "savefig.dpi":       180,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  GRID_C,
        "legend.framealpha": 0.3,
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
    out = [r for r in records
           if any(t.lower() in str(r.get("category", "")).lower() for t in target)]
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
                "token":          clean,
                "word_orig":      word,
                "word_idx":       idx,
                "marker_type":    info["type"],
                "expected_window": info.get("window"),
                "expected_stem":  info.get("expected_stem"),
            })
    return found

# ==============================================================================
# 5) SEGMENT POSITIONS
# ==============================================================================

def segment_positions(seg_boundaries: List[Dict], n_seg: int) -> np.ndarray:
    """
    Posizione normalizzata [0,1] del centro di ogni segmento.
    Usa i boundary reali dal JSON; fallback uniforme se mancanti.
    """
    if seg_boundaries and len(seg_boundaries) >= n_seg:
        starts, ends = [], []
        for d in seg_boundaries[:n_seg]:
            s = d.get("segment_start_sec")
            e = d.get("segment_end_sec")
            starts.append(float(s) if s is not None else float("nan"))
            ends.append(float(e) if e is not None else float("nan"))
        s_a, e_a = np.array(starts), np.array(ends)
        if not np.any(np.isnan(s_a)) and not np.any(np.isnan(e_a)) and e_a[-1] > 0:
            return np.clip((s_a + e_a) / 2.0 / e_a[-1], 0.0, 1.0)
    return np.linspace(1.0 / (2 * n_seg), 1.0 - 1.0 / (2 * n_seg), n_seg)

# ==============================================================================
# 6) CLASSIFICAZIONE
# ==============================================================================

def classify_temporal(
    mi2_w: float,
    mi1: np.ndarray,
    peak_pos: float,
    peak_concentration: float,
    marker_type: str,
    expected_window: Optional[Tuple[float, float]],
) -> str:
    """
    Classifica grounding temporale.
    Ritorna: pseudo_text | diffuse | correct | wrong
    """
    mi1_range = float(np.max(mi1) - np.min(mi1))

    # 1. Pseudo-text: il token conta nel testo MA l'audio è piatto
    if abs(mi2_w) > THRESH_TEXT_IMPORTANCE and mi1_range < THRESH_MI_FLATNESS:
        return "pseudo_text"

    # 2. Diffuse: nessun peak dominante
    if peak_concentration < THRESH_PEAK_CONCENTRATION:
        # Speciale: throughout diffuso è semanticamente CORRECT
        if marker_type == "temporal_diffuse":
            return "correct"
        return "diffuse"

    # 3. Localized: peak chiaro
    if marker_type == "temporal_diffuse":
        # throughout NON dovrebbe avere un peak → wrong
        return "wrong"

    if expected_window is not None:
        lo, hi = expected_window
        return "correct" if lo <= peak_pos <= hi else "wrong"

    return "diffuse"  # fallback


def classify_timbral(
    mi2_w: float,
    mi1: np.ndarray,
    mi1_stem: List[float],
    stem_names: List[str],
    expected_stem: str,
) -> Tuple[str, str, float, float]:
    """
    Classifica grounding timbrico.

    Verifica se lo stem atteso ha MI più alto di tutti gli altri.
    Ritorna: (pattern, dominant_stem, mi_expected, mi_dominant)

    Logica:
      pseudo_text : mi2[w] alto MA tutti gli stem hanno MI quasi uguale (piatto)
      diffuse     : MI varia tra stem MA il margine tra primo e secondo è troppo piccolo
      correct     : lo stem atteso è il dominante con margine sufficiente
      wrong       : uno stem diverso da quello atteso è dominante con margine sufficiente
    """
    mi1_range = float(np.max(mi1) - np.min(mi1))

    # Pseudo-text: il token conta MA l'audio (complessivo) è insensibile
    if abs(mi2_w) > THRESH_TEXT_IMPORTANCE and mi1_range < THRESH_MI_FLATNESS:
        return "pseudo_text", "none", 0.0, 0.0

    stem_arr = np.array(mi1_stem, dtype=float)
    if len(stem_arr) == 0:
        return "diffuse", "none", 0.0, 0.0

    # Normalizza per confronto relativo
    stem_abs = np.abs(stem_arr)
    total = float(np.sum(stem_abs))
    if total < 1e-9:
        return "diffuse", "none", 0.0, 0.0

    # Stem dominante (max MI assoluto)
    dominant_idx = int(np.argmax(stem_abs))
    dominant_stem = stem_names[dominant_idx] if dominant_idx < len(stem_names) else "unknown"
    mi_dominant = float(stem_abs[dominant_idx])

    # MI dello stem atteso
    expected_idx = next((i for i, n in enumerate(stem_names) if n == expected_stem), None)
    mi_expected = float(stem_abs[expected_idx]) if expected_idx is not None else 0.0

    # Margine tra primo e secondo
    sorted_vals = np.sort(stem_abs)[::-1]
    margin = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else float(sorted_vals[0])

    # Diffuse: margine troppo piccolo per affermare dominanza
    if margin < THRESH_STEM_DOMINANCE:
        return "diffuse", dominant_stem, mi_expected, mi_dominant

    # Correct/Wrong: il dominante è quello atteso?
    pattern = "correct" if dominant_stem == expected_stem else "wrong"
    return pattern, dominant_stem, mi_expected, mi_dominant


def compute_grounding(
    marker: Dict,
    mi2_global_word: List[float],
    mi1_time: List[float],
    mi1_stem: List[float],
    stem_names: List[str],
    seg_positions_arr: np.ndarray,
) -> Dict:
    w_idx = marker["word_idx"]
    mi2 = np.array(mi2_global_word, dtype=float)
    mi1 = np.array(mi1_time, dtype=float)
    mi2_w = float(mi2[w_idx]) if w_idx < len(mi2) else 0.0

    # Grounding vector temporale (prodotto esterno dei marginali)
    # NOTA METODOLOGICA: proxy dei marginali DIME separati, non vera cross-modale
    g_vec = mi2_w * mi1
    g_sum = float(np.sum(np.abs(g_vec)))
    g_norm = g_vec / (g_sum + 1e-12)

    peak_idx = int(np.argmax(np.abs(g_norm)))
    peak_val = float(g_norm[peak_idx])
    peak_pos = float(seg_positions_arr[peak_idx]) if peak_idx < len(seg_positions_arr) else 0.5
    peak_concentration = float(abs(peak_val))

    mtype = marker["marker_type"]

    if mtype == "timbral":
        pattern, dominant_stem, mi_expected, mi_dominant = classify_timbral(
            mi2_w=mi2_w,
            mi1=mi1,
            mi1_stem=mi1_stem,
            stem_names=stem_names,
            expected_stem=marker["expected_stem"],
        )
        return {
            "token":            marker["token"],
            "word_orig":        marker["word_orig"],
            "word_idx":         w_idx,
            "marker_type":      mtype,
            "expected_stem":    marker["expected_stem"],
            "mi2_weight":       mi2_w,
            "mi1_time_vec":     mi1.tolist(),
            "mi1_stem":         list(mi1_stem),
            "stem_names":       list(stem_names),
            "dominant_stem":    dominant_stem,
            "mi_expected_stem": mi_expected,
            "mi_dominant_stem": mi_dominant,
            # Temporale comunque calcolato per G-D1
            "grounding_vec":        g_vec.tolist(),
            "grounding_vec_norm":   g_norm.tolist(),
            "grounding_score":      float(np.max(np.abs(g_norm))),
            "peak_segment_idx":     peak_idx,
            "peak_segment_pos":     peak_pos,
            "pattern":              pattern,
        }

    else:  # temporal_absolute, temporal_diffuse
        pattern = classify_temporal(
            mi2_w=mi2_w,
            mi1=mi1,
            peak_pos=peak_pos,
            peak_concentration=peak_concentration,
            marker_type=mtype,
            expected_window=marker["expected_window"],
        )
        return {
            "token":            marker["token"],
            "word_orig":        marker["word_orig"],
            "word_idx":         w_idx,
            "marker_type":      mtype,
            "expected_window":  marker["expected_window"],
            "mi2_weight":       mi2_w,
            "mi1_time_vec":     mi1.tolist(),
            "mi1_stem":         list(mi1_stem),
            "stem_names":       list(stem_names),
            "grounding_vec":        g_vec.tolist(),
            "grounding_vec_norm":   g_norm.tolist(),
            "grounding_score":      float(np.max(np.abs(g_norm))),
            "peak_segment_idx":     peak_idx,
            "peak_segment_pos":     peak_pos,
            "peak_concentration":   peak_concentration,
            "mi1_range":            float(np.max(mi1) - np.min(mi1)),
            "pattern":              pattern,
        }

# ==============================================================================
# 7) PROCESSING
# ==============================================================================

def process_sample(record: Dict) -> Optional[Dict]:
    prompt_words  = record.get("prompt_words", [])
    mi2           = record.get("mi2_global_word", [])
    mi1           = record.get("mi1_time", [])
    mi1_stem      = record.get("mi1_stem", [])
    stem_names    = record.get("stem_names", ["drums", "bass", "other", "vocals"])
    seg_bounds    = record.get("segment_boundaries_sec", [])

    if not prompt_words or not mi2 or not mi1:
        return None

    markers = detect_markers(prompt_words)
    if not markers:
        return None

    n_seg = len(mi1)
    seg_pos = segment_positions(seg_bounds, n_seg)

    results = []
    for mk in markers:
        r = compute_grounding(mk, mi2, mi1, mi1_stem, stem_names, seg_pos)
        results.append(r)

    priority = {"pseudo_text": 0, "wrong": 1, "diffuse": 2, "correct": 3}
    overall = sorted(results, key=lambda x: priority.get(x["pattern"], 99))[0]["pattern"]

    return {
        "sample_id":       record.get("sample_id", ""),
        "category":        record.get("category", ""),
        "difficulty":      record.get("difficulty", ""),
        "macro_family":    record.get("macro_family", ""),
        "prompt_words":    prompt_words,
        "mi2_global_word": mi2,
        "mi1_time":        mi1,
        "mi1_stem":        mi1_stem,
        "stem_names":      stem_names,
        "seg_positions":   seg_pos.tolist(),
        "mi1_stem_x_seg":  record.get("mi1_stem_x_seg", []),
        "marker_results":  results,
        "overall_pattern": overall,
        "n_markers":       len(results),
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
# 8) GRAFICI
# ==============================================================================

# ── G-D1: Heatmap grounding + barplot stem per campione ──────────────────────

def plot_gd1(processed: List[Dict], output_path: str, max_samples: int = 12):
    """
    G-D1 — Due pannelli affiancati per ogni campione selezionato:
    SINISTRA: heatmap grounding temporale (token × segmento)
    DESTRA: barplot MI per stem (con evidenziazione stem atteso)
    I token marcatori sono evidenziati in giallo.
    """
    _style()

    by_pattern: Dict[str, List[Dict]] = {}
    for s in processed:
        for mk in s["marker_results"]:
            p = mk["pattern"]
            if p not in by_pattern:
                by_pattern[p] = []
            if s not in by_pattern[p]:
                by_pattern[p].append(s)

    selected = []
    per_pat = max(1, max_samples // max(len(by_pattern), 1))
    for pat in PATTERN_ORDER:
        selected.extend(by_pattern.get(pat, [])[:per_pat])
    selected = selected[:max_samples]

    if not selected:
        print("[G-D1] Nessun campione.")
        return

    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 9, nrows * 5.5), facecolor=BG)
    fig.suptitle(
        "G-D1 — Grounding temporale (heatmap) + Grounding timbrico (stem MI)",
        color=TEXT_C, fontsize=13, fontweight="bold", y=0.99,
    )

    for i, sample in enumerate(selected):
        # GridSpec: 2 colonne per campione (heatmap | stembar)
        gs_outer = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.35)
        gs_inner = gs_outer[i // ncols, i % ncols].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.08)
        ax_heat = fig.add_subplot(gs_inner[0])
        ax_stem = fig.add_subplot(gs_inner[1])

        words   = sample["prompt_words"]
        mi2     = np.array(sample["mi2_global_word"])
        mi1     = np.array(sample["mi1_time"])
        n_seg   = len(mi1)
        max_w   = min(18, len(words))
        G       = np.outer(mi2[:max_w], mi1)
        seg_pos = np.array(sample["seg_positions"])

        marker_idxs = {mk["word_idx"] for mk in sample["marker_results"] if mk["word_idx"] < max_w}

        vmax = max(float(np.max(np.abs(G))), 1e-9)
        im = ax_heat.imshow(G, aspect="auto", cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax, origin="upper")
        ax_heat.set_xticks(range(n_seg))
        ax_heat.set_xticklabels([f"S{j}\n{seg_pos[j]:.0%}" for j in range(n_seg)], fontsize=5.5)
        ax_heat.set_yticks(range(max_w))
        ylabels = [f"▶ {words[wi]}" if wi in marker_idxs else words[wi]
                   for wi in range(max_w)]
        ax_heat.set_yticklabels(ylabels, fontsize=6)
        for wi in marker_idxs:
            ax_heat.axhline(wi, color="#f1c40f", linewidth=1.0, alpha=0.5)

        pat = sample["overall_pattern"]
        ax_heat.set_title(
            f"{sample['category'][:20]} | [{PATTERN_LABELS.get(pat, pat)}]",
            color=PATTERN_COLORS.get(pat, TEXT_C), fontsize=7.5, pad=3,
        )
        plt.colorbar(im, ax=ax_heat, shrink=0.55, pad=0.02)

        # Barplot stem MI
        stem_names = sample.get("stem_names", ["drums", "bass", "other", "vocals"])
        mi1_stem   = np.abs(np.array(sample.get("mi1_stem", [0.0] * len(stem_names))))

        # Individua stem attesi dai marker timbrici di questo campione
        expected_stems = {
            mk["expected_stem"]
            for mk in sample["marker_results"]
            if mk["marker_type"] == "timbral" and mk.get("expected_stem")
        }

        bar_colors = []
        for sn in stem_names:
            if sn in expected_stems:
                bar_colors.append("#f1c40f")          # giallo = atteso
            else:
                bar_colors.append(STEM_COLORS.get(sn, "#aaaaaa"))

        ax_stem.barh(range(len(stem_names)), mi1_stem[:len(stem_names)],
                     color=bar_colors, edgecolor=BG, linewidth=0.5)
        ax_stem.set_yticks(range(len(stem_names)))
        ax_stem.set_yticklabels(stem_names, fontsize=6.5)
        ax_stem.set_xlabel("MI stem", fontsize=6)
        ax_stem.set_title("Stems", color=TEXT_C, fontsize=7)
        ax_stem.tick_params(axis="x", labelsize=5.5)

        # Segna il dominante con asterisco
        if len(mi1_stem) > 0:
            dom_idx = int(np.argmax(mi1_stem))
            ax_stem.annotate(
                "★", xy=(mi1_stem[dom_idx], dom_idx),
                fontsize=9, color="#f1c40f", ha="left", va="center",
            )

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[G-D1] {output_path}")


# ── G-D2: Pattern distribution separata per tipo marker ──────────────────────

def plot_gd2(processed: List[Dict], output_path: str):
    """
    G-D2 — Due barplot 100% stacked affiancati:
    SINISTRA: distribuzione pattern per i marcatori TEMPORALI (per token)
    DESTRA:   distribuzione pattern per i marcatori TIMBRICI (per stem atteso)
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle(
        "G-D2 — Distribuzione pattern: temporale (per token) | timbrico (per stem atteso)",
        color=TEXT_C, fontsize=12, fontweight="bold",
    )

    all_markers = [mk for s in processed for mk in s["marker_results"]]

    # -- Temporali: raggruppati per token --
    temp_data: Dict[str, Counter] = {}
    for mk in all_markers:
        if mk["marker_type"] in ("temporal_absolute", "temporal_diffuse"):
            tok = mk["token"]
            if tok not in temp_data:
                temp_data[tok] = Counter()
            temp_data[tok][mk["pattern"]] += 1

    # -- Timbrici: raggruppati per stem atteso --
    timbral_data: Dict[str, Counter] = {}
    for mk in all_markers:
        if mk["marker_type"] == "timbral":
            stem = mk.get("expected_stem", "unknown")
            if stem not in timbral_data:
                timbral_data[stem] = Counter()
            timbral_data[stem][mk["pattern"]] += 1

    def _stacked(ax, data_dict, title, xlabel=""):
        labels = sorted(data_dict.keys(), key=lambda l: -sum(data_dict[l].values()))
        totals = {l: sum(data_dict[l].values()) for l in labels}
        labels = [l for l in labels if totals[l] > 0]
        if not labels:
            ax.set_title(title, color=TEXT_C)
            ax.text(0.5, 0.5, "nessun dato", ha="center", va="center",
                    transform=ax.transAxes, color=TEXT_C, alpha=0.5)
            return

        bottoms = np.zeros(len(labels))
        for pat in PATTERN_ORDER:
            vals = np.array([
                data_dict[l].get(pat, 0) / max(totals[l], 1) * 100
                for l in labels
            ])
            ax.bar(range(len(labels)), vals, bottom=bottoms,
                   color=PATTERN_COLORS[pat], label=PATTERN_LABELS[pat],
                   edgecolor=BG, linewidth=0.4)
            for xi, (v, b) in enumerate(zip(vals, bottoms)):
                if v > 9:
                    ax.text(xi, b + v / 2, f"{v:.0f}%",
                            ha="center", va="center", fontsize=6.5,
                            color="white", fontweight="bold")
            bottoms += vals

        # Annotazione n sotto ogni barra
        for xi, l in enumerate(labels):
            ax.text(xi, -6, f"n={totals[l]}", ha="center", va="top",
                    fontsize=6, color=TEXT_C, alpha=0.6)

        ax.set_ylim(-10, 108)
        ax.set_ylabel("Proporzione (%)")
        ax.set_title(title, color=TEXT_C)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.grid(axis="x", alpha=0)
        ax.set_axisbelow(True)

    _stacked(axes[0], temp_data, "Marcatori TEMPORALI assoluti — per token",
             xlabel="Token")
    _stacked(axes[1], timbral_data, "Marcatori TIMBRICI — per stem atteso (htdemucs)",
             xlabel="Stem atteso")

    fig.text(
        0.5, 0.01,
        "Temporal: correct = peak nel segmento atteso  |  "
        "Timbral: correct = stem atteso è dominante con margine > soglia  |  "
        "* prodotto esterno dei marginali DIME",
        ha="center", fontsize=6.5, color=TEXT_C, alpha=0.5,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[G-D2] {output_path}")


# ── G-D3: Profili temporali (solo temporal) + Stem MI profiles (solo timbral) ─

def plot_gd3(processed: List[Dict], output_path: str, min_n: int = 3):
    """
    G-D3 — Due sezioni verticali:
    TOP:    profili temporali MI_audio medi per token temporale
            (come nella v2, asse X = posizione normalizzata reale)
    BOTTOM: stem MI profile per token timbrico
            (barplot a raggruppamento: corretto vs sbagliato)
    """
    _style()

    all_markers = [mk for s in processed for mk in s["marker_results"]]

    # --- Raccolta profili temporali ---
    temp_profiles: Dict[str, Dict[str, List]] = {}
    temp_positions: Dict[str, List] = {}
    for s in processed:
        mi1 = np.array(s["mi1_time"], dtype=float)
        m   = float(mi1.max())
        mi1_norm = mi1 / m if m > 1e-9 else mi1
        seg_pos = np.array(s["seg_positions"])
        for mk in s["marker_results"]:
            if mk["marker_type"] not in ("temporal_absolute", "temporal_diffuse"):
                continue
            tok = mk["token"]
            pat = mk["pattern"]
            if tok not in temp_profiles:
                temp_profiles[tok] = {}
                temp_positions[tok] = []
            if pat not in temp_profiles[tok]:
                temp_profiles[tok][pat] = []
            temp_profiles[tok][pat].append(mi1_norm[:len(mi1_norm)])
            temp_positions[tok].append(seg_pos[:len(mi1_norm)])

    # --- Raccolta profili timbrici ---
    # Per ogni token timbrico: mi1_stem per pattern correct vs wrong
    timbral_stem_profiles: Dict[str, Dict[str, List]] = {}
    timbral_stem_names: Dict[str, List[str]] = {}
    for s in processed:
        stems = s.get("stem_names", ["drums", "bass", "other", "vocals"])
        mi1_stem = np.abs(np.array(s.get("mi1_stem", [0.0] * len(stems))))
        # normalizza per stem
        total = float(mi1_stem.sum())
        mi1_stem_norm = mi1_stem / total if total > 1e-9 else mi1_stem
        for mk in s["marker_results"]:
            if mk["marker_type"] != "timbral":
                continue
            tok = mk["token"]
            pat = mk["pattern"]
            if tok not in timbral_stem_profiles:
                timbral_stem_profiles[tok] = {}
                timbral_stem_names[tok] = stems
            if pat not in timbral_stem_profiles[tok]:
                timbral_stem_profiles[tok][pat] = []
            timbral_stem_profiles[tok][pat].append(mi1_stem_norm.tolist())

    # --- Adattamento min_n ---
    n_total = len(processed)
    eff_min_n = min_n
    if n_total < 30 and min_n > 1:
        eff_min_n = max(1, min_n - 1)

    valid_temp = sorted([
        tok for tok, pats in temp_profiles.items()
        if sum(len(v) for v in pats.values()) >= eff_min_n
    ])
    valid_timbral = sorted([
        tok for tok, pats in timbral_stem_profiles.items()
        if sum(len(v) for v in pats.values()) >= eff_min_n
    ])

    n_top = len(valid_temp)
    n_bot = len(valid_timbral)

    if n_top == 0 and n_bot == 0:
        print("[G-D3] Nessun marker con abbastanza campioni.")
        return

    ncols = max(max(n_top, n_bot), 1)
    fig, axes = plt.subplots(
        2, ncols,
        figsize=(max(ncols * 4.5, 8), 9),
        facecolor=BG, squeeze=False,
    )
    fig.suptitle("G-D3 — Profili MI: temporale (top) | timbrico/stem (bottom)",
                 color=TEXT_C, fontsize=12, fontweight="bold")

    line_styles = {
        "pseudo_text": ("--", PATTERN_COLORS["pseudo_text"]),
        "diffuse":     (":",  PATTERN_COLORS["diffuse"]),
        "correct":     ("-",  PATTERN_COLORS["correct"]),
        "wrong":       ("-.", PATTERN_COLORS["wrong"]),
    }

    # TOP: profili temporali
    for mi_idx in range(ncols):
        ax = axes[0][mi_idx]
        ax.set_facecolor(PANEL)
        if mi_idx >= n_top:
            ax.set_visible(False)
            continue
        tok = valid_temp[mi_idx]
        all_pos = temp_positions[tok]
        n_seg_ref = len(all_pos[0]) if all_pos else 8
        x_mean = np.nanmean(
            [p[:n_seg_ref] if len(p) >= n_seg_ref
             else np.pad(p, (0, n_seg_ref - len(p)), constant_values=np.nan)
             for p in all_pos], axis=0,
        )
        for i, v in enumerate(x_mean):
            if not np.isfinite(v):
                x_mean[i] = (i + 0.5) / n_seg_ref

        for pat in PATTERN_ORDER:
            vecs = temp_profiles[tok].get(pat, [])
            if not vecs:
                continue
            arr = np.stack([
                v[:n_seg_ref] if len(v) >= n_seg_ref
                else np.pad(v, (0, n_seg_ref - len(v)))
                for v in vecs
            ])
            mean_v, std_v = arr.mean(axis=0), arr.std(axis=0)
            ls, col_p = line_styles[pat]
            ax.plot(x_mean, mean_v, color=col_p, lw=1.8, ls=ls,
                    label=f"{PATTERN_LABELS[pat]} (n={len(vecs)})", zorder=3)
            ax.fill_between(x_mean, mean_v - std_v, mean_v + std_v,
                            color=col_p, alpha=0.12, zorder=2)
            ax.scatter(x_mean, mean_v, color=col_p, s=20, zorder=4, alpha=0.7)

        mk_info = MARKER_DICT.get(tok, {})
        mtype = mk_info.get("type", "")
        mwin  = mk_info.get("window")
        if mwin and mtype == "temporal_absolute":
            lo, hi = mwin
            seg_in = [xm for xm in x_mean if lo <= xm <= hi]
            if seg_in:
                step = (x_mean[1] - x_mean[0]) if len(x_mean) > 1 else 0.1
                ax.axvspan(min(seg_in) - step * 0.5, max(seg_in) + step * 0.5,
                           color="#f1c40f", alpha=0.08, label=f"Atteso [{lo:.0%}–{hi:.0%}]")

        n_tot = sum(len(v) for v in temp_profiles[tok].values())
        ax.set_title(f'"{tok}" (n={n_tot})',
                     color=TYPE_COLORS.get(mtype, TEXT_C), fontsize=8)
        ax.set_xticks(x_mean)
        ax.set_xticklabels([f"{v:.0%}" for v in x_mean], fontsize=5.5, rotation=30)
        ax.set_ylabel("MI_audio norm.", fontsize=7)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=6, loc="upper right")

    axes[0][0].set_title(
        axes[0][0].get_title(),
        color=TYPE_COLORS.get("temporal_absolute", TEXT_C), fontsize=8,
    )

    # BOTTOM: stem profiles per token timbrico
    for mi_idx in range(ncols):
        ax = axes[1][mi_idx]
        ax.set_facecolor(PANEL)
        if mi_idx >= n_bot:
            ax.set_visible(False)
            continue
        tok = valid_timbral[mi_idx]
        stems = timbral_stem_names.get(tok, ["drums", "bass", "other", "vocals"])
        expected_stem = MARKER_DICT.get(tok, {}).get("expected_stem", "?")

        x = np.arange(len(stems))
        width = 0.35
        offset = 0

        for pat in ["correct", "wrong", "pseudo_text", "diffuse"]:
            vecs = timbral_stem_profiles[tok].get(pat, [])
            if not vecs:
                continue
            arr = np.array(vecs)
            mean_v = arr.mean(axis=0)
            col_p  = PATTERN_COLORS[pat]
            bars = ax.bar(x + offset * width, mean_v[:len(stems)],
                          width * 0.8, label=f"{PATTERN_LABELS[pat]} (n={len(vecs)})",
                          color=col_p, alpha=0.8, edgecolor=BG, linewidth=0.4)
            offset += 1

        # Evidenzia stem atteso con linea verticale
        if expected_stem in stems:
            exp_x = stems.index(expected_stem)
            ax.axvline(exp_x + (offset - 1) * width / 2,
                       color="#f1c40f", linewidth=1.5, linestyle="--",
                       label=f"Atteso: {expected_stem}", alpha=0.7)

        n_tot = sum(len(v) for v in timbral_stem_profiles[tok].values())
        ax.set_title(f'"{tok}" → {expected_stem} (n={n_tot})',
                     color=TYPE_COLORS["timbral"], fontsize=8)
        ax.set_xticks(x + (offset - 1) * width / 2)
        ax.set_xticklabels(stems, fontsize=7.5)
        ax.set_ylabel("MI stem (norm.)", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper right")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[G-D3] {output_path}")


# ── G-D4: Scatter unificato MI_text vs grounding score ───────────────────────

def plot_gd4(processed: List[Dict], output_path: str):
    """
    G-D4 — Scatter unificato:
    X  = mi2_weight (importanza token nel testo)
    Y  = grounding score (peak temporale per temporali / MI_expected_stem/MI_dominant per timbrici)
    Colore = pattern (pseudo_text/diffuse/correct/wrong)
    Forma  = tipo (◆ temporal | ▲ timbral)
    Annotazione = token

    Per i timbrici: Y = mi_expected_stem / (mi_dominant_stem + 1e-9)
                    → 1.0 = perfetto (atteso == dominante)
                    → < 1.0 = stem sbagliato è più alto
    """
    _style()
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor=BG)

    shape_map = {"temporal": "D", "timbral": "^"}

    data_by_pat: Dict[str, Dict] = {}
    all_x, all_y = [], []

    for s in processed:
        for mk in s["marker_results"]:
            p    = mk["pattern"]
            mtype = mk["marker_type"]
            macro = "temporal" if mtype in ("temporal_absolute", "temporal_diffuse") else "timbral"

            x_val = mk["mi2_weight"]

            if macro == "timbral":
                mi_exp = mk.get("mi_expected_stem", 0.0)
                mi_dom = mk.get("mi_dominant_stem", 1e-9)
                # Rapporto: quanto il match è buono
                y_val = float(mi_exp / (mi_dom + 1e-9))
                y_val = min(y_val, 1.05)   # cap a 1.05 per visualizzazione
            else:
                y_val = mk.get("grounding_score", 0.0)

            if p not in data_by_pat:
                data_by_pat[p] = {"x": [], "y": [], "tok": [], "shape": []}
            data_by_pat[p]["x"].append(x_val)
            data_by_pat[p]["y"].append(y_val)
            data_by_pat[p]["tok"].append(mk["token"])
            data_by_pat[p]["shape"].append(shape_map[macro])
            all_x.append(x_val)
            all_y.append(y_val)

    for pat in PATTERN_ORDER:
        d = data_by_pat.get(pat, {})
        if not d.get("x"):
            continue
        col = PATTERN_COLORS[pat]
        xs, ys = d["x"], d["y"]

        for mshape in set(d["shape"]):
            idxs = [i for i, sh in enumerate(d["shape"]) if sh == mshape]
            ax.scatter(
                [xs[i] for i in idxs], [ys[i] for i in idxs],
                c=col, marker=mshape, s=75, alpha=0.78,
                edgecolors=BG, linewidths=0.5,
                label=PATTERN_LABELS[pat] if mshape == "D" else None,
                zorder=3,
            )

        seen: List[Tuple] = []
        for xi, yi, lbl in zip(xs, ys, d["tok"]):
            too_close = any(abs(xi - px) < 0.025 and abs(yi - py) < 0.04 for px, py in seen)
            if not too_close:
                ax.annotate(lbl, (xi, yi), xytext=(4, 3),
                            textcoords="offset points", fontsize=5.5,
                            color=col, alpha=0.85)
                seen.append((xi, yi))

    # Soglie
    ax.axvline(THRESH_TEXT_IMPORTANCE, color=GRID_C, lw=1, ls="--", alpha=0.5)
    ax.axhline(THRESH_PEAK_CONCENTRATION, color=GRID_C, lw=0.8, ls=":",
               alpha=0.4, label="soglia temporal")
    ax.axhline(1.0, color="#f1c40f", lw=0.8, ls=":", alpha=0.35,
               label="timbral match perfetto (y=1)")

    # Quadranti
    xr = max(all_x) * 1.05 if all_x else 1.0
    yr = max(all_y) * 1.05 if all_y else 1.0
    kw = dict(fontsize=6.5, color=TEXT_C, alpha=0.22, ha="center", va="center")
    ax.text(xr * 0.78, yr * 0.90, "Vero grounding\ninterattivo", **kw)
    ax.text(xr * 0.12, yr * 0.90, "Pseudo-grounding\ntestuale", **kw)
    ax.text(xr * 0.78, yr * 0.08, "Grounding senza\ndriver testuale", **kw)
    ax.text(xr * 0.12, yr * 0.08, "Token e audio\nnon correlati", **kw)

    shape_handles = [
        mpatches.Patch(color=TEXT_C, label="◆ Temporale assoluto / throughout"),
        mpatches.Patch(color=TEXT_C, label="▲ Timbrico (Y = mi_expected/mi_dominant)"),
    ]
    leg1 = ax.legend(loc="upper left", fontsize=7.5, title="Pattern")
    leg2 = ax.legend(handles=shape_handles, loc="lower right", fontsize=6.5, title="Tipo")
    ax.add_artist(leg1)

    ax.set_xlabel("MI_text weight  [mi2_global_word(w)]", fontsize=10)
    ax.set_ylabel("Grounding score  [temporal: peak norm. | timbral: mi_exp/mi_dom]", fontsize=9)
    ax.set_title("G-D4 — MI_text vs Grounding score",
                 color=TEXT_C, fontsize=12, fontweight="bold")

    fig.text(
        0.5, 0.005,
        "Temporal: G[w,s] ≈ mi2[w]×mi1_time[s]  |  "
        "Timbral: dominanza stem verificata su mi1_stem htdemucs (drums/bass/other/vocals)  |  "
        "prodotto esterno = approssimazione dei marginali DIME",
        ha="center", fontsize=6, color=TEXT_C, alpha=0.45,
    )
    plt.tight_layout(rect=[0, 0.025, 1, 1])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[G-D4] {output_path}")


# ==============================================================================
# 9) REPORT STATISTICO
# ==============================================================================

def print_summary(processed: List[Dict]):
    all_markers = [mk for s in processed for mk in s["marker_results"]]
    temp_markers   = [mk for mk in all_markers if mk["marker_type"] in ("temporal_absolute","temporal_diffuse")]
    timbral_markers = [mk for mk in all_markers if mk["marker_type"] == "timbral"]

    print(f"\n{'='*65}")
    print("ESPERIMENTO D v3 — REPORT SOMMARIO")
    print(f"{'='*65}")
    print(f"Campioni con marker : {len(processed)}")
    print(f"Marker totali       : {len(all_markers)}")
    print(f"  Temporali         : {len(temp_markers)}")
    print(f"  Timbrici          : {len(timbral_markers)}")

    def _show(label, markers):
        if not markers:
            print(f"\n  [{label}] nessun dato")
            return
        pc = Counter(mk["pattern"] for mk in markers)
        tot = len(markers)
        print(f"\n--- {label} (n={tot}) ---")
        for p in PATTERN_ORDER:
            n = pc.get(p, 0)
            print(f"  {PATTERN_LABELS[p]:<15} {n:4d}  ({n/tot*100:.1f}%)")

        # Dettaglio per token/stem
        by_tok = {}
        for mk in markers:
            key = mk["token"]
            if key not in by_tok: by_tok[key] = Counter()
            by_tok[key][mk["pattern"]] += 1
        for tok in sorted(by_tok):
            pc2 = by_tok[tok]
            n   = sum(pc2.values())
            parts = "  ".join(f"{PATTERN_LABELS[p][0]}={pc2.get(p,0)}" for p in PATTERN_ORDER)
            print(f"    {tok:<15} n={n:3d}  {parts}")

    _show("TEMPORALI ASSOLUTI", temp_markers)
    _show("TIMBRICI", timbral_markers)

    print(f"\n--- Per categoria ---")
    for cat in sorted({s["category"] for s in processed}):
        mks = [mk for s in processed if s["category"] == cat for mk in s["marker_results"]]
        if not mks: continue
        pc3 = Counter(mk["pattern"] for mk in mks)
        print(f"  {cat[:30]:<30} n={len(mks):3d}  "
              f"pseudo={pc3.get('pseudo_text',0)/len(mks)*100:.0f}%  "
              f"diffuse={pc3.get('diffuse',0)/len(mks)*100:.0f}%  "
              f"correct={pc3.get('correct',0)/len(mks)*100:.0f}%  "
              f"wrong={pc3.get('wrong',0)/len(mks)*100:.0f}%")
    print(f"{'='*65}\n")

# ==============================================================================
# 10) ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp D v3 — Grounding temporale + timbrico stem-based"
    )
    parser.add_argument("--batch-dir",  required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--category",   nargs="+", default=None)
    parser.add_argument("--max-gd1-samples",    type=int, default=12)
    parser.add_argument("--min-marker-samples", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.batch_dir, "exp_d_v3_results")
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
    plot_gd1(processed,
             os.path.join(output_dir, "G-D1_grounding_heatmap_stem.png"),
             args.max_gd1_samples)
    plot_gd2(processed,
             os.path.join(output_dir, "G-D2_pattern_distribution.png"))
    plot_gd3(processed,
             os.path.join(output_dir, "G-D3_temporal_and_stem_profiles.png"),
             args.min_marker_samples)
    plot_gd4(processed,
             os.path.join(output_dir, "G-D4_mi_vs_grounding.png"))
    print(f"\n[Done] {output_dir}")


if __name__ == "__main__":
    main()