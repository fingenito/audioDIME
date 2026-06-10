"""
Esperimento D — Grounding domanda-audio [v4]
===========================================

Output principali:
  D-Table 1: distribuzione marker trovati
  D-Table 2: pattern distribution per tipo marker
  D-Table 3: temporal accuracy sopra chance
  D-Table 4: timbral accuracy sopra chance
  D-Table 5: confusion matrix expected stem vs dominant stem

Figure:
  G-D1: pattern distribution compatta
  G-D2: observed accuracy vs chance con CI bootstrap
  G-D3: confusion matrix timbral
  G-D4: esempi qualitativi, uno per pattern

Uso:
  python -m QA_analysis.experiments.expD.plots_exp_d --batch-dir /nas/home/fingenito/Thesis_project/QA_analysis/Results_QA/experiments/exp_A/batch_run_00

STEM_MARKERS = {
    "drums": [
        "drums", "drum", "drummer", "kick", "snare",
        "hats", "cymbal", "brushes", "sticks",
        "shaker", "percussion",
    ],
    "bass": [
        "bass",
    ],
    "vocals": [
        "vocal", "vocals", "voice", "voices",
        "singer", "vocalist", "choir", "singing",
    ],
    "other": [
        "guitar", "guitars", "guitarist",
        "piano", "keyboard", "organ",
        "saxophone", "sax",
        "trombone", "brass",
        "synth", "synthesizer", "synths",
        "ukulele", "flutist", "string",
    ],
}
TEMPORAL_MARKERS = {
    "beginning": [
        "intro", "introduction", "beginning",
        "start", "starts", "begin",
    ],
    "ending": [
        "outro", "end", "final",
    ],
}
"""

import os
import re
import csv
import json
import glob
import math
import argparse
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# 1) CONFIG
# ==============================================================================

MARKER_DICT: Dict[str, Dict[str, Any]] = {
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

    "throughout": {"type": "temporal_diffuse", "window": None},

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

    "guitar":      {"type": "timbral", "expected_stem": "other"},
    "guitars":     {"type": "timbral", "expected_stem": "other"},
    "guitarist":   {"type": "timbral", "expected_stem": "other"},
    "piano":       {"type": "timbral", "expected_stem": "other"},
    "keyboard":    {"type": "timbral", "expected_stem": "other"},
    "organ":       {"type": "timbral", "expected_stem": "other"},
    "saxophone":   {"type": "timbral", "expected_stem": "other"},
    "sax":         {"type": "timbral", "expected_stem": "other"},
    "trombone":    {"type": "timbral", "expected_stem": "other"},
    "brass":       {"type": "timbral", "expected_stem": "other"},
    "synth":       {"type": "timbral", "expected_stem": "other"},
    "synthesizer": {"type": "timbral", "expected_stem": "other"},
    "synths":      {"type": "timbral", "expected_stem": "other"},
    "ukulele":     {"type": "timbral", "expected_stem": "other"},
    "flutist":     {"type": "timbral", "expected_stem": "other"},
    "strings":     {"type": "timbral", "expected_stem": "other"},
}

TARGET_CATEGORIES = {
    "Instrumentation",
    "Sound Texture",
    "Metre and Rhythm",
    "Musical Texture",
    "Structure",
    "Performance",
}

THRESH_TEXT_IMPORTANCE = 0.05
THRESH_MI_FLATNESS = 0.04
THRESH_PEAK_CONCENTRATION = 0.22
THRESH_STEM_DOMINANCE = 0.10

PATTERN_ORDER = ["pseudo_text", "diffuse", "correct", "wrong"]
PATTERN_LABELS = {
    "pseudo_text": "Pseudo-text",
    "diffuse": "Diffuse",
    "correct": "Correct",
    "wrong": "Wrong",
}
PATTERN_COLORS = {
    "pseudo_text": "#E74C3C",
    "diffuse": "#F39C12",
    "correct": "#2ECC71",
    "wrong": "#8E44AD",
}

TYPE_LABELS = {
    "temporal_absolute": "Temporal absolute",
    "temporal_diffuse": "Throughout",
    "timbral": "Timbral",
}

STEM_ORDER = ["drums", "bass", "other", "vocals", "none", "unknown"]

# ==============================================================================
# 2) IO UTILS
# ==============================================================================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[Table] {path}")


def save_markdown_table(rows: List[Dict[str, Any]], path: str, title: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if not rows:
            f.write("No data.\n")
            return
        keys = list(rows[0].keys())
        f.write("| " + " | ".join(keys) + " |\n")
        f.write("| " + " | ".join(["---"] * len(keys)) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |\n")
    print(f"[Table] {path}")


def load_exp_d_jsons(batch_dir: str) -> List[Dict]:
    per_sample_dir = os.path.join(batch_dir, "per_sample")
    if not os.path.isdir(per_sample_dir):
        raise FileNotFoundError(f"per_sample dir non trovata: {per_sample_dir}")

    files = sorted(glob.glob(os.path.join(per_sample_dir, "*_exp_d.json")))
    if not files:
        raise FileNotFoundError(f"Nessun *_exp_d.json in {per_sample_dir}")

    records = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            print(f"[WARN] skip {os.path.basename(path)}: {e}")

    print(f"[Load] {len(records)} campioni")
    return records


def filter_by_category(records: List[Dict], target: Optional[set] = None) -> List[Dict]:
    if target is None:
        target = TARGET_CATEGORIES
    out = [
        r for r in records
        if any(t.lower() in str(r.get("category", "")).lower() for t in target)
    ]
    print(f"[Filter] {len(out)} campioni nelle categorie target")
    return out

# ==============================================================================
# 3) MARKER + CLASSIFICAZIONE
# ==============================================================================

def detect_markers(prompt_words: List[str]) -> List[Dict]:
    found = []
    for idx, word in enumerate(prompt_words):
        clean = re.sub(r"[^a-zA-Z]", "", str(word)).lower()
        if clean in MARKER_DICT:
            info = MARKER_DICT[clean]
            found.append({
                "token": clean,
                "word_orig": word,
                "word_idx": idx,
                "marker_type": info["type"],
                "expected_window": info.get("window"),
                "expected_stem": info.get("expected_stem"),
            })
    return found


def segment_positions(seg_boundaries: List[Dict], n_seg: int) -> np.ndarray:
    if seg_boundaries and len(seg_boundaries) >= n_seg:
        starts, ends = [], []
        for d in seg_boundaries[:n_seg]:
            starts.append(float(d.get("segment_start_sec", float("nan"))))
            ends.append(float(d.get("segment_end_sec", float("nan"))))
        s_a, e_a = np.array(starts), np.array(ends)
        if not np.any(np.isnan(s_a)) and not np.any(np.isnan(e_a)) and e_a[-1] > 0:
            return np.clip((s_a + e_a) / 2.0 / e_a[-1], 0.0, 1.0)
    return np.linspace(1.0 / (2 * n_seg), 1.0 - 1.0 / (2 * n_seg), n_seg)


def classify_temporal(
    mi2_w: float,
    mi1: np.ndarray,
    peak_pos: float,
    peak_concentration: float,
    marker_type: str,
    expected_window: Optional[Tuple[float, float]],
) -> str:
    mi1_range = float(np.max(mi1) - np.min(mi1))

    if abs(mi2_w) > THRESH_TEXT_IMPORTANCE and mi1_range < THRESH_MI_FLATNESS:
        return "pseudo_text"

    if peak_concentration < THRESH_PEAK_CONCENTRATION:
        if marker_type == "temporal_diffuse":
            return "correct"
        return "diffuse"

    if marker_type == "temporal_diffuse":
        return "wrong"

    if expected_window is not None:
        lo, hi = expected_window
        return "correct" if lo <= peak_pos <= hi else "wrong"

    return "diffuse"


def classify_timbral(
    mi2_w: float,
    mi1: np.ndarray,
    mi1_stem: List[float],
    stem_names: List[str],
    expected_stem: str,
) -> Tuple[str, str, float, float, float]:
    mi1_range = float(np.max(mi1) - np.min(mi1))

    if abs(mi2_w) > THRESH_TEXT_IMPORTANCE and mi1_range < THRESH_MI_FLATNESS:
        return "pseudo_text", "none", 0.0, 0.0, 0.0

    stem_arr = np.array(mi1_stem, dtype=float)
    if stem_arr.size == 0:
        return "diffuse", "none", 0.0, 0.0, 0.0

    stem_abs = np.abs(stem_arr)
    total = float(np.sum(stem_abs))
    if total < 1e-9:
        return "diffuse", "none", 0.0, 0.0, 0.0

    dominant_idx = int(np.argmax(stem_abs))
    dominant_stem = stem_names[dominant_idx] if dominant_idx < len(stem_names) else "unknown"
    mi_dominant = float(stem_abs[dominant_idx])

    expected_idx = next((i for i, n in enumerate(stem_names) if n == expected_stem), None)
    mi_expected = float(stem_abs[expected_idx]) if expected_idx is not None else 0.0

    sorted_vals = np.sort(stem_abs)[::-1]
    margin = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else float(sorted_vals[0])

    if margin < THRESH_STEM_DOMINANCE:
        return "diffuse", dominant_stem, mi_expected, mi_dominant, margin

    pattern = "correct" if dominant_stem == expected_stem else "wrong"
    return pattern, dominant_stem, mi_expected, mi_dominant, margin


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

    g_vec = mi2_w * mi1
    g_sum = float(np.sum(np.abs(g_vec)))
    g_norm = g_vec / (g_sum + 1e-12)

    peak_idx = int(np.argmax(np.abs(g_norm)))
    peak_val = float(g_norm[peak_idx])
    peak_pos = float(seg_positions_arr[peak_idx]) if peak_idx < len(seg_positions_arr) else 0.5
    peak_concentration = float(abs(peak_val))

    mtype = marker["marker_type"]

    base = {
        "token": marker["token"],
        "word_orig": marker["word_orig"],
        "word_idx": w_idx,
        "marker_type": mtype,
        "mi2_weight": mi2_w,
        "mi1_time_vec": mi1.tolist(),
        "mi1_stem": list(mi1_stem),
        "stem_names": list(stem_names),
        "grounding_vec": g_vec.tolist(),
        "grounding_vec_norm": g_norm.tolist(),
        "grounding_score": float(np.max(np.abs(g_norm))),
        "peak_segment_idx": peak_idx,
        "peak_segment_pos": peak_pos,
        "peak_concentration": peak_concentration,
        "mi1_range": float(np.max(mi1) - np.min(mi1)),
    }

    if mtype == "timbral":
        pattern, dominant_stem, mi_expected, mi_dominant, margin = classify_timbral(
            mi2_w=mi2_w,
            mi1=mi1,
            mi1_stem=mi1_stem,
            stem_names=stem_names,
            expected_stem=marker["expected_stem"],
        )
        base.update({
            "expected_stem": marker["expected_stem"],
            "dominant_stem": dominant_stem,
            "mi_expected_stem": mi_expected,
            "mi_dominant_stem": mi_dominant,
            "stem_margin": margin,
            "pattern": pattern,
        })
        return base

    pattern = classify_temporal(
        mi2_w=mi2_w,
        mi1=mi1,
        peak_pos=peak_pos,
        peak_concentration=peak_concentration,
        marker_type=mtype,
        expected_window=marker["expected_window"],
    )
    base.update({
        "expected_window": marker["expected_window"],
        "pattern": pattern,
    })
    return base

# ==============================================================================
# 4) PROCESSING
# ==============================================================================

def process_sample(record: Dict) -> Optional[Dict]:
    prompt_words = record.get("prompt_words", [])
    mi2 = record.get("mi2_global_word", [])
    mi1 = record.get("mi1_time", [])
    mi1_stem = record.get("mi1_stem", [])
    stem_names = record.get("stem_names", ["drums", "bass", "other", "vocals"])
    seg_bounds = record.get("segment_boundaries_sec", [])

    if not prompt_words or not mi2 or not mi1:
        return None

    markers = detect_markers(prompt_words)
    if not markers:
        return None

    seg_pos = segment_positions(seg_bounds, len(mi1))

    marker_results = [
        compute_grounding(m, mi2, mi1, mi1_stem, stem_names, seg_pos)
        for m in markers
    ]

    priority = {"pseudo_text": 0, "wrong": 1, "diffuse": 2, "correct": 3}
    overall = sorted(marker_results, key=lambda x: priority.get(x["pattern"], 99))[0]["pattern"]

    return {
        "sample_id": record.get("sample_id", ""),
        "category": record.get("category", ""),
        "difficulty": record.get("difficulty", ""),
        "macro_family": record.get("macro_family", ""),
        "prompt_words": prompt_words,
        "mi2_global_word": mi2,
        "mi1_time": mi1,
        "mi1_stem": mi1_stem,
        "stem_names": stem_names,
        "seg_positions": seg_pos.tolist(),
        "mi1_stem_x_seg": record.get("mi1_stem_x_seg", []),
        "marker_results": marker_results,
        "overall_pattern": overall,
        "n_markers": len(marker_results),
    }


def process_all(records: List[Dict]) -> List[Dict]:
    processed, skipped = [], 0
    for r in records:
        out = process_sample(r)
        if out is None:
            skipped += 1
        else:
            processed.append(out)
    print(f"[Process] {len(processed)} con marker | {skipped} senza marker")
    return processed


def flatten_markers(processed: List[Dict]) -> List[Dict]:
    rows = []
    for s in processed:
        for mk in s["marker_results"]:
            row = {
                "sample_id": s["sample_id"],
                "category": s["category"],
                "difficulty": s["difficulty"],
                "macro_family": s["macro_family"],
                **mk,
            }
            rows.append(row)
    return rows

# ==============================================================================
# 5) STATISTICA
# ==============================================================================

def bootstrap_ci_binary(values: List[int], n_boot: int = 5000, alpha: float = 0.05, seed: int = 13) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats.append(float(np.mean(sample)))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def binomial_tail_p_value(k: int, n: int, p0: float) -> float:
    if n <= 0:
        return 1.0
    if n > 1000:
        # normal approximation fallback
        mu = n * p0
        sigma = math.sqrt(n * p0 * (1 - p0) + 1e-12)
        z = (k - mu) / sigma
        return float(0.5 * math.erfc(z / math.sqrt(2)))
    p = 0.0
    for i in range(k, n + 1):
        p += math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
    return float(min(max(p, 0.0), 1.0))


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def build_d_tables(marker_rows: List[Dict], output_dir: str) -> Dict[str, List[Dict]]:
    tables_dir = ensure_dir(os.path.join(output_dir, "tables"))

    # D-Table 1
    token_counts = Counter((r["marker_type"], r["token"]) for r in marker_rows)
    table1 = []
    total_markers = len(marker_rows)
    for (mtype, token), n in sorted(token_counts.items(), key=lambda x: (-x[1], x[0])):
        table1.append({
            "marker_type": mtype,
            "token": token,
            "n": n,
            "percent_total": pct(n / max(total_markers, 1)),
        })

    # D-Table 2
    table2 = []
    by_type = defaultdict(list)
    for r in marker_rows:
        by_type[r["marker_type"]].append(r)

    for mtype in ["temporal_absolute", "temporal_diffuse", "timbral"]:
        rows = by_type.get(mtype, [])
        n = len(rows)
        c = Counter(r["pattern"] for r in rows)
        table2.append({
            "marker_type": mtype,
            "n": n,
            "pseudo_text_n": c.get("pseudo_text", 0),
            "pseudo_text_pct": pct(c.get("pseudo_text", 0) / max(n, 1)),
            "diffuse_n": c.get("diffuse", 0),
            "diffuse_pct": pct(c.get("diffuse", 0) / max(n, 1)),
            "correct_n": c.get("correct", 0),
            "correct_pct": pct(c.get("correct", 0) / max(n, 1)),
            "wrong_n": c.get("wrong", 0),
            "wrong_pct": pct(c.get("wrong", 0) / max(n, 1)),
        })

    # D-Table 3 temporal accuracy above chance
    temporal_abs = [r for r in marker_rows if r["marker_type"] == "temporal_absolute"]
    table3 = []

    for token in sorted({r["token"] for r in temporal_abs}):
        rows = [r for r in temporal_abs if r["token"] == token]
        correct = [1 if r["pattern"] == "correct" else 0 for r in rows]
        n = len(correct)
        k = sum(correct)
        window = MARKER_DICT[token]["window"]
        chance = float(window[1] - window[0]) if window else 0.0
        ci_lo, ci_hi = bootstrap_ci_binary(correct)
        pval = binomial_tail_p_value(k, n, chance)

        table3.append({
            "token": token,
            "n": n,
            "correct_n": k,
            "observed_accuracy_pct": pct(k / max(n, 1)),
            "chance_pct": pct(chance),
            "ci95_low_pct": pct(ci_lo),
            "ci95_high_pct": pct(ci_hi),
            "binomial_p_value": f"{pval:.4g}",
        })

    if temporal_abs:
        correct = [1 if r["pattern"] == "correct" else 0 for r in temporal_abs]
        n = len(correct)
        k = sum(correct)
        chance_values = [
            float(MARKER_DICT[r["token"]]["window"][1] - MARKER_DICT[r["token"]]["window"][0])
            for r in temporal_abs
        ]
        chance = float(np.mean(chance_values))
        ci_lo, ci_hi = bootstrap_ci_binary(correct)
        pval = binomial_tail_p_value(k, n, chance)
        table3.insert(0, {
            "token": "ALL_TEMPORAL_ABSOLUTE",
            "n": n,
            "correct_n": k,
            "observed_accuracy_pct": pct(k / max(n, 1)),
            "chance_pct": pct(chance),
            "ci95_low_pct": pct(ci_lo),
            "ci95_high_pct": pct(ci_hi),
            "binomial_p_value": f"{pval:.4g}",
        })

    # D-Table 4 timbral accuracy above chance
    timbral = [r for r in marker_rows if r["marker_type"] == "timbral"]
    table4 = []

    for stem in ["drums", "bass", "other", "vocals"]:
        rows = [r for r in timbral if r.get("expected_stem") == stem]
        correct = [1 if r["pattern"] == "correct" else 0 for r in rows]
        n = len(correct)
        if n == 0:
            continue
        k = sum(correct)
        chance = 0.25
        ci_lo, ci_hi = bootstrap_ci_binary(correct)
        pval = binomial_tail_p_value(k, n, chance)

        table4.append({
            "expected_stem": stem,
            "n": n,
            "correct_n": k,
            "observed_accuracy_pct": pct(k / max(n, 1)),
            "chance_pct": pct(chance),
            "ci95_low_pct": pct(ci_lo),
            "ci95_high_pct": pct(ci_hi),
            "binomial_p_value": f"{pval:.4g}",
        })

    if timbral:
        correct = [1 if r["pattern"] == "correct" else 0 for r in timbral]
        n = len(correct)
        k = sum(correct)
        chance = 0.25
        ci_lo, ci_hi = bootstrap_ci_binary(correct)
        pval = binomial_tail_p_value(k, n, chance)
        table4.insert(0, {
            "expected_stem": "ALL_TIMBRAL",
            "n": n,
            "correct_n": k,
            "observed_accuracy_pct": pct(k / max(n, 1)),
            "chance_pct": pct(chance),
            "ci95_low_pct": pct(ci_lo),
            "ci95_high_pct": pct(ci_hi),
            "binomial_p_value": f"{pval:.4g}",
        })

    # D-Table 5 confusion matrix
    table5 = []
    expected_stems = ["drums", "bass", "other", "vocals"]
    dominant_stems = ["drums", "bass", "other", "vocals", "none", "unknown"]

    for exp in expected_stems:
        row_markers = [r for r in timbral if r.get("expected_stem") == exp]
        n = len(row_markers)
        c = Counter(r.get("dominant_stem", "unknown") for r in row_markers)
        row = {"expected_stem": exp, "n": n}
        for dom in dominant_stems:
            row[f"dominant_{dom}"] = c.get(dom, 0)
        table5.append(row)

    all_tables = {
        "D-Table1_marker_distribution": table1,
        "D-Table2_pattern_by_marker_type": table2,
        "D-Table3_temporal_accuracy_above_chance": table3,
        "D-Table4_timbral_accuracy_above_chance": table4,
        "D-Table5_timbral_confusion_matrix": table5,
    }

    for name, rows in all_tables.items():
        save_csv(rows, os.path.join(tables_dir, f"{name}.csv"))
        save_markdown_table(rows, os.path.join(tables_dir, f"{name}.md"), name)

    with open(os.path.join(tables_dir, "all_marker_rows.json"), "w", encoding="utf-8") as f:
        json.dump(marker_rows, f, indent=2, ensure_ascii=False)

    return all_tables

# ==============================================================================
# 6) PLOT
# ==============================================================================

def _plot_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 180,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def plot_gd1_pattern_distribution(marker_rows: List[Dict], output_path: str):
    """
    G-D1: pattern distribution compatta.
    """
    _plot_style()

    groups = ["temporal_absolute", "temporal_diffuse", "timbral"]
    counts = {
        g: Counter(r["pattern"] for r in marker_rows if r["marker_type"] == g)
        for g in groups
    }
    totals = {g: sum(counts[g].values()) for g in groups}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))

    for pat in PATTERN_ORDER:
        vals = np.array([
            counts[g].get(pat, 0) / max(totals[g], 1) * 100
            for g in groups
        ])
        ax.bar(x, vals, bottom=bottom, label=PATTERN_LABELS[pat], color=PATTERN_COLORS[pat])
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[g] for g in groups])
    ax.set_ylabel("% marker")
    ax.set_ylim(0, 100)
    ax.set_title("G-D1 — Pattern distribution by marker type")
    ax.legend(loc="upper right", fontsize=8)

    for i, g in enumerate(groups):
        ax.text(i, -5, f"n={totals[g]}", ha="center", va="top", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {output_path}")


def _extract_accuracy_rows(table3: List[Dict], table4: List[Dict]) -> List[Dict]:
    rows = []
    for r in table3:
        if r["token"] == "ALL_TEMPORAL_ABSOLUTE":
            rows.append({
                "label": "Temporal",
                "observed": float(r["observed_accuracy_pct"]),
                "chance": float(r["chance_pct"]),
                "lo": float(r["ci95_low_pct"]),
                "hi": float(r["ci95_high_pct"]),
                "n": int(r["n"]),
            })
    for r in table4:
        if r["expected_stem"] == "ALL_TIMBRAL":
            rows.append({
                "label": "Timbral",
                "observed": float(r["observed_accuracy_pct"]),
                "chance": float(r["chance_pct"]),
                "lo": float(r["ci95_low_pct"]),
                "hi": float(r["ci95_high_pct"]),
                "n": int(r["n"]),
            })
    return rows


def plot_gd2_accuracy_above_chance(table3: List[Dict], table4: List[Dict], output_path: str):
    """
    G-D2: observed accuracy vs chance con CI bootstrap.
    """
    _plot_style()
    rows = _extract_accuracy_rows(table3, table4)
    if not rows:
        print("[G-D2] skip: no accuracy data.")
        return

    labels = [r["label"] for r in rows]
    observed = np.array([r["observed"] for r in rows])
    chance = np.array([r["chance"] for r in rows])
    lo = np.array([r["lo"] for r in rows])
    hi = np.array([r["hi"] for r in rows])
    yerr = np.vstack([observed - lo, hi - observed])

    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, observed, yerr=yerr, capsize=5, color=["#00BCD4", "#AB47BC"], alpha=0.85, label="Observed")
    ax.scatter(x, chance, color="black", marker="D", s=55, label="Chance baseline", zorder=4)

    for i, r in enumerate(rows):
        ax.text(i, observed[i] + 4, f"n={r['n']}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, min(100, max(100, np.max(hi) + 10)))
    ax.set_title("G-D2 — Observed accuracy above chance")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {output_path}")


def plot_gd3_timbral_confusion(table5: List[Dict], output_path: str):
    """
    G-D3: confusion matrix expected stem vs dominant stem.
    """
    _plot_style()
    expected = ["drums", "bass", "other", "vocals"]
    dominant = ["drums", "bass", "other", "vocals", "none", "unknown"]

    mat = np.zeros((len(expected), len(dominant)), dtype=float)
    counts = {r["expected_stem"]: r for r in table5}

    for i, exp in enumerate(expected):
        row = counts.get(exp, {})
        for j, dom in enumerate(dominant):
            mat[i, j] = float(row.get(f"dominant_{dom}", 0))

    row_sums = mat.sum(axis=1, keepdims=True)
    mat_pct = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0) * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mat_pct, cmap="Blues", vmin=0, vmax=max(1, np.max(mat_pct)))

    ax.set_xticks(np.arange(len(dominant)))
    ax.set_xticklabels(dominant)
    ax.set_yticks(np.arange(len(expected)))
    ax.set_yticklabels(expected)

    ax.set_xlabel("Dominant stem")
    ax.set_ylabel("Expected stem")
    ax.set_title("G-D3 — Timbral confusion matrix")

    for i in range(len(expected)):
        for j in range(len(dominant)):
            count = int(mat[i, j])
            val = mat_pct[i, j]
            if count > 0:
                ax.text(j, i, f"{count}\n{val:.0f}%", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label="% by expected stem")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {output_path}")


def plot_gd4_representative_examples(processed: List[Dict], output_path: str):
    """
    G-D4: esempi qualitativi, uno per pattern.
    Mostra un piccolo profilo temporale/stem, non milioni di campioni.
    """
    _plot_style()

    selected = {}
    for pat in PATTERN_ORDER:
        for s in processed:
            hit = next((m for m in s["marker_results"] if m["pattern"] == pat), None)
            if hit is not None:
                selected[pat] = (s, hit)
                break

    if not selected:
        print("[G-D4] skip: no examples.")
        return

    fig, axes = plt.subplots(len(selected), 2, figsize=(10, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.array([axes])

    for row_idx, pat in enumerate(PATTERN_ORDER):
        if pat not in selected:
            continue

        s, mk = selected[pat]
        ax_time = axes[row_idx, 0]
        ax_stem = axes[row_idx, 1]

        mi1 = np.array(mk["mi1_time_vec"], dtype=float)
        seg_pos = np.array(s["seg_positions"], dtype=float)
        ax_time.plot(seg_pos, mi1, marker="o", color=PATTERN_COLORS[pat])
        ax_time.axvline(mk["peak_segment_pos"], color="black", linestyle="--", alpha=0.5)

        if mk["marker_type"] == "temporal_absolute" and mk.get("expected_window"):
            lo, hi = mk["expected_window"]
            ax_time.axvspan(lo, hi, color="green", alpha=0.12, label="expected window")
        elif mk["marker_type"] == "temporal_diffuse":
            ax_time.set_title("Expected: diffuse/throughout")

        ax_time.set_title(
            f"{PATTERN_LABELS[pat]} — {mk['token']} | {s['sample_id']}",
            fontsize=9,
        )
        ax_time.set_xlabel("Normalized time")
        ax_time.set_ylabel("MI audio time")

        stems = mk.get("stem_names", ["drums", "bass", "other", "vocals"])
        vals = np.abs(np.array(mk.get("mi1_stem", [0.0] * len(stems)), dtype=float))
        colors = ["#CCCCCC"] * len(stems)

        if mk["marker_type"] == "timbral" and mk.get("expected_stem") in stems:
            colors[stems.index(mk["expected_stem"])] = "#2ECC71"

        ax_stem.bar(stems, vals[:len(stems)], color=colors)
        if mk.get("dominant_stem") in stems:
            ax_stem.text(stems.index(mk["dominant_stem"]), max(vals) if len(vals) else 0, "★", ha="center", va="bottom")
        ax_stem.set_ylabel("MI stem")
        ax_stem.set_title("Stem profile", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {output_path}")

# ==============================================================================
# 7) SUMMARY
# ==============================================================================

def print_summary(marker_rows: List[Dict], tables: Dict[str, List[Dict]]) -> None:
    print("\n" + "=" * 70)
    print("EXP D v4 — REPORT PRINCIPALE")
    print("=" * 70)
    print(f"Marker totali: {len(marker_rows)}")

    for r in tables["D-Table2_pattern_by_marker_type"]:
        print(
            f"{r['marker_type']:<20} n={r['n']:<4} "
            f"correct={r['correct_pct']}%  wrong={r['wrong_pct']}%  "
            f"diffuse={r['diffuse_pct']}%  pseudo={r['pseudo_text_pct']}%"
        )

    print("\nAccuracy sopra chance:")
    for r in tables["D-Table3_temporal_accuracy_above_chance"]:
        if r["token"] == "ALL_TEMPORAL_ABSOLUTE":
            print(
                f"Temporal absolute: observed={r['observed_accuracy_pct']}% "
                f"chance={r['chance_pct']}% p={r['binomial_p_value']}"
            )

    for r in tables["D-Table4_timbral_accuracy_above_chance"]:
        if r["expected_stem"] == "ALL_TIMBRAL":
            print(
                f"Timbral: observed={r['observed_accuracy_pct']}% "
                f"chance={r['chance_pct']}% p={r['binomial_p_value']}"
            )

    print("=" * 70 + "\n")

# ==============================================================================
# 8) MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Exp D v4 — Tabelle statistiche + plot compatti")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--category", nargs="+", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.batch_dir, "exp_d_v4_results")
    ensure_dir(output_dir)
    plots_dir = ensure_dir(os.path.join(output_dir, "plots"))

    print(f"[Output] {output_dir}")

    records = load_exp_d_jsons(args.batch_dir)
    filtered = filter_by_category(records, set(args.category) if args.category else None)

    if not filtered:
        print("[ERROR] Nessun campione dopo filtro categoria.")
        return

    processed = process_all(filtered)
    if not processed:
        print("[ERROR] Nessun campione con marker.")
        return

    marker_rows = flatten_markers(processed)

    tables = build_d_tables(marker_rows, output_dir)
    print_summary(marker_rows, tables)

    plot_gd1_pattern_distribution(
        marker_rows,
        os.path.join(plots_dir, "G-D1_pattern_distribution_compact.png"),
    )
    plot_gd2_accuracy_above_chance(
        tables["D-Table3_temporal_accuracy_above_chance"],
        tables["D-Table4_timbral_accuracy_above_chance"],
        os.path.join(plots_dir, "G-D2_accuracy_above_chance.png"),
    )
    plot_gd3_timbral_confusion(
        tables["D-Table5_timbral_confusion_matrix"],
        os.path.join(plots_dir, "G-D3_timbral_confusion_matrix.png"),
    )
    plot_gd4_representative_examples(
        processed,
        os.path.join(plots_dir, "G-D4_representative_examples.png"),
    )

    with open(os.path.join(output_dir, "processed_samples.json"), "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] Risultati salvati in: {output_dir}")


if __name__ == "__main__":
    main()