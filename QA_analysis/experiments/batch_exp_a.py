"""
Experiment A — Batch runner
===========================
Itera su tutti i sample HumMusQA, esegue MM-SHAP + DIME per ciascuno,
estrae i campi necessari per Exp A e li salva come JSON compatti.
Nessun plot per sample. I plot vengono rimossi subito dopo l'estrazione.

Output layout:
    Results_QA/exp_A/batch_run_XX/
        run_info.json               ← config + env del batch
        progress.json               ← checkpoint: completed / failed / pending
        per_sample/
            {sample_id}_exp_a.json  ← campi Exp A estratti (leggero)
        aggregated/
            exp_a_results.parquet   ← tabella aggregata finale
            exp_a_results.csv       ← copia CSV per ispezione rapida

Utilizzo:
    python batch_exp_a.py
    python batch_exp_a.py --resume          # riprende dal checkpoint
    python batch_exp_a.py --only-aggregate  # solo aggregazione dei JSON esistenti
"""

import os
import re
import gc
import sys
import json
import glob
import time
import shutil
import hashlib
import logging
import argparse
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

# =============================================================================
# 0) ENV — deve stare prima di qualsiasi import del progetto
# =============================================================================

def _apply_env() -> None:
    env_cfg = {
        "TRANSFORMERS_NO_TF": "1",
        "DIME_BG_AUDIO_DIR": "...",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "DIME_VALUE_MODE": "logit",
        "DIME_PROMPTS_FILE": "",
        "DIME_NUM_EXPECTATION_SAMPLES": "16",
        "DIME_NUM_LIME_SAMPLES": "512",
        "DIME_LIME_NUM_FEATURES_AUDIO": "16",
        "DIME_LIME_NUM_FEATURES_TEXT": "10",
        "DIME_QUEUE_WINDOW": "320",
        "DIME_L_BATCH_SIZE": "8",
        "DIME_LIME_BATCH_SIZE": "8",
        "DIME_WORKER_INNER_BATCH_SIZE": "8",
        "DIME_STEP5_AUDIO_PERTURB_BATCH": "128",
        "DIME_STEP5_TEXT_PERTURB_BATCH": "256",
        "DIME_AUDIO_FEATURE_MODE": "audiolime_demucs",
        "DIME_AUDIOLIME_DEMUCS_MODEL": "htdemucs",
        "DIME_AUDIOLIME_USE_PRECOMPUTED": "1",
        "DIME_AUDIOLIME_RECOMPUTE": "0",
        "DIME_AUDIOLIME_DEMUCS_DEVICE": "cpu",
        "DIME_AUDIOLIME_DEMUCS_SEGMENT": "8",
        "DIME_AUDIOLIME_DEMUCS_SPLIT": "1",
        "DIME_AUDIOLIME_DEMUCS_OVERLAP": "0.25",
        "DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS": "8",
        "DIME_AUDIOLIME_MAX_FEATURES": "40",
        "DIME_AUDIOLIME_MIN_R2": "0.25",
        "DIME_AUDIOLIME_PRECOMPUTED_DIR": "/nas/home/fingenito/Thesis_project/QA_analysis/data/demucs_cache",
        "MMSHAP_QUEUE_WINDOW": "128",
        "DIME_WORKER_EMPTYCACHE_EVERY": "32",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8",
        "DIME_STEP4A_AUDIO_IO_MODE": "auto",
        "DIME_STEP4A_AUDIO_EQ_ATOL": "1e-5",
        "DIME_STEP4A_AUDIO_EQ_RTOL": "1e-4",
        "DIME_STEP4A_RUNNER_AUDIO_TRANSPORT": "shared_memory",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "DIME_FIXED_AUDIO_TRANSPORT_CACHE": "1",
        "DIME_FIXED_AUDIO_TRANSPORT_MODE": "shared_memory",
        "DIME_QWEN_TEXT_CACHE_SIZE": "4096",
        "DIME_TEXT_PKV_CACHE": "1",
        "DIME_TEXT_PKV_CACHE_VERIFY": "0",
        "DIME_TEXT_PKV_CACHE_MIN_COMMON_PREFIX": "8",
        "DIME_TEXT_PKV_CACHE_ATOL": "1e-5",
        "DIME_TEXT_PKV_CACHE_RTOL": "1e-4",
        "DIME_TEXT_PKV_CACHE_DEBUG": "0",
        "DIME_REUSE_PERTURBATIONS_ACROSS_TOKENS": "1",
    }
    for k, v in env_cfg.items():
        os.environ[k] = str(v)

_apply_env()

# =============================================================================
# 1) Import progetto
# =============================================================================

from transformers import Qwen2_5OmniProcessor

from QA_analysis.utils.analysis_1 import analysis_1_start
from QA_analysis.utils.analysis_2 import analyze_dime, get_dime_module_config_snapshot
from QA_analysis.utils.gpu_utils import try_create_parallel_runner, get_available_gpus_with_memory
from QA_analysis.utils.shared_utils import (
    load_hummusqa_entries_parquet,
    build_hummusqa_qwen25_prompt,
)

# =============================================================================
# 2) Costanti
# =============================================================================

HUMMUSQA_ROOT = "/nas/home/fingenito/HumMusQA/data"
EXPERIMENT_RESULTS_ROOT = "/nas/home/fingenito/Thesis_project/QA_analysis/Results_QA"
MODEL_PATH = "/nas/home/fingenito/Models/Qwen2.5-Omni-7B"

MAX_GPUS_TO_USE = 8
MIN_FREE_GB_RUNNER = 21.0

# Macrofamiglie per l'analisi aggregata.
# ⚠ VERIFICA: controlla che i valori corrispondano esattamente ai
#   valori del campo 'category' nel parquet HumMusQA.
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
# 3) Logging
# =============================================================================

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("ExpA-Batch")
_h = logging.StreamHandler(sys.stdout)
_h.setLevel(logging.INFO)
_h.setFormatter(logging.Formatter("%(asctime)s [ExpA] %(levelname)s — %(message)s"))
logger.handlers = [_h]
logger.setLevel(logging.INFO)
logger.propagate = False

for _noisy in ["transformers", "huggingface_hub", "datasets", "urllib3",
               "qwen_omni_utils", "QwenOmni"]:
    _lg = logging.getLogger(_noisy)
    _lg.setLevel(logging.ERROR)
    _lg.propagate = False

try:
    from transformers.utils import logging as _hf_log
    _hf_log.set_verbosity_error()
    _hf_log.disable_progress_bar()
except Exception:
    pass

# =============================================================================
# 4) Helpers generali
# =============================================================================

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _next_batch_dir(exp_root: str) -> str:
    _ensure_dir(exp_root)
    existing = [
        int(n[10:]) for n in os.listdir(exp_root)
        if n.startswith("batch_run_") and n[10:].isdigit()
    ]
    nxt = (max(existing) + 1) if existing else 0
    d = os.path.join(exp_root, f"batch_run_{nxt:02d}")
    os.makedirs(d, exist_ok=False)
    return d


def _atomic_json_dump(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=os.path.dirname(os.path.abspath(path))
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(_make_json_safe(data), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _make_json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    return str(obj)


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha1_bytes(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def _sha1_array(arr: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(arr, dtype=np.float32).tobytes())
    return h.hexdigest()

# =============================================================================
# 5) HumMusQA helpers
# =============================================================================

# Nomi di campo tentati per category/difficulty.
# ⚠ Se il tuo parquet usa nomi diversi, aggiungili qui.
_CATEGORY_FIELDS = ["category", "Category", "music_category", "task_category", "type", "task"]
_DIFFICULTY_FIELDS = ["difficulty", "Difficulty", "level", "Level", "skill_level"]


def _get_field(entry: dict, candidates: List[str], default: str = "") -> str:
    for key in candidates:
        val = entry.get(key, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    return default


def extract_hummusqa_meta(entry: dict, idx: int) -> Dict[str, Any]:
    """
    Estrae i metadati di un entry HumMusQA, inclusi category e difficulty.
    Usa fallback multipli per robustezza rispetto a variazioni nei nomi di campo.
    """
    sample_id = str(
        entry.get("question_id") or
        entry.get("id") or
        entry.get("sample_id") or
        f"sample_{idx}"
    )

    question = str(entry.get("question", "")).strip()
    correct_answer_str = str(entry.get("answer", "")).strip()
    options = [
        correct_answer_str,
        str(entry.get("distractor_1", "")).strip(),
        str(entry.get("distractor_2", "")).strip(),
        str(entry.get("distractor_3", "")).strip(),
    ]

    category = _get_field(entry, _CATEGORY_FIELDS, default="unknown")
    difficulty = _get_field(entry, _DIFFICULTY_FIELDS, default="unknown")

    # Macrofamiglia
    macro = "unknown"
    for keyword, family in MACRO_FAMILY_MAP.items():
        if keyword.lower() in category.lower():
            macro = family
            break

    return {
        "sample_id": sample_id,
        "idx": idx,
        "question": question,
        "options": options,
        "correct_answer_str": correct_answer_str,
        "category": category,
        "difficulty": difficulty,
        "macro_family": macro,
    }


def _entry_has_valid_mcqa(entry: dict) -> bool:
    q = str(entry.get("question", "")).strip()
    opts = [
        str(entry.get("answer", "")).strip(),
        str(entry.get("distractor_1", "")).strip(),
        str(entry.get("distractor_2", "")).strip(),
        str(entry.get("distractor_3", "")).strip(),
    ]
    return bool(q) and len(opts) == 4 and all(opts)


def _infer_audio_ext(entry: dict) -> str:
    audio_field = entry.get("audio", None)
    if isinstance(audio_field, dict):
        for k in ["path", "audio_path", "filename", "file"]:
            v = audio_field.get(k, "")
            if isinstance(v, str) and v.strip():
                _, ext = os.path.splitext(v.lower())
                if ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                    return ext
    return ".mp3"


def materialize_audio(entry: dict, cache_dir: str, idx: int) -> str:
    """
    Restituisce un path WAV canonico per il sample.
    Uguale alla logica in main.py (_materialize_hummusqa_audio).
    """
    os.makedirs(cache_dir, exist_ok=True)
    audio_field = entry.get("audio", None)

    def _exists(v) -> Optional[str]:
        if isinstance(v, str) and v.strip() and os.path.exists(v.strip()):
            return v.strip()
        return None

    p = _exists(audio_field)
    if p:
        return p

    if isinstance(audio_field, dict):
        for k in ["path", "audio_path", "filename", "file"]:
            p = _exists(audio_field.get(k))
            if p:
                return p

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"sample_{idx}")[:48]

    if isinstance(audio_field, dict):
        arr = audio_field.get("array")
        sr = audio_field.get("sampling_rate")
        if arr is not None and sr is not None:
            y = np.asarray(arr, dtype=np.float32).reshape(-1)
            digest = _sha1_array(y)[:16]
            out = os.path.join(cache_dir, f"{safe_name}_{digest}.wav")
            if not os.path.exists(out):
                sf.write(out, y, int(sr))
            return out

        raw = audio_field.get("bytes")
        if raw is not None:
            digest = _sha1_bytes(raw)[:16]
            ext = _infer_audio_ext(entry)
            raw_path = os.path.join(cache_dir, f"{safe_name}_{digest}{ext}")
            decoded_path = os.path.join(cache_dir, f"{safe_name}_{digest}_decoded.wav")
            if not os.path.exists(raw_path):
                with open(raw_path, "wb") as f:
                    f.write(raw)
            if not os.path.exists(decoded_path):
                y, sr_loaded = librosa.load(raw_path, sr=None, mono=True)
                sf.write(decoded_path, y, int(sr_loaded))
            return decoded_path

    raise RuntimeError(f"Impossibile materializzare audio per sample idx={idx}")

# =============================================================================
# 6) Estrazione campi Exp A dai JSON
# =============================================================================

def _answer_to_letter(correct_answer_str: str, options: List[str]) -> str:
    """Converte la stringa della risposta corretta nella lettera corrispondente."""
    for i, opt in enumerate(options):
        if str(opt).strip().lower() == correct_answer_str.strip().lower():
            return chr(65 + i)
    return "A"  # fallback: in HumMusQA la risposta corretta è sempre options[0]


def extract_exp_a_fields(
    dime_json: dict,
    mmshap_json: dict,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Estrae tutti i campi necessari per Exp A da un run DIME + MM-SHAP.

    Chiavi reali verificate su run_53:
      mmshap.json:
        "A-SHAP", "T-SHAP", "baseline_answer_raw", "correct_answer", "options"
      dime_results_step4.json:
        "audio_global_aggregations": {
            "uc1_stem": [4],  "mi1_stem": [4]   ← per stem, aggregato su time
            "uc1_time": [8],  "mi1_time": [8]   ← per segment, aggregato su stems
            "uc1_global_stem_segment": [4][8],  "mi1_global_stem_segment": [4][8]
            "stem_names": [4]
            "segment_boundaries_sec": [9]
        }
        "wordlevel": { "prompt": {
            "words": [...],
            "uc2_global_word": [...],
            "mi2_global_word": [...]
        }}
        "explanations": { "0": { "base_ucmi": float, "token": str } }
    """
    # ---- MM-SHAP ----
    a_shap = float(mmshap_json.get("A-SHAP", 0.0))
    t_shap = float(mmshap_json.get("T-SHAP", 0.0))
    model_answer = str(mmshap_json.get("baseline_answer_raw", "")).strip().upper()
    options = mmshap_json.get("options", [])
    correct_answer_str = str(mmshap_json.get("correct_answer", "")).strip()
    correct_letter = _answer_to_letter(correct_answer_str, options)
    correct = (model_answer == correct_letter)

    # ---- DIME audio aggregations ----
    agg = dime_json.get("audio_global_aggregations", {})
    uc1_stem: List[float] = [float(x) for x in agg.get("uc1_stem", [0.0] * 4)]
    mi1_stem: List[float] = [float(x) for x in agg.get("mi1_stem", [0.0] * 4)]
    uc1_time: List[float] = [float(x) for x in agg.get("uc1_time", [0.0] * 8)]
    mi1_time: List[float] = [float(x) for x in agg.get("mi1_time", [0.0] * 8)]
    stem_names: List[str] = agg.get("stem_names", ["drums", "bass", "other", "vocals"])
    seg_boundaries: List[float] = [float(x) for x in agg.get("segment_boundaries_sec", [])]

    # matrice 4×8 stems × segments
    uc1_matrix = [[float(v) for v in row] for row in agg.get("uc1_global_stem_segment", [])]
    mi1_matrix = [[float(v) for v in row] for row in agg.get("mi1_global_stem_segment", [])]

    # ---- DIME text ----
    wl = dime_json.get("wordlevel", {})
    prompt_wl = wl.get("prompt", {})
    question_words: List[str] = prompt_wl.get("words", [])
    uc2_words: List[float] = [float(x) for x in prompt_wl.get("uc2_global_word", [])]
    mi2_words: List[float] = [float(x) for x in prompt_wl.get("mi2_global_word", [])]

    # ---- base logit ----
    explanations = dime_json.get("explanations", {})
    base_ucmi = None
    if "0" in explanations:
        base_ucmi = explanations["0"].get("base_ucmi", None)
    if base_ucmi is not None:
        base_ucmi = float(base_ucmi)

    # ---- Scalari aggregati (norma L1 sul valore assoluto) ----
    # Usiamo L1 perché LIME con ridge restituisce valori positivi e negativi.
    # Per "quanto conta" una modalità, la norma L1 è più stabile della somma algebrica.
    uc_audio_l1 = float(np.sum(np.abs(uc1_stem)))
    mi_audio_l1 = float(np.sum(np.abs(mi1_stem)))
    uc_text_l1 = float(np.sum(np.abs(uc2_words)))
    mi_text_l1 = float(np.sum(np.abs(mi2_words)))

    # Somma algebrica (può essere negativa, utile per analisi di segno)
    uc_audio_sum = float(np.sum(uc1_stem))
    mi_audio_sum = float(np.sum(mi1_stem))
    uc_text_sum = float(np.sum(uc2_words))
    mi_text_sum = float(np.sum(mi2_words))

    return {
        # Identifiers
        "sample_id": meta["sample_id"],
        "idx": meta["idx"],
        "question": meta["question"],
        "category": meta["category"],
        "difficulty": meta["difficulty"],
        "macro_family": meta["macro_family"],

        # Risposta
        "model_answer": model_answer,
        "correct_answer_letter": correct_letter,
        "correct": correct,

        # MM-SHAP
        "a_shap": a_shap,
        "t_shap": t_shap,

        # DIME scalari (L1)
        "uc_audio_l1": uc_audio_l1,
        "mi_audio_l1": mi_audio_l1,
        "uc_text_l1": uc_text_l1,
        "mi_text_l1": mi_text_l1,

        # DIME scalari (somma algebrica)
        "uc_audio_sum": uc_audio_sum,
        "mi_audio_sum": mi_audio_sum,
        "uc_text_sum": uc_text_sum,
        "mi_text_sum": mi_text_sum,

        # DIME vettori per stem
        "stem_names": stem_names,
        "uc1_stem": uc1_stem,
        "mi1_stem": mi1_stem,

        # DIME vettori per segmento temporale
        "uc1_time": uc1_time,
        "mi1_time": mi1_time,
        "segment_boundaries_sec": seg_boundaries,

        # DIME matrici stems × segments
        "uc1_stem_x_seg": uc1_matrix,
        "mi1_stem_x_seg": mi1_matrix,

        # DIME testo per parola
        "question_words": question_words,
        "uc2_words": uc2_words,
        "mi2_words": mi2_words,

        # Valore base DIME
        "base_ucmi": base_ucmi,
    }

# =============================================================================
# 7) Cleanup artifacts (no plot, no L_tables)
# =============================================================================

def cleanup_run_artifacts(run_dir: str) -> None:
    """
    Rimuove tutti i file non essenziali da una directory di run singolo:
    - file .png (plot)
    - file .npy (L_tables)
    - directory L_tables/
    Mantiene solo i .json.
    """
    for ext in ("*.png", "*.npy"):
        for f in glob.glob(os.path.join(run_dir, "**", ext), recursive=True):
            try:
                os.remove(f)
            except Exception:
                pass

    ltables = os.path.join(run_dir, "L_tables")
    if os.path.isdir(ltables):
        try:
            shutil.rmtree(ltables)
        except Exception:
            pass

# =============================================================================
# 8) Checkpoint
# =============================================================================

def _load_progress(batch_dir: str) -> Dict[str, Any]:
    path = os.path.join(batch_dir, "progress.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "pending": []}


def _save_progress(batch_dir: str, progress: Dict[str, Any]) -> None:
    _atomic_json_dump(progress, os.path.join(batch_dir, "progress.json"))

# =============================================================================
# 9) Aggregazione finale
# =============================================================================

def aggregate_exp_a_results(batch_dir: str) -> str:
    """
    Legge tutti i {sample_id}_exp_a.json in per_sample/ e li
    aggrega in un parquet e un CSV nella cartella aggregated/.
    Restituisce il path del parquet.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas richiesto per l'aggregazione. pip install pandas")

    per_sample_dir = os.path.join(batch_dir, "per_sample")
    records = []
    for jf in sorted(glob.glob(os.path.join(per_sample_dir, "*_exp_a.json"))):
        with open(jf, "r") as f:
            records.append(json.load(f))

    if not records:
        logger.warning("Nessun file _exp_a.json trovato — aggregazione vuota.")
        return ""

    # Flatten: i campi lista (uc1_stem, mi1_stem, ecc.) vengono espansi
    # come colonne separate per facilitare l'analisi con pandas.
    flat_records = []
    for r in records:
        row = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}

        # Espandi uc1_stem / mi1_stem per stem
        for stem_idx, stem in enumerate(r.get("stem_names", ["drums", "bass", "other", "vocals"])):
            uc = r.get("uc1_stem", [])
            mi = r.get("mi1_stem", [])
            row[f"uc_stem_{stem}"] = float(uc[stem_idx]) if stem_idx < len(uc) else float("nan")
            row[f"mi_stem_{stem}"] = float(mi[stem_idx]) if stem_idx < len(mi) else float("nan")

        # Espandi uc1_time / mi1_time per segmento
        for seg_idx in range(8):
            uc_t = r.get("uc1_time", [])
            mi_t = r.get("mi1_time", [])
            row[f"uc_time_seg{seg_idx}"] = float(uc_t[seg_idx]) if seg_idx < len(uc_t) else float("nan")
            row[f"mi_time_seg{seg_idx}"] = float(mi_t[seg_idx]) if seg_idx < len(mi_t) else float("nan")

        flat_records.append(row)

    df = pd.DataFrame(flat_records)

    agg_dir = _ensure_dir(os.path.join(batch_dir, "aggregated"))
    parquet_path = os.path.join(agg_dir, "exp_a_results.parquet")
    csv_path = os.path.join(agg_dir, "exp_a_results.csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    logger.info(f"Aggregazione completata: {len(df)} sample → {parquet_path}")
    return parquet_path

# =============================================================================
# 10) Main batch loop
# =============================================================================

def run_batch_exp_a(
    resume_dir: Optional[str] = None,
    only_aggregate: bool = False,
) -> str:
    """
    Esegue il batch completo per Exp A.
    Se resume_dir è fornito, riprende il batch esistente.
    Se only_aggregate è True, salta il batch e aggrega solo i JSON esistenti.
    Restituisce il path della batch_dir.
    """

    # ----- Directory setup -----
    exp_root = os.path.join(EXPERIMENT_RESULTS_ROOT, "exp_A")
    _ensure_dir(exp_root)

    if resume_dir:
        batch_dir = resume_dir
        logger.info(f"Riprendendo batch esistente: {batch_dir}")
    else:
        batch_dir = _next_batch_dir(exp_root)
        logger.info(f"Nuova batch dir: {batch_dir}")

    per_sample_dir = _ensure_dir(os.path.join(batch_dir, "per_sample"))
    audio_cache_dir = _ensure_dir(os.path.join(batch_dir, "_audio_cache"))

    # ----- Solo aggregazione -----
    if only_aggregate:
        logger.info("Modalità only-aggregate: salto inferenza.")
        aggregate_exp_a_results(batch_dir)
        return batch_dir

    # ----- Carica entries HumMusQA -----
    logger.info(f"Caricamento HumMusQA da: {HUMMUSQA_ROOT}")
    entries, parquet_files = load_hummusqa_entries_parquet(HUMMUSQA_ROOT)
    valid_entries = [(i, e) for i, e in enumerate(entries) if _entry_has_valid_mcqa(e)]
    logger.info(f"Entry valide: {len(valid_entries)} / {len(entries)}")

    # Verifica colonne category/difficulty (early warning)
    if valid_entries:
        _, sample_entry = valid_entries[0]
        cat_found = any(k in sample_entry for k in _CATEGORY_FIELDS)
        diff_found = any(k in sample_entry for k in _DIFFICULTY_FIELDS)
        if not cat_found:
            logger.warning(
                f"⚠ Nessun campo category trovato nell'entry. "
                f"Campi disponibili: {list(sample_entry.keys())}. "
                f"Aggiorna _CATEGORY_FIELDS nel codice."
            )
        if not diff_found:
            logger.warning(
                f"⚠ Nessun campo difficulty trovato nell'entry. "
                f"Campi disponibili: {list(sample_entry.keys())}. "
                f"Aggiorna _DIFFICULTY_FIELDS nel codice."
            )

    # ----- Modello e runner -----
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path non trovato: {MODEL_PATH}")

    gpu_ids, _ = get_available_gpus_with_memory(min_free_memory_gb=MIN_FREE_GB_RUNNER)
    gpu_ids = gpu_ids[:MAX_GPUS_TO_USE]
    if not gpu_ids:
        raise RuntimeError(f"Nessuna GPU con >= {MIN_FREE_GB_RUNNER} GB liberi.")
    logger.info(f"GPU selezionate: {gpu_ids}")

    runner = try_create_parallel_runner(
        model_path=MODEL_PATH,
        min_free_memory_gb=MIN_FREE_GB_RUNNER,
        gpu_ids_physical=gpu_ids,
    )
    if runner is None:
        raise RuntimeError("Runner non attivo.")

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

    # ----- Salva run_info -----
    _atomic_json_dump(
        {
            "batch_dir": batch_dir,
            "exp": "A",
            "hummusqa_root": HUMMUSQA_ROOT,
            "parquet_files": [str(p) for p in parquet_files],
            "n_valid_entries": len(valid_entries),
            "gpu_ids": gpu_ids,
            "model_path": MODEL_PATH,
            "dime_module_config": get_dime_module_config_snapshot(),
        },
        os.path.join(batch_dir, "run_info.json"),
    )

    # ----- Pre-materializza tutti gli audio (con cache) -----
    logger.info("Pre-materializzazione audio (con cache disco)...")
    all_audio_paths: List[str] = []
    all_prompts: List[str] = []
    all_metas: List[Dict[str, Any]] = []

    for orig_idx, entry in valid_entries:
        meta = extract_hummusqa_meta(entry, orig_idx)
        try:
            audio_path = materialize_audio(entry, audio_cache_dir, orig_idx)
        except Exception as e:
            logger.warning(f"Audio skip per idx={orig_idx}: {e}")
            all_audio_paths.append("")
            all_prompts.append("")
            all_metas.append(meta)
            continue
        prompt = build_hummusqa_qwen25_prompt(meta["question"], meta["options"])
        all_audio_paths.append(audio_path)
        all_prompts.append(prompt)
        all_metas.append(meta)

    # ----- Checkpoint -----
    progress = _load_progress(batch_dir)
    completed_ids = set(progress.get("completed", []))
    failed_ids = set(progress.get("failed", []))

    total = len(valid_entries)
    logger.info(
        f"Stato checkpoint: {len(completed_ids)} completati, "
        f"{len(failed_ids)} falliti, "
        f"{total - len(completed_ids) - len(failed_ids)} da fare."
    )

    # ----- Loop principale -----
    try:
        for batch_pos, (orig_idx, entry) in enumerate(valid_entries):
            meta = all_metas[batch_pos]
            sample_id = meta["sample_id"]
            audio_path = all_audio_paths[batch_pos]
            prompt = all_prompts[batch_pos]

            if sample_id in completed_ids:
                logger.info(f"[{batch_pos+1}/{total}] SKIP (già completato): {sample_id}")
                continue

            if not audio_path or not os.path.exists(audio_path):
                logger.warning(f"[{batch_pos+1}/{total}] SKIP (audio mancante): {sample_id}")
                failed_ids.add(sample_id)
                progress["failed"] = list(failed_ids)
                _save_progress(batch_dir, progress)
                continue

            logger.info(
                f"[{batch_pos+1}/{total}] {sample_id} | "
                f"cat={meta['category']} | diff={meta['difficulty']} | "
                f"q={meta['question'][:60]}..."
            )

            t0 = time.time()
            try:
                # Directory temporanea per questo sample (verrà ripulita)
                tmp_run_dir = _ensure_dir(
                    os.path.join(batch_dir, "_tmp_run", sample_id)
                )
                dime_dir = _ensure_dir(os.path.join(tmp_run_dir, "dime"))
                mmshap_dir = _ensure_dir(os.path.join(tmp_run_dir, "mmshap"))

                # Background pairs: tutti gli altri audio del pool
                bg_audio = [p for i, p in enumerate(all_audio_paths)
                            if i != batch_pos and p and os.path.exists(p)]
                bg_prompts = [p for i, p in enumerate(all_prompts)
                              if i != batch_pos and p]
                # Limita a NUM_EXPECTATION_SAMPLES come nel run singolo
                n_bg = int(os.environ.get("DIME_NUM_EXPECTATION_SAMPLES", "16"))
                rng = np.random.RandomState(0)
                if len(bg_audio) > n_bg - 1:
                    sel = rng.choice(len(bg_audio), size=n_bg - 1, replace=False).tolist()
                    bg_audio = [bg_audio[i] for i in sel]
                    bg_prompts = [bg_prompts[i] for i in sel]

                # ---- Baseline condivisa ----
                caption = runner.generate_caption(
                    audio_path=audio_path,
                    prompt=prompt,
                )
                shared_baseline = {
                    "baseline_answer": caption,
                    "prompt": prompt,
                    "audio_path": audio_path,
                    "input_ids": None,
                    "output_ids": None,
                }

                # ---- MM-SHAP ----
                qa_entry = {
                    "sample_id": sample_id,
                    "audio_path": audio_path,
                    "question": meta["question"],
                    "options": meta["options"],
                    "correct_answer": meta["correct_answer_str"],
                }
                mmshap_results = analysis_1_start(
                    model=None,
                    processor=processor,
                    results_dir=mmshap_dir,
                    runner=runner,
                    qa_entry=qa_entry,
                    shared_baseline=shared_baseline,
                )

                # ---- DIME ----
                dime_results = analyze_dime(
                    model=None,
                    processor=processor,
                    audio_path=audio_path,
                    prompt=prompt,
                    caption=caption,
                    results_dir=dime_dir,
                    runner=runner,
                    background_audio_paths=bg_audio,
                    background_prompts=bg_prompts,
                    question=meta["question"],
                    options=meta["options"],
                    num_lime_samples=int(os.environ.get("DIME_NUM_LIME_SAMPLES", "512")),
                    num_features=int(os.environ.get("DIME_LIME_NUM_FEATURES_AUDIO", "16")),
                )

                # ---- Rimuovi plot e L_tables ----
                cleanup_run_artifacts(dime_dir)
                cleanup_run_artifacts(mmshap_dir)

                # ---- Leggi i JSON salvati dalle analisi ----
                dime_json_path = dime_results.get("json_path", "")
                mmshap_json_path = mmshap_results.get("json_path", "")

                with open(dime_json_path, "r") as f:
                    dime_json = json.load(f)
                with open(mmshap_json_path, "r") as f:
                    mmshap_json = json.load(f)

                # ---- Estrai campi Exp A ----
                exp_a_row = extract_exp_a_fields(dime_json, mmshap_json, meta)

                # ---- Salva per-sample JSON ----
                out_path = os.path.join(per_sample_dir, f"{sample_id}_exp_a.json")
                _atomic_json_dump(exp_a_row, out_path)

                # ---- Cleanup tmp ----
                shutil.rmtree(tmp_run_dir, ignore_errors=True)

                elapsed = time.time() - t0
                logger.info(
                    f"  ✓ {sample_id} completato in {elapsed:.0f}s | "
                    f"A-SHAP={exp_a_row['a_shap']:.3f} | "
                    f"correct={exp_a_row['correct']}"
                )

                completed_ids.add(sample_id)
                progress["completed"] = list(completed_ids)
                _save_progress(batch_dir, progress)

                gc.collect()

            except Exception as e:
                elapsed = time.time() - t0
                tb = traceback.format_exc()
                logger.error(
                    f"  ✗ {sample_id} FALLITO dopo {elapsed:.0f}s:\n{tb}"
                )
                failed_ids.add(sample_id)
                progress["failed"].append({
                    "sample_id": sample_id,
                    "error": str(e),
                    "traceback": tb[:2000],
                })
                _save_progress(batch_dir, progress)
                # Cleanup tmp anche in caso di errore
                try:
                    shutil.rmtree(
                        os.path.join(batch_dir, "_tmp_run", sample_id),
                        ignore_errors=True,
                    )
                except Exception:
                    pass
                gc.collect()
                continue

    finally:
        try:
            runner.stop()
        except Exception:
            pass
        # Cleanup cartella tmp generale
        try:
            shutil.rmtree(os.path.join(batch_dir, "_tmp_run"), ignore_errors=True)
        except Exception:
            pass

    # ----- Aggregazione finale -----
    logger.info("Batch completato. Avvio aggregazione...")
    aggregate_exp_a_results(batch_dir)

    logger.info(
        f"\n{'='*60}\n"
        f"BATCH EXP A COMPLETATO\n"
        f"  completati : {len(completed_ids)}\n"
        f"  falliti    : {len(failed_ids)}\n"
        f"  batch dir  : {batch_dir}\n"
        f"{'='*60}"
    )
    return batch_dir


# =============================================================================
# 11) Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch runner Esperimento A")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="BATCH_DIR",
        help="Riprendi un batch esistente dalla sua directory.",
    )
    parser.add_argument(
        "--only-aggregate",
        action="store_true",
        help="Salta il batch, aggrega solo i JSON già esistenti nella batch dir.",
    )
    args = parser.parse_args()

    if args.only_aggregate and not args.resume:
        raise ValueError("--only-aggregate richiede --resume <batch_dir>")

    run_batch_exp_a(
        resume_dir=args.resume,
        only_aggregate=args.only_aggregate,
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
