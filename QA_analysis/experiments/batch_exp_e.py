"""
Esperimento E — Batch runner
============================
Test di sufficienza e necessità delle spiegazioni DIME/MM-SHAP.

Per ogni sample HumMusQA:
  1) Legge i ranking di feature da {sample_id}_exp_a.json (Exp A già completato)
  2) Per ogni (feature_source ∈ {UC_audio, UC_text, MI}, k ∈ {3,5,8}):
       - Costruisce input perturbati per sufficiency e necessity
       - Inferenza in batch sul runner multi-GPU
       - Salva logit, risposta, score, verdict
  3) Aggrega tutto in parquet long-format

Output:
    Results_QA/experiments/exp_E/batch_run_XX/
        run_info.json
        progress.json
        per_sample/{sample_id}_exp_e.json
        aggregated/exp_e_long.parquet + .csv

Utilizzo:
    python -m QA_analysis.experiments.batch_exp_e --exp-a-dir <PATH_BATCH_A>
    python -m QA_analysis.experiments.batch_exp_e --exp-a-dir <PATH_BATCH_A> --resume <PATH_BATCH_E>
"""

import os
import re
import gc
import sys
import json
import glob
import time
import shutil
import logging
import argparse
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# 0a) Soppressione warning verbose
# =============================================================================
import warnings as _warnings
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=FutureWarning)
_warnings.filterwarnings("ignore", category=UserWarning)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_LOGGING_VERBOSITY", "3")
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning"
)

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    for _mod in ("aifc", "audioop", "sunau"):
        try:
            __import__(_mod)
        except Exception:
            pass

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold("error")
except Exception:
    pass

# =============================================================================
# 0b) Filtro stderr (riusa la logica di batch_exp_a)
# =============================================================================
_NOISE_PATTERNS = (
    "aifc", "audioop", "sunau",
    "is deprecated and slated for removal",
    "DeprecationWarning", "FutureWarning",
    "oneDNN custom operations are on", "oneDNN",
    "absl::InitializeLog", "All log messages before",
    "I0000 00:", "port.cc:",
)

class _StderrFilter:
    def __init__(self, real):
        self._real = real
        self._buf = ""

    def write(self, s):
        if not s:
            return
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit_line(line + "\n")

    def _emit_line(self, line):
        stripped = line.strip()
        if not stripped:
            self._real.write(line)
            return
        if any(k in stripped for k in ("Traceback", "Error:", "Exception:", "CRITICAL", "FAIL")):
            self._real.write(line)
            return
        if any(p in stripped for p in _NOISE_PATTERNS):
            return
        self._real.write(line)

    def flush(self):
        if self._buf:
            self._emit_line(self._buf)
            self._buf = ""
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)

sys.stderr = _StderrFilter(sys.stderr)

# =============================================================================
# 0c) ENV — coerente con Exp A
# =============================================================================
def _apply_env() -> None:
    env_cfg = {
        "TRANSFORMERS_NO_TF": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "DIME_VALUE_MODE": "logit",

        # Audio masking — STESSI DEFAULT di DIME
        "DIME_AUDIO_FEATURE_MODE": "audiolime_demucs",
        "DIME_AUDIO_MASK_MODE": "bg_random_energy",   # <— DEFAULT DIME
        "DIME_SILENCE_RMS": "1e-3",
        "DIME_CROSSFADE": "1",
        "DIME_CROSSFADE_MS": "5.0",

        # AudioLIME — riusa cache demucs di Exp A
        "DIME_AUDIOLIME_DEMUCS_MODEL": "htdemucs",
        "DIME_AUDIOLIME_USE_PRECOMPUTED": "1",
        "DIME_AUDIOLIME_RECOMPUTE": "0",
        "DIME_AUDIOLIME_DEMUCS_DEVICE": "cpu",
        "DIME_AUDIOLIME_DEMUCS_SEGMENT": "8",
        "DIME_AUDIOLIME_DEMUCS_SPLIT": "1",
        "DIME_AUDIOLIME_DEMUCS_OVERLAP": "0.25",
        "DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS": "8",
        "DIME_AUDIOLIME_PRECOMPUTED_DIR": "/nas/home/fingenito/Thesis_project/QA_analysis/data/demucs_cache",
        "DIME_AUDIOLIME_NORMALIZE_COMPOSITION": "1",

        # Runtime
        "MMSHAP_QUEUE_WINDOW": "128",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "TORCH_HOME": "/nas/home/fingenito/Thesis_project/QA_analysis/data/torch_cache",
        "TORCH_HUB": "/nas/home/fingenito/Thesis_project/QA_analysis/data/torch_cache/hub",
    }
    for k, v in env_cfg.items():
        os.environ[k] = str(v)

_apply_env()

# =============================================================================
# 1) Import progetto
# =============================================================================
from transformers import Qwen2_5OmniProcessor

from QA_analysis.utils.gpu_utils import (
    try_create_parallel_runner,
    get_available_gpus_with_memory,
)
from QA_analysis.experiments.perturbations import (
    rank_audio_features,
    rank_text_words,
    select_top_k,
    build_audio_binary_mask,
    materialize_perturbed_audio,
    build_perturbed_prompt,
    get_letter_token_ids,
    answer_letter_from_logits,
    N_AUDIO_FEATURES,
)

# =============================================================================
# 2) Costanti
# =============================================================================
EXPERIMENT_RESULTS_ROOT = "/nas/home/fingenito/Thesis_project/QA_analysis/Results_QA/experiments"
MODEL_PATH = "/nas/home/fingenito/Models/Qwen2.5-Omni-7B"

MAX_GPUS_TO_USE = 8
MIN_FREE_GB_RUNNER = 21.0

# Iperparametri Exp E
K_VALUES: List[int] = [3, 5, 8]
FEATURE_SOURCES: List[str] = ["UC_audio", "UC_text", "MI"]

# Soglie per la classificazione del verdetto
SUFFICIENCY_THRESHOLD: float = 0.7   # >= 0.7 → sufficiente
NECESSITY_THRESHOLD: float = 0.4     # <= 0.4 → necessaria

# TEST MODE
TEST_FIRST_N_SAMPLES: Optional[int] = 5

# =============================================================================
# 3) Logging
# =============================================================================
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("ExpE-Batch")
_h = logging.StreamHandler(sys.stdout)
_h.setLevel(logging.INFO)
_h.setFormatter(logging.Formatter("%(asctime)s [ExpE] %(levelname)s — %(message)s"))
logger.handlers = [_h]
logger.setLevel(logging.INFO)
logger.propagate = False

for _noisy in ["transformers", "huggingface_hub", "datasets", "urllib3",
               "qwen_omni_utils", "QwenOmni", "tensorflow", "jax", "grpc",
               "absl", "torch", "matplotlib", "demucs", "demucs.api"]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)
    logging.getLogger(_noisy).propagate = False

try:
    from transformers.utils import logging as _hf_log
    _hf_log.set_verbosity_error()
    _hf_log.disable_progress_bar()
except Exception:
    pass


# =============================================================================
# 4) Helpers
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
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json",
                                dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def _json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def _load_progress(batch_dir: str) -> Dict[str, Any]:
    path = os.path.join(batch_dir, "progress.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def _save_progress(batch_dir: str, progress: Dict[str, Any]) -> None:
    _atomic_json_dump(progress, os.path.join(batch_dir, "progress.json"))


# =============================================================================
# 5) Carica i risultati di Exp A per un sample
# =============================================================================
def load_exp_a_sample(exp_a_dir: str, sample_id: str) -> Dict[str, Any]:
    """Carica il JSON di Exp A per un singolo sample."""
    path = os.path.join(exp_a_dir, "per_sample", f"{sample_id}_exp_a.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Exp A JSON non trovato: {path}")
    with open(path, "r") as f:
        return json.load(f)


def list_available_exp_a_samples(exp_a_dir: str) -> List[str]:
    """Lista tutti i sample_id per cui esiste un risultato Exp A."""
    pattern = os.path.join(exp_a_dir, "per_sample", "*_exp_a.json")
    files = sorted(glob.glob(pattern))
    return [os.path.basename(f).replace("_exp_a.json", "") for f in files]


# =============================================================================
# 6) Estrazione ranking per le 3 feature_source
# =============================================================================
def get_rankings_from_exp_a(exp_a_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per il sample, costruisce 3 ranking ordinati per |valore|:
      - UC_audio: 32 feature (4 stems × 8 segments) da uc1_stem_x_seg
      - UC_text:  N parole da uc2_words
      - MI:       sia 32 audio (mi1_stem_x_seg) che N testo (mi2_words),
                  fusi in un unico ranking ordinato per |valore|

    Nota su MI: per coerenza con il design, "MI" è una feature_source unica che
    può estrarre top-k che mescola audio e testo. Quando perturbiamo, separiamo
    il top-k in audio_part e text_part.
    """
    uc_audio_ranking = rank_audio_features(exp_a_data.get("uc1_stem_x_seg", []))
    mi_audio_ranking = rank_audio_features(exp_a_data.get("mi1_stem_x_seg", []))

    word_strings = exp_a_data.get("question_words", [])
    uc_text_ranking = rank_text_words(exp_a_data.get("uc2_words", []), word_strings)
    mi_text_ranking = rank_text_words(exp_a_data.get("mi2_words", []), word_strings)

    # MI: fonde audio + testo in un unico ranking
    mi_combined = []
    for f in mi_audio_ranking:
        mi_combined.append({**f, "modality": "audio"})
    for f in mi_text_ranking:
        mi_combined.append({**f, "modality": "text"})
    mi_combined.sort(key=lambda d: d["abs_value"], reverse=True)

    return {
        "UC_audio": [{**f, "modality": "audio"} for f in uc_audio_ranking],
        "UC_text":  [{**f, "modality": "text"}  for f in uc_text_ranking],
        "MI":       mi_combined,
    }


def split_topk_by_modality(top_k: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """Separa una lista mista in (audio_features, text_words)."""
    audio_part = [f for f in top_k if f.get("modality") == "audio"]
    text_part  = [f for f in top_k if f.get("modality") == "text"]
    return audio_part, text_part


# =============================================================================
# 7) Costruzione input perturbato (audio + testo) per una singola condizione
# =============================================================================
def build_perturbation_inputs(
    exp_a_data: Dict[str, Any],
    feature_source: str,
    k: int,
    mode: str,
    rankings: Dict[str, Any],
    audio_path: str,
    options: List[str],
    tokenizer,
    perturb_audio_dir: str,
) -> Dict[str, Any]:
    """
    Costruisce (perturbed_audio_path, perturbed_prompt) per una singola condizione.

    Logica per le 3 feature_source:
      - UC_audio: solo audio viene perturbato; testo invariato (prompt originale)
      - UC_text:  solo testo viene perturbato; audio invariato (path originale)
      - MI:       sia audio che testo vengono perturbati (top-k spalmati su entrambi)
    """
    question = exp_a_data.get("question", "")
    n_words_total = len(exp_a_data.get("question_words", []))

    ranked = rankings[feature_source]
    top_k = select_top_k(ranked, k)

    audio_part, text_part = split_topk_by_modality(top_k)

    # ---- AUDIO ----
    if feature_source == "UC_text":
        # nessuna perturbazione audio
        out_audio_path = audio_path
    else:
        if not audio_part:
            # Top-k tutto testo? Allora audio invariato (caso bordo per MI)
            out_audio_path = audio_path
        else:
            am = build_audio_binary_mask(audio_part, mode=mode)
            tag = f"_{feature_source}_k{k}_{mode}"
            out_audio_path = materialize_perturbed_audio(
                audio_path=audio_path,
                audio_binary_mask=am,
                out_dir=perturb_audio_dir,
                suffix=tag,
            )

    # ---- TESTO ----
    if feature_source == "UC_audio":
        # nessuna perturbazione testo
        from QA_analysis.utils.shared_utils import build_hummusqa_qwen25_prompt
        out_prompt = build_hummusqa_qwen25_prompt(question, options)
    else:
        if not text_part and feature_source == "MI":
            # Top-k tutto audio? Allora testo invariato
            from QA_analysis.utils.shared_utils import build_hummusqa_qwen25_prompt
            out_prompt = build_hummusqa_qwen25_prompt(question, options)
        else:
            out_prompt = build_perturbed_prompt(
                question=question,
                options=options,
                top_k_words=text_part if text_part else [],
                mode=mode,
                tokenizer=tokenizer,
                n_words_total=n_words_total,
            )

    return {
        "audio_path": out_audio_path,
        "prompt": out_prompt,
        "n_audio_features_perturbed": len(audio_part),
        "n_text_words_perturbed": len(text_part),
    }


# =============================================================================
# 8) Calcolo metriche e verdetto causale
# =============================================================================
def compute_scores_and_verdict(
    logit_orig: float,
    logit_suf: float,
    logit_nec: float,
    answer_orig: str,
    answer_suf: str,
    answer_nec: str,
) -> Dict[str, Any]:
    """
    Score:
      sufficiency_score = logit_suf / logit_orig  → alto = top-k da sole bastano
      necessity_score   = logit_nec / logit_orig  → basso = togliere top-k distrugge

    Edge cases:
      - logit_orig <= 0: si lavora in spazio logit, può capitare. In quel caso
        usiamo delta normalizzato dal max(|logit_orig|, |logit_suf|, |logit_nec|).
    """
    eps = 1e-6
    if abs(logit_orig) > eps:
        sufficiency_score = logit_suf / logit_orig
        necessity_score = logit_nec / logit_orig
    else:
        denom = max(abs(logit_orig), abs(logit_suf), abs(logit_nec), eps)
        sufficiency_score = (logit_suf - logit_orig) / denom + 1.0
        necessity_score = (logit_nec - logit_orig) / denom + 1.0

    suf_delta = float(logit_suf - logit_orig)
    nec_delta = float(logit_nec - logit_orig)
    suf_changed = (answer_suf != answer_orig)
    nec_changed = (answer_nec != answer_orig)

    # Verdetto
    is_sufficient = (sufficiency_score >= SUFFICIENCY_THRESHOLD) and (not suf_changed)
    is_necessary = (necessity_score <= NECESSITY_THRESHOLD) or nec_changed

    if is_sufficient and is_necessary:
        verdict = "causal_strong"
    elif is_sufficient and not is_necessary:
        verdict = "redundant"
    elif is_necessary and not is_sufficient:
        verdict = "critical_not_sufficient"
    else:
        verdict = "decorative"

    return {
        "logit_original": float(logit_orig),
        "logit_sufficiency": float(logit_suf),
        "logit_necessity": float(logit_nec),
        "sufficiency_score": float(sufficiency_score),
        "necessity_score": float(necessity_score),
        "suf_delta_logit": suf_delta,
        "nec_delta_logit": nec_delta,
        "answer_original": answer_orig,
        "answer_sufficiency": answer_suf,
        "answer_necessity": answer_nec,
        "suf_answer_changed": suf_changed,
        "nec_answer_changed": nec_changed,
        "is_sufficient": is_sufficient,
        "is_necessary": is_necessary,
        "causal_verdict": verdict,
    }


# =============================================================================
# 9) Process singolo sample
# =============================================================================
def process_sample(
    sample_id: str,
    exp_a_dir: str,
    audio_cache_dir: str,
    perturb_audio_dir: str,
    runner,
    tokenizer,
    letter_token_ids: List[int],
) -> Dict[str, Any]:
    """
    Esegue Exp E completo su un sample.
    Restituisce un dict serializzabile pronto per il salvataggio.
    """
    exp_a_data = load_exp_a_sample(exp_a_dir, sample_id)

    # Risolvi audio_path: i WAV sono in audio_cache di Exp A
    # Usiamo una convenzione: audio_cache è copiato dal batch di Exp A
    # Il sample.id non basta — dobbiamo trovare il file giusto
    audio_path = _resolve_audio_path(audio_cache_dir, exp_a_data)
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio non trovato per {sample_id} in {audio_cache_dir}. "
            f"Verifica che il batch di Exp A sia stato eseguito correttamente."
        )

    options = [
        exp_a_data.get("correct_answer_letter", "A"),  # placeholder, vero answer è in options
    ]
    # Ricostruisci options dalla domanda originale — ma non è in JSON Exp A.
    # Workaround: leggi dal HumMusQA parquet direttamente.
    options = _resolve_options_for_sample(sample_id)

    rankings = get_rankings_from_exp_a(exp_a_data)

    # ---- Calcola logit baseline (input non perturbato) ----
    from QA_analysis.utils.shared_utils import build_hummusqa_qwen25_prompt
    base_prompt = build_hummusqa_qwen25_prompt(exp_a_data["question"], options)

    base_logits = runner.get_mmshap_logits(
        audio_path=audio_path,
        prompt=base_prompt,
        target_ids=letter_token_ids,
    )
    answer_orig = answer_letter_from_logits(base_logits)
    correct_letter = exp_a_data.get("correct_answer_letter", "A")
    correct_idx = ord(correct_letter) - 65
    logit_orig_correct = float(base_logits[correct_idx])

    # ---- Costruisci tutte le 18 perturbazioni ----
    items: List[Dict[str, Any]] = []
    item_index_to_meta: List[Tuple[str, int, str]] = []  # (feature_source, k, mode)

    for fs in FEATURE_SOURCES:
        for k in K_VALUES:
            for mode in ["sufficiency", "necessity"]:
                inp = build_perturbation_inputs(
                    exp_a_data=exp_a_data,
                    feature_source=fs,
                    k=k,
                    mode=mode,
                    rankings=rankings,
                    audio_path=audio_path,
                    options=options,
                    tokenizer=tokenizer,
                    perturb_audio_dir=perturb_audio_dir,
                )
                items.append({
                    "audio_path": inp["audio_path"],
                    "prompt": inp["prompt"],
                    "target_ids": letter_token_ids,
                })
                item_index_to_meta.append((fs, k, mode))

    # ---- Inferenza in batch parallela su tutte le GPU ----
    all_logits = runner.get_mmshap_logits_batch(items=items)

    # ---- Costruisci risultati per (fs, k) accoppiando suf+nec ----
    results: List[Dict[str, Any]] = []

    # Mappa (fs, k, mode) → (logit_correct, answer)
    by_key = {}
    for i, (fs, k, mode) in enumerate(item_index_to_meta):
        logits_4 = all_logits[i]
        if not logits_4 or len(logits_4) < 4:
            logger.warning(f"  Logit incompleti per {fs}/k={k}/{mode} — skip")
            continue
        ans = answer_letter_from_logits(logits_4)
        logit_correct = float(logits_4[correct_idx])
        by_key[(fs, k, mode)] = (logit_correct, ans, logits_4)

    for fs in FEATURE_SOURCES:
        for k in K_VALUES:
            if (fs, k, "sufficiency") not in by_key or (fs, k, "necessity") not in by_key:
                continue
            l_suf, a_suf, _ = by_key[(fs, k, "sufficiency")]
            l_nec, a_nec, _ = by_key[(fs, k, "necessity")]

            # Top-k feature usate (per riferimento)
            top_k = select_top_k(rankings[fs], k)
            top_k_repr = [
                {
                    "modality": f.get("modality", ""),
                    "stem_idx": f.get("stem_idx", None),
                    "segment_idx": f.get("segment_idx", None),
                    "word_idx": f.get("word_idx", None),
                    "word": f.get("word", None),
                    "value": f.get("value", 0.0),
                }
                for f in top_k
            ]

            scores = compute_scores_and_verdict(
                logit_orig=logit_orig_correct,
                logit_suf=l_suf,
                logit_nec=l_nec,
                answer_orig=answer_orig,
                answer_suf=a_suf,
                answer_nec=a_nec,
            )

            results.append({
                "feature_source": fs,
                "k": k,
                "top_features": top_k_repr,
                **scores,
            })

    return {
        "sample_id": sample_id,
        "category": exp_a_data.get("category", "unknown"),
        "difficulty": exp_a_data.get("difficulty", "unknown"),
        "macro_family": exp_a_data.get("macro_family", "unknown"),
        "correct_answer_letter": correct_letter,
        "model_answer_baseline": answer_orig,
        "correct_baseline": (answer_orig == correct_letter),
        "logit_original_correct": logit_orig_correct,

        "masking_strategy": {
            "audio_mode": os.environ.get("DIME_AUDIO_MASK_MODE", "bg_random_energy"),
            "audio_crossfade_ms": float(os.environ.get("DIME_CROSSFADE_MS", "5.0")),
            "text_mask_token": "[MASK]",
            "options_never_masked": True,
        },
        "thresholds": {
            "sufficiency": SUFFICIENCY_THRESHOLD,
            "necessity": NECESSITY_THRESHOLD,
        },

        "results": results,
    }


# =============================================================================
# 10) Helpers per risolvere audio_path e options del sample
# =============================================================================
_audio_path_cache: Dict[str, str] = {}
_options_cache: Dict[str, List[str]] = {}
_hummusqa_entries_cache: Optional[List[dict]] = None


def _load_hummusqa_entries():
    """Lazy load del parquet HumMusQA per options + audio."""
    global _hummusqa_entries_cache
    if _hummusqa_entries_cache is None:
        from QA_analysis.utils.shared_utils import load_hummusqa_entries_parquet
        entries, _ = load_hummusqa_entries_parquet("/nas/home/fingenito/HumMusQA/data")
        _hummusqa_entries_cache = entries
    return _hummusqa_entries_cache


def _resolve_audio_path(audio_cache_dir: str, exp_a_data: Dict[str, Any]) -> Optional[str]:
    """
    Cerca il WAV materializzato del sample nella audio_cache di Exp A.
    Convenzione di naming: sample_{idx}_{sha16}_decoded.wav
    """
    sample_id = exp_a_data.get("sample_id", "")
    if sample_id in _audio_path_cache:
        return _audio_path_cache[sample_id]

    idx = exp_a_data.get("idx", -1)
    if idx < 0:
        return None

    pattern = os.path.join(audio_cache_dir, f"sample_{idx}_*_decoded.wav")
    matches = glob.glob(pattern)
    if not matches:
        # fallback senza _decoded
        pattern = os.path.join(audio_cache_dir, f"sample_{idx}_*.wav")
        matches = glob.glob(pattern)

    if matches:
        _audio_path_cache[sample_id] = matches[0]
        return matches[0]
    return None


def _resolve_options_for_sample(sample_id: str) -> List[str]:
    """Risolve le options A/B/C/D dal parquet HumMusQA originale."""
    if sample_id in _options_cache:
        return _options_cache[sample_id]

    entries = _load_hummusqa_entries()
    for entry in entries:
        ident = str(entry.get("identifier") or entry.get("question_id") or entry.get("id") or "")
        if ident == sample_id:
            opts = [
                str(entry.get("answer", "")).strip(),
                str(entry.get("distractor_1", "")).strip(),
                str(entry.get("distractor_2", "")).strip(),
                str(entry.get("distractor_3", "")).strip(),
            ]
            _options_cache[sample_id] = opts
            return opts

    raise RuntimeError(f"Options non trovate per sample {sample_id} nel parquet HumMusQA")


# =============================================================================
# 11) Aggregazione long format
# =============================================================================
def aggregate_exp_e_long(batch_dir: str) -> str:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas richiesto. pip install pandas")

    per_sample_dir = os.path.join(batch_dir, "per_sample")
    rows = []
    for jf in sorted(glob.glob(os.path.join(per_sample_dir, "*_exp_e.json"))):
        with open(jf, "r") as f:
            data = json.load(f)
        common = {
            "sample_id": data.get("sample_id"),
            "category": data.get("category"),
            "difficulty": data.get("difficulty"),
            "macro_family": data.get("macro_family"),
            "correct_baseline": data.get("correct_baseline"),
            "logit_original_correct": data.get("logit_original_correct"),
        }
        for r in data.get("results", []):
            row = dict(common)
            for key in ["feature_source", "k",
                        "logit_sufficiency", "logit_necessity",
                        "sufficiency_score", "necessity_score",
                        "suf_delta_logit", "nec_delta_logit",
                        "suf_answer_changed", "nec_answer_changed",
                        "is_sufficient", "is_necessary", "causal_verdict"]:
                row[key] = r.get(key)
            rows.append(row)

    if not rows:
        logger.warning("Nessun risultato Exp E da aggregare.")
        return ""

    import pandas as pd
    df = pd.DataFrame(rows)

    agg_dir = _ensure_dir(os.path.join(batch_dir, "aggregated"))
    parquet_path = os.path.join(agg_dir, "exp_e_long.parquet")
    csv_path = os.path.join(agg_dir, "exp_e_long.csv")
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    logger.info(f"Aggregazione Exp E: {len(df)} righe → {parquet_path}")
    return parquet_path


# =============================================================================
# 12) Main batch loop
# =============================================================================
def run_batch_exp_e(
    exp_a_dir: str,
    resume_dir: Optional[str] = None,
    only_aggregate: bool = False,
) -> str:
    if not os.path.isdir(exp_a_dir):
        raise FileNotFoundError(f"Exp A dir non trovata: {exp_a_dir}")

    exp_root = os.path.join(EXPERIMENT_RESULTS_ROOT, "exp_E")
    _ensure_dir(exp_root)

    if resume_dir:
        batch_dir = resume_dir
        logger.info(f"Riprendendo batch esistente: {batch_dir}")
    else:
        batch_dir = _next_batch_dir(exp_root)
        logger.info(f"Nuova batch dir: {batch_dir}")

    per_sample_dir = _ensure_dir(os.path.join(batch_dir, "per_sample"))
    perturb_audio_dir = _ensure_dir(os.path.join(batch_dir, "_perturbed_audio"))
    audio_cache_dir = os.path.join(exp_a_dir, "_audio_cache")

    if only_aggregate:
        aggregate_exp_e_long(batch_dir)
        return batch_dir

    # ---- Sample disponibili ----
    sample_ids = list_available_exp_a_samples(exp_a_dir)
    logger.info(f"Sample disponibili da Exp A: {len(sample_ids)}")

    if TEST_FIRST_N_SAMPLES is not None:
        sample_ids = sample_ids[:int(TEST_FIRST_N_SAMPLES)]
        logger.info(f"[TEST MODE] Uso solo i primi {len(sample_ids)} sample.")

    # ---- Runner ----
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
    tokenizer = processor.tokenizer
    letter_token_ids = get_letter_token_ids(tokenizer)
    logger.info(f"Letter token ids (A,B,C,D): {letter_token_ids}")

    # ---- run_info ----
    _atomic_json_dump(
        {
            "batch_dir": batch_dir,
            "exp": "E",
            "exp_a_dir": exp_a_dir,
            "n_sample_ids": len(sample_ids),
            "test_first_n_samples": TEST_FIRST_N_SAMPLES,
            "feature_sources": FEATURE_SOURCES,
            "k_values": K_VALUES,
            "sufficiency_threshold": SUFFICIENCY_THRESHOLD,
            "necessity_threshold": NECESSITY_THRESHOLD,
            "masking": {
                "audio_mode": os.environ.get("DIME_AUDIO_MASK_MODE"),
                "text_token": "[MASK]",
            },
            "gpu_ids": gpu_ids,
            "model_path": MODEL_PATH,
            "letter_token_ids": letter_token_ids,
        },
        os.path.join(batch_dir, "run_info.json"),
    )

    # ---- Checkpoint ----
    progress = _load_progress(batch_dir)
    completed_ids = set(progress.get("completed", []))
    total = len(sample_ids)
    logger.info(f"Stato: {len(completed_ids)} completati, "
                f"{total - len(completed_ids)} da fare.")

    # ---- Loop ----
    try:
        for pos, sid in enumerate(sample_ids):
            if sid in completed_ids:
                logger.info(f"[{pos+1}/{total}] SKIP (già completato): {sid}")
                continue

            logger.info(f"[{pos+1}/{total}] Processing: {sid}")
            t0 = time.time()
            try:
                result = process_sample(
                    sample_id=sid,
                    exp_a_dir=exp_a_dir,
                    audio_cache_dir=audio_cache_dir,
                    perturb_audio_dir=perturb_audio_dir,
                    runner=runner,
                    tokenizer=tokenizer,
                    letter_token_ids=letter_token_ids,
                )

                out_path = os.path.join(per_sample_dir, f"{sid}_exp_e.json")
                _atomic_json_dump(result, out_path)

                # Cleanup audio perturbati di questo sample
                for f in glob.glob(os.path.join(perturb_audio_dir, "*")):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

                elapsed = time.time() - t0
                n_strong = sum(1 for r in result["results"]
                               if r.get("causal_verdict") == "causal_strong")
                n_decor = sum(1 for r in result["results"]
                              if r.get("causal_verdict") == "decorative")
                logger.info(
                    f"  ✓ {sid} in {elapsed:.0f}s | "
                    f"strong={n_strong}/18 | decor={n_decor}/18"
                )

                completed_ids.add(sid)
                progress["completed"] = sorted(completed_ids)
                _save_progress(batch_dir, progress)
                gc.collect()

            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"  ✗ {sid} FAILED in {time.time()-t0:.0f}s:\n{tb}")
                if not isinstance(progress.get("failed"), list):
                    progress["failed"] = []
                progress["failed"].append({
                    "sample_id": sid,
                    "error": str(e),
                    "traceback": tb[:2000],
                })
                _save_progress(batch_dir, progress)
                gc.collect()
                continue

    finally:
        try:
            runner.stop()
        except Exception:
            pass
        try:
            shutil.rmtree(perturb_audio_dir, ignore_errors=True)
        except Exception:
            pass

    # ---- Aggregazione ----
    logger.info("Batch completato. Aggregazione...")
    aggregate_exp_e_long(batch_dir)

    logger.info(
        f"\n{'='*60}\n"
        f"BATCH EXP E COMPLETATO\n"
        f"  completati : {len(completed_ids)}\n"
        f"  batch dir  : {batch_dir}\n"
        f"{'='*60}"
    )
    return batch_dir


# =============================================================================
# 13) Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Batch runner Esperimento E")
    parser.add_argument("--exp-a-dir", required=True, type=str,
                        help="Path della batch dir di Exp A (es. .../exp_A/batch_run_00)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Riprendi una batch dir di Exp E esistente.")
    parser.add_argument("--only-aggregate", action="store_true")
    args = parser.parse_args()

    if args.only_aggregate and not args.resume:
        raise ValueError("--only-aggregate richiede --resume")

    run_batch_exp_e(
        exp_a_dir=args.exp_a_dir,
        resume_dir=args.resume,
        only_aggregate=args.only_aggregate,
    )


if __name__ == "__main__":
    main()