import os
import re
import json
import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from transformers import (
    Qwen2_5OmniProcessor,
)

# SOSTITUISCI con:
_CATEGORY_FIELDS = ["main_category", "category", "Category"]
_DIFFICULTY_FIELDS = ["difficulty", "Difficulty"]

MACRO_FAMILY_MAP: Dict[str, str] = {
    "Instrumentation":   "percettiva",
    "Sound Texture":     "percettiva",
    "Metre and Rhythm":  "percettiva",
    "Musical Texture":   "percettiva",
    "Harmony":           "analitica",
    "Melody":            "analitica",
    "Structure":         "analitica",
    "Performance":       "analitica",
    "Genre and Style":   "knowledge",
    "Historical":        "knowledge",
    "Mood and Expression": "knowledge",
}

# =============================================================================
# 0) ENV: TUTTI i settaggi DEVONO essere fissati qui, prima degli import progetto
# =============================================================================

def _apply_reproducible_env() -> None:
    """
    Imposta in modo RIPRODUCIBILE tutte le env rilevanti per il run.
    Non usa setdefault sui parametri sperimentali, così evitiamo environment leakage
    da shell/tmux/export precedenti.
    """

    env_cfg = {
        "TRANSFORMERS_NO_TF": "1",
        "DIME_BG_AUDIO_DIR": "...",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "DIME_VALUE_MODE": "logit",
        "DIME_PROMPTS_FILE": "",

        # ==========================================================
        # QUALITY — invariati, scelta metodologica
        # ==========================================================
        "DIME_NUM_EXPECTATION_SAMPLES": "16",
        "DIME_NUM_LIME_SAMPLES": "512",
        "DIME_LIME_NUM_FEATURES_AUDIO": "16",
        "DIME_LIME_NUM_FEATURES_TEXT": "10",

        # ==========================================================
        # SCHEDULING / BATCHING — ottimizzati per 8 GPU
        # ==========================================================
        # QUEUE_WINDOW: più largo = GPUs sempre occupate
        # Con 8 GPU e task da ~25s ognuno, vogliamo almeno 3-4 wave
        # pre-queued → 8 GPU × 4 wave = 32 task → ma task size=8 audio
        # → 32 × 8 = 256 audio pre-queued. Mettiamo 320 per sicurezza.
        "DIME_QUEUE_WINDOW": "320",  # era 160

        # L_BATCH_SIZE: N=16, 256 pairs totali.
        # Con batch=8 → 32 chunk → 8 GPU ottengono 4 round ciascuna.
        # Con batch=16 → 16 chunk → 8 GPU ottengono 2 round.
        # Più fine-grained = meglio per load balancing → teniamo 8.
        "DIME_L_BATCH_SIZE": "8",

        # LIME_BATCH_SIZE: chunk size per GPU per task.
        # 8 audio per task × PKV: ~25s per task → ottimale per granularità.
        # Non aumentare: task più grandi = più code lag.
        "DIME_LIME_BATCH_SIZE": "8",

        # INNER_BATCH_SIZE: usato solo nel fallback (PKV disabilitato).
        # Teniamo 8 come safety net.
        "DIME_WORKER_INNER_BATCH_SIZE": "8",

        # AUDIO_PERTURB_BATCH: quanti audio perturbati mandare in blocco.
        # 128 → 128/8 = 16 task chunk → 8 GPU ricevono 2 round di lavoro
        # pre-caricato → migliore pipeline vs 64 (1 round solo).
        "DIME_STEP5_AUDIO_PERTURB_BATCH": "128",  # era 64

        # TEXT_PERTURB_BATCH: quanti prompt mascherati mandare in blocco.
        # 256 → 256/8 = 32 task chunk → 8 GPU × 4 round pre-caricato.
        # Più granulare = GPU non aspettano mai.
        "DIME_STEP5_TEXT_PERTURB_BATCH": "256",  # era 128

        # ==========================================================
        # AUDIO / DEMUCS — invariati
        # ==========================================================
        "DIME_AUDIO_FEATURE_MODE": "audiolime_demucs",
        "DIME_AUDIOLIME_DEMUCS_MODEL": "htdemucs",
        "DIME_AUDIOLIME_USE_PRECOMPUTED": "1",
        "DIME_AUDIOLIME_RECOMPUTE": "0",
        "DIME_AUDIOLIME_PRECOMPUTED_DIR": "/nas/home/fingenito/Thesis_project/QA_analysis/data/demucs_cache",
        "DIME_AUDIOLIME_DEMUCS_DEVICE": "cpu",
        "DIME_AUDIOLIME_DEMUCS_SEGMENT": "8",
        "DIME_AUDIOLIME_DEMUCS_SPLIT": "1",
        "DIME_AUDIOLIME_DEMUCS_OVERLAP": "0.25",

        # ==========================================================
        # AUDIO FEATURES — invariati
        # ==========================================================
        "DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS": "8",
        "DIME_AUDIOLIME_MAX_FEATURES": "40",
        "DIME_AUDIOLIME_MIN_R2": "0.25",

        # ==========================================================
        # MM-SHAP — più finestra per 8 GPU
        # ==========================================================
        "MMSHAP_QUEUE_WINDOW": "128",  # era 64 → 8 GPU × 16 task pre-queued

        # ==========================================================
        # MEMORY / STABILITY
        # ==========================================================
        # PKV deepcopy crea molti tensori GPU temporanei.
        # Cache cleanup ogni 32 task invece di 96 evita OOM per accumulo.
        "DIME_WORKER_EMPTYCACHE_EVERY": "32",  # era 96

        # max_split_size_mb più grande: deepcopy alloca blocchi ~84MB
        # → con 128 venivano spezzati. 256 li lascia contigui → meno frammentazione.
        # garbage_collection_threshold: libera aggressivamente tra le operazioni.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8",

        "DIME_STEP4A_AUDIO_IO_MODE": "auto",
        "DIME_STEP4A_AUDIO_EQ_ATOL": "1e-5",
        "DIME_STEP4A_AUDIO_EQ_RTOL": "1e-4",
        "DIME_STEP4A_RUNNER_AUDIO_TRANSPORT": "shared_memory",

        # CPU threads: con PKV cache il CPU fa meno preprocessing.
        # Possiamo permetterci più thread per il tokenizer/mel spec residuo.
        "OMP_NUM_THREADS": "4",  # era 2
        "MKL_NUM_THREADS": "4",  # era 2
        "NUMEXPR_NUM_THREADS": "4",  # era 2

        # ==========================================================
        # LEVEL 1: transport cache
        # ==========================================================
        "DIME_FIXED_AUDIO_TRANSPORT_CACHE": "1",
        "DIME_FIXED_AUDIO_TRANSPORT_MODE": "shared_memory",
        "DIME_QWEN_TEXT_CACHE_SIZE": "4096",

        # ==========================================================
        # LEVEL 3: PKV cache
        # ==========================================================
        "DIME_TEXT_PKV_CACHE": "1",
        "DIME_TEXT_PKV_CACHE_VERIFY": "0",  # CRITICO: verify=1 annulla tutto
        "DIME_TEXT_PKV_CACHE_MIN_COMMON_PREFIX": "8",
        "DIME_TEXT_PKV_CACHE_ATOL": "1e-5",
        "DIME_TEXT_PKV_CACHE_RTOL": "1e-4",
        "DIME_TEXT_PKV_CACHE_DEBUG": "0",

        # ==========================================================
        # LEVEL 2: reuse perturbations
        # ==========================================================
        "DIME_REUSE_PERTURBATIONS_ACROSS_TOKENS": "1",
    }

    for k, v in env_cfg.items():
        os.environ[k] = str(v)

##"DIME_QUEUE_WINDOW": "128",  # ↑ più lavoro pronto
        #"DIME_L_BATCH_SIZE": "8",  # ↓ più granularità
        #"DIME_LIME_BATCH_SIZE": "10",  # ↓ meno stragglers

        #"DIME_STEP5_AUDIO_PERTURB_BATCH": "80",  # ↓ task più piccoli
        #"DIME_STEP5_TEXT_PERTURB_BATCH": "104",


def _collect_reproducible_env_snapshot() -> Dict[str, str]:
    """
    Salva TUTTE le env rilevanti al run, non solo un sottoinsieme.
    """
    keys = [
        "DIME_TEXT_PKV_CACHE",
        "DIME_TEXT_PKV_CACHE_VERIFY",
        "DIME_TEXT_PKV_CACHE_MIN_COMMON_PREFIX",
        "DIME_TEXT_PKV_CACHE_ATOL",
        "DIME_TEXT_PKV_CACHE_RTOL",
        "DIME_REUSE_PERTURBATIONS_ACROSS_TOKENS",
        "DIME_FIXED_AUDIO_TRANSPORT_CACHE",
        "DIME_FIXED_AUDIO_TRANSPORT_MODE",
        "DIME_QWEN_TEXT_CACHE_SIZE",
        "DIME_STEP4A_RUNNER_AUDIO_TRANSPORT",
        "DIME_STEP4A_AUDIO_IO_MODE",
        "DIME_STEP4A_AUDIO_EQ_ATOL",
        "DIME_STEP4A_AUDIO_EQ_RTOL",
        "DIME_BG_AUDIO_DIR",
        "DIME_PROMPTS_FILE",
        "DIME_VALUE_MODE",
        "DIME_NUM_EXPECTATION_SAMPLES",
        "DIME_QUEUE_WINDOW",
        "DIME_L_BATCH_SIZE",
        "DIME_LIME_BATCH_SIZE",
        "DIME_STEP5_AUDIO_PERTURB_BATCH",
        "DIME_STEP5_TEXT_PERTURB_BATCH",
        "DIME_TEMP_AUDIO_DIR",
        "DIME_AUDIO_FEATURE_MODE",
        "DIME_AUDIOLIME_DEMUCS_MODEL",
        "DIME_AUDIOLIME_USE_PRECOMPUTED",
        "DIME_AUDIOLIME_PRECOMPUTED_DIR",
        "DIME_AUDIOLIME_RECOMPUTE",
        "DIME_AUDIOLIME_DEMUCS_DEVICE",
        "DIME_AUDIOLIME_DEMUCS_SEGMENT",
        "DIME_AUDIOLIME_DEMUCS_SHIFTS",
        "DIME_AUDIOLIME_DEMUCS_SPLIT",
        "DIME_AUDIOLIME_DEMUCS_OVERLAP",
        "DIME_AUDIOLIME_DEMUCS_JOBS",
        "DIME_AUDIOLIME_DEMUCS_PROGRESS",
        "DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS",
        "DIME_AUDIOLIME_MAX_FEATURES",
        "DIME_AUDIOLIME_MIN_R2",
        "DIME_AUDIOLIME_SAVE_DEBUG_AUDIO",
        "DIME_AUDIOLIME_DEBUG_AUDIO_DIR",
        "DIME_AUDIOLIME_NORMALIZE_COMPOSITION",
        "DIME_AUDIOLIME_PEAK_TARGET",
        "DIME_LIME_KERNEL_WIDTH",
        "DIME_LIME_NUM_FEATURES_AUDIO",
        "DIME_LIME_NUM_FEATURES_TEXT",
        "DIME_NUM_LIME_SAMPLES",
        "DIME_FAST_DEBUG",
        "DIME_DETERMINISTIC",
        "TRANSFORMERS_VERBOSITY",
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TF_CPP_MIN_LOG_LEVEL",
        "TF_ENABLE_ONEDNN_OPTS",
        "DIME_AUDIOLIME_SEGMENTATION_MODE",
        "DIME_AUDIOLIME_ONSET_MIN_SEGMENT_SEC",
        "DIME_AUDIOLIME_ONSET_MAX_SEGMENT_SEC",
        "DIME_AUDIOLIME_ONSET_BACKTRACK",
    ]
    return {k: os.environ.get(k, "") for k in keys}

_apply_reproducible_env()

# =============================================================================
# 1) IMPORT PROGETTO
# =============================================================================

from QA_analysis.utils.analysis_1 import analysis_1_start
from QA_analysis.utils.analysis_2 import analyze_dime, get_dime_module_config_snapshot
from QA_analysis.utils.gpu_utils import (
    try_create_parallel_runner,
    get_available_gpus_with_memory,
)
from QA_analysis.utils.shared_utils import (
    ask_yes_no,
    load_hummusqa_entries_parquet,
    build_hummusqa_qwen25_prompt,
)
os.environ["TRANSFORMERS_NO_TF"] = "1"
# =============================================================================
# 2) Setup progetto / logger
# =============================================================================

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("Main Logger")

def _configure_quiet_logging():
    """
    Mantiene visibili i log utili del progetto:
    - GPU trovate / runner attivo
    - avanzamento run
    - info dei logger del progetto

    Sopprime invece:
    - warning ripetitivi di Qwen
    - warning Demucs fastidiosi
    - rumore di transformers / datasets / urllib3 / hub
    """
    import sys
    import warnings

    # ==========================================================
    # 1) Root logger: WARNING
    #    così il rumore esterno si abbassa, ma non si spegne tutto
    # ==========================================================
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
        stream=sys.stdout,
    )

    # ==========================================================
    # 2) Logger del progetto: INFO visibile
    # ==========================================================
    main_handler = logging.StreamHandler(sys.stdout)
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.handlers = []
    logger.addHandler(main_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    gpu_logger = logging.getLogger("GPU Utils")
    gpu_logger.handlers = []
    gpu_logger.addHandler(main_handler)
    gpu_logger.setLevel(logging.INFO)
    gpu_logger.propagate = False

    dime_logger = logging.getLogger("DIME-Analysis")
    dime_logger.handlers = []
    dime_logger.addHandler(main_handler)
    dime_logger.setLevel(logging.INFO)
    dime_logger.propagate = False

    mmshap_logger = logging.getLogger("MM-SHAP")
    mmshap_logger.handlers = []
    mmshap_logger.addHandler(main_handler)
    mmshap_logger.setLevel(logging.INFO)
    mmshap_logger.propagate = False

    # ==========================================================
    # 3) Warning Python: filtra SOLO quelli noti e inutili
    # ==========================================================
    warnings.resetwarnings()
    warnings.simplefilter("default")


    warnings.filterwarnings("ignore", message=".*audio output may not work as expected.*")
    warnings.filterwarnings("ignore", message=".*System prompt modified.*")
    warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
    warnings.filterwarnings("ignore", message=".*last .* samples are ignored.*")
    warnings.filterwarnings("ignore", message=".*Demucs factorization config.*")
    warnings.filterwarnings("ignore", message=".*Requested segmentation exceeds DIME_AUDIOLIME_MAX_FEATURES.*")

    # ==========================================================
    # 4) Filtro logger per i messaggi davvero fastidiosi
    # ==========================================================
    class _DropKnownNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()

            blocked_substrings = [
                "audio output may not work as expected",
                "System prompt modified",
                "process_mm_info",
                "Demucs factorization config",
                "Requested segmentation exceeds DIME_AUDIOLIME_MAX_FEATURES",
            ]

            for s in blocked_substrings:
                if s in msg:
                    return False
            return True

    drop_filter = _DropKnownNoise()

    logging.getLogger().addFilter(drop_filter)
    logger.addFilter(drop_filter)
    gpu_logger.addFilter(drop_filter)
    dime_logger.addFilter(drop_filter)
    mmshap_logger.addFilter(drop_filter)

    # ==========================================================
    # 5) Librerie esterne rumorose: ERROR
    # ==========================================================
    noisy_logger_names = [
        "transformers",
        "transformers.generation.utils",
        "transformers.modeling_utils",
        "transformers.tokenization_utils_base",
        "transformers.configuration_utils",
        "huggingface_hub",
        "datasets",
        "urllib3",
        "qwen_omni_utils",
        "QwenOmni",
        "Qwen2.5Omni",
    ]

    for name in noisy_logger_names:
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.propagate = False
        lg.addFilter(drop_filter)

    # ==========================================================
    # 6) Transformers / datasets helper
    # ==========================================================
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    try:
        import datasets
        datasets.disable_progress_bar()
    except Exception:
        pass

HUMMUSQA_ROOT = "/nas/home/fingenito/HumMusQA/data"
EXPERIMENT_RESULTS_ROOT = "/nas/home/fingenito/Thesis_project/QA_analysis/Results_QA"


# =============================================================================
# HELPERS GENERALI
# =============================================================================

def _choose_index(title: str, items, label_fn, idx_base: int = 0):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for i, it in enumerate(items):
        shown = i + idx_base
        print(f"[{shown}] {label_fn(it, i)}")
    print("=" * 100)

    lo = idx_base
    hi = len(items) - 1 + idx_base

    while True:
        raw = input(f"Inserisci indice ({lo}..{hi}): ").strip()
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v - idx_base
        except Exception:
            pass
        print("Indice non valido. Riprova.")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _next_run_dir(root: str) -> str:
    _ensure_dir(root)

    existing = []
    for name in os.listdir(root):
        if not name.startswith("run_"):
            continue
        suf = name[4:]
        if suf.isdigit():
            existing.append(int(suf))

    nxt = (max(existing) + 1) if existing else 0
    run_name = f"run_{nxt:02d}"
    run_dir = os.path.join(root, run_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def _write_run_info(run_dir: str, info: dict):
    path = os.path.join(run_dir, "run_info.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# =============================================================================
# HELPERS HUMMUSQA
# =============================================================================

def _sha1_bytes(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def _sha1_array(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    h = hashlib.sha1()
    h.update(arr.tobytes())
    return h.hexdigest()


def _sha1_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_audio_stem_name(s: str, max_len: int = 48) -> str:
    s = str(s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    if not s:
        s = "audio"
    return s[:max_len]


def _get_audio_source_debug_string(entry: dict) -> str:
    audio_field = entry.get("audio", None)

    if isinstance(audio_field, str) and audio_field.strip():
        return audio_field.strip()

    if isinstance(audio_field, dict):
        for k in ["path", "audio_path", "filename", "file"]:
            v = audio_field.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        if audio_field.get("bytes", None) is not None:
            return "embedded_bytes"

        if audio_field.get("array", None) is not None:
            return "embedded_array"

    for k in ["audio_path", "audio_file", "file", "wav", "path"]:
        v = entry.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return "unknown_audio_source"

def _extract_sample_id(entry: dict, fallback_idx: int) -> str:
    return str(
        entry.get("question_id")
        or entry.get("id")
        or entry.get("sample_id")
        or f"sample_{fallback_idx}"
    )


def _extract_question(entry: dict) -> str:
    return str(entry.get("question", "")).strip()


def _extract_options(entry: dict) -> List[str]:
    return [
        str(entry.get("answer", "")).strip(),
        str(entry.get("distractor_1", "")).strip(),
        str(entry.get("distractor_2", "")).strip(),
        str(entry.get("distractor_3", "")).strip(),
    ]


def _entry_has_valid_mcqa(entry: dict) -> bool:
    q = _extract_question(entry)
    opts = _extract_options(entry)
    return bool(q) and len(opts) == 4 and all(isinstance(x, str) for x in opts) and all(x != "" for x in opts)

def _extract_entry_category(entry: dict) -> str:
    """
    Estrae il campo 'category' da un entry HumMusQA.
    Prova multipli nomi di campo per robustezza.
    Restituisce 'unknown' se nessun campo trovato.
    """
    for key in _CATEGORY_FIELDS:
        val = entry.get(key, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "unknown"


def _extract_entry_difficulty(entry: dict) -> str:
    """
    Estrae il campo 'difficulty' da un entry HumMusQA.
    Prova multipli nomi di campo per robustezza.
    Restituisce 'unknown' se nessun campo trovato.
    """
    for key in _DIFFICULTY_FIELDS:
        val = entry.get(key, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    return "unknown"


def _extract_entry_macro_family(category: str) -> str:
    """Mappa category → macrofamiglia."""
    for keyword, fam in MACRO_FAMILY_MAP.items():
        if keyword.lower() in category.lower():
            return fam
    return "unknown"

def _infer_audio_extension_from_entry(entry: dict) -> str:
    """
    Cerca di inferire l'estensione originale dell'audio embedded.
    Se non riesce, usa .mp3 come default ragionevole per HumMusQA/Jamendo.
    """
    src = _get_audio_source_debug_string(entry)
    _, ext = os.path.splitext(str(src).strip())
    ext = ext.lower().strip()

    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    if ext in valid_exts:
        return ext

    return ".mp3"

def _materialize_hummusqa_audio(
    entry: dict,
    cache_dir: str,
    fallback_idx: Optional[int] = None,
) -> str:
    """
    Restituisce un path audio reale e CANONICO da usare nel progetto.

    FIX IMPORTANTE:
    - il nome del file non dipende dal sample_id
    - dipende dal contenuto audio (hash)
    - se l'audio arriva come bytes compressi (es. mp3), NON lo salviamo
      fittiziamente come .wav: prima lo scriviamo con la sua estensione reale,
      poi lo decodifichiamo e lo risalviamo come WAV PCM canonico.
    - così evitiamo:
        1) collisioni tra sample diversi
        2) riuso accidentale dell'audio sbagliato
        3) comportamenti strani del decoder causati da estensione falsa
    """
    os.makedirs(cache_dir, exist_ok=True)

    sample_id = _extract_sample_id(entry, fallback_idx if fallback_idx is not None else -1)
    audio_field = entry.get("audio", None)

    def _existing_path(v) -> Optional[str]:
        if isinstance(v, str) and v.strip():
            p = v.strip()
            if os.path.exists(p):
                return p
        return None

    # --------------------------------------------------
    # Caso 1: audio direttamente come path esistente
    # --------------------------------------------------
    p = _existing_path(audio_field)
    if p is not None:
        return p

    # --------------------------------------------------
    # Caso 2: audio dict con path esistente
    # --------------------------------------------------
    if isinstance(audio_field, dict):
        for k in ["path", "audio_path", "filename", "file"]:
            p = _existing_path(audio_field.get(k))
            if p is not None:
                return p

    # --------------------------------------------------
    # Caso 3: audio embedded come array + sampling_rate
    # --------------------------------------------------
    if isinstance(audio_field, dict):
        arr = audio_field.get("array", None)
        sr = audio_field.get("sampling_rate", None)

        if arr is not None and sr is not None:
            y = np.asarray(arr, dtype=np.float32).reshape(-1)
            digest = _sha1_array(y)[:16]

            source_label = _safe_audio_stem_name(_get_audio_source_debug_string(entry))
            out_path = os.path.join(cache_dir, f"{source_label}_{digest}.wav")

            if not os.path.exists(out_path):
                sf.write(out_path, y, int(sr))

            return out_path

    # --------------------------------------------------
    # Caso 4: audio embedded come bytes compressi
    # --------------------------------------------------
    if isinstance(audio_field, dict):
        raw_bytes = audio_field.get("bytes", None)
        if raw_bytes is not None:
            digest = _sha1_bytes(raw_bytes)[:16]
            source_label = _safe_audio_stem_name(_get_audio_source_debug_string(entry))
            original_ext = _infer_audio_extension_from_entry(entry)

            raw_path = os.path.join(cache_dir, f"{source_label}_{digest}{original_ext}")
            decoded_wav_path = os.path.join(cache_dir, f"{source_label}_{digest}_decoded.wav")

            if not os.path.exists(raw_path):
                with open(raw_path, "wb") as f:
                    f.write(raw_bytes)

            # Decodifica canonica -> WAV PCM vero
            if not os.path.exists(decoded_wav_path):
                try:
                    y, sr_loaded = librosa.load(raw_path, sr=None, mono=True)
                    y = np.asarray(y, dtype=np.float32).reshape(-1)

                    if y.size == 0 or int(sr_loaded) <= 0:
                        raise RuntimeError(
                            f"Decoded embedded audio is empty/invalid for sample_id={sample_id}"
                        )

                    sf.write(decoded_wav_path, y, int(sr_loaded))
                except Exception as e:
                    warnings.warn(
                        f"[HumMusQA audio materialization] Impossibile decodificare bytes embedded "
                        f"per sample_id={sample_id} da {raw_path}. "
                        f"Uso il file raw originale. Errore: {repr(e)}"
                    )
                    return raw_path

            return decoded_wav_path

    # --------------------------------------------------
    # fallback su chiavi top-level alternative
    # --------------------------------------------------
    for k in ["audio_path", "audio_file", "file", "wav", "path"]:
        p = _existing_path(entry.get(k))
        if p is not None:
            return p

    raise RuntimeError(
        f"Impossibile materializzare l'audio per sample_id={sample_id} | "
        f"audio_field_type={type(audio_field)} | audio_field={audio_field}"
    )


def _build_entry_selection_list(entries: List[dict]) -> List[Tuple[int, dict]]:
    valid = []
    for i, e in enumerate(entries):
        if _entry_has_valid_mcqa(e):
            valid.append((i, e))
    return valid


def _entry_label_for_menu(item, _i: int) -> str:
    """
    Versione aggiornata di _entry_label_for_menu che mostra
    category e difficulty nella lista di selezione.
    """
    idx, e = item
    sample_id = str(
        e.get("identifier") or e.get("question_id") or e.get("id") or f"sample_{idx}"
    )
    q = str(e.get("question", "")).strip()
    short_q = q if len(q) <= 80 else (q[:77] + "...")

    category = _extract_entry_category(e)
    difficulty = _extract_entry_difficulty(e)

    return f"{sample_id} | [{category} / {difficulty}] | {short_q}"

def _build_hummusqa_background_pairs_local(
    entries: List[dict],
    target_sample_id: str,
    k: int,
    seed: int,
    cache_dir: str,
    include_target_first: bool = True,
) -> List[Tuple[str, str]]:
    rng = np.random.RandomState(int(seed))
    k = max(1, int(k))

    valid_pairs = []
    target_pair = None

    for idx, e in enumerate(entries):
        if not _entry_has_valid_mcqa(e):
            continue

        try:
            audio_path = _materialize_hummusqa_audio(
                e,
                cache_dir=cache_dir,
                fallback_idx=idx,
            )
        except Exception:
            continue

        sample_id = _extract_sample_id(e, idx)
        question = _extract_question(e)
        options = _extract_options(e)
        prompt = build_hummusqa_qwen25_prompt(question, options)

        pair = (audio_path, prompt)

        if str(sample_id) == str(target_sample_id):
            target_pair = pair
        else:
            valid_pairs.append(pair)

    if target_pair is None:
        raise RuntimeError(f"Target sample_id={target_sample_id} non trovato nelle entry valide HumMusQA.")

    if len(valid_pairs) > 0:
        perm = rng.permutation(len(valid_pairs)).tolist()
        sampled = [valid_pairs[i] for i in perm[: max(0, k - 1 if include_target_first else k)]]
    else:
        sampled = []

    if include_target_first:
        return [target_pair] + sampled
    return sampled


# =============================================================================
# MAIN
# =============================================================================

def main():
    _configure_quiet_logging()
    logger.info("=" * 80)
    logger.info("ANALISI MM-SHAP + DIME (PARALLELO MULTI-GPU) — RUN_XX structure")
    logger.info("=" * 80)

    # -------------------------
    # Create run_XX
    # -------------------------
    run_dir = _next_run_dir(EXPERIMENT_RESULTS_ROOT)
    dime_dir = os.path.join(run_dir, "dime")
    mmshap_dir = os.path.join(run_dir, "mm-shap")
    audio_cache_dir = os.path.join(run_dir, "_audio_cache")

    logger.info(f"Run directory creata: {run_dir}")

    # -------------------------
    # Load HumMusQA entries
    # -------------------------
    entries, hummusqa_meta_path = load_hummusqa_entries_parquet(HUMMUSQA_ROOT)
    valid_entries = _build_entry_selection_list(entries)

    print(f"\nHumMusQA metadata file: {hummusqa_meta_path}")
    print(f"Numero entry lette: {len(entries)}")
    print(f"Numero entry MCQA valide: {len(valid_entries)}")

    if not valid_entries:
        logger.error("Nessuna entry valida trovata in HumMusQA.")
        return

    # -------------------------
    # Model path
    # -------------------------
    model_path = "/nas/home/fingenito/Models/Qwen2.5-Omni-7B"
    if not os.path.exists(model_path):
        logger.error(f"Model path non esiste: {model_path}")
        return

    # -------------------------
    # Select HumMusQA sample directly
    # -------------------------
    selected_idx = _choose_index(
        title="SELEZIONE ENTRY HUMMUSQA",
        items=valid_entries,
        label_fn=_entry_label_for_menu,
        idx_base=0,
    )

    orig_idx, target_entry = valid_entries[selected_idx]

    sample_id = _extract_sample_id(target_entry, orig_idx)
    target_question = _extract_question(target_entry)
    target_options = _extract_options(target_entry)
    target_correct_answer = target_options[0]
    target_category = _extract_entry_category(target_entry)
    target_difficulty = _extract_entry_difficulty(target_entry)
    target_macro_family = _extract_entry_macro_family(target_category)

    try:
        target_audio_path = _materialize_hummusqa_audio(
            target_entry,
            cache_dir=audio_cache_dir,
            fallback_idx=orig_idx,
        )
    except Exception as e:
        logger.error(f"Errore nel recupero audio della entry selezionata: {e}")
        return

    try:
        target_audio_sha1 = _sha1_file(target_audio_path)
    except Exception:
        target_audio_sha1 = ""
    target_prompt = build_hummusqa_qwen25_prompt(target_question, target_options)

    print("\n" + "=" * 60)
    print("🎯 TARGET SELEZIONATO")
    print("=" * 60)
    print(f"Run dir:       {run_dir}")
    print(f"Sample id:     {sample_id}")
    print(f"Audio path:    {target_audio_path}")
    print(f"Question:      {target_question}")
    print("Options:")
    for i, opt in enumerate(target_options):
        print(f"  {chr(65 + i)}. {opt}")
    print(f"Correct:       {target_correct_answer}")
    print(f"Category:      {target_category}")
    print(f"Difficulty:    {target_difficulty}")
    print("=" * 60)

    _write_run_info(run_dir, {
        "sample_id": sample_id,
        "target_audio_path": target_audio_path,
        "target_question": target_question,
        "target_options": target_options,
        "target_correct_answer": target_correct_answer,
        "target_prompt": target_prompt,
        "target_audio_sha1": target_audio_sha1,
        "target_audio_source_debug": _get_audio_source_debug_string(target_entry),
        "hummusqa_meta_path": hummusqa_meta_path,
        "env": _collect_reproducible_env_snapshot(),
        "analysis_2_module_config": get_dime_module_config_snapshot(),
        "target_category": target_category,
        "target_difficulty": target_difficulty,
        "target_macro_family": target_macro_family,
    })

    # -------------------------
    # GPU + Runner
    # -------------------------
    MAX_GPUS_TO_USE = 8
    MIN_FREE_GB_RUNNER = 21.0

    gpu_ids, _details = get_available_gpus_with_memory(min_free_memory_gb=MIN_FREE_GB_RUNNER)
    gpu_ids = gpu_ids[:MAX_GPUS_TO_USE]
    logger.info(f"GPU candidate trovate: {gpu_ids}")
    if not gpu_ids:
        logger.error(
            f"Nessuna GPU con >= {MIN_FREE_GB_RUNNER}GB liberi. "
            f"Libera VRAM o abbassa MIN_FREE_GB_RUNNER."
        )
        return

    logger.info(f"GPU selezionate per runner: {gpu_ids}")

    runner = try_create_parallel_runner(
        model_path=model_path,
        min_free_memory_gb=MIN_FREE_GB_RUNNER,
        gpu_ids_physical=gpu_ids,
    )
    if runner is None:
        logger.error("Runner multiprocess NON attivo: impossibile procedere.")
        return

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    _local_model = {"obj": None}


    did_anything = False
    caption = None

    try:
        do_dime = ask_yes_no("Vuoi eseguire DIME su questo target?")
        do_mmshap = ask_yes_no("Vuoi eseguire MM-SHAP su questo target?")

        if not do_dime and not do_mmshap:
            print("\nNessuna analisi selezionata. Processo terminato.")
            return

        did_anything = True

        local_model = None

        # --------------------------------------------------
        # Shared baseline generation (RUNNER ONLY)
        # --------------------------------------------------
        shared_baseline = None
        caption = None

        if do_mmshap or do_dime:
            print("\nGenerazione baseline condivisa usando SOLO runner...")

            caption = runner.generate_caption(
                audio_path=target_audio_path,
                prompt=target_prompt,
            )

            shared_baseline = {
                "baseline_answer": caption,
                "prompt": target_prompt,
                "audio_path": target_audio_path,

                # 🔥 IMPORTANTISSIMO
                "input_ids": None,
                "output_ids": None,
            }
        else:
            caption = None

        # --------------------------------------------------
        # DIME
        # --------------------------------------------------
        if do_dime:
            _ensure_dir(dime_dir)

            bg_pairs = _build_hummusqa_background_pairs_local(
                entries=entries,
                target_sample_id=sample_id,
                k=int(os.environ.get("DIME_NUM_EXPECTATION_SAMPLES", "24")),
                seed=int(os.environ.get("DIME_SEED", "0")),
                cache_dir=audio_cache_dir,
                include_target_first=True,
            )
            bg_audio_paths = [a for a, _p in bg_pairs]
            bg_prompts = [p for _a, p in bg_pairs]

            if caption is None:
                print("Generazione risposta base (runner)...")
                caption = runner.generate_caption(
                    audio_path=target_audio_path,
                    prompt=target_prompt,
                )
                print(f"Risposta generata: {caption}")
            else:
                print(f"Uso baseline condivisa come risposta base per DIME: {caption}")

            print("\nAvvio DIME (target-first)...")
            t0 = time.time()
            results_2 = analyze_dime(
                model=None,
                processor=processor,
                audio_path=target_audio_path,
                prompt=target_prompt,
                caption=caption,
                results_dir=dime_dir,
                runner=runner,
                background_audio_paths=bg_audio_paths,
                background_prompts=bg_prompts,
                question=target_question,
                options=target_options,
                num_lime_samples=int(os.environ.get("DIME_NUM_LIME_SAMPLES", "256")),
                num_features=int(os.environ.get("DIME_LIME_NUM_FEATURES_AUDIO", "12")),
            )
            t1 = time.time()
            elapsed = t1 - t0
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)

            logger.info(f"DIME completato in {hours} ore e {minutes} minuti.")
            print("\n✓ DIME completato.")
            print(f"  Cartella risultati DIME: {dime_dir}")
            print(f"  Output JSON: {results_2.get('json_path', '')}")
            print(f"  Output PLOT: {results_2.get('plot_path', '')}")

        # --------------------------------------------------
        # MM-SHAP
        # --------------------------------------------------
        if do_mmshap:
            _ensure_dir(mmshap_dir)

            qa_entry = {
                "sample_id": sample_id,
                "audio_path": target_audio_path,
                "question": target_question,
                "options": target_options,
                "correct_answer": target_correct_answer,
            }

            print("\nAvvio MM-SHAP QA...")

            results_1 = analysis_1_start(
                model=None,  # 🔥 niente modello locale
                processor=processor,
                results_dir=mmshap_dir,
                runner=runner,  # 🔥 passi runner
                qa_entry=qa_entry,
                shared_baseline=shared_baseline,
            )

            print("\n✓ MM-SHAP completato.")
            print(f"  Cartella risultati MM-SHAP: {mmshap_dir}")
            print(f"  Output JSON: {results_1.get('json_path', '')}")
            print(f"  Output PLOT: {results_1.get('plot_path', '')}")

        if did_anything:
            print("\nProcesso terminato: analisi completata.")

    finally:
        if runner is not None:
            try:
                runner.stop()
            except Exception:
                pass
        if _local_model["obj"] is not None:
            try:
                del _local_model["obj"]
            except Exception:
                pass


if __name__ == "__main__":
    import warnings

    warnings.resetwarnings()
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", message=".*audio output may not work as expected.*")
    warnings.filterwarnings("ignore", message=".*System prompt modified.*")
    warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
    warnings.filterwarnings("ignore", message=".*last .* samples are ignored.*")
    warnings.filterwarnings("ignore", message=".*Demucs factorization config.*")

    main()