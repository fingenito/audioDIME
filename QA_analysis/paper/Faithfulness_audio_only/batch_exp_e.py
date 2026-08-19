"""
Esperimento E — Batch runner (v4 — FIX MCQA LOGITS)
=====================================================
Faithfulness causale delle spiegazioni multimodali DIME.

BUGFIX rispetto alla versione originale
========================================
PROBLEMA: get_mmshap_logits(target_ids=[id_A, id_B, id_C, id_D]) passava
  quattro ID come se fossero una *sequenza autoregressiva*.
  In gpu_utils.py, quando T > 1 il worker fa:
      outputs = model.generate(max_new_tokens=T+2, output_logits=True)
      step_logits[0, id_A]   # logit di A al passo 0
      step_logits[1, id_B]   # logit di B al passo 1 (dopo aver già emesso qualcosa)
      step_logits[2, id_C]   # ...
      step_logits[3, id_D]   # ...
  Questi QUATTRO numeri sono misurati in QUATTRO punti temporali diversi
  della catena autoregressiva → l'argmax non ha senso → accuracy ~25%.

SOLUZIONE: usare il fast-path T==1 del worker per ogni lettera.
  Quando T==1, il worker fa un SINGOLO forward pass (no generate()) e legge
  logits[0, -1, target_id] → il logit corretto al punto decisionale.

  Per non fare 4 round-trip IPC seriali (lento), usiamo
  score_items_mcqa_batch() che manda i 4 item in parallelo sulla coda
  del runner (sfruttando la finestra di accodamento già esistente).

  Costo: 4 forward pass invece di 1, MA ogni forward pass è ~3-5x più
  veloce di model.generate() e soprattutto produce NUMERI CORRETTI.

DESIGN
======
Esperimento PRINCIPALE: solo MI.
  - MI è l'unica componente DIME che rappresenta l'interazione audio↔testo.
  - Per ogni sample, calcoliamo CURVE PROGRESSIVE di sufficiency e necessity
    al variare di k ∈ {1, 2, 3, 4, 5, 6, ..., 16}.
  - target = risposta del modello (la spiegazione testa la decisione effettiva).
  - ranking MI audio+testo SCALE-NORMALIZED (fusione cross-modale equa).
  - selezione: SOLO feature MI positive che supportano la risposta del modello.

Esperimenti DIAGNOSTICI: UC_audio, UC_text.
  - Inclusi opzionalmente per controllo.
  - Disattivabili con flag --no-diagnostic per dimezzare il costo runtime.

METRICHE NORMALIZZATE SUL CHANCE LEVEL (0.25 per MCQA 4-way)
============================================================
  sufficiency_score = (p_suf - 0.25) / (p_orig - 0.25)
  necessity_score   = (p_orig - p_nec) / (p_orig - 0.25)
"""

import os
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
_warnings.filterwarnings("ignore", category=ResourceWarning)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_LOGGING_VERBOSITY", "3")
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::DeprecationWarning,ignore::FutureWarning,"
    "ignore::UserWarning,ignore::ResourceWarning",
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
# 0b) Filtro stderr
# =============================================================================
_NOISE_PATTERNS = (
    "aifc", "audioop", "sunau",
    "is deprecated and slated for removal",
    "DeprecationWarning", "FutureWarning", "ResourceWarning",
    "oneDNN custom operations are on", "oneDNN",
    "absl::InitializeLog", "All log messages before",
    "I0000 00:", "port.cc:", "Implicitly cleaning up",
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
            self._real.write(line); return
        if any(k in stripped for k in ("Traceback", "Error:", "Exception:", "CRITICAL", "FAIL")):
            self._real.write(line); return
        if any(p in stripped for p in _NOISE_PATTERNS):
            return
        self._real.write(line)
    def flush(self):
        if self._buf:
            self._emit_line(self._buf); self._buf = ""
        self._real.flush()
    def __getattr__(self, name):
        return getattr(self._real, name)

sys.stderr = _StderrFilter(sys.stderr)

# =============================================================================
# 0c) ENV
# =============================================================================
def _apply_env() -> None:
    env_cfg = {
        "TRANSFORMERS_NO_TF": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "DIME_VALUE_MODE": "logit",
        "DIME_AUDIO_FEATURE_MODE": "audiolime_demucs",
        "DIME_AUDIO_MASK_MODE": "bg_random_energy",
        "DIME_SILENCE_RMS": "1e-3",
        "DIME_CROSSFADE": "1",
        "DIME_CROSSFADE_MS": "5.0",
        "DIME_AUDIOLIME_DEMUCS_MODEL": "htdemucs",
        "DIME_AUDIOLIME_USE_PRECOMPUTED": "1",
        "DIME_AUDIOLIME_RECOMPUTE": "0",
        "DIME_AUDIOLIME_DEMUCS_DEVICE": "cpu",
        "DIME_AUDIOLIME_DEMUCS_SEGMENT": "8",
        "DIME_AUDIOLIME_DEMUCS_SPLIT": "1",
        "DIME_AUDIOLIME_DEMUCS_OVERLAP": "0.25",
        "DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS": "8",
        "DIME_AUDIOLIME_PRECOMPUTED_DIR":
            "/nas/home/fingenito/Thesis_project/QA_analysis/data/demucs_cache",
        "DIME_AUDIOLIME_NORMALIZE_COMPOSITION": "1",
        "MMSHAP_QUEUE_WINDOW": "128",
        "PYTORCH_CUDA_ALLOC_CONF":
            "expandable_segments:True,max_split_size_mb:256,"
            "garbage_collection_threshold:0.8",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "TORCH_HOME":
            "/nas/home/fingenito/Thesis_project/QA_analysis/data/torch_cache",
        "TORCH_HUB":
            "/nas/home/fingenito/Thesis_project/QA_analysis/data/torch_cache/hub",
    }
    for k, v in env_cfg.items():
        os.environ[k] = str(v)

_apply_env()

# =============================================================================
# 1) Import progetto
# =============================================================================
from transformers import Qwen2_5OmniProcessor

from QA_analysis.utils.gpu_utils import (
    get_available_gpus_with_memory,
    # Funzioni pure usate dal McqaParallelRunner interno a expE
    Task,
    configure_runtime,
    _configure_worker_quiet_mode,
    _build_messages,
    _materialize_worker_audio_input,
    _close_worker_audio_handle,
)
from QA_analysis.experiments.expE.perturbations_exp_e import (
    rank_audio_features,
    rank_text_words,
    scale_normalize_ranked,
    build_audio_binary_mask,
    materialize_perturbed_audio,
    build_perturbed_prompt,
    get_letter_token_ids,
    answer_letter_from_logits,
)

# =============================================================================
# 2) Costanti
# =============================================================================
EXPERIMENT_RESULTS_ROOT = (
    "/nas/home/fingenito/Thesis_project/QA_analysis/Results_QA/experiments"
)
MODEL_PATH = "/nas/home/fingenito/Models/Qwen2.5-Omni-7B"
HUMMUSQA_PATH = "/nas/home/fingenito/HumMusQA/data"

MAX_GPUS_TO_USE = 8
MIN_FREE_GB_RUNNER = 21.0

K_MAX_MAIN: int = 16
K_VALUES_MAIN: List[int] = list(range(1, K_MAX_MAIN + 1))

INCLUDE_BALANCED_MI: bool = True
K_VALUES_BALANCED_MI: List[int] = list(range(1, K_MAX_MAIN + 1))

P_CHANCE: float = 0.25

ALPHA_SUFFICIENT: float = 0.5
BETA_NECESSARY:   float = 0.5

EPS_BASELINE: float = 0.02

K_CANONICAL_MAIN: int = 3

TEST_FIRST_N_SAMPLES: Optional[int] = None

DIAGNOSTIC_FIRST_N_SAMPLES: int = 3

INCLUDE_DIAGNOSTIC_SOURCES_DEFAULT: bool = True
DIAGNOSTIC_K_VALUES_UC_AUDIO: List[int] = list(range(1, K_MAX_MAIN + 1))
DIAGNOSTIC_K_VALUES_UC_TEXT:  List[int] = list(range(1, K_MAX_MAIN + 1))

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
# 4) Helpers I/O
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
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
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
# 5) Macrofamily map
# =============================================================================
MACRO_FAMILY_MAP: Dict[str, str] = {
    "Instrumentation":               "percettiva",
    "Sound Texture":                 "percettiva",
    "Metre and Rhythm":              "percettiva",
    "Musical Texture":               "percettiva",
    "Harmony":                       "analitica",
    "Melody":                        "analitica",
    "Structure":                     "analitica",
    "Performance":                   "analitica",
    "Genre and Style":               "knowledge",
    "Mood and Expression":           "knowledge",
    "Functional Context":            "knowledge",
    "Historical and Cultural Context": "knowledge",
    "Lyrics":                        "knowledge",
}

def _macro_family_from_category(category: str) -> str:
    cat = (category or "").strip()
    if cat in MACRO_FAMILY_MAP:
        return MACRO_FAMILY_MAP[cat]
    cat_lower = cat.lower()
    for k, v in MACRO_FAMILY_MAP.items():
        if k.lower() in cat_lower:
            return v
    return "unknown"

# =============================================================================
# 6) Carica risultati Exp A
# =============================================================================
def load_exp_a_sample(exp_a_dir: str, sample_id: str) -> Dict[str, Any]:
    path = os.path.join(exp_a_dir, "per_sample", f"{sample_id}_exp_a.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Exp A JSON non trovato: {path}")
    with open(path, "r") as f:
        return json.load(f)

def list_available_exp_a_samples(exp_a_dir: str) -> List[str]:
    pattern = os.path.join(exp_a_dir, "per_sample", "*_exp_a.json")
    files = sorted(glob.glob(pattern))
    return [os.path.basename(f).replace("_exp_a.json", "") for f in files]

# =============================================================================
# 7) McqaParallelRunner — runner custom interno a expE
# =============================================================================
#
# PERCHÉ esiste questo runner
# ----------------------------
# Il ParallelTokenRunner di gpu_utils gestisce task kind fissi nel worker loop.
# Per MCQA serve: 1 forward pass → leggere logit di N lettere dallo stesso
# tensore logits[0,-1]. Non possiamo aggiungere questo task kind al worker di
# gpu_utils senza modificarlo. Quindi definiamo qui un runner indipendente
# con il proprio worker loop che fa esattamente quella cosa.
#
# COSA NON MODIFICA
# ------------------
# - gpu_utils.py: invariato
# - ParallelTokenRunner DIME: invariato, non usato qui
# - Metodologia: stessi logit, stesso punto decisionale del vecchio codice corretto
#
# COME FUNZIONA
# --------------
# _mcqa_worker_loop: loop custom. Carica Qwen2.5-Omni una volta, poi per ogni
#   task "mcqa_logits" fa 1 forward pass e restituisce
#   [logits_last[tid] for tid in target_ids] — tutti dallo stesso tensore.
# McqaParallelRunner: replica il meccanismo IPC di ParallelTokenRunner
#   (Queue, req_id, stash, window) ma solo per mcqa_logits.
# Le utility di basso livello (prepare_qwen25_omni_inputs, _build_messages,
#   _materialize_worker_audio_input) vengono importate da gpu_utils.
#
# COSTO vs versione v1 (4 forward pass per perturbazione)
# ---------------------------------------------------------
#   v1: score_items_mcqa espandeva ogni item in 4 sotto-item con T=1 ciascuno
#       → 4 forward pass separati per perturbazione → ~8h per 320 sample
#   v3: McqaParallelRunner fa 1 forward pass per perturbazione e legge
#       tutti e 4 i logit dallo stesso tensore → ~2h per 320 sample
# =============================================================================

import multiprocessing as _mp


def _mcqa_worker_loop(
    gpu_id: int,
    model_path: str,
    task_q,
    result_q,
    torch_dtype: str = "bf16",
) -> None:
    """
    Worker loop custom per McqaParallelRunner.

    Gestisce un solo task kind: "mcqa_logits".

    Payload:
        audio_path  : str | shared_memory_audio | inline_audio
        prompt      : str
        target_ids  : List[int]  token id delle lettere da leggere
        batch_id    : int | None
        req_id      : int

    Risposta:
        ("mcqa_logits", {"req_id": int, "batch_id": int|None, "vals": List[float]})

    Fa UN SOLO forward pass (model(**inputs)) e legge tutti i target_ids
    da logits[0, -1] — corretto per MCQA, 4x più veloce di 4 forward separati.
    """
    import os
    import gc
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("DIME_VALUE_MODE", "logit")
    configure_runtime()
    _configure_worker_quiet_mode()

    from transformers import (
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )
    from QA_analysis.utils.shared_utils import prepare_qwen25_omni_inputs

    import logging
    _wlog = logging.getLogger("McqaWorker")

    dtype = torch.bfloat16 if torch_dtype == "bf16" else torch.float16

    try:
        _wlog.info(f"[mcqa_worker gpu={gpu_id}] Caricamento modello...")
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        ).eval().to("cuda:0")
        _wlog.info(f"[mcqa_worker gpu={gpu_id}] Pronto.")
    except Exception as e:
        result_q.put(("__mcqa_worker_init_error__", {"gpu": gpu_id, "error": str(e)}))
        return

    torch.set_grad_enabled(False)

    while True:
        task = task_q.get()
        if task is None:
            break

        req_id = task.payload.get("req_id", None)

        try:
            if task.kind != "mcqa_logits":
                raise ValueError(f"McqaWorker: task kind sconosciuto '{task.kind}'")

            audio_path_raw = task.payload["audio_path"]
            prompt         = task.payload["prompt"]
            target_ids     = task.payload["target_ids"]
            batch_id       = task.payload.get("batch_id", None)

            audio_path, audio_handle = _materialize_worker_audio_input(audio_path_raw)
            try:
                messages = _build_messages(audio_path, prompt)
                _text, inputs = prepare_qwen25_omni_inputs(
                    processor=processor,
                    conversation=messages,
                    device=model.device,
                    dtype=getattr(model, "dtype", None),
                    use_audio_in_video=False,
                )
                # UN SOLO forward pass — nessun generate()
                with torch.inference_mode():
                    outputs = model(**inputs, use_cache=False, return_dict=True)
                logits_last = outputs.logits[0, -1]
                vocab_size  = int(logits_last.shape[-1])
                vals = []
                for tid in target_ids:
                    tid = int(tid)
                    if 0 <= tid < vocab_size:
                        vals.append(float(logits_last[tid].item()))
                    else:
                        vals.append(float(logits_last.mean().item()))
                del outputs
            finally:
                _close_worker_audio_handle(audio_handle)

            result_q.put(("mcqa_logits", {
                "req_id":   req_id,
                "batch_id": batch_id,
                "vals":     vals,
            }))
            gc.collect()

        except torch.cuda.OutOfMemoryError as e:
            gc.collect()
            try: torch.cuda.empty_cache()
            except Exception: pass
            result_q.put(("__mcqa_task_error__", {
                "req_id": req_id, "gpu": gpu_id,
                "error": f"CUDA OOM: {e}",
            }))
        except Exception as e:
            import traceback as _tb
            result_q.put(("__mcqa_task_error__", {
                "req_id": req_id, "gpu": gpu_id,
                "error": f"{e}\n{_tb.format_exc()[:1000]}",
            }))


class McqaParallelRunner:
    """
    Runner multiprocesso per MCQA logit-probing in expE.
    Indipendente da ParallelTokenRunner. Gestisce solo "mcqa_logits":
    1 forward pass per item → N logit, uno per lettera target.
    Non modifica gpu_utils.py.
    """

    def __init__(
        self,
        model_path: str,
        gpu_ids: List[int],
        torch_dtype: str = "bf16",
    ) -> None:
        self.model_path  = model_path
        self.gpu_ids     = gpu_ids
        self.torch_dtype = torch_dtype
        self._ctx        = _mp.get_context("spawn")
        self._task_q     = self._ctx.Queue()
        self._result_q   = self._ctx.Queue()
        self._procs: List[_mp.Process] = []
        self._next_req_id = 1
        self._stash: Dict[int, List[Tuple[str, dict]]] = {}

    def start(self) -> None:
        for gid in self.gpu_ids:
            p = self._ctx.Process(
                target=_mcqa_worker_loop,
                args=(gid, self.model_path, self._task_q, self._result_q,
                      self.torch_dtype),
                daemon=True,
            )
            p.start()
            self._procs.append(p)

    def stop(self) -> None:
        for _ in self._procs:
            self._task_q.put(None)
        for p in self._procs:
            p.join(timeout=15)

    def is_alive(self) -> bool:
        return all(p.is_alive() for p in self._procs)

    def get_mcqa_logits(
        self,
        audio_path: str,
        prompt: str,
        target_ids: List[int],
    ) -> List[float]:
        """Singolo forward pass → [logit_A, logit_B, logit_C, logit_D]."""
        req_id = self._new_req_id()
        self._put_task("mcqa_logits", {
            "audio_path": audio_path,
            "prompt":     prompt,
            "target_ids": list(target_ids),
        }, req_id=req_id)
        kind, payload = self._get_for_req(req_id)
        if kind != "mcqa_logits":
            raise RuntimeError(f"Risposta inattesa: {kind}")
        return list(payload["vals"])

    def get_mcqa_logits_batch(
        self,
        items: List[Dict[str, Any]],
        target_ids: List[int],
        window: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Batch di N forward pass → N vettori [logit_A, logit_B, logit_C, logit_D].
        Finestra IPC per tenere i worker sempre occupati.
        """
        if not items:
            return []
        if window is None:
            env_w = int(os.environ.get("MMSHAP_QUEUE_WINDOW", "0"))
            window = env_w if env_w > 0 else max(len(self._procs), 1) * 8
        window = max(1, int(window))

        req_id = self._new_req_id()
        total  = len(items)
        out: List[Optional[List[float]]] = [None] * total
        sent = got = 0

        while sent < total and sent < window:
            self._put_task("mcqa_logits", {
                "audio_path": items[sent]["audio_path"],
                "prompt":     items[sent]["prompt"],
                "target_ids": list(target_ids),
                "batch_id":   sent,
            }, req_id=req_id)
            sent += 1

        while got < total:
            kind, payload = self._get_for_req(req_id)
            if kind != "mcqa_logits":
                raise RuntimeError(f"Risposta inattesa nel batch: {kind}")
            bid = payload.get("batch_id")
            if bid is None:
                raise RuntimeError(f"batch_id mancante: {payload}")
            out[int(bid)] = list(payload["vals"])
            got += 1
            if sent < total:
                self._put_task("mcqa_logits", {
                    "audio_path": items[sent]["audio_path"],
                    "prompt":     items[sent]["prompt"],
                    "target_ids": list(target_ids),
                    "batch_id":   sent,
                }, req_id=req_id)
                sent += 1

        return [x if x is not None else [] for x in out]

    def _new_req_id(self) -> int:
        rid = int(self._next_req_id)
        self._next_req_id += 1
        return rid

    def _put_task(self, kind: str, payload: dict, req_id: int) -> None:
        d = dict(payload)
        d["req_id"] = int(req_id)
        self._task_q.put(Task(kind, d))

    def _get_for_req(self, req_id: int) -> Tuple[str, dict]:
        req_id = int(req_id)
        q = self._stash.get(req_id, [])
        if q:
            kind, payload = q.pop(0)
            if not q:
                self._stash.pop(req_id, None)
            return kind, payload
        while True:
            kind, payload = self._result_q.get()
            if kind == "__mcqa_worker_init_error__":
                raise RuntimeError(f"McqaWorker init error: {payload}")
            if kind == "__mcqa_task_error__":
                raise RuntimeError(f"McqaWorker task error: {payload}")
            rid = int(payload.get("req_id", -1))
            if rid == req_id:
                return kind, payload
            self._stash.setdefault(rid, []).append((kind, payload))


def _create_mcqa_runner(model_path: str, gpu_ids: List[int]) -> McqaParallelRunner:
    """Crea, avvia e verifica McqaParallelRunner."""
    import time as _time
    runner = McqaParallelRunner(model_path=model_path, gpu_ids=gpu_ids)
    runner.start()
    _time.sleep(1.5)
    if not runner.is_alive():
        try: runner.stop()
        except Exception: pass
        raise RuntimeError("McqaParallelRunner: worker morti in avvio.")
    logger.info(f"McqaParallelRunner attivo su GPU fisiche: {gpu_ids}")
    return runner


# Wrappers pubblici (interfaccia identica per il resto di expE)
def get_mcqa_letter_logits(
    runner: McqaParallelRunner,
    audio_path: str,
    prompt: str,
    letter_token_ids: List[int],
) -> List[float]:
    """[logit_A, logit_B, logit_C, logit_D] in un singolo forward pass."""
    return runner.get_mcqa_logits(
        audio_path=audio_path,
        prompt=prompt,
        target_ids=letter_token_ids,
    )


def score_items_mcqa(
    runner: McqaParallelRunner,
    items: List[Dict[str, Any]],
    letter_token_ids: List[int],
) -> List[List[float]]:
    """N forward pass → N vettori [logit_A, logit_B, logit_C, logit_D]."""
    return runner.get_mcqa_logits_batch(items=items, target_ids=letter_token_ids)


# 8) Costruzione ranking MI
# =============================================================================
def build_mi_ranking(exp_a_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    mi_audio = rank_audio_features(exp_a_data.get("mi1_stem_x_seg", []))
    word_strings = exp_a_data.get("question_words", [])
    mi_text  = rank_text_words(exp_a_data.get("mi2_words", []), word_strings)

    mi_audio_norm = scale_normalize_ranked(mi_audio)
    mi_text_norm  = scale_normalize_ranked(mi_text)

    combined = []
    for f in mi_audio_norm:
        combined.append({**f, "modality": "audio"})
    for f in mi_text_norm:
        combined.append({**f, "modality": "text"})
    combined.sort(key=lambda d: d.get("abs_value_norm", 0.0), reverse=True)
    return combined

def build_mi_balanced_topks(
    exp_a_data: Dict[str, Any],
    k_values: List[int],
) -> Dict[int, List[Dict[str, Any]]]:
    mi_audio = rank_audio_features(exp_a_data.get("mi1_stem_x_seg", []))
    word_strings = exp_a_data.get("question_words", [])
    mi_text = rank_text_words(exp_a_data.get("mi2_words", []), word_strings)

    mi_audio_norm = scale_normalize_ranked(mi_audio)
    mi_text_norm = scale_normalize_ranked(mi_text)

    def _positive_sorted(items: List[Dict[str, Any]], modality: str) -> List[Dict[str, Any]]:
        out = []
        for f in items:
            v = float(f.get("value_norm", f.get("value", 0.0)))
            if v > 0.0:
                out.append({**f, "modality": modality})
        out.sort(key=lambda d: float(d.get("value_norm", d.get("value", 0.0))), reverse=True)
        return out

    audio_pos = _positive_sorted(mi_audio_norm, "audio")
    text_pos = _positive_sorted(mi_text_norm, "text")

    topks: Dict[int, List[Dict[str, Any]]] = {}
    for k in k_values:
        kk = int(k)
        topks[kk] = audio_pos[:kk] + text_pos[:kk]

    return topks

def build_uc_audio_ranking(exp_a_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranked = rank_audio_features(exp_a_data.get("uc1_stem_x_seg", []))
    return [{**f, "modality": "audio"} for f in ranked]

def build_uc_text_ranking(exp_a_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    word_strings = exp_a_data.get("question_words", [])
    ranked = rank_text_words(exp_a_data.get("uc2_words", []), word_strings)
    return [{**f, "modality": "text"} for f in ranked]

def select_top_k_positive(
    ranked: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    def _v(f):
        return float(f.get("value_norm", f.get("value", 0.0)))
    positives = [f for f in ranked if _v(f) > 0.0]
    positives.sort(key=_v, reverse=True)
    return positives[:int(k)]

def split_topk_by_modality(
    top_k: List[Dict[str, Any]],
) -> Tuple[List[Dict], List[Dict]]:
    audio_part = [f for f in top_k if f.get("modality") == "audio"]
    text_part  = [f for f in top_k if f.get("modality") == "text"]
    return audio_part, text_part

# =============================================================================
# 9) Costruzione input perturbato
# =============================================================================
def build_perturbation_inputs(
    feature_source: str,
    selected_features: List[Dict[str, Any]],
    mode: str,
    question: str,
    options: List[str],
    n_words_total: int,
    audio_path: str,
    tokenizer,
    perturb_audio_dir: str,
    k_tag: int,
) -> Dict[str, Any]:
    from QA_analysis.utils.shared_utils import build_hummusqa_qwen25_prompt

    audio_part, text_part = split_topk_by_modality(selected_features)

    # ---- AUDIO ----
    if feature_source == "UC_text":
        out_audio_path = audio_path
    else:
        if not audio_part:
            out_audio_path = audio_path
        else:
            am = build_audio_binary_mask(audio_part, mode=mode)
            tag = f"_{feature_source}_k{k_tag}_{mode}"
            out_audio_path = materialize_perturbed_audio(
                audio_path=audio_path,
                audio_binary_mask=am,
                out_dir=perturb_audio_dir,
                suffix=tag,
            )

    # ---- TESTO ----
    if feature_source == "UC_audio":
        out_prompt = build_hummusqa_qwen25_prompt(question, options)
    else:
        if not text_part:
            out_prompt = build_hummusqa_qwen25_prompt(question, options)
        else:
            out_prompt = build_perturbed_prompt(
                question=question,
                options=options,
                selected_words=text_part,
                mode=mode,
                tokenizer=tokenizer,
                n_words_total=n_words_total,
            )

    return {
        "audio_path": out_audio_path,
        "prompt":     out_prompt,
        "n_audio":    len(audio_part),
        "n_text":     len(text_part),
    }

# =============================================================================
# 10) Metriche chance-normalized
# =============================================================================
def softmax_4(logits_4: List[float]) -> np.ndarray:
    x = np.asarray(logits_4, dtype=float)
    if x.size == 0:
        return np.ones(4, dtype=float) / 4.0
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-12)

def chance_normalized_sufficiency(
    p_suf: float, p_orig: float, p_chance: float = P_CHANCE,
) -> float:
    denom = p_orig - p_chance
    if denom <= EPS_BASELINE:
        return float("nan")
    return float((p_suf - p_chance) / denom)

def chance_normalized_necessity(
    p_nec: float, p_orig: float, p_chance: float = P_CHANCE,
) -> float:
    denom = p_orig - p_chance
    if denom <= EPS_BASELINE:
        return float("nan")
    return float((p_orig - p_nec) / denom)

# =============================================================================
# 11) Costruzione curve progressive
# =============================================================================
def build_curves_for_source(
    feature_source: str,
    ranked_full: List[Dict[str, Any]],
    k_values: List[int],
    p_orig: float,
    target_idx: int,
    target_letter: str,
    question: str,
    options: List[str],
    n_words_total: int,
    audio_path: str,
    tokenizer,
    perturb_audio_dir: str,
    letter_token_ids: List[int],
    runner,
) -> Dict[str, Any]:
    def _v(f):
        return float(f.get("value_norm", f.get("value", 0.0)))
    positives = [f for f in ranked_full if _v(f) > 0.0]
    positives.sort(key=_v, reverse=True)
    n_pos_available = len(positives)

    # Costruiamo gli item da scorare — NOTA: niente "target_ids" qui,
    # lo gestisce score_items_mcqa internamente
    items: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []

    for k in k_values:
        topk = positives[:k]
        if len(topk) == 0:
            meta.append({"k": k, "mode": None, "skip_reason": "no_positive_features"})
            continue

        for mode in ("sufficiency", "necessity"):
            inp = build_perturbation_inputs(
                feature_source=feature_source,
                selected_features=topk,
                mode=mode,
                question=question,
                options=options,
                n_words_total=n_words_total,
                audio_path=audio_path,
                tokenizer=tokenizer,
                perturb_audio_dir=perturb_audio_dir,
                k_tag=k,
            )
            # BUGFIX: non mettiamo più target_ids nell'item.
            # score_items_mcqa farà il probing corretto.
            items.append({
                "audio_path": inp["audio_path"],
                "prompt":     inp["prompt"],
            })
            meta.append({
                "k": k,
                "mode": mode,
                "n_audio_perturbed": inp["n_audio"],
                "n_text_perturbed":  inp["n_text"],
                "topk_size_actual":  len(topk),
            })

    # BUGFIX: usa score_items_mcqa invece di get_mmshap_logits_batch
    if items:
        all_logits_4 = score_items_mcqa(
            runner=runner,
            items=items,
            letter_token_ids=letter_token_ids,
        )
    else:
        all_logits_4 = []

    # Riassembla per k
    by_k: Dict[int, Dict[str, Any]] = {}

    logits_iter = iter(all_logits_4)
    for m in meta:
        if m.get("skip_reason") is not None:
            continue
        k = m["k"]
        mode = m["mode"]
        logits_4 = next(logits_iter)
        if not logits_4 or len(logits_4) < 4:
            continue
        probs = softmax_4(logits_4)
        p_t = float(probs[target_idx])
        ans = answer_letter_from_logits(logits_4)
        if k not in by_k:
            by_k[k] = {"k": int(k)}
        by_k[k][mode] = {
            "p_target": p_t,
            "answer": ans,
            "logits": [float(x) for x in logits_4],
            "probs":  [float(x) for x in probs],
            "n_audio_perturbed": int(m["n_audio_perturbed"]),
            "n_text_perturbed":  int(m["n_text_perturbed"]),
            "topk_size_actual":  int(m["topk_size_actual"]),
        }

    curve_sufficiency = []
    curve_necessity   = []
    topk_composition  = []

    for k in k_values:
        topk = positives[:k]
        n_audio = sum(1 for f in topk if f.get("modality") == "audio")
        n_text  = sum(1 for f in topk if f.get("modality") == "text")
        topk_composition.append({
            "k": int(k),
            "topk_size_actual": int(len(topk)),
            "n_audio": int(n_audio),
            "n_text":  int(n_text),
            "audio_fraction": float(n_audio / max(1, len(topk))),
            "text_fraction":  float(n_text  / max(1, len(topk))),
            "is_multimodal":    bool(n_audio > 0 and n_text > 0),
            "is_unimodal_audio": bool(n_audio > 0 and n_text == 0),
            "is_unimodal_text":  bool(n_text  > 0 and n_audio == 0),
        })

        pack = by_k.get(k, {})
        suf = pack.get("sufficiency")
        nec = pack.get("necessity")

        if suf is not None:
            s_score = chance_normalized_sufficiency(suf["p_target"], p_orig)
            curve_sufficiency.append({
                "k": int(k),
                "topk_size_actual": int(len(topk)),
                "p_target": float(suf["p_target"]),
                "score":   float(s_score) if not np.isnan(s_score) else None,
                "answer":  suf["answer"],
                "answer_matches_target": bool(suf["answer"] == target_letter),
            })
        else:
            curve_sufficiency.append({
                "k": int(k), "topk_size_actual": int(len(topk)),
                "p_target": None, "score": None,
                "answer": None, "answer_matches_target": None,
            })

        if nec is not None:
            n_score = chance_normalized_necessity(nec["p_target"], p_orig)
            curve_necessity.append({
                "k": int(k),
                "topk_size_actual": int(len(topk)),
                "p_target": float(nec["p_target"]),
                "score":   float(n_score) if not np.isnan(n_score) else None,
                "answer":  nec["answer"],
                "target_lost": bool(nec["answer"] != target_letter),
            })
        else:
            curve_necessity.append({
                "k": int(k), "topk_size_actual": int(len(topk)),
                "p_target": None, "score": None,
                "answer": None, "target_lost": None,
            })

    def _first_k_above(curve, threshold):
        for pt in curve:
            s = pt.get("score")
            if s is not None and s >= threshold:
                return int(pt["k"])
        return None

    return {
        "feature_source": feature_source,
        "n_positive_available": int(n_pos_available),
        "k_values": [int(k) for k in k_values],
        "curve_sufficiency": curve_sufficiency,
        "curve_necessity":   curve_necessity,
        "topk_composition":  topk_composition,
        "minimal_sufficient_k": _first_k_above(curve_sufficiency, ALPHA_SUFFICIENT),
        "minimal_necessary_k":  _first_k_above(curve_necessity,   BETA_NECESSARY),
    }

def build_curves_for_explicit_topks(
    feature_source: str,
    topks_by_k: Dict[int, List[Dict[str, Any]]],
    k_values: List[int],
    p_orig: float,
    target_idx: int,
    target_letter: str,
    question: str,
    options: List[str],
    n_words_total: int,
    audio_path: str,
    tokenizer,
    perturb_audio_dir: str,
    letter_token_ids: List[int],
    runner,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    meta: List[Dict[str, Any]] = []

    n_positive_available = max(
        (len(v) for v in topks_by_k.values()),
        default=0,
    )

    for k in k_values:
        topk = topks_by_k.get(int(k), [])
        if len(topk) == 0:
            meta.append({"k": k, "mode": None, "skip_reason": "no_positive_features"})
            continue

        for mode in ("sufficiency", "necessity"):
            inp = build_perturbation_inputs(
                feature_source="MI",
                selected_features=topk,
                mode=mode,
                question=question,
                options=options,
                n_words_total=n_words_total,
                audio_path=audio_path,
                tokenizer=tokenizer,
                perturb_audio_dir=perturb_audio_dir,
                k_tag=k,
            )
            # BUGFIX: niente target_ids
            items.append({
                "audio_path": inp["audio_path"],
                "prompt":     inp["prompt"],
            })
            meta.append({
                "k": k,
                "mode": mode,
                "n_audio_perturbed": inp["n_audio"],
                "n_text_perturbed": inp["n_text"],
                "topk_size_actual": len(topk),
            })

    # BUGFIX: usa score_items_mcqa
    if items:
        all_logits_4 = score_items_mcqa(
            runner=runner,
            items=items,
            letter_token_ids=letter_token_ids,
        )
    else:
        all_logits_4 = []

    by_k: Dict[int, Dict[str, Any]] = {}
    logits_iter = iter(all_logits_4)

    for m in meta:
        if m.get("skip_reason") is not None:
            continue
        k = int(m["k"])
        mode = m["mode"]
        logits_4 = next(logits_iter)
        if not logits_4 or len(logits_4) < 4:
            continue
        probs = softmax_4(logits_4)
        p_t = float(probs[target_idx])
        ans = answer_letter_from_logits(logits_4)
        if k not in by_k:
            by_k[k] = {"k": k}
        by_k[k][mode] = {
            "p_target": p_t,
            "answer": ans,
            "logits": [float(x) for x in logits_4],
            "probs": [float(x) for x in probs],
            "n_audio_perturbed": int(m["n_audio_perturbed"]),
            "n_text_perturbed": int(m["n_text_perturbed"]),
            "topk_size_actual": int(m["topk_size_actual"]),
        }

    curve_sufficiency = []
    curve_necessity = []
    topk_composition = []

    for k in k_values:
        kk = int(k)
        topk = topks_by_k.get(kk, [])
        n_audio = sum(1 for f in topk if f.get("modality") == "audio")
        n_text = sum(1 for f in topk if f.get("modality") == "text")

        topk_composition.append({
            "k": kk, "topk_size_actual": int(len(topk)),
            "n_audio": int(n_audio), "n_text": int(n_text),
            "audio_fraction": float(n_audio / max(1, len(topk))),
            "text_fraction": float(n_text / max(1, len(topk))),
            "is_multimodal": bool(n_audio > 0 and n_text > 0),
            "is_unimodal_audio": bool(n_audio > 0 and n_text == 0),
            "is_unimodal_text": bool(n_text > 0 and n_audio == 0),
        })

        pack = by_k.get(kk, {})
        suf = pack.get("sufficiency")
        nec = pack.get("necessity")

        if suf is not None:
            s_score = chance_normalized_sufficiency(suf["p_target"], p_orig)
            curve_sufficiency.append({
                "k": kk, "topk_size_actual": int(len(topk)),
                "p_target": float(suf["p_target"]),
                "score": float(s_score) if not np.isnan(s_score) else None,
                "answer": suf["answer"],
                "answer_matches_target": bool(suf["answer"] == target_letter),
            })
        else:
            curve_sufficiency.append({
                "k": kk, "topk_size_actual": int(len(topk)),
                "p_target": None, "score": None,
                "answer": None, "answer_matches_target": None,
            })

        if nec is not None:
            n_score = chance_normalized_necessity(nec["p_target"], p_orig)
            curve_necessity.append({
                "k": kk, "topk_size_actual": int(len(topk)),
                "p_target": float(nec["p_target"]),
                "score": float(n_score) if not np.isnan(n_score) else None,
                "answer": nec["answer"],
                "target_lost": bool(nec["answer"] != target_letter),
            })
        else:
            curve_necessity.append({
                "k": kk, "topk_size_actual": int(len(topk)),
                "p_target": None, "score": None,
                "answer": None, "target_lost": None,
            })

    def _first_k_above(curve, threshold):
        for pt in curve:
            s = pt.get("score")
            if s is not None and s >= threshold:
                return int(pt["k"])
        return None

    return {
        "feature_source": feature_source,
        "selection_policy": "balanced_top_k_audio_plus_top_k_text",
        "n_positive_available": int(n_positive_available),
        "k_values": [int(k) for k in k_values],
        "curve_sufficiency": curve_sufficiency,
        "curve_necessity": curve_necessity,
        "topk_composition": topk_composition,
        "minimal_sufficient_k": _first_k_above(curve_sufficiency, ALPHA_SUFFICIENT),
        "minimal_necessary_k": _first_k_above(curve_necessity, BETA_NECESSARY),
    }

# =============================================================================
# 12) Process singolo sample
# =============================================================================
_n_diagnostic_printed = 0

def process_sample(
    sample_id: str,
    exp_a_dir: str,
    audio_cache_dir: str,
    perturb_audio_dir: str,
    runner,
    tokenizer,
    letter_token_ids: List[int],
    include_diagnostic: bool,
) -> Dict[str, Any]:
    global _n_diagnostic_printed

    exp_a_data = load_exp_a_sample(exp_a_dir, sample_id)

    audio_path = _resolve_audio_path(audio_cache_dir, exp_a_data)
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio non trovato per {sample_id} in {audio_cache_dir}."
        )

    options = _resolve_options_for_sample(sample_id)
    question = exp_a_data.get("question", "")
    n_words_total = len(exp_a_data.get("question_words", []))

    category = exp_a_data.get("category", "unknown")
    macro_family = _macro_family_from_category(category)

    # ---- Baseline: usa get_mcqa_letter_logits (BUGFIX) ----
    from QA_analysis.utils.shared_utils import build_hummusqa_qwen25_prompt
    base_prompt = build_hummusqa_qwen25_prompt(question, options)

    # BUGFIX: get_mcqa_letter_logits invece di get_mmshap_logits con 4 id
    base_logits = get_mcqa_letter_logits(
        runner=runner,
        audio_path=audio_path,
        prompt=base_prompt,
        letter_token_ids=letter_token_ids,
    )

    answer_orig = answer_letter_from_logits(base_logits)
    correct_letter = exp_a_data.get("correct_answer_letter", "A")

    target_letter = answer_orig
    target_idx = ord(target_letter) - 65
    correct_idx = ord(correct_letter) - 65

    base_probs = softmax_4(base_logits)
    p_orig_target  = float(base_probs[target_idx])
    p_orig_correct = float(base_probs[correct_idx])

    baseline_above_chance = bool((p_orig_target - P_CHANCE) > EPS_BASELINE)

    if _n_diagnostic_printed < DIAGNOSTIC_FIRST_N_SAMPLES:
        logger.info(
            f"  [DIAG {_n_diagnostic_printed+1}/{DIAGNOSTIC_FIRST_N_SAMPLES}] "
            f"{sample_id} | logits=[{base_logits[0]:.2f},{base_logits[1]:.2f},"
            f"{base_logits[2]:.2f},{base_logits[3]:.2f}] | target={target_letter} | "
            f"p_target={p_orig_target:.3f} | correct={correct_letter} | "
            f"model_correct={answer_orig == correct_letter} | "
            f"above_chance={baseline_above_chance}"
        )

    # ---- MI (PRINCIPALE) ----
    mi_ranking = build_mi_ranking(exp_a_data)
    mi_results = build_curves_for_source(
        feature_source="MI",
        ranked_full=mi_ranking,
        k_values=K_VALUES_MAIN,
        p_orig=p_orig_target,
        target_idx=target_idx,
        target_letter=target_letter,
        question=question,
        options=options,
        n_words_total=n_words_total,
        audio_path=audio_path,
        tokenizer=tokenizer,
        perturb_audio_dir=perturb_audio_dir,
        letter_token_ids=letter_token_ids,
        runner=runner,
    )

    # ---- MI BALANCED ----
    mi_balanced_results = None
    if INCLUDE_BALANCED_MI:
        mi_balanced_topks = build_mi_balanced_topks(
            exp_a_data=exp_a_data,
            k_values=K_VALUES_BALANCED_MI,
        )
        mi_balanced_results = build_curves_for_explicit_topks(
            feature_source="MI_balanced",
            topks_by_k=mi_balanced_topks,
            k_values=K_VALUES_BALANCED_MI,
            p_orig=p_orig_target,
            target_idx=target_idx,
            target_letter=target_letter,
            question=question,
            options=options,
            n_words_total=n_words_total,
            audio_path=audio_path,
            tokenizer=tokenizer,
            perturb_audio_dir=perturb_audio_dir,
            letter_token_ids=letter_token_ids,
            runner=runner,
        )

    diag_to_print = (_n_diagnostic_printed < DIAGNOSTIC_FIRST_N_SAMPLES)
    if diag_to_print:
        for pt_s, pt_n in zip(mi_results["curve_sufficiency"],
                              mi_results["curve_necessity"]):
            k = pt_s["k"]
            s = pt_s["score"]; n = pt_n["score"]
            s_str = f"{s:.3f}" if s is not None else "  nan"
            n_str = f"{n:.3f}" if n is not None else "  nan"
            logger.info(
                f"    MI k={k} | suf={s_str} | nec={n_str} | "
                f"ans_suf={pt_s['answer']} | ans_nec={pt_n['answer']}"
            )
        logger.info(
            f"    MI minimal_sufficient_k = {mi_results['minimal_sufficient_k']} | "
            f"minimal_necessary_k = {mi_results['minimal_necessary_k']}"
        )

    out: Dict[str, Any] = {
        "sample_id": sample_id,
        "category":   category,
        "difficulty": exp_a_data.get("difficulty", "unknown"),
        "macro_family": macro_family,
        "correct_answer_letter": correct_letter,
        "model_answer_baseline": answer_orig,
        "correct_baseline": bool(answer_orig == correct_letter),
        "target_letter": target_letter,
        "target_idx":    int(target_idx),
        "target_policy": "model_prediction",

        "logits_original_all": [float(x) for x in base_logits],
        "probs_original_all":  [float(x) for x in base_probs],
        "prob_original_target":  p_orig_target,
        "prob_original_correct": p_orig_correct,

        "p_chance": P_CHANCE,
        "eps_baseline": EPS_BASELINE,
        "baseline_above_chance": baseline_above_chance,

        "alpha_sufficient": ALPHA_SUFFICIENT,
        "beta_necessary":   BETA_NECESSARY,

        "metric_definitions": {
            "sufficiency_score": "(p_suf - p_chance) / (p_orig - p_chance)",
            "necessity_score":   "(p_orig - p_nec) / (p_orig - p_chance)",
            "minimal_sufficient_k":
                f"min k with sufficiency_score >= {ALPHA_SUFFICIENT}",
            "minimal_necessary_k":
                f"min k with necessity_score >= {BETA_NECESSARY}",
        },

        "mi_results": mi_results,
        "mi_balanced_results": mi_balanced_results,
    }

    # ---- UC_audio, UC_text (DIAGNOSTICI, opzionali) ----
    if include_diagnostic:
        uc_audio_ranking = build_uc_audio_ranking(exp_a_data)
        uc_text_ranking  = build_uc_text_ranking(exp_a_data)

        out["diagnostic_uc_audio_results"] = build_curves_for_source(
            feature_source="UC_audio",
            ranked_full=uc_audio_ranking,
            k_values=DIAGNOSTIC_K_VALUES_UC_AUDIO,
            p_orig=p_orig_target,
            target_idx=target_idx,
            target_letter=target_letter,
            question=question,
            options=options,
            n_words_total=n_words_total,
            audio_path=audio_path,
            tokenizer=tokenizer,
            perturb_audio_dir=perturb_audio_dir,
            letter_token_ids=letter_token_ids,
            runner=runner,
        )
        out["diagnostic_uc_text_results"] = build_curves_for_source(
            feature_source="UC_text",
            ranked_full=uc_text_ranking,
            k_values=DIAGNOSTIC_K_VALUES_UC_TEXT,
            p_orig=p_orig_target,
            target_idx=target_idx,
            target_letter=target_letter,
            question=question,
            options=options,
            n_words_total=n_words_total,
            audio_path=audio_path,
            tokenizer=tokenizer,
            perturb_audio_dir=perturb_audio_dir,
            letter_token_ids=letter_token_ids,
            runner=runner,
        )

    if _n_diagnostic_printed < DIAGNOSTIC_FIRST_N_SAMPLES:
        _n_diagnostic_printed += 1

    return out

# =============================================================================
# 13) Helpers risoluzione audio_path e options
# =============================================================================
_audio_path_cache: Dict[str, str] = {}
_options_cache: Dict[str, List[str]] = {}
_hummusqa_entries_cache: Optional[List[dict]] = None

def _load_hummusqa_entries():
    global _hummusqa_entries_cache
    if _hummusqa_entries_cache is None:
        from QA_analysis.utils.shared_utils import load_hummusqa_entries_parquet
        entries, _ = load_hummusqa_entries_parquet(HUMMUSQA_PATH)
        _hummusqa_entries_cache = entries
    return _hummusqa_entries_cache

def _resolve_audio_path(audio_cache_dir: str, exp_a_data: Dict[str, Any]) -> Optional[str]:
    sample_id = exp_a_data.get("sample_id", "")
    if sample_id in _audio_path_cache:
        return _audio_path_cache[sample_id]
    idx = exp_a_data.get("idx", -1)
    if idx < 0:
        return None
    pattern = os.path.join(audio_cache_dir, f"sample_{idx}_*_decoded.wav")
    matches = glob.glob(pattern)
    if not matches:
        pattern = os.path.join(audio_cache_dir, f"sample_{idx}_*.wav")
        matches = glob.glob(pattern)
    if matches:
        _audio_path_cache[sample_id] = matches[0]
        return matches[0]
    return None

def _resolve_options_for_sample(sample_id: str) -> List[str]:
    if sample_id in _options_cache:
        return _options_cache[sample_id]
    entries = _load_hummusqa_entries()
    for entry in entries:
        ident = str(entry.get("identifier") or entry.get("question_id")
                    or entry.get("id") or "")
        if ident == sample_id:
            opts = [
                str(entry.get("answer", "")).strip(),
                str(entry.get("distractor_1", "")).strip(),
                str(entry.get("distractor_2", "")).strip(),
                str(entry.get("distractor_3", "")).strip(),
            ]
            _options_cache[sample_id] = opts
            return opts
    raise RuntimeError(f"Options non trovate per {sample_id} nel parquet HumMusQA")

# =============================================================================
# 14) Aggregazione
# =============================================================================
def aggregate_exp_e(batch_dir: str) -> Tuple[str, str]:
    import pandas as pd

    per_sample_dir = os.path.join(batch_dir, "per_sample")
    curve_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for jf in sorted(glob.glob(os.path.join(per_sample_dir, "*_exp_e.json"))):
        with open(jf, "r") as f:
            d = json.load(f)

        common = {
            "sample_id":   d.get("sample_id"),
            "category":    d.get("category"),
            "difficulty":  d.get("difficulty"),
            "macro_family": d.get("macro_family"),
            "correct_baseline": d.get("correct_baseline"),
            "target_letter": d.get("target_letter"),
            "prob_original_target":  d.get("prob_original_target"),
            "prob_original_correct": d.get("prob_original_correct"),
            "baseline_above_chance": d.get("baseline_above_chance"),
        }

        for src_key, src_label in [
            ("mi_results", "MI"),
            ("mi_balanced_results", "MI_balanced"),
            ("diagnostic_uc_audio_results", "UC_audio"),
            ("diagnostic_uc_text_results", "UC_text"),
        ]:
            r = d.get(src_key)
            if not r:
                continue
            n_pos = r.get("n_positive_available", 0)
            for pt_s, pt_n, pt_c in zip(
                r.get("curve_sufficiency", []),
                r.get("curve_necessity",   []),
                r.get("topk_composition",  []),
            ):
                row = dict(common)
                row.update({
                    "feature_source": src_label,
                    "k": int(pt_s["k"]),
                    "topk_size_actual": int(pt_s.get("topk_size_actual",
                                                     pt_c.get("topk_size_actual", 0))),
                    "n_positive_available": int(n_pos),
                    "p_suf_target":  pt_s.get("p_target"),
                    "p_nec_target":  pt_n.get("p_target"),
                    "sufficiency_score": pt_s.get("score"),
                    "necessity_score":   pt_n.get("score"),
                    "answer_suf":   pt_s.get("answer"),
                    "answer_nec":   pt_n.get("answer"),
                    "answer_suf_matches_target": pt_s.get("answer_matches_target"),
                    "target_lost_in_necessity": pt_n.get("target_lost"),
                    "topk_n_audio": int(pt_c.get("n_audio", 0)),
                    "topk_n_text":  int(pt_c.get("n_text", 0)),
                    "topk_audio_fraction": float(pt_c.get("audio_fraction", 0.0)),
                    "topk_text_fraction":  float(pt_c.get("text_fraction", 0.0)),
                    "topk_is_multimodal":     pt_c.get("is_multimodal"),
                    "topk_is_unimodal_audio": pt_c.get("is_unimodal_audio"),
                    "topk_is_unimodal_text":  pt_c.get("is_unimodal_text"),
                })
                curve_rows.append(row)

            srow = dict(common)
            srow.update({
                "feature_source": src_label,
                "n_positive_available": int(n_pos),
                "minimal_sufficient_k": r.get("minimal_sufficient_k"),
                "minimal_necessary_k":  r.get("minimal_necessary_k"),
            })
            ks = [pt["k"] for pt in r.get("curve_sufficiency", [])
                  if pt.get("score") is not None]
            s_vals = [pt["score"] for pt in r.get("curve_sufficiency", [])
                      if pt.get("score") is not None]
            n_vals = [pt["score"] for pt in r.get("curve_necessity", [])
                      if pt.get("score") is not None]
            if len(ks) >= 2:
                width = ks[-1] - ks[0]
                srow["auc_sufficiency"] = float(np.trapz(s_vals, ks) / max(1, width))
                srow["auc_necessity"]   = float(np.trapz(n_vals, ks) / max(1, width))
            else:
                srow["auc_sufficiency"] = None
                srow["auc_necessity"]   = None
            summary_rows.append(srow)

    if not curve_rows:
        logger.warning("Nessun risultato Exp E da aggregare.")
        return "", ""

    agg_dir = _ensure_dir(os.path.join(batch_dir, "aggregated"))
    df_curves = pd.DataFrame(curve_rows)
    df_summary = pd.DataFrame(summary_rows)

    curves_pq = os.path.join(agg_dir, "exp_e_curves.parquet")
    curves_cs = os.path.join(agg_dir, "exp_e_curves.csv")
    summary_pq = os.path.join(agg_dir, "exp_e_summary.parquet")
    summary_cs = os.path.join(agg_dir, "exp_e_summary.csv")

    df_curves.to_parquet(curves_pq, index=False)
    df_curves.to_csv(curves_cs, index=False)
    df_summary.to_parquet(summary_pq, index=False)
    df_summary.to_csv(summary_cs, index=False)

    logger.info(f"Aggregazione: {len(df_curves)} righe curve → {curves_pq}")
    logger.info(f"Aggregazione: {len(df_summary)} righe summary → {summary_pq}")
    return curves_pq, summary_pq

# =============================================================================
# 15) Main batch loop
# =============================================================================
def run_batch_exp_e(
    exp_a_dir: str,
    resume_dir: Optional[str] = None,
    only_aggregate: bool = False,
    include_diagnostic: bool = INCLUDE_DIAGNOSTIC_SOURCES_DEFAULT,
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
        aggregate_exp_e(batch_dir)
        return batch_dir

    sample_ids = list_available_exp_a_samples(exp_a_dir)
    logger.info(f"Sample disponibili da Exp A: {len(sample_ids)}")
    if TEST_FIRST_N_SAMPLES is not None:
        sample_ids = sample_ids[:int(TEST_FIRST_N_SAMPLES)]
        logger.info(f"[TEST MODE] Uso solo i primi {len(sample_ids)} sample.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model path non trovato: {MODEL_PATH}")

    gpu_ids, _ = get_available_gpus_with_memory(min_free_memory_gb=MIN_FREE_GB_RUNNER)
    gpu_ids = gpu_ids[:MAX_GPUS_TO_USE]
    if not gpu_ids:
        raise RuntimeError(f"Nessuna GPU con >= {MIN_FREE_GB_RUNNER} GB liberi.")
    logger.info(f"GPU selezionate: {gpu_ids}")

    runner = _create_mcqa_runner(
        model_path=MODEL_PATH,
        gpu_ids=gpu_ids,
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer
    letter_token_ids = get_letter_token_ids(tokenizer)
    logger.info(f"Letter token ids (A,B,C,D): {letter_token_ids}")
    logger.info(
        f"Design: MI=PRIMARIO (k={K_VALUES_MAIN}) | "
        f"UC_audio,UC_text=DIAGNOSTICI ({'on' if include_diagnostic else 'off'})"
    )
    logger.info(
        "[BUGFIX ATTIVO] MCQA logits via score_items_mcqa "
        "(T=1 fast-path per ogni lettera, nessun branch generate())"
    )

    _atomic_json_dump(
        {
            "batch_dir": batch_dir, "exp": "E_v4_fixed",
            "exp_a_dir": exp_a_dir,
            "n_sample_ids": len(sample_ids),
            "test_first_n_samples": TEST_FIRST_N_SAMPLES,
            "main_feature_source": "MI",
            "k_values_main": K_VALUES_MAIN,
            "include_diagnostic_sources": include_diagnostic,
            "diagnostic_k_values_uc_audio": DIAGNOSTIC_K_VALUES_UC_AUDIO,
            "diagnostic_k_values_uc_text":  DIAGNOSTIC_K_VALUES_UC_TEXT,
            "k_canonical_main": K_CANONICAL_MAIN,
            "p_chance": P_CHANCE,
            "eps_baseline": EPS_BASELINE,
            "alpha_sufficient": ALPHA_SUFFICIENT,
            "beta_necessary":   BETA_NECESSARY,
            "metric_space_primary": "chance_normalized_probability",
            "metric_definitions": {
                "sufficiency_score": "(p_suf - 0.25) / (p_orig - 0.25)",
                "necessity_score":   "(p_orig - p_nec) / (p_orig - 0.25)",
            },
            "topk_selection_policy":
                "positive_supporting_features_only (sort by value_norm desc)",
            "mi_ranking_scale_normalized": True,
            "target_policy": "model_prediction",
            "masking": {
                "audio_mode": os.environ.get("DIME_AUDIO_MASK_MODE"),
                "text_token": "[MASK]",
            },
            "gpu_ids": gpu_ids,
            "model_path": MODEL_PATH,
            "letter_token_ids": letter_token_ids,
            "macro_family_map": MACRO_FAMILY_MAP,
            "include_balanced_mi": INCLUDE_BALANCED_MI,
            "k_values_balanced_mi": K_VALUES_BALANCED_MI,
            "mi_balanced_policy": "top-k audio MI positive + top-k text MI positive",
            "bugfix": {
                "version": "v4_fixed_v3_internal_runner",
                "issue": "target_ids=[A,B,C,D] passato a get_mmshap_logits() "
                         "entrava nel branch generate() autoregressivo: "
                         "step_logits[t, target_ids[t]] misurava 4 punti temporali diversi.",
                "fix_v1": "score_items_mcqa() espandeva ogni item in 4 sotto-item con T=1 "
                          "ciascuno → corretto ma 4x forward pass per perturbazione (~8h).",
                "fix_v3": "McqaParallelRunner custom definito interamente in expE: "
                          "worker loop con 1 forward pass per perturbazione che legge "
                          "tutti i logit da logits[0,-1] → corretto E veloce (~2h). "
                          "gpu_utils.py invariato.",
            },
        },
        os.path.join(batch_dir, "run_info.json"),
    )

    progress = _load_progress(batch_dir)
    completed_ids = set(progress.get("completed", []))
    total = len(sample_ids)
    logger.info(f"Stato: {len(completed_ids)} completati, "
                f"{total - len(completed_ids)} da fare.")

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
                    include_diagnostic=include_diagnostic,
                )
                out_path = os.path.join(per_sample_dir, f"{sid}_exp_e.json")
                _atomic_json_dump(result, out_path)

                for f in glob.glob(os.path.join(perturb_audio_dir, "*")):
                    try: os.remove(f)
                    except Exception: pass

                elapsed = time.time() - t0
                mi_r = result.get("mi_results", {})
                msk = mi_r.get("minimal_sufficient_k")
                mnk = mi_r.get("minimal_necessary_k")
                logger.info(
                    f"  ✓ {sid} in {elapsed:.0f}s | "
                    f"MI: min_suf_k={msk} min_nec_k={mnk} | "
                    f"above_chance={result.get('baseline_above_chance')}"
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
        try: runner.stop()
        except Exception: pass
        try: shutil.rmtree(perturb_audio_dir, ignore_errors=True)
        except Exception: pass

    logger.info("Batch completato. Aggregazione...")
    aggregate_exp_e(batch_dir)
    logger.info(
        f"\n{'='*60}\n"
        f"BATCH EXP E v4 (FIXED) COMPLETATO\n"
        f"  completati : {len(completed_ids)}\n"
        f"  batch dir  : {batch_dir}\n"
        f"{'='*60}"
    )
    return batch_dir

# =============================================================================
# 16) Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batch runner Esperimento E v4-fixed (MCQA logits corretti)"
    )
    parser.add_argument("--exp-a-dir", required=True, type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--only-aggregate", action="store_true")
    parser.add_argument("--no-diagnostic", action="store_true",
                        help="Disabilita UC_audio e UC_text (più veloce).")
    args = parser.parse_args()

    if args.only_aggregate and not args.resume:
        raise ValueError("--only-aggregate richiede --resume")

    include_diagnostic = (not args.no_diagnostic)

    run_batch_exp_e(
        exp_a_dir=args.exp_a_dir,
        resume_dir=args.resume,
        only_aggregate=args.only_aggregate,
        include_diagnostic=include_diagnostic,
    )

if __name__ == "__main__":
    main()
