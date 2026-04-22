import os
import time
import logging
import subprocess
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import sys
import warnings
import contextlib

logger = logging.getLogger("GPU Utils")

def _configure_worker_quiet_mode() -> None:
    """
    Silenzia il rumore inutile nei worker, ma lascia visibili
    i log informativi del progetto.
    """
    import sys

    warnings.resetwarnings()
    warnings.simplefilter("default")
    warnings.filterwarnings("ignore", message=".*audio output may not work as expected.*")
    warnings.filterwarnings("ignore", message=".*System prompt modified.*")
    warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
    warnings.filterwarnings("ignore", message=".*last .* samples are ignored.*")
    warnings.filterwarnings("ignore", message=".*Demucs factorization config.*")
    warnings.filterwarnings("ignore", message=".*Requested segmentation exceeds DIME_AUDIOLIME_MAX_FEATURES.*")

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
        stream=sys.stdout,
    )

    worker_logger = logging.getLogger("GPU Utils")
    worker_logger.setLevel(logging.INFO)
    worker_logger.propagate = False

    if not worker_logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        worker_logger.addHandler(h)

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
    worker_logger.addFilter(drop_filter)

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

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """
    Sopprime stampe dirette su stdout/stderr di librerie rumorose.
    Utile soprattutto durante from_pretrained / generate / demucs init.
    """
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

# ==========================================================
# Runtime config
# ==========================================================
def configure_runtime() -> None:
    os.environ.setdefault("DIME_VALUE_MODE", "logit")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

    deterministic = os.environ.get("DIME_DETERMINISTIC", "0") in ("1", "true", "True")

    try:
        import torch
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = (not deterministic)

            if deterministic:
                torch.backends.cudnn.deterministic = True
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
    except Exception:
        pass


# ==========================================================
# GPU inventory
# ==========================================================
def get_gpu_inventory() -> List[Dict[str, Any]]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]

        details: List[Dict[str, Any]] = []
        for ln in lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 4:
                continue
            gid = int(parts[0])
            used = int(parts[1])
            total = int(parts[2])
            util = int(parts[3])
            free = int(total - used)

            details.append({
                "id": gid,
                "memory_free": free,
                "memory_used": used,
                "memory_total": total,
                "utilization": util,
            })

        details.sort(key=lambda d: (-d["memory_free"], d["utilization"]))
        return details
    except Exception as e:
        logger.warning(f"Impossibile usare nvidia-smi per inventario GPU: {e}")
        return []


def get_available_gpus_with_memory(
    min_free_memory_gb: float = 20.0,
    allowed_gpu_ids: Optional[List[int]] = None,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    min_free_mib = int(float(min_free_memory_gb) * 1024)
    allow = set(allowed_gpu_ids) if allowed_gpu_ids is not None else None

    try:
        inv = get_gpu_inventory()
        if not inv:
            return [], []

        details: List[Dict[str, Any]] = []
        for d in inv:
            gid = int(d["id"])
            if allow is not None and gid not in allow:
                continue
            if int(d["memory_free"]) >= min_free_mib:
                details.append(d)

        gpu_ids = [d["id"] for d in details]

        logger.info(f"GPU con >= {min_free_memory_gb}GB liberi: {gpu_ids}")
        for d in details:
            logger.info(
                f"  GPU {d['id']}: free={d['memory_free']} MiB / total={d['memory_total']} MiB | util={d['utilization']}%"
            )

        return gpu_ids, details

    except Exception as e:
        logger.warning(f"Impossibile rilevare memoria GPU: {e}")
        return [], []


# ==========================================================
# Task protocol
# ==========================================================
@dataclass
class Task:
    kind: str
    payload: dict


# ==========================================================
# Qwen2.5-Omni helpers
# ==========================================================
def _build_messages(audio_path: str, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _trim_generated_ids_at_first_im_end(gen_ids_1d, processor):
    ids = gen_ids_1d.detach().clone()
    tok = processor.tokenizer

    im_end_id = getattr(tok, "eos_token_id", None)
    if im_end_id is None:
        try:
            im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            im_end_id = 151645

    cut = len(ids)
    for i in range(len(ids)):
        if int(ids[i].item()) == int(im_end_id):
            cut = i
            break

    return ids[:cut]


def _generate_text_response_qwen25_omni(model, processor, audio_path: str, prompt: str) -> str:
    import torch
    from QA_analysis.utils.shared_utils import prepare_qwen25_omni_inputs

    messages = _build_messages(audio_path, prompt)

    _text, inputs = prepare_qwen25_omni_inputs(
        processor=processor,
        conversation=messages,
        device=model.device,
        dtype=getattr(model, "dtype", None),
        use_audio_in_video=False,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            return_dict_in_generate=True,
            output_scores=False,
            output_logits=False,
            use_cache=False,
        )

    sequences = outputs.sequences
    input_len = int(inputs["input_ids"].shape[1])
    gen_ids = sequences[0, input_len:]
    gen_ids = _trim_generated_ids_at_first_im_end(gen_ids, processor)

    text_out = processor.tokenizer.decode(
        gen_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return str(text_out).strip()

# ==========================================================
# Worker
# ==========================================================
def _worker_loop(
    gpu_id: int,
    model_path: str,
    task_q: mp.Queue,
    result_q: mp.Queue,
    torch_dtype: str = "bf16",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("DIME_VALUE_MODE", "logit")
    configure_runtime()
    _configure_worker_quiet_mode()

    import gc
    import torch
    from transformers import (
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    dtype = torch.bfloat16 if torch_dtype == "bf16" else torch.float16

    try:
        logger.info(f"[worker gpu={gpu_id}] Caricamento processor...")
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

        logger.info(f"[worker gpu={gpu_id}] Caricamento modello Qwen2.5-Omni...")
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        ).eval().to("cuda:0")

        logger.info(f"[worker gpu={gpu_id}] Modello caricato correttamente.")
    except Exception as e:
        result_q.put(("__worker_init_error__", {"gpu": gpu_id, "error": str(e)}))
        return

    torch.set_grad_enabled(False)

    from QA_analysis.utils.analysis_2 import dime_token_value
    from QA_analysis.utils.shared_utils import (
        prepare_qwen25_omni_inputs,
        get_token_logit_autoregressive_id_batch,
    )

    CACHE_EVERY_N_TASKS = int(os.environ.get("DIME_WORKER_EMPTYCACHE_EVERY", "0"))
    task_counter = 0

    TRUE_BATCH_SIZE = max(1, int(os.environ.get("DIME_QWEN_TRUE_BATCH_SIZE", "4")))

    def _eval_dime_items_serial(items: List[Dict[str, Any]]) -> List[float]:
        vals = []
        for item in items:
            with torch.inference_mode():
                v = dime_token_value(
                    model=model,
                    processor=processor,
                    audio_path=item["audio_path"],
                    prompt=item["prompt"],
                    caption_ids=item["caption_ids"],
                    token_index=int(item["token_index"]),
                )
            vals.append(float(v))
        return vals

    def _eval_dime_items_batched_once(items: List[Dict[str, Any]]) -> List[float]:
        """
        Un singolo forward batchato Qwen per un micro-batch di esempi.
        """
        batch_payload = []
        for item in items:
            caption_ids = item["caption_ids"]
            token_index = int(item["token_index"])
            target_id = int(caption_ids[token_index])
            prefix_ids = caption_ids[:token_index]

            batch_payload.append({
                "audio_path": item["audio_path"],
                "prompt": item["prompt"],
                "prefix_ids": prefix_ids,
                "target_id": target_id,
            })

        return get_token_logit_autoregressive_id_batch(
            model=model,
            processor=processor,
            batch_items=batch_payload,
        )

    def _eval_dime_items_adaptive(items: List[Dict[str, Any]]) -> List[float]:
        """
        Valuta gli item usando vero batching Qwen.
        Se il micro-batch va in OOM, dimezza ricorsivamente fino al fallback seriale.
        """
        if not items:
            return []

        try:
            return _eval_dime_items_batched_once(items)

        except torch.cuda.OutOfMemoryError:
            _cleanup_cuda(force=True)

            if len(items) == 1:
                return _eval_dime_items_serial(items)

            mid = len(items) // 2
            left = _eval_dime_items_adaptive(items[:mid])
            right = _eval_dime_items_adaptive(items[mid:])
            return left + right

        except RuntimeError as e:
            msg = str(e).lower()
            # alcuni errori del processor/model batching si manifestano come RuntimeError
            if "out of memory" in msg or "cuda" in msg:
                _cleanup_cuda(force=True)
                if len(items) == 1:
                    return _eval_dime_items_serial(items)
                mid = len(items) // 2
                left = _eval_dime_items_adaptive(items[:mid])
                right = _eval_dime_items_adaptive(items[mid:])
                return left + right
            raise

    def _eval_dime_items_chunked(items: List[Dict[str, Any]]) -> List[float]:
        """
        Spezza il task in micro-batch reali di dimensione TRUE_BATCH_SIZE.
        Ogni micro-batch prova un vero forward batchato.
        """
        out: List[float] = []
        for b0 in range(0, len(items), TRUE_BATCH_SIZE):
            chunk = items[b0:b0 + TRUE_BATCH_SIZE]
            out.extend(_eval_dime_items_adaptive(chunk))
        return out

    def _cleanup_cuda(force: bool = False):
        nonlocal task_counter
        task_counter += 1
        gc.collect()
        if force:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return
        if CACHE_EVERY_N_TASKS > 0 and (task_counter % CACHE_EVERY_N_TASKS == 0):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    while True:
        task = task_q.get()
        if task is None:
            break

        req_id = task.payload.get("req_id", None)

        try:
            if task.kind == "caption_chat":
                audio_path = task.payload["audio_path"]
                prompt = task.payload["prompt"]

                response = _generate_text_response_qwen25_omni(
                    model=model,
                    processor=processor,
                    audio_path=audio_path,
                    prompt=prompt,
                )

                result_q.put(("caption_chat", {
                    "req_id": req_id,
                    "batch_id": task.payload.get("batch_id", None),
                    "response": str(response),
                }))
                _cleanup_cuda(force=False)

            elif task.kind == "dime_L_batch":
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])
                audio_paths = task.payload["audio_paths"]
                prompts = task.payload["prompts"]
                ij_list = task.payload["ij_list"]
                batch_id = int(task.payload["batch_id"])
                items = []
                for (i, j) in ij_list:
                    items.append({
                        "audio_path": audio_paths[int(i)],
                        "prompt": prompts[int(j)],
                        "caption_ids": caption_ids,
                        "token_index": token_index,
                    })
                vals = _eval_dime_items_chunked(items)

                result_q.put(("dime_L_batch", {
                    "req_id": req_id,
                    "batch_id": batch_id,
                    "ij_list": ij_list,
                    "vals": vals,
                }))
                _cleanup_cuda(force=False)


            elif task.kind == "dime_row_values_batch":
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])
                audio_paths = task.payload["audio_paths"]
                prompts = task.payload["prompts"]
                batch_id = int(task.payload["batch_id"])
                items = []
                shape = []
                for ap in audio_paths:
                    row_len = 0
                    for p in prompts:
                        items.append({
                            "audio_path": ap,
                            "prompt": p,
                            "caption_ids": caption_ids,
                            "token_index": token_index,
                        })
                        row_len += 1
                    shape.append(row_len)
                flat_vals = _eval_dime_items_chunked(items)
                rows = []
                cursor = 0
                for row_len in shape:
                    rows.append([float(x) for x in flat_vals[cursor:cursor + row_len]])
                    cursor += row_len
                result_q.put(("dime_row_values_batch", {
                    "req_id": req_id,
                    "batch_id": batch_id,
                    "rows": rows,
                }))
                _cleanup_cuda(force=False)

            elif task.kind == "dime_col_values_batch":
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])
                audio_paths = task.payload["audio_paths"]
                prompts = task.payload["prompts"]
                batch_id = int(task.payload["batch_id"])
                items = []
                shape = []
                for p in prompts:
                    col_len = 0
                    for ap in audio_paths:
                        items.append({
                            "audio_path": ap,
                            "prompt": p,
                            "caption_ids": caption_ids,
                            "token_index": token_index,
                        })
                        col_len += 1
                    shape.append(col_len)
                flat_vals = _eval_dime_items_chunked(items)
                cols = []
                cursor = 0
                for col_len in shape:
                    cols.append([float(x) for x in flat_vals[cursor:cursor + col_len]])
                    cursor += col_len
                result_q.put(("dime_col_values_batch", {
                    "req_id": req_id,
                    "batch_id": batch_id,
                    "cols": cols,
                }))
                _cleanup_cuda(force=False)


            elif task.kind == "mmshap_logits":
                audio_path = task.payload["audio_path"]
                prompt = task.payload["prompt"]
                target_ids = task.payload["target_ids"]
                batch_id = task.payload.get("batch_id", None)
                messages = _build_messages(audio_path, prompt)

                _text, inputs = prepare_qwen25_omni_inputs(
                    processor=processor,
                    conversation=messages,
                    device=model.device,
                    dtype=getattr(model, "dtype", None),
                    use_audio_in_video=False,
                )
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max(1, len(target_ids) + 2),
                        return_dict_in_generate=True,
                        output_logits=True,
                        use_cache=False,
                    )

                step_logits = torch.stack([x[0] for x in outputs.logits], dim=0)
                T = min(step_logits.shape[0], len(target_ids))
                vals = []

                for t in range(T):
                    tid = int(target_ids[t])
                    if 0 <= tid < step_logits.shape[1]:
                        vals.append(float(step_logits[t, tid].item()))
                    else:
                        vals.append(float(step_logits[t].mean().item()))

                result_q.put(("mmshap_logits", {
                    "req_id": req_id,
                    "batch_id": batch_id,
                    "vals": vals,
                }))
                _cleanup_cuda(force=False)

            else:
                raise ValueError(f"Unknown task kind: {task.kind}")

        except torch.cuda.OutOfMemoryError as e:
            _cleanup_cuda(force=True)
            result_q.put(("__task_error__", {
                "req_id": req_id,
                "gpu": gpu_id,
                "kind": task.kind,
                "error": f"CUDA OOM: {str(e)}",
            }))
        except Exception as e:
            _cleanup_cuda(force=True)
            result_q.put(("__task_error__", {
                "req_id": req_id,
                "gpu": gpu_id,
                "kind": task.kind,
                "error": str(e),
            }))

# ==========================================================
# Runner
# ==========================================================
class ParallelTokenRunner:
    def __init__(self, model_path: str, gpu_ids: List[int], torch_dtype: str = "bf16"):
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.torch_dtype = torch_dtype

        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.procs: List[mp.Process] = []

        self._next_req_id = 1
        self._stash: Dict[int, List[Tuple[str, dict]]] = {}

    def run_single(self, kind: str, payload: dict):
        req_id = self._new_req_id()
        self._put_task(kind, payload, req_id=req_id)

        while True:
            k, out = self._get_for_req(req_id)
            if k == kind:
                return out["vals"]

    def get_mmshap_logits(self, audio_path: str, prompt: str, target_ids: List[int]) -> List[float]:
        req_id = self._new_req_id()

        self._put_task(
            "mmshap_logits",
            {
                "audio_path": audio_path,
                "prompt": prompt,
                "target_ids": list(target_ids),
            },
            req_id=req_id,
        )

        kind, payload = self._get_for_req(req_id)

        if kind != "mmshap_logits":
            raise RuntimeError(f"Unexpected response kind: {kind}")

        return payload["vals"]

    def get_mmshap_logits_batch(
        self,
        items: List[Dict[str, Any]],
        window: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Esegue più richieste MM-SHAP logits in parallelo sui worker/GPU.

        items: lista di dict con chiavi:
            - audio_path
            - prompt
            - target_ids

        Ritorna:
            lista ordinata di vettori logits, uno per item.
        """
        if not items:
            return []

        req_id = self._new_req_id()
        total = len(items)
        out: List[Optional[List[float]]] = [None] * total

        if window is None:
            env_window = int(os.environ.get("MMSHAP_QUEUE_WINDOW", "0"))
            if env_window > 0:
                window = env_window
            else:
                # abbastanza grande da tenere occupati tutti i worker
                window = max(len(self.procs), 1) * 8

        window = int(max(1, window))

        sent = 0
        got = 0

        # prefill finestra
        while sent < total and sent < window:
            payload = {
                "audio_path": items[sent]["audio_path"],
                "prompt": items[sent]["prompt"],
                "target_ids": list(items[sent]["target_ids"]),
                "batch_id": int(sent),
            }
            self._put_task("mmshap_logits", payload, req_id=req_id)
            sent += 1

        while got < total:
            kind, payload = self._get_for_req(req_id)

            if kind != "mmshap_logits":
                raise RuntimeError(f"Unexpected kind for mmshap batch: {kind}")

            batch_id = payload.get("batch_id", None)
            if batch_id is None:
                raise RuntimeError(f"Missing batch_id in mmshap batch payload: {payload}")

            out[int(batch_id)] = list(payload["vals"])
            got += 1

            if sent < total:
                next_payload = {
                    "audio_path": items[sent]["audio_path"],
                    "prompt": items[sent]["prompt"],
                    "target_ids": list(items[sent]["target_ids"]),
                    "batch_id": int(sent),
                }
                self._put_task("mmshap_logits", next_payload, req_id=req_id)
                sent += 1

        return [x if x is not None else [] for x in out]

    def start(self):
        for gid in self.gpu_ids:
            p = self.ctx.Process(
                target=_worker_loop,
                args=(gid, self.model_path, self.task_q, self.result_q, self.torch_dtype),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def stop(self):
        for _ in self.procs:
            self.task_q.put(None)
        for p in self.procs:
            p.join(timeout=10)

    def _new_req_id(self) -> int:
        rid = int(self._next_req_id)
        self._next_req_id += 1
        return rid

    def _put_task(self, kind: str, payload: dict, req_id: int):
        d = dict(payload)
        d["req_id"] = int(req_id)
        self.task_q.put(Task(kind, d))

    def _get_for_req(self, req_id: int) -> Tuple[str, dict]:
        req_id = int(req_id)

        q = self._stash.get(req_id, [])
        if q:
            kind, payload = q.pop(0)
            if q:
                self._stash[req_id] = q
            else:
                self._stash.pop(req_id, None)
            return kind, payload

        while True:
            kind, payload = self.result_q.get()

            if kind == "__worker_init_error__":
                raise RuntimeError(f"Worker init error: {payload}")
            if kind == "__task_error__":
                raise RuntimeError(f"Worker task error: {payload}")

            rid = payload.get("req_id", None)
            if rid is None:
                raise RuntimeError(f"Missing req_id in payload: kind={kind}, payload={payload}")

            rid = int(rid)
            if rid == req_id:
                return kind, payload

            self._stash.setdefault(rid, []).append((kind, payload))

    def generate_caption(self, audio_path: str, prompt: str) -> str:
        req_id = self._new_req_id()
        self._put_task("caption_chat", {"audio_path": audio_path, "prompt": prompt}, req_id=req_id)

        while True:
            kind, payload = self._get_for_req(req_id)
            if kind == "caption_chat":
                return str(payload["response"])

    def generate_captions_batch(
        self,
        items: List[Dict[str, str]],
        window: Optional[int] = None,
    ) -> List[str]:
        if not items:
            return []

        req_id = self._new_req_id()
        total = len(items)
        out: List[Optional[str]] = [None] * total

        if window is None:
            env_window = int(os.environ.get("PROMPT_QWEN_BATCH_WINDOW", "0"))
            if env_window > 0:
                window = env_window
            else:
                window = max(len(self.procs), 1) * 4
        window = int(max(1, window))

        sent = 0
        got = 0

        while sent < total and sent < window:
            payload = {
                "audio_path": items[sent]["audio_path"],
                "prompt": items[sent]["prompt"],
                "batch_id": int(sent),
            }
            self._put_task("caption_chat", payload, req_id=req_id)
            sent += 1

        while got < total:
            kind, payload = self._get_for_req(req_id)
            if kind != "caption_chat":
                raise RuntimeError(f"Unexpected kind for caption batch: {kind}")

            batch_id = payload.get("batch_id", None)
            if batch_id is None:
                raise RuntimeError(f"Missing batch_id in caption batch payload: {payload}")

            out[int(batch_id)] = str(payload["response"])
            got += 1

            if sent < total:
                next_payload = {
                    "audio_path": items[sent]["audio_path"],
                    "prompt": items[sent]["prompt"],
                    "batch_id": int(sent),
                }
                self._put_task("caption_chat", next_payload, req_id=req_id)
                sent += 1

        return [x if x is not None else "" for x in out]

    def run_dime_L_table(
        self,
        bg_audio_paths: List[str],
        bg_prompts: List[str],
        caption_ids: List[int],
        token_index: int,
        batch_size: int = 1,
    ) -> List[List[float]]:
        N = min(len(bg_audio_paths), len(bg_prompts))
        if N <= 0:
            return []

        req_id = self._new_req_id()
        ij_all = [(i, j) for i in range(N) for j in range(N)]

        chunks = []
        bs = max(1, int(batch_size))
        for b0 in range(0, len(ij_all), bs):
            chunks.append(ij_all[b0:b0 + bs])

        for batch_id, ij_list in enumerate(chunks):
            self._put_task(
                "dime_L_batch",
                {
                    "caption_ids": list(caption_ids),
                    "token_index": int(token_index),
                    "audio_paths": list(bg_audio_paths[:N]),
                    "prompts": list(bg_prompts[:N]),
                    "ij_list": ij_list,
                    "batch_id": int(batch_id),
                },
                req_id=req_id,
            )

        L = [[0.0 for _ in range(N)] for _ in range(N)]
        received = 0

        while received < len(chunks):
            kind, payload = self._get_for_req(req_id)
            if kind != "dime_L_batch":
                raise RuntimeError(f"Unexpected kind for run_dime_L_table: {kind}")

            ij_list = payload["ij_list"]
            vals = payload["vals"]
            for (i, j), v in zip(ij_list, vals):
                L[int(i)][int(j)] = float(v)
            received += 1

        return L

    def run_dime_row_values(
        self,
        audio_paths: List[str],
        prompts: List[str],
        caption_ids: List[int],
        token_index: int,
        batch_size: int = 1,
    ) -> List[List[float]]:
        if not audio_paths:
            return []

        req_id = self._new_req_id()
        bs = max(1, int(batch_size))
        chunks = [audio_paths[b0:b0 + bs] for b0 in range(0, len(audio_paths), bs)]

        for batch_id, chunk in enumerate(chunks):
            self._put_task(
                "dime_row_values_batch",
                {
                    "caption_ids": list(caption_ids),
                    "token_index": int(token_index),
                    "audio_paths": list(chunk),
                    "prompts": list(prompts),
                    "batch_id": int(batch_id),
                },
                req_id=req_id,
            )

        out = [None] * len(chunks)
        received = 0

        while received < len(chunks):
            kind, payload = self._get_for_req(req_id)
            if kind != "dime_row_values_batch":
                raise RuntimeError(f"Unexpected kind for run_dime_row_values: {kind}")
            out[int(payload["batch_id"])] = payload["rows"]
            received += 1

        rows = []
        for chunk_rows in out:
            rows.extend(chunk_rows)
        return rows

    def run_dime_col_values(
        self,
        audio_paths: List[str],
        prompts: List[str],
        caption_ids: List[int],
        token_index: int,
        batch_size: int = 1,
    ) -> List[List[float]]:
        if not prompts:
            return []

        req_id = self._new_req_id()
        bs = max(1, int(batch_size))
        chunks = [prompts[b0:b0 + bs] for b0 in range(0, len(prompts), bs)]

        for batch_id, chunk in enumerate(chunks):
            self._put_task(
                "dime_col_values_batch",
                {
                    "caption_ids": list(caption_ids),
                    "token_index": int(token_index),
                    "audio_paths": list(audio_paths),
                    "prompts": list(chunk),
                    "batch_id": int(batch_id),
                },
                req_id=req_id,
            )

        out = [None] * len(chunks)
        received = 0

        while received < len(chunks):
            kind, payload = self._get_for_req(req_id)
            if kind != "dime_col_values_batch":
                raise RuntimeError(f"Unexpected kind for run_dime_col_values: {kind}")
            out[int(payload["batch_id"])] = payload["cols"]
            received += 1

        cols = []
        for chunk_cols in out:
            cols.extend(chunk_cols)
        return cols


# ==========================================================
# factory
# ==========================================================
def try_create_parallel_runner(
    model_path: str,
    max_gpus: int = 4,
    min_free_memory_gb: float = 24.0,
    allowed_gpu_ids: Optional[List[int]] = None,
    gpu_ids_physical: Optional[List[int]] = None,
):
    os.environ.setdefault("DIME_VALUE_MODE", "logit")

    if gpu_ids_physical is not None:
        gpu_ids = list(gpu_ids_physical)
    else:
        gpu_ids, _ = get_available_gpus_with_memory(
            min_free_memory_gb=min_free_memory_gb,
            allowed_gpu_ids=allowed_gpu_ids,
        )
        if max_gpus is not None and max_gpus > 0:
            gpu_ids = gpu_ids[:max_gpus]

    if not gpu_ids:
        logger.warning("Runner multiprocess non attivato (memoria libera insufficiente).")
        return None

    if max_gpus is not None and max_gpus > 0:
        gpu_ids = gpu_ids[: int(max_gpus)]

    runner = ParallelTokenRunner(model_path=model_path, gpu_ids=gpu_ids, torch_dtype="bf16")
    runner.start()

    time.sleep(1.0)
    for p in runner.procs:
        if not p.is_alive():
            logger.error("Worker morto in init: disabilito runner.")
            try:
                runner.stop()
            except Exception:
                pass
            return None

    logger.info(f"ParallelTokenRunner attivo su GPU fisiche: {gpu_ids}")
    return runner