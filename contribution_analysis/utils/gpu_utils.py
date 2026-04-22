import os
import time
import logging
import subprocess
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GPU Utils")


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
# Worker
# ==========================================================
def _worker_loop(
    gpu_id: int,
    model_path: str,
    task_q: mp.Queue,
    result_q: mp.Queue,
    torch_dtype: str = "bf16",
):
    # Ogni worker vede UNA sola GPU (isolamento totale)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("DIME_VALUE_MODE", "logit")
    configure_runtime()

    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch_dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        ).eval().to("cuda:0")
    except Exception as e:
        result_q.put(("__worker_init_error__", {"gpu": gpu_id, "error": str(e)}))
        return

    torch.set_grad_enabled(False)

    # Worker-side imports
    from contribution_analysis.utils.analysis_1 import compute_mmshap_for_token
    from contribution_analysis.utils.analysis_2 import dime_token_value

    # cleanup policy
    CACHE_EVERY_N_TASKS = int(os.environ.get("DIME_WORKER_EMPTYCACHE_EVERY", "0"))  # 0 = mai
    task_counter = 0

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
        if task is not None:
            rid = task.payload.get("req_id", None)
            bid = task.payload.get("batch_id", None)
        if task is None:
            break

        # payload MUST contain req_id for routing
        req_id = task.payload.get("req_id", None)

        try:
            if task.kind == "caption_chat":
                audio_path = task.payload["audio_path"]
                prompt = task.payload["prompt"]

                query = tokenizer.from_list_format([
                    {"audio": audio_path},
                    {"text": prompt},
                ])
                audio_info = tokenizer.process_audio(query)

                with torch.inference_mode():
                    response, _ = model.chat(tokenizer, query=query, audio_info=audio_info, history=None)

                result_q.put(("caption_chat", {
                    "req_id": req_id,
                    "batch_id": task.payload.get("batch_id", None),
                    "response": str(response),
                }))
                _cleanup_cuda(force=False)

            elif task.kind == "mmshap_token":
                with torch.inference_mode():
                    a_shap, t_shap = compute_mmshap_for_token(
                        model=model,
                        tokenizer=tokenizer,
                        audio_path=task.payload["audio_path"],
                        prompt=task.payload["prompt"],
                        caption_ids=task.payload["caption_ids"],
                        token_index=int(task.payload["token_index"]),
                        num_permutations=int(task.payload.get("num_permutations", 10)),
                    )
                result_q.put(("mmshap_token", {
                    "req_id": req_id,
                    "token_index": int(task.payload["token_index"]),
                    "a_shap": float(a_shap),
                    "t_shap": float(t_shap),
                }))
                _cleanup_cuda(force=False)

            elif task.kind == "dime_L_batch":
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])

                audio_paths = task.payload["audio_paths"]
                prompts = task.payload["prompts"]
                ij_list = task.payload["ij_list"]
                batch_id = int(task.payload["batch_id"])

                vals = []
                for (i, j) in ij_list:
                    ai = audio_paths[int(i)]
                    tj = prompts[int(j)]
                    with torch.inference_mode():
                        v = dime_token_value(
                            model=model,
                            tokenizer=tokenizer,
                            audio_path=ai,
                            prompt=tj,
                            caption_ids=caption_ids,
                            token_index=token_index,
                        )
                    vals.append(float(v))

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
                audio_paths = task.payload["audio_paths"]   # batch of perturbed audios
                prompts = task.payload["prompts"]           # full bg_prompts
                batch_id = int(task.payload["batch_id"])

                rows = []
                for ap in audio_paths:
                    row = []
                    for p in prompts:
                        with torch.inference_mode():
                            v = dime_token_value(
                                model=model,
                                tokenizer=tokenizer,
                                audio_path=ap,
                                prompt=p,
                                caption_ids=caption_ids,
                                token_index=token_index,
                            )
                        row.append(float(v))
                    rows.append(row)

                result_q.put(("dime_row_values_batch", {
                    "req_id": req_id,
                    "batch_id": batch_id,
                    "rows": rows,
                }))
                _cleanup_cuda(force=False)

            elif task.kind == "dime_col_values_batch":
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])
                audio_paths = task.payload["audio_paths"]   # full bg_audio_paths
                prompts = task.payload["prompts"]           # batch of perturbed prompts
                batch_id = int(task.payload["batch_id"])

                cols = []
                for p in prompts:
                    col = []
                    for ap in audio_paths:
                        with torch.inference_mode():
                            v = dime_token_value(
                                model=model,
                                tokenizer=tokenizer,
                                audio_path=ap,
                                prompt=p,
                                caption_ids=caption_ids,
                                token_index=token_index,
                            )
                        col.append(float(v))
                    cols.append(col)

                result_q.put(("dime_col_values_batch", {
                    "req_id": req_id,
                    "batch_id": batch_id,
                    "cols": cols,
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

        # routing state
        self._next_req_id = 1
        self._stash: Dict[int, List[Tuple[str, dict]]] = {}  # req_id -> [(kind,payload), ...]

    # ------------------------
    # lifecycle
    # ------------------------
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

    # ------------------------
    # routing helpers
    # ------------------------
    def _new_req_id(self) -> int:
        rid = int(self._next_req_id)
        self._next_req_id += 1
        return rid

    def _put_task(self, kind: str, payload: dict, req_id: int):
        d = dict(payload)
        d["req_id"] = int(req_id)
        self.task_q.put(Task(kind, d))

    def _get_for_req(self, req_id: int) -> Tuple[str, dict]:
        """
        Returns next (kind,payload) for given req_id.
        Stashes messages for other req_id without losing them.
        """
        req_id = int(req_id)

        # if already stashed
        q = self._stash.get(req_id, [])
        if q:
            kind, payload = q.pop(0)
            if q:
                self._stash[req_id] = q
            else:
                self._stash.pop(req_id, None)
            return kind, payload

        # otherwise pull from result_q until match
        while True:
            kind, payload = self.result_q.get()

            # init / task errors may miss req_id; handle anyway
            if kind == "__worker_init_error__":
                raise RuntimeError(f"Worker init error: {payload}")
            if kind == "__task_error__":
                # may carry req_id; if not, still raise
                raise RuntimeError(f"Worker task error: {payload}")

            rid = payload.get("req_id", None)
            if rid is None:
                # unexpected, but don't drop it: treat as fatal
                raise RuntimeError(f"Missing req_id in payload: kind={kind}, payload={payload}")

            rid = int(rid)
            if rid == req_id:
                return kind, payload

            # stash for other request
            self._stash.setdefault(rid, []).append((kind, payload))

    # ------------------------
    # API
    # ------------------------
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
        """
        Esegue più task `caption_chat` indipendenti in parallelo sui worker/GPU.
        L'ordine di output coincide con l'ordine di input.
        Ogni item deve contenere:
          - audio_path
          - prompt
        """
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

    def run_mmshap_tokens(
        self,
        audio_path: str,
        prompt: str,
        caption_ids: List[int],
        num_permutations: int = 10
    ) -> Dict[int, Dict[str, float]]:
        req_id = self._new_req_id()
        n = len(caption_ids)

        for i in range(n):
            self._put_task("mmshap_token", {
                "audio_path": audio_path,
                "prompt": prompt,
                "caption_ids": caption_ids,
                "token_index": i,
                "num_permutations": num_permutations,
            }, req_id=req_id)

        out: Dict[int, Dict[str, float]] = {}
        got = 0
        while got < n:
            kind, payload = self._get_for_req(req_id)
            if kind == "mmshap_token":
                out[int(payload["token_index"])] = {
                    "a_shap": float(payload["a_shap"]),
                    "t_shap": float(payload["t_shap"]),
                }
                got += 1
        return out

    # ==========================================================
    # Windowed scheduler (core for stable multi-GPU utilization)
    # ==========================================================
    def _run_windowed_batches(
        self,
        req_id: int,
        task_kind_send: str,
        task_kind_recv: str,
        make_payload_fn,
        total_batches: int,
        window: int,
    ) -> List[dict]:
        """
        Send at most `window` batches at a time. When one batch returns, send next.
        Returns list of payloads (unordered). You reconstruct via batch_id.
        """
        total_batches = int(total_batches)
        window = int(max(1, window))
        sent = 0
        got = 0
        results: List[dict] = []

        # prime the pipeline
        while sent < total_batches and sent < window:
            self._put_task(task_kind_send, make_payload_fn(sent), req_id=req_id)
            sent += 1

        # drain + refill
        while got < total_batches:
            kind, payload = self._get_for_req(req_id)
            if kind == task_kind_recv:
                results.append(payload)
                got += 1
                if sent < total_batches:
                    self._put_task(task_kind_send, make_payload_fn(sent), req_id=req_id)
                    sent += 1
            else:
                # should not happen for this req_id, but be defensive
                raise RuntimeError(f"Unexpected kind for req_id={req_id}: {kind}")

        return results

    # ==========================================================
    # STEP 2: L-table (now windowed too)
    # ==========================================================
    def run_dime_L_table(
        self,
        bg_audio_paths: List[str],
        bg_prompts: List[str],
        caption_ids: List[int],
        token_index: int,
        batch_size: int = 1,
    ) -> List[List[float]]:
        req_id = self._new_req_id()

        audio_paths = list(bg_audio_paths)
        prompts = list(bg_prompts)
        N = min(len(audio_paths), len(prompts))
        if N <= 0:
            raise RuntimeError("Empty background sets for L table.")
        audio_paths = audio_paths[:N]
        prompts = prompts[:N]

        all_pairs: List[Tuple[int, int]] = [(i, j) for i in range(N) for j in range(N)]
        bs = int(max(1, batch_size))
        batches: List[List[Tuple[int, int]]] = [all_pairs[p0:p0 + bs] for p0 in range(0, len(all_pairs), bs)]
        total_batches = len(batches)

        window = int(os.environ.get("DIME_QUEUE_WINDOW", "16"))

        def make_payload(bid: int) -> dict:
            return {
                "batch_id": int(bid),
                "audio_paths": audio_paths,
                "prompts": prompts,
                "caption_ids": caption_ids,
                "token_index": int(token_index),
                "ij_list": batches[bid],
            }

        payloads = self._run_windowed_batches(
            req_id=req_id,
            task_kind_send="dime_L_batch",
            task_kind_recv="dime_L_batch",
            make_payload_fn=make_payload,
            total_batches=total_batches,
            window=window,
        )

        L = [[0.0 for _ in range(N)] for __ in range(N)]
        for pl in payloads:
            ij_list = pl["ij_list"]
            vals = pl["vals"]
            if len(ij_list) != len(vals):
                raise RuntimeError("Bad L-batch payload sizes.")
            for (i, j), v in zip(ij_list, vals):
                L[int(i)][int(j)] = float(v)
        return L

    # ==========================================================
    # STEP 4A: row updates (windowed pipeline)
    # ==========================================================
    def run_dime_row_values(
        self,
        audio_paths: List[str],       # perturbed audio paths length S
        prompts: List[str],           # bg_prompts length N
        caption_ids: List[int],
        token_index: int,
        batch_size: int = 1,
    ) -> List[List[float]]:
        req_id = self._new_req_id()

        S = len(audio_paths)
        if S == 0:
            return []

        bs = int(max(1, batch_size))
        offsets: List[Tuple[int, int]] = [(s0, min(S, s0 + bs)) for s0 in range(0, S, bs)]
        total_batches = len(offsets)

        window = int(os.environ.get("DIME_QUEUE_WINDOW", "16"))

        def make_payload(bid: int) -> dict:
            s0, s1 = offsets[bid]
            return {
                "batch_id": int(bid),
                "audio_paths": audio_paths[s0:s1],
                "prompts": prompts,
                "caption_ids": caption_ids,
                "token_index": int(token_index),
            }

        payloads = self._run_windowed_batches(
            req_id=req_id,
            task_kind_send="dime_row_values_batch",
            task_kind_recv="dime_row_values_batch",
            make_payload_fn=make_payload,
            total_batches=total_batches,
            window=window,
        )

        out_rows: List[Optional[List[float]]] = [None] * S
        for pl in payloads:
            bid = int(pl["batch_id"])
            rows = pl["rows"]
            s0, s1 = offsets[bid]
            if len(rows) != (s1 - s0):
                raise RuntimeError("Bad row batch size.")
            for i, r in enumerate(rows):
                out_rows[s0 + i] = [float(x) for x in r]

        return [r for r in out_rows if r is not None]

    # ==========================================================
    # STEP 4B: col updates (windowed pipeline)
    # ==========================================================
    def run_dime_col_values(
        self,
        audio_paths: List[str],       # bg_audio_paths length N
        prompts: List[str],           # perturbed prompts length S
        caption_ids: List[int],
        token_index: int,
        batch_size: int = 1,
    ) -> List[List[float]]:
        req_id = self._new_req_id()

        S = len(prompts)
        if S == 0:
            return []

        bs = int(max(1, batch_size))
        offsets: List[Tuple[int, int]] = [(s0, min(S, s0 + bs)) for s0 in range(0, S, bs)]
        total_batches = len(offsets)

        window = int(os.environ.get("DIME_QUEUE_WINDOW", "16"))

        def make_payload(bid: int) -> dict:
            s0, s1 = offsets[bid]
            return {
                "batch_id": int(bid),
                "audio_paths": audio_paths,
                "prompts": prompts[s0:s1],
                "caption_ids": caption_ids,
                "token_index": int(token_index),
            }

        payloads = self._run_windowed_batches(
            req_id=req_id,
            task_kind_send="dime_col_values_batch",
            task_kind_recv="dime_col_values_batch",
            make_payload_fn=make_payload,
            total_batches=total_batches,
            window=window,
        )

        out_cols: List[Optional[List[float]]] = [None] * S
        for pl in payloads:
            bid = int(pl["batch_id"])
            cols = pl["cols"]
            s0, s1 = offsets[bid]
            if len(cols) != (s1 - s0):
                raise RuntimeError("Bad col batch size.")
            for i, c in enumerate(cols):
                out_cols[s0 + i] = [float(x) for x in c]

        return [c for c in out_cols if c is not None]


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

    # hard cap safety
    if max_gpus is not None and max_gpus > 0:
        gpu_ids = gpu_ids[: int(max_gpus)]

    runner = ParallelTokenRunner(model_path=model_path, gpu_ids=gpu_ids, torch_dtype="bf16")
    runner.start()

    time.sleep(0.5)
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