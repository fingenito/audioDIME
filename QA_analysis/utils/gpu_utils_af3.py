"""
gpu_utils_af3.py — equivalente di McqaParallelRunner/_mcqa_worker_loop
(definiti in paper/Faithfulness_correct/batch_exp_e.py) ma per Audio
Flamingo 3 invece di Qwen2.5-Omni.

Riusa da utils/gpu_utils.py tutto cio' che e' generico (non specifico di
un modello): Task, configure_runtime, _configure_worker_quiet_mode,
_materialize_worker_audio_input, _close_worker_audio_handle.

NOTA: _materialize_worker_audio_input converte descrittori shared-memory
/inline in un dict {"array":..., "sampling_rate":...} pensato per la
pipeline di Qwen. _build_af3_conversation gestisce entrambi i casi: se
riceve un path (str) lo passa cosi' com'e' al processor AF3; se riceve il
dict {"array",...} (MM-SHAP e DIME, che perturbano l'audio in RAM e non
passano mai per un file su disco) estrae l'ndarray nudo, che e' l'unico
altro formato accettato da load_audio() oltre alla stringa.
"""

import multiprocessing as _mp
from typing import Any, Dict, List, Optional, Tuple

from QA_analysis.utils.gpu_utils import (
    Task,
    configure_runtime,
    _configure_worker_quiet_mode,
    _materialize_worker_audio_input,
    _close_worker_audio_handle,
    ParallelTokenRunner,
)

AF3_MODEL_PATH = "/nas/home/fingenito/Models/audio-flamingo-3-hf"


def _build_af3_conversation(audio_path: Any, prompt: str) -> list:
    audio_value = audio_path
    if isinstance(audio_value, dict) and "array" in audio_value:
        # Audio materializzato da _materialize_worker_audio_input (shared-memory
        # o inline -- es. MM-SHAP, che perturba l'audio in RAM e non passa mai
        # per un file su disco). Il processor AF3 (load_audio) accetta solo un
        # numpy array nudo o una str (path/url/base64), non questo dict.
        # sampling_rate non viene riletta qui: load_audio non ricampiona un
        # ndarray, si fida che sia gia' alla rate del modello -- ed e' la
        # stessa rate (16kHz) gia' usata per l'encoder Whisper-based di Qwen.
        import numpy as np
        audio_value = np.asarray(audio_value["array"], dtype=np.float32).reshape(-1)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": audio_value},
            ],
        }
    ]


def _resolve_audio_array_af3(ap: Any, target_sr: int) -> Any:
    """
    Converte l'audio materializzato da _materialize_worker_audio_input in un
    array numpy nudo alla sampling rate del modello (resample se necessario).

    ap puo' essere: un path su disco (str) -- caption/DIME su audio reale --
    oppure un dict {"array","sampling_rate"} da shared-memory/inline, come
    l'audio perturbato da audioLIME per le righe/colonne della L-table.
    """
    import numpy as np
    if isinstance(ap, dict) and "array" in ap:
        audio_arr = np.asarray(ap["array"], dtype=np.float32).reshape(-1)
        arr_sr = int(ap.get("sampling_rate", target_sr))
        if arr_sr != target_sr:
            import librosa as _librosa
            audio_arr = np.asarray(
                _librosa.resample(audio_arr, orig_sr=arr_sr, target_sr=target_sr),
                dtype=np.float32,
            )
        return audio_arr
    import librosa as _librosa
    audio_arr, _ = _librosa.load(str(ap), sr=target_sr, mono=True)
    return np.asarray(audio_arr, dtype=np.float32)


def _af3_configure_worker_logger(name: str):
    """
    _configure_worker_quiet_mode() (condivisa con Qwen, in gpu_utils.py) alza
    il livello di logging globale a WARNING e riabilita esplicitamente solo
    il logger "GPU Utils" a INFO. I worker AF3 usano nomi di logger diversi
    ("Af3Worker"/"Af3TokenWorker") per distinguerli nei log da quelli di
    Qwen: senza questa funzione restano silenziati a WARNING e i loro
    messaggi INFO (incluso quale attn_implementation e' stata caricata)
    non si vedono mai. Replica qui lo stesso setup fatto per "GPU Utils".
    """
    import sys
    import logging
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.INFO)
    logger_obj.propagate = False
    if not logger_obj.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger_obj.addHandler(h)
    return logger_obj


def _af3_worker_loop(
    gpu_id: int,
    model_path: str,
    task_q,
    result_q,
    torch_dtype: str = "bf16",
) -> None:
    """
    Worker loop per AF3. Gestisce un solo task kind: "mcqa_logits".

    Payload atteso (identico a _mcqa_worker_loop di Qwen):
        audio_path  : str | shared_memory_audio | inline_audio
        prompt      : str
        target_ids  : List[int]  token id da leggere dai logit
        batch_id    : int | None
        req_id      : int

    Risposta:
        ("mcqa_logits", {"req_id": int, "batch_id": int|None, "vals": List[float]})

    Un solo forward pass (model(**inputs)), nessun generate() — stesso
    principio del worker Qwen, per correttezza e velocita'.
    """
    import os
    # CUDA_VISIBLE_DEVICES va impostata PRIMA di importare torch, altrimenti
    # il runtime CUDA ha gia' enumerato le GPU e il processo le vede tutte
    # (bug reale osservato 2026-08-25: 8 worker finiti tutti sulla stessa
    # GPU fisica invece che uno ciascuno).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import gc
    import torch

    configure_runtime()
    _configure_worker_quiet_mode()

    _wlog = _af3_configure_worker_logger("Af3Worker")

    dtype = torch.bfloat16 if torch_dtype == "bf16" else torch.float16

    try:
        _wlog.info(f"[af3_worker gpu={gpu_id}] Caricamento modello...")
        processor, model, attn_impl_used = _load_af3_model(model_path, dtype)
        _wlog.info(f"[af3_worker gpu={gpu_id}] Pronto (attn_implementation={attn_impl_used}).")
    except Exception as e:
        result_q.put(("__af3_worker_init_error__", {"gpu": gpu_id, "error": str(e)}))
        return

    torch.set_grad_enabled(False)

    while True:
        task = task_q.get()
        if task is None:
            break

        req_id = task.payload.get("req_id", None)

        try:
            if task.kind != "mcqa_logits":
                raise ValueError(f"Af3Worker: task kind sconosciuto '{task.kind}'")

            audio_path_raw = task.payload["audio_path"]
            prompt         = task.payload["prompt"]
            target_ids     = task.payload["target_ids"]
            batch_id       = task.payload.get("batch_id", None)

            audio_path, audio_handle = _materialize_worker_audio_input(audio_path_raw)
            try:
                conversation = _build_af3_conversation(audio_path, prompt)
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                ).to(model.device)
                for k in inputs:
                    if hasattr(inputs[k], "dtype") and inputs[k].dtype.is_floating_point:
                        inputs[k] = inputs[k].to(model.dtype)

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
            result_q.put(("__af3_task_error__", {
                "req_id": req_id, "gpu": gpu_id,
                "error": f"CUDA OOM: {e}",
            }))
        except Exception as e:
            import traceback as _tb
            result_q.put(("__af3_task_error__", {
                "req_id": req_id, "gpu": gpu_id,
                "error": f"{e}\n{_tb.format_exc()[:1000]}",
            }))


class Af3ParallelRunner:
    """
    Runner multiprocesso per MCQA logit-probing con Audio Flamingo 3.
    Stessa logica di IPC di McqaParallelRunner (Qwen): Queue, req_id,
    stash, finestra di accodamento. Cambia solo il worker loop.
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
        import os
        for gid in self.gpu_ids:
            # Fix isolamento GPU (2026-08-25): CUDA_VISIBLE_DEVICES va
            # impostata nel PADRE subito prima di ogni Process.start(), cosi'
            # viene ereditata dal figlio alla nascita del processo -- prima
            # che qualunque import (incluso transformers, che in questo
            # ambiente tocca CUDA durante l'import) possa vedere le GPU
            # sbagliate. Vedi paper_af3/isolated_gpu_test.py per la
            # bisezione che ha isolato la causa.
            _old = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
            try:
                p = self._ctx.Process(
                    target=_af3_worker_loop,
                    args=(gid, self.model_path, self._task_q, self._result_q,
                          self.torch_dtype),
                    daemon=True,
                )
                p.start()
            finally:
                if _old is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = _old
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
        """Singolo forward pass -> [logit_A, logit_B, logit_C, logit_D]."""
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
        Batch di N forward pass -> N vettori [logit_A, logit_B, logit_C, logit_D].
        Finestra IPC per tenere i worker sempre occupati.
        """
        if not items:
            return []
        if window is None:
            import os as _os
            env_w = int(_os.environ.get("MMSHAP_QUEUE_WINDOW", "0"))
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
            if kind == "__af3_worker_init_error__":
                raise RuntimeError(f"Af3Worker init error: {payload}")
            if kind == "__af3_task_error__":
                raise RuntimeError(f"Af3Worker task error: {payload}")
            rid = int(payload.get("req_id", -1))
            if rid == req_id:
                return kind, payload
            self._stash.setdefault(rid, []).append((kind, payload))


def create_af3_runner(model_path: str, gpu_ids: List[int]) -> Af3ParallelRunner:
    """Crea, avvia e verifica Af3ParallelRunner."""
    import time as _time
    runner = Af3ParallelRunner(model_path=model_path, gpu_ids=gpu_ids)
    runner.start()
    _time.sleep(1.5)
    if not runner.is_alive():
        try: runner.stop()
        except Exception: pass
        raise RuntimeError("Af3ParallelRunner: worker morti in avvio.")
    return runner


# =============================================================================
# Af3TokenRunner — equivalente di ParallelTokenRunner (usato da Exp A per
# MM-SHAP e DIME: L-table, row/col values, caption), non solo MCQA.
#
# NOTA SULLE PRESTAZIONI (aggiornato 2026-08-27): dime_row_values_batch,
# dime_col_values_batch E dime_L_batch (Step 2, L-table) usano tutti lo
# stesso meccanismo di cache PKV (past-key-value) di Qwen -- per ogni audio
# fisso, un solo forward pass sul prefisso condiviso tra i prompt (con
# use_cache=True), poi un forward economico per ciascun suffisso divergente
# riusando quel KV cache, invece di rifare da zero (incluso il ri-encoding
# audio) per ogni combinazione (audio, prompt). Per dime_L_batch le coppie
# (i,j) ricevute in un task vengono raggruppate per i (stesso audio) prima
# di applicare la cache. Vedi _batch_dime_logit_same_audio_with_prefix_kv_inner_af3.
# Controllato da DIME_TEXT_PKV_CACHE (default "1"); con "0" si torna al
# fallback originale, un forward pass indipendente per ogni combinazione.
# =============================================================================

def dime_token_value_af3(
    model,
    processor,
    audio_path: Any,
    prompt: str,
    caption_ids: list,
    token_index: int,
) -> float:
    """
    Equivalente di dime_token_value (analysis_2.py) + get_token_logit_autoregressive_id
    (shared_utils.py) per AF3: stesso schema "appendi il prefisso gia' generato
    come testo al prompt, un solo forward pass, leggi il logit del target al
    last position" -- solo con l'input-prep di AF3 (apply_chat_template)
    invece che di Qwen.
    """
    import torch

    target_id = int(caption_ids[token_index])
    prefix_ids = caption_ids[:token_index]

    tok = processor.tokenizer
    if prefix_ids:
        prefix_tokens = tok.convert_ids_to_tokens(prefix_ids)
        prefix_text = tok.convert_tokens_to_string(prefix_tokens)
        full_prompt = prompt + " " + prefix_text if prefix_text else prompt
    else:
        full_prompt = prompt

    conversation = _build_af3_conversation(audio_path, full_prompt)
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
    ).to(model.device)
    for k in inputs:
        if hasattr(inputs[k], "dtype") and inputs[k].dtype.is_floating_point:
            inputs[k] = inputs[k].to(model.dtype)

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)

    logits = outputs.logits[0, -1]
    vocab_size = int(logits.size(-1))
    if not isinstance(target_id, int) or target_id < 0 or target_id >= vocab_size:
        return float(logits.mean().item())
    return float(logits[target_id].item())


def _generate_text_response_af3(model, processor, audio_path: Any, prompt: str) -> str:
    """
    Genera la risposta MCQA di AF3 come lettera singola (A/B/C/D), non testo
    libero -- allinea il formato di risposta a quello naturale di Qwen (che
    risponde gia' con la lettera nuda), cosi' MM-SHAP e DIME lavorano sullo
    stesso identico numero di "token di risposta" (1) per entrambi i modelli,
    invece che 1 per Qwen e diversi per AF3 (es. "(A) RIFF" -> piu' token).
    Questo e' il moltiplicatore dominante del costo di DIME (che ripete
    Step2/4/5 una volta per token di caption), non solo un dettaglio di
    MM-SHAP.

    Un solo forward pass (niente generate() multi-step): si legge il logit
    dei 4 token lettera (get_letter_token_ids_af3 -- gia' validato: AF3
    risponde sistematicamente col pattern "(X" come primo token) e si prende
    quello con probabilita' piu' alta. Non e' una scelta diversa da quella
    che il modello farebbe comunque come primo token in generazione libera:
    ci fermiamo li' invece di lasciarlo elaborare oltre.
    """
    import torch
    from QA_analysis.utils.shared_utils_af3 import get_letter_token_ids_af3

    conversation = _build_af3_conversation(audio_path, prompt)
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
    ).to(model.device)
    for k in inputs:
        if hasattr(inputs[k], "dtype") and inputs[k].dtype.is_floating_point:
            inputs[k] = inputs[k].to(model.dtype)

    letters = ("A", "B", "C", "D")
    letter_ids = get_letter_token_ids_af3(processor.tokenizer, letters=letters)

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False, return_dict=True)

    logits_last = outputs.logits[0, -1]
    letter_logits = [float(logits_last[tid].item()) for tid in letter_ids]
    best_idx = max(range(len(letter_logits)), key=lambda i: letter_logits[i])
    return letters[best_idx]


def _build_prefix_forward_inputs_af3(full_input: Dict[str, Any], prefix_text_len: int) -> Dict[str, Any]:
    """
    Equivalente di _build_prefix_forward_inputs (Qwen, gpu_utils.py) ma con
    le chiavi audio di AF3: input_features/input_features_mask (Qwen usa
    invece feature_attention_mask). Il resto -- slicing di input_ids/
    attention_mask sul prefisso comune, inclusione dei tensori audio nel
    forward del prefisso -- e' identico.
    """
    out: Dict[str, Any] = {
        "input_ids": full_input["input_ids"][:, :prefix_text_len],
    }
    if "attention_mask" in full_input:
        out["attention_mask"] = full_input["attention_mask"][:, :prefix_text_len]
    for k in ("input_features", "input_features_mask"):
        if k in full_input:
            out[k] = full_input[k]
    return out


def _prepare_per_prompt_inputs_same_audio_af3(
    model,
    processor,
    audio_array: "Any",
    full_prompts: List[str],
) -> List[Dict[str, Any]]:
    """
    Equivalente di _prepare_per_prompt_inputs_same_audio (Qwen) per AF3:
    stesso audio (array numpy nudo, gia' alla sample rate del modello),
    prompt diversi -- un apply_chat_template per prompt, tensori tenuti su
    CPU (lo spostamento su device/dtype lo fa il chiamante al momento del
    forward, come per Qwen).
    """
    per_item_inputs: List[Dict[str, Any]] = []
    for p in full_prompts:
        conversation = _build_af3_conversation(audio_array, p)
        inp = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
        )
        cloned = {}
        for k, v in inp.items():
            if hasattr(v, "detach"):
                cloned[k] = v.detach().clone()
            else:
                cloned[k] = v
        per_item_inputs.append(cloned)
    return per_item_inputs


def _batch_dime_logit_same_audio_with_prefix_kv_inner_af3(
    model,
    processor,
    audio_array: "Any",
    full_prompts: List[str],
    target_id: int,
) -> List[float]:
    """
    Equivalente AF3 di _batch_dime_logit_same_audio_with_prefix_kv_inner
    (Qwen, gpu_utils.py): stesso audio fisso, N prompt divergenti -- un solo
    forward pass sul prefisso comune (audio + testo condiviso, use_cache=True),
    poi un forward pass economico per ciascun suffisso divergente riusando il
    KV cache del prefisso invece di ricalcolare tutto da zero (incluso il
    ri-encoding audio, che altrimenti si ripete per ogni singola valutazione).

    Le funzioni di supporto generiche (LCP sugli input_ids, spostamento
    device/dtype, lettura/clonazione del past_key_values, forward del
    suffisso) sono quelle di Qwen in gpu_utils.py: operano solo su tensori,
    senza alcun riferimento a nomi di chiave specifici di Qwen, quindi sono
    direttamente riusabili. Solo _build_prefix_forward_inputs va rifatto
    per AF3 (chiavi audio diverse) -- vedi _build_prefix_forward_inputs_af3.
    """
    import os
    import torch
    from QA_analysis.utils.gpu_utils import (
        _effective_seq_len_from_inputs,
        _longest_common_prefix_len_input_ids,
        _move_model_inputs_to_device_dtype,
        _normalize_past_key_values_for_reuse,
        _infer_pkv_seq_len,
        _build_suffix_forward_inputs,
        _single_logit_from_full_prepared_inputs,
        logger as _gpu_logger,
    )

    if not full_prompts:
        return []

    per_item_inputs = _prepare_per_prompt_inputs_same_audio_af3(
        model=model, processor=processor, audio_array=audio_array, full_prompts=full_prompts,
    )

    if len(per_item_inputs) == 1:
        return [_single_logit_from_full_prepared_inputs(model, per_item_inputs[0], target_id)]

    lcp_text_len = _longest_common_prefix_len_input_ids(per_item_inputs)
    min_common_prefix = int(os.environ.get("DIME_TEXT_PKV_CACHE_MIN_COMMON_PREFIX", "8"))
    min_eff_text_len = min(_effective_seq_len_from_inputs(x) for x in per_item_inputs)

    if lcp_text_len >= min_eff_text_len:
        lcp_text_len = max(0, min_eff_text_len - 1)

    if lcp_text_len < min_common_prefix:
        return [
            _single_logit_from_full_prepared_inputs(model, inp, target_id)
            for inp in per_item_inputs
        ]

    try:
        prefix_inputs = _build_prefix_forward_inputs_af3(per_item_inputs[0], lcp_text_len)
        prefix_inputs = _move_model_inputs_to_device_dtype(model, prefix_inputs)

        with torch.inference_mode():
            prefix_out = model(**prefix_inputs, use_cache=True, return_dict=True)

        past_key_values = _normalize_past_key_values_for_reuse(prefix_out.past_key_values)
        prefix_cache_len = _infer_pkv_seq_len(past_key_values)

        if prefix_cache_len < lcp_text_len:
            raise RuntimeError(
                f"Invalid PKV lengths: prefix_cache_len={prefix_cache_len} < lcp_text_len={lcp_text_len}"
            )

        prefix_logits_last = prefix_out.logits[0, -1]
        prefix_vocab_size = int(prefix_logits_last.shape[-1])
        if not isinstance(target_id, int) or target_id < 0 or target_id >= prefix_vocab_size:
            prefix_last_value = float(prefix_logits_last.mean().item())
        else:
            prefix_last_value = float(prefix_logits_last[target_id].item())

        vals: List[float] = []
        for inp in per_item_inputs:
            suffix_inputs = _build_suffix_forward_inputs(
                full_input=inp,
                prefix_text_len=lcp_text_len,
                prefix_cache_len=prefix_cache_len,
                past_key_values=past_key_values,
            )
            if suffix_inputs is None:
                vals.append(float(prefix_last_value))
                continue

            suffix_inputs = _move_model_inputs_to_device_dtype(model, suffix_inputs)
            with torch.inference_mode():
                out = model(**suffix_inputs, use_cache=False, return_dict=True)

            logits = out.logits[0, -1]
            vocab_size = int(logits.shape[-1])
            if not isinstance(target_id, int) or target_id < 0 or target_id >= vocab_size:
                vals.append(float(logits.mean().item()))
            else:
                vals.append(float(logits[target_id].item()))

        verify = os.environ.get("DIME_TEXT_PKV_CACHE_VERIFY", "1").lower() in ("1", "true", "yes")
        debug_all = os.environ.get("DIME_TEXT_PKV_CACHE_DEBUG", "0").lower() in ("1", "true", "yes")
        if verify:
            atol = float(os.environ.get("DIME_TEXT_PKV_CACHE_ATOL", "1e-5"))
            rtol = float(os.environ.get("DIME_TEXT_PKV_CACHE_RTOL", "1e-4"))
            verify_indices = [0]
            if len(per_item_inputs) > 1:
                verify_indices.append(len(per_item_inputs) - 1)
            for idx in verify_indices:
                baseline = _single_logit_from_full_prepared_inputs(model, per_item_inputs[idx], target_id)
                diff = abs(float(vals[idx]) - float(baseline))
                thr = atol + rtol * max(1.0, abs(float(vals[idx])), abs(float(baseline)))
                rel_diff = diff / max(1.0, abs(float(vals[idx])), abs(float(baseline)))
                # DIME_TEXT_PKV_CACHE_DEBUG=1: logga OGNI confronto (anche i pass),
                # per costruire la distribuzione completa dei diff osservati e non
                # vedere solo la coda che supera la soglia -- utile per calibrare
                # ATOL/RTOL con dati reali invece di tarare alla cieca sui warning.
                if debug_all:
                    _gpu_logger.info(
                        "[AF3 DIME PKV CACHE] verify | "
                        f"idx={idx} n_items={len(per_item_inputs)} "
                        f"lcp_text_len={lcp_text_len} prefix_cache_len={prefix_cache_len} "
                        f"val_pkv={float(vals[idx]):.6f} val_full={float(baseline):.6f} "
                        f"diff={diff:.6f} rel_diff={rel_diff:.6f} thr={thr:.6f} "
                        f"pass={diff <= thr}"
                    )
                if diff > thr:
                    _gpu_logger.warning(
                        "[AF3 DIME PKV CACHE] verify failed -> fallback full path | "
                        f"idx={idx} n_items={len(per_item_inputs)} "
                        f"lcp_text_len={lcp_text_len} prefix_cache_len={prefix_cache_len} "
                        f"val_pkv={float(vals[idx]):.6f} val_full={float(baseline):.6f} "
                        f"diff={diff:.6f} rel_diff={rel_diff:.6f} thr={thr:.6f}"
                    )
                    return [
                        _single_logit_from_full_prepared_inputs(model, inp, target_id)
                        for inp in per_item_inputs
                    ]

        return vals

    except Exception as e:
        _gpu_logger.warning(f"[AF3 DIME PKV CACHE] exception -> fallback full path: {e}")
        return [
            _single_logit_from_full_prepared_inputs(model, inp, target_id)
            for inp in per_item_inputs
        ]


def _load_af3_model(model_path: str, dtype):
    """Carica processor+modello AF3 col fallback flash_attention_2 -> sdpa -> eager."""
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    model = None
    attn_impl_used = None
    attempt_errors: List[str] = []
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=None,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
            ).to("cuda:0").eval()
            attn_impl_used = attn_impl
            break
        except Exception as e:
            import traceback as _tb
            attempt_errors.append(f"[{attn_impl}] {type(e).__name__}: {e}\n{_tb.format_exc()[-500:]}")
            model = None
            # Pulizia tra un tentativo e l'altro: senza questo, allocazioni
            # parziali del tentativo fallito restano occupate e il tentativo
            # successivo (es. sdpa dopo flash_attention_2) parte gia' in
            # svantaggio, aggravando l'OOM invece di ripartire pulito.
            try:
                import gc as _gc
                import torch as _torch
                _gc.collect()
                _torch.cuda.empty_cache()
            except Exception:
                pass
    if model is None:
        detail = "\n---\n".join(attempt_errors)
        raise RuntimeError(f"Impossibile caricare AF3 con nessun attn_implementation disponibile:\n{detail}")
    return processor, model, attn_impl_used


def _af3_worker_loop_full(
    gpu_id: int,
    model_path: str,
    task_q,
    result_q,
    torch_dtype: str = "bf16",
) -> None:
    """
    Worker loop AF3 equivalente a _worker_loop (Qwen), stesso set di task
    kind e stessi nomi/formati di payload e risposta:
        caption_chat, dime_probe_audio_io, dime_L_batch,
        dime_row_values_batch, dime_col_values_batch, mmshap_logits

    IMPORTANTE: usa i sentinel di errore "__worker_init_error__" e
    "__task_error__" -- NON nomi custom -- perche' Af3TokenRunner eredita
    _get_for_req da ParallelTokenRunner senza modificarlo, e quel metodo
    riconosce solo questi due nomi esatti.
    """
    import os
    # CUDA_VISIBLE_DEVICES va impostata PRIMA di importare torch (vedi nota
    # in _af3_worker_loop piu' sopra nello stesso file).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("DIME_VALUE_MODE", "logit")

    import gc
    import torch

    configure_runtime()
    _configure_worker_quiet_mode()

    _wlog = _af3_configure_worker_logger("Af3TokenWorker")
    dtype = torch.bfloat16 if torch_dtype == "bf16" else torch.float16

    try:
        _wlog.info(f"[af3_token_worker gpu={gpu_id}] Caricamento modello...")
        processor, model, attn_impl_used = _load_af3_model(model_path, dtype)
        _wlog.info(f"[af3_token_worker gpu={gpu_id}] Pronto (attn_implementation={attn_impl_used}).")
    except Exception as e:
        result_q.put(("__worker_init_error__", {"gpu": gpu_id, "error": str(e)}))
        return

    torch.set_grad_enabled(False)

    CACHE_EVERY_N_TASKS = int(os.environ.get("DIME_WORKER_EMPTYCACHE_EVERY", "0"))
    task_counter = 0

    def _cleanup_cuda(force: bool = False):
        nonlocal task_counter
        task_counter += 1
        gc.collect()
        if force:
            try: torch.cuda.empty_cache()
            except Exception: pass
            return
        if CACHE_EVERY_N_TASKS > 0 and (task_counter % CACHE_EVERY_N_TASKS == 0):
            try: torch.cuda.empty_cache()
            except Exception: pass

    while True:
        task = task_q.get()
        if task is None:
            break

        req_id = task.payload.get("req_id", None)

        try:
            if task.kind == "caption_chat":
                audio_path = task.payload["audio_path"]
                prompt = task.payload["prompt"]
                response = _generate_text_response_af3(model, processor, audio_path, prompt)
                result_q.put(("caption_chat", {
                    "req_id": req_id,
                    "batch_id": task.payload.get("batch_id", None),
                    "response": str(response),
                }))
                _cleanup_cuda(force=False)

            elif task.kind == "dime_probe_audio_io":
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])
                file_audio_path = task.payload["file_audio_path"]
                inline_audio = task.payload["inline_audio"]
                prompt = task.payload["prompt"]
                inline_audio_input, inline_handle = _materialize_worker_audio_input(inline_audio)
                try:
                    v_file = dime_token_value_af3(
                        model, processor, file_audio_path, prompt, caption_ids, token_index,
                    )
                    v_inline = dime_token_value_af3(
                        model, processor, inline_audio_input, prompt, caption_ids, token_index,
                    )
                finally:
                    _close_worker_audio_handle(inline_handle)
                result_q.put(("dime_probe_audio_io", {
                    "req_id": req_id,
                    "file_value": float(v_file),
                    "inline_value": float(v_inline),
                }))
                _cleanup_cuda(force=False)

            elif task.kind == "dime_L_batch":
                # PKV/prefix-cache anche per la L-table (Step 2): le coppie
                # (i,j) ricevute in un singolo task_.dime_L_batch condividono
                # spesso lo stesso i (stesso audio di sfondo) perche' il
                # chiamante (run_dime_L_table, gpu_utils.py) chunk-izza
                # ij_all = [(i,j) for i in range(N) for j in range(N)] in
                # blocchi contigui di DEFAULT_L_BATCH_SIZE -- per costruzione,
                # se DEFAULT_L_BATCH_SIZE <= N ogni blocco resta dentro un
                # solo i. Raggruppiamo esplicitamente per i (senza assumere
                # nulla su come sono ordinate le coppie: funziona comunque
                # anche se un chunk dovesse coprire piu' di un i) cosi' da
                # applicare lo stesso identico principio gia' validato per
                # dime_row_values_batch: un solo forward pass sul prefisso
                # condiviso per ogni i, invece di un forward pass pieno per
                # ciascuna singola coppia (i,j). Stessa matematica, stesso
                # risultato atteso (entro il normale rumore bf16 gia'
                # misurato), percorso di calcolo piu' veloce.
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])
                audio_paths = task.payload["audio_paths"]
                prompts = task.payload["prompts"]
                ij_list = task.payload["ij_list"]
                batch_id = int(task.payload["batch_id"])

                target_id = int(caption_ids[token_index])
                prefix_ids = caption_ids[:token_index]
                tok = processor.tokenizer
                prefix_text = ""
                if prefix_ids:
                    prefix_text = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(prefix_ids))

                target_sr = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000))
                use_pkv_cache = os.environ.get("DIME_TEXT_PKV_CACHE", "1").lower() in ("1", "true", "yes")

                groups: Dict[int, List[int]] = {}
                for (i, j) in ij_list:
                    groups.setdefault(int(i), []).append(int(j))

                vals_by_ij: Dict[Tuple[int, int], float] = {}

                for i, j_list in groups.items():
                    ai_raw = audio_paths[i]
                    ai, ai_handle = _materialize_worker_audio_input(ai_raw)
                    try:
                        audio_arr = _resolve_audio_array_af3(ai, target_sr)
                        group_prompts = [prompts[j] for j in j_list]

                        if use_pkv_cache:
                            full_prompts = [
                                (p + " " + prefix_text).strip() if prefix_text else p
                                for p in group_prompts
                            ]
                            group_vals = _batch_dime_logit_same_audio_with_prefix_kv_inner_af3(
                                model=model, processor=processor,
                                audio_array=audio_arr, full_prompts=full_prompts, target_id=target_id,
                            )
                        else:
                            group_vals = [
                                float(dime_token_value_af3(model, processor, audio_arr, p, caption_ids, token_index))
                                for p in group_prompts
                            ]
                    finally:
                        _close_worker_audio_handle(ai_handle)

                    for j, v in zip(j_list, group_vals):
                        vals_by_ij[(i, j)] = float(v)

                vals = [vals_by_ij[(int(i), int(j))] for (i, j) in ij_list]

                result_q.put(("dime_L_batch", {
                    "req_id": req_id,
                    "batch_id": batch_id,
                    "ij_list": ij_list,
                    "vals": vals,
                }))
                _cleanup_cuda(force=False)

            elif task.kind == "dime_row_values_batch":
                # PKV/prefix-cache (DIME_TEXT_PKV_CACHE): per ogni audio fisso,
                # un solo forward pass sul prefisso comune a tutti i prompt
                # (audio + testo condiviso), poi un forward economico per
                # ciascun prompt divergente riusando il KV cache. Fallback
                # (DIME_TEXT_PKV_CACHE=0): un forward pass indipendente per
                # ogni coppia (audio, prompt), come prima.
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])
                audio_paths = task.payload["audio_paths"]
                prompts = task.payload["prompts"]
                batch_id = int(task.payload["batch_id"])

                tok = processor.tokenizer
                target_id = int(caption_ids[token_index])
                prefix_ids = caption_ids[:token_index]
                prefix_text = ""
                if prefix_ids:
                    prefix_text = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(prefix_ids))
                full_prompts = [(p + " " + prefix_text).strip() if prefix_text else p for p in prompts]

                target_sr = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000))
                use_pkv_cache = os.environ.get("DIME_TEXT_PKV_CACHE", "1").lower() in ("1", "true", "yes")

                rows = []
                for ap_raw in audio_paths:
                    ap, ap_handle = _materialize_worker_audio_input(ap_raw)
                    try:
                        audio_arr = _resolve_audio_array_af3(ap, target_sr)
                        if use_pkv_cache:
                            row = _batch_dime_logit_same_audio_with_prefix_kv_inner_af3(
                                model=model, processor=processor,
                                audio_array=audio_arr, full_prompts=full_prompts, target_id=target_id,
                            )
                        else:
                            row = [
                                float(dime_token_value_af3(model, processor, audio_arr, p, caption_ids, token_index))
                                for p in prompts
                            ]
                    finally:
                        _close_worker_audio_handle(ap_handle)
                    rows.append([float(x) for x in row])

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

                tok = processor.tokenizer
                target_id = int(caption_ids[token_index])
                prefix_ids = caption_ids[:token_index]
                prefix_text = ""
                if prefix_ids:
                    prefix_text = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(prefix_ids))
                full_prompts = [(p + " " + prefix_text).strip() if prefix_text else p for p in prompts]

                target_sr = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000))
                use_pkv_cache = os.environ.get("DIME_TEXT_PKV_CACHE", "1").lower() in ("1", "true", "yes")

                audio_handles = []
                rows_by_audio: List[List[float]] = []
                try:
                    for ap_raw in audio_paths:
                        ap, ap_handle = _materialize_worker_audio_input(ap_raw)
                        audio_handles.append(ap_handle)
                        audio_arr = _resolve_audio_array_af3(ap, target_sr)
                        if use_pkv_cache:
                            row_vals = _batch_dime_logit_same_audio_with_prefix_kv_inner_af3(
                                model=model, processor=processor,
                                audio_array=audio_arr, full_prompts=full_prompts, target_id=target_id,
                            )
                        else:
                            row_vals = [
                                float(dime_token_value_af3(model, processor, audio_arr, p, caption_ids, token_index))
                                for p in prompts
                            ]
                        rows_by_audio.append([float(x) for x in row_vals])
                finally:
                    for h in audio_handles:
                        _close_worker_audio_handle(h)

                num_prompts = len(prompts)
                num_audios = len(audio_paths)
                cols = [
                    [float(rows_by_audio[i][j]) for i in range(num_audios)]
                    for j in range(num_prompts)
                ]
                result_q.put(("dime_col_values_batch", {
                    "req_id": req_id,
                    "batch_id": batch_id,
                    "cols": cols,
                }))
                _cleanup_cuda(force=False)

            elif task.kind == "mmshap_logits":
                audio_path_raw = task.payload["audio_path"]
                prompt = task.payload["prompt"]
                target_ids = task.payload["target_ids"]
                batch_id = task.payload.get("batch_id", None)
                audio_path, audio_handle = _materialize_worker_audio_input(audio_path_raw)
                try:
                    conversation = _build_af3_conversation(audio_path, prompt)
                    inputs = processor.apply_chat_template(
                        conversation, tokenize=True, add_generation_prompt=True, return_dict=True,
                    ).to(model.device)
                    for k in inputs:
                        if hasattr(inputs[k], "dtype") and inputs[k].dtype.is_floating_point:
                            inputs[k] = inputs[k].to(model.dtype)

                    T = len(target_ids)
                    vals = []
                    if T == 1:
                        with torch.inference_mode():
                            outputs = model(**inputs, use_cache=False, return_dict=True)
                        logits_last = outputs.logits[0, -1]
                        tid = int(target_ids[0])
                        vocab_size = int(logits_last.shape[-1])
                        if 0 <= tid < vocab_size:
                            vals.append(float(logits_last[tid].item()))
                        else:
                            vals.append(float(logits_last.mean().item()))
                    else:
                        # use_cache=True (a differenza del path Qwen, che lo
                        # tiene a False ovunque): per AF3 questo path multi-
                        # token e' la norma, non un caso raro -- baseline_answer
                        # e' il testo libero completo del modello (es. "(A)
                        # RIFF"), non la lettera nuda di Qwen, quindi quasi
                        # ogni valutazione MM-SHAP finisce qui. Con
                        # use_cache=False, generate() rifà il forward
                        # sull'intera sequenza per OGNI token generato (T+2
                        # forward pass pieni); con use_cache=True ne serve
                        # uno pieno + T+1 incrementali economici -- stesso
                        # principio della cache PKV, qui applicato al posto
                        # giusto. Non tocca gpu_utils.py (Qwen): li' questo
                        # path e' quasi mai esercitato, nessun bisogno di
                        # cambiarlo.
                        with torch.inference_mode():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max(1, T + 2),
                                return_dict_in_generate=True,
                                output_logits=True,
                                use_cache=True,
                            )
                        step_logits = torch.stack([x[0] for x in outputs.logits], dim=0)
                        T_actual = min(step_logits.shape[0], T)
                        for t in range(T_actual):
                            tid = int(target_ids[t])
                            if 0 <= tid < step_logits.shape[1]:
                                vals.append(float(step_logits[t, tid].item()))
                            else:
                                vals.append(float(step_logits[t].mean().item()))
                finally:
                    _close_worker_audio_handle(audio_handle)

                result_q.put(("mmshap_logits", {
                    "req_id": req_id,
                    "batch_id": batch_id,
                    "vals": vals,
                }))
                _cleanup_cuda(force=False)

            else:
                raise ValueError(f"Af3TokenWorker: task kind sconosciuto '{task.kind}'")

        except torch.cuda.OutOfMemoryError as e:
            _cleanup_cuda(force=True)
            result_q.put(("__task_error__", {
                "req_id": req_id, "gpu": gpu_id, "kind": task.kind,
                "error": f"CUDA OOM: {e}",
            }))
        except Exception as e:
            _cleanup_cuda(force=True)
            import traceback as _tb
            result_q.put(("__task_error__", {
                "req_id": req_id, "gpu": gpu_id, "kind": task.kind,
                "error": f"{e}\n{_tb.format_exc()[:1000]}",
            }))


class Af3TokenRunner(ParallelTokenRunner):
    """
    Sottoclasse di ParallelTokenRunner (Qwen): eredita INVARIATA tutta la
    logica di IPC/batching -- run_dime_L_table, run_dime_row_values,
    run_dime_col_values, get_mmshap_logits_batch, generate_caption,
    run_single, stop, _put_task, _get_for_req -- e sovrascrive solo
    start(), per avviare _af3_worker_loop_full invece del worker Qwen.
    """

    def start(self):
        import os
        for gid in self.gpu_ids:
            # Fix isolamento GPU (2026-08-25): vedi commento identico in
            # Af3ParallelRunner.start() poco sopra.
            _old = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
            try:
                p = self.ctx.Process(
                    target=_af3_worker_loop_full,
                    args=(gid, self.model_path, self.task_q, self.result_q, self.torch_dtype),
                    daemon=True,
                )
                p.start()
            finally:
                if _old is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = _old
            self.procs.append(p)


def try_create_af3_token_runner(
    model_path: str,
    max_gpus: int = 8,
    min_free_memory_gb: float = 24.0,
    allowed_gpu_ids: Optional[List[int]] = None,
    gpu_ids_physical: Optional[List[int]] = None,
):
    """Equivalente di try_create_parallel_runner (Qwen) per Af3TokenRunner."""
    import os as _os
    import time as _time
    from QA_analysis.utils.gpu_utils import get_available_gpus_with_memory, logger as _gpu_logger

    _os.environ.setdefault("DIME_VALUE_MODE", "logit")

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
        _gpu_logger.warning("Af3TokenRunner non attivato (memoria libera insufficiente).")
        return None

    if max_gpus is not None and max_gpus > 0:
        gpu_ids = gpu_ids[: int(max_gpus)]

    runner = Af3TokenRunner(model_path=model_path, gpu_ids=gpu_ids, torch_dtype="bf16")
    runner.start()

    _time.sleep(1.0)
    for p in runner.procs:
        if not p.is_alive():
            _gpu_logger.error("Worker AF3 morto in init: disabilito runner.")
            try: runner.stop()
            except Exception: pass
            return None

    _gpu_logger.info(f"Af3TokenRunner attivo su GPU fisiche: {gpu_ids}")
    return runner
