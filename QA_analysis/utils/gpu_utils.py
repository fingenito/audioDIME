import os
import time
import logging
import subprocess
import multiprocessing as mp
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import sys
import warnings
import copy
import contextlib
from multiprocessing import shared_memory

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
def _build_messages(audio_path: Any, prompt: str):
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

def _is_shared_memory_audio_descriptor(x: Any) -> bool:
    return isinstance(x, dict) and x.get("kind") == "shared_memory_audio"

def _is_inline_audio_descriptor(x: Any) -> bool:
    return isinstance(x, dict) and (
        x.get("kind") == "inline_audio" or ("array" in x and "sampling_rate" in x)
    )

def _materialize_worker_audio_input(audio_obj: Any) -> Tuple[Any, Optional[shared_memory.SharedMemory]]:
    """
    Converte l'oggetto ricevuto dal main in un input consumabile da Qwen nel worker.

    Ritorna:
      - audio_input pronto per shared_utils._build_qwen25_audio_messages / prepare_qwen25_omni_inputs
      - eventuale handle SharedMemory da chiudere dopo l'uso

    FIX Level-1:
    - per shared_memory NON facciamo copy() immediata;
    - manteniamo vivo l'handle fino a fine uso;
    - prepare_qwen25_omni_inputs consumerà il buffer durante la chiamata.
    """
    import numpy as np

    if _is_shared_memory_audio_descriptor(audio_obj):
        shm_name = str(audio_obj["shm_name"])
        shape = tuple(int(x) for x in audio_obj["shape"])
        dtype = np.dtype(str(audio_obj["dtype"]))
        sr = int(audio_obj["sampling_rate"])

        shm = shared_memory.SharedMemory(name=shm_name)
        arr_view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        return {
            "array": arr_view,   # zero-copy view sul buffer shared memory
            "sampling_rate": sr,
        }, shm

    if _is_inline_audio_descriptor(audio_obj):
        return {
            "array": np.asarray(audio_obj["array"], dtype=np.float32).reshape(-1),
            "sampling_rate": int(audio_obj["sampling_rate"]),
        }, None

    return audio_obj, None

def _close_worker_audio_handle(handle: Optional[shared_memory.SharedMemory]) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except Exception:
        pass

def _generate_text_response_qwen25_omni(model, processor, audio_path: Any, prompt: str) -> str:
    import torch
    from QA_analysis.utils.shared_utils import prepare_qwen25_omni_inputs

    audio_input, shm_handle = _materialize_worker_audio_input(audio_path)
    try:
        messages = _build_messages(audio_input, prompt)

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
    finally:
        _close_worker_audio_handle(shm_handle)

def _move_model_inputs_to_device_dtype(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            vv = v.to(model.device)
            if vv.is_floating_point():
                vv = vv.to(model.dtype)
            out[k] = vv
        else:
            out[k] = v
    return out


def _effective_seq_len_from_inputs(inp: Dict[str, Any]) -> int:
    import torch

    if "attention_mask" in inp and torch.is_tensor(inp["attention_mask"]):
        return int(inp["attention_mask"][0].sum().item())
    return int(inp["input_ids"].shape[1])


def _longest_common_prefix_len_input_ids(per_item_inputs: List[Dict[str, Any]]) -> int:
    if not per_item_inputs:
        return 0

    seqs = []
    for inp in per_item_inputs:
        eff_len = _effective_seq_len_from_inputs(inp)
        ids = inp["input_ids"][0, :eff_len].detach().cpu().tolist()
        seqs.append(ids)

    min_len = min(len(x) for x in seqs)
    if min_len <= 0:
        return 0

    lcp = 0
    for i in range(min_len):
        tok0 = seqs[0][i]
        same = True
        for s in seqs[1:]:
            if s[i] != tok0:
                same = False
                break
        if not same:
            break
        lcp += 1

    return int(lcp)


def _build_prefix_forward_inputs(full_input: Dict[str, Any], prefix_text_len: int) -> Dict[str, Any]:
    """
    Forward del prefisso multimodale.
    Qui vanno inclusi anche i tensori audio/video, perché servono per costruire
    correttamente il contesto cached del modello.
    """
    out: Dict[str, Any] = {
        "input_ids": full_input["input_ids"][:, :prefix_text_len],
    }

    if "attention_mask" in full_input:
        out["attention_mask"] = full_input["attention_mask"][:, :prefix_text_len]

    for k in (
        "input_features",
        "feature_attention_mask",
        "audio_feature_lengths",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "rope_deltas",
        "video_second_per_grid",
    ):
        if k in full_input:
            out[k] = full_input[k]

    return out


def _normalize_past_key_values_for_reuse(past_key_values):
    """
    NON convertire a legacy format: il modello si aspetta DynamicCache.
    Restituisce il cache così com'è.
    """
    return past_key_values


def _infer_pkv_seq_len(past_key_values) -> int:
    """
    Ricava la lunghezza del cache. Gestisce sia DynamicCache che legacy tuple.
    """
    if past_key_values is None:
        raise RuntimeError("past_key_values is None")

    # DynamicCache (transformers >= 4.38, usato da Qwen2.5-Omni)
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            pass

    # DynamicCache: accesso diretto key_cache
    if hasattr(past_key_values, "key_cache"):
        try:
            key_cache = past_key_values.key_cache
            if key_cache and key_cache[0] is not None:
                return int(key_cache[0].shape[-2])
        except Exception:
            pass

    # Legacy tuple format
    try:
        layer0 = past_key_values[0]
        if isinstance(layer0, (tuple, list)):
            return int(layer0[0].shape[-2])
    except Exception:
        pass

    raise RuntimeError("Unable to infer KV cache sequence length from past_key_values")


def _clone_past_key_values_for_safe_reuse(past_key_values):
    """
    Clona il KV cache per evitare modifiche in-place tra prompt successivi.

    DynamicCache: usa copy.deepcopy (sicuro, mantiene il tipo corretto).
    Legacy tuple: clone tensore per tensore.
    """
    import torch
    import copy

    if past_key_values is None:
        return None

    # DynamicCache: deepcopy mantiene la classe e i metodi (.update, ecc.)
    if hasattr(past_key_values, "key_cache") or hasattr(past_key_values, "get_seq_length"):
        try:
            return copy.deepcopy(past_key_values)
        except Exception:
            pass

    # Legacy tuple format
    cloned = []
    for layer in past_key_values:
        if isinstance(layer, (tuple, list)):
            cloned_layer = []
            for x in layer:
                if torch.is_tensor(x):
                    cloned_layer.append(x.clone())
                else:
                    cloned_layer.append(x)
            cloned.append(tuple(cloned_layer))
        else:
            cloned.append(layer)
    return tuple(cloned)


def _build_suffix_forward_inputs(
    full_input: Dict[str, Any],
    prefix_text_len: int,
    prefix_cache_len: int,
    past_key_values,
) -> Optional[Dict[str, Any]]:
    """
    Costruisce il forward del suffix in modo coerente con il cache multimodale reale.
    """
    import torch

    total_text_len = _effective_seq_len_from_inputs(full_input)
    suffix_len = int(total_text_len - prefix_text_len)

    if suffix_len <= 0:
        return None

    suffix_input_ids = full_input["input_ids"][:, prefix_text_len:total_text_len]

    if "attention_mask" in full_input and torch.is_tensor(full_input["attention_mask"]):
        attn_dtype = full_input["attention_mask"].dtype
    else:
        attn_dtype = torch.long

    attention_mask = torch.ones(
        (1, int(prefix_cache_len + suffix_len)),
        dtype=attn_dtype,
        device=suffix_input_ids.device,
    )

    cache_position = torch.arange(
        int(prefix_cache_len),
        int(prefix_cache_len + suffix_len),
        dtype=torch.long,
        device=suffix_input_ids.device,
    )

    out: Dict[str, Any] = {
        "input_ids": suffix_input_ids,
        "attention_mask": attention_mask,
        "past_key_values": _clone_past_key_values_for_safe_reuse(past_key_values),
        "cache_position": cache_position,
    }

    if "rope_deltas" in full_input:
        out["rope_deltas"] = full_input["rope_deltas"]

    if "video_second_per_grid" in full_input:
        out["video_second_per_grid"] = full_input["video_second_per_grid"]

    return out


def _single_logit_from_full_prepared_inputs(
    model,
    full_input: Dict[str, Any],
    target_id: int,
) -> float:
    import torch

    inp = _move_model_inputs_to_device_dtype(model, full_input)
    with torch.inference_mode():
        outputs = model(**inp, use_cache=False, return_dict=True)

    logits = outputs.logits[0, -1]
    vocab_size = int(logits.shape[-1])

    if not isinstance(target_id, int) or target_id < 0 or target_id >= vocab_size:
        return float(logits.mean().item())

    return float(logits[target_id].item())


def _prepare_per_prompt_inputs_same_audio(
    model,
    processor,
    audio_array: "np.ndarray",
    target_sr: int,
    full_prompts: List[str],
) -> List[Dict[str, Any]]:
    import numpy as np
    import torch
    from QA_analysis.utils.shared_utils import (
        prepare_qwen25_omni_inputs,
        _build_qwen25_audio_messages,
    )

    audio_np = np.asarray(audio_array, dtype=np.float32).reshape(-1)
    audio_input = {"array": audio_np, "sampling_rate": int(target_sr)}

    _fe_cache: dict = {}

    def _audio_fingerprint(arr: np.ndarray, sr: int) -> tuple:
        n = int(arr.shape[0]) if arr.ndim > 0 else 0
        head = arr.ravel()[:min(64, n)].tobytes()
        tail = arr.ravel()[-min(16, n):].tobytes() if n >= 16 else b""
        return (n, head + tail, int(sr))

    original_fe_call = processor.feature_extractor.__call__

    def _cached_fe(audio, sampling_rate=None, **kwargs):
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            arr = np.asarray(audio[0], dtype=np.float32).reshape(-1)
        elif isinstance(audio, np.ndarray):
            arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        else:
            return original_fe_call(audio, sampling_rate=sampling_rate, **kwargs)

        sr = int(sampling_rate) if sampling_rate is not None else int(target_sr)
        key = _audio_fingerprint(arr, sr)

        if key in _fe_cache:
            return _fe_cache[key]

        result = original_fe_call(audio, sampling_rate=sampling_rate, **kwargs)
        _fe_cache[key] = result
        return result

    processor.feature_extractor.__call__ = _cached_fe

    try:
        per_item_inputs: List[Dict[str, Any]] = []
        for fp in full_prompts:
            messages = _build_qwen25_audio_messages(audio_input, fp)
            _, inp = prepare_qwen25_omni_inputs(
                processor=processor,
                conversation=messages,
                device=None,
                dtype=None,
            )

            cloned = {}
            for k, v in inp.items():
                if torch.is_tensor(v):
                    cloned[k] = v.detach().clone()
                else:
                    cloned[k] = v
            per_item_inputs.append(cloned)

        return per_item_inputs

    finally:
        processor.feature_extractor.__call__ = original_fe_call


def _batch_dime_logit_same_audio_with_prefix_kv_inner(
    model,
    processor,
    audio_array: "np.ndarray",
    target_sr: int,
    full_prompts: List[str],
    target_id: int,
) -> List[float]:
    """
    Reuse corretto del prefisso cached per Qwen2.5-Omni.

    La chiave è:
    - il prefisso comune si trova sui token testuali
    - la lunghezza vera del contesto cached si legge da past_key_values
    - il suffix va costruito con cache_position coerente col cache reale
    """
    import torch

    if not full_prompts:
        return []

    per_item_inputs = _prepare_per_prompt_inputs_same_audio(
        model=model,
        processor=processor,
        audio_array=audio_array,
        target_sr=target_sr,
        full_prompts=full_prompts,
    )

    if len(per_item_inputs) == 1:
        return [_single_logit_from_full_prepared_inputs(model, per_item_inputs[0], target_id)]

    lcp_text_len = _longest_common_prefix_len_input_ids(per_item_inputs)
    min_common_prefix = int(os.environ.get("DIME_TEXT_PKV_CACHE_MIN_COMMON_PREFIX", "8"))

    min_eff_text_len = min(_effective_seq_len_from_inputs(x) for x in per_item_inputs)

    # deve rimanere almeno 1 token nel suffix
    if lcp_text_len >= min_eff_text_len:
        lcp_text_len = max(0, min_eff_text_len - 1)

    if lcp_text_len < min_common_prefix:
        return [
            _single_logit_from_full_prepared_inputs(model, inp, target_id)
            for inp in per_item_inputs
        ]

    try:
        # 1) prefisso comune: una sola volta
        prefix_inputs = _build_prefix_forward_inputs(per_item_inputs[0], lcp_text_len)
        prefix_inputs = _move_model_inputs_to_device_dtype(model, prefix_inputs)

        with torch.inference_mode():
            prefix_out = model(**prefix_inputs, use_cache=True, return_dict=True)

        past_key_values = _normalize_past_key_values_for_reuse(prefix_out.past_key_values)
        prefix_cache_len = _infer_pkv_seq_len(past_key_values)

        # se il cache è più corto del prefisso testuale, qualcosa non torna
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

        # verifica exact su 2 elementi del batch
        verify = os.environ.get("DIME_TEXT_PKV_CACHE_VERIFY", "1").lower() in ("1", "true", "yes")
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

                if diff > thr:
                    logger.warning(
                        "[DIME PKV CACHE] verify failed -> fallback full path | "
                        f"idx={idx} diff={diff:.6e} thr={thr:.6e} "
                        f"lcp_text_len={lcp_text_len} prefix_cache_len={prefix_cache_len}"
                    )
                    return [
                        _single_logit_from_full_prepared_inputs(model, inp, target_id)
                        for inp in per_item_inputs
                    ]

        return vals

    except Exception as e:
        logger.warning(
            "[DIME PKV CACHE] fallback full path after error: "
            f"{e} | num_prompts={len(full_prompts)} | lcp_text_len={lcp_text_len}"
        )
        return [
            _single_logit_from_full_prepared_inputs(model, inp, target_id)
            for inp in per_item_inputs
        ]

# ==========================================================
# Batch logit helpers — Bottleneck 2 fix (v2: manual tensor batching)
# ==========================================================

def _batch_dime_logit_same_audio_inner(
    model,
    processor,
    audio_array: "np.ndarray",
    target_sr: int,
    full_prompts: List[str],
    target_id: int,
) -> List[float]:
    """
    Batch forward pass: stesso audio, N prompt diversi.

    v2 — mel spectrogram cached: processor.feature_extractor viene patchato
    temporaneamente per restituire il risultato cached per le chiamate 2..B-1
    con lo stesso audio. Riduce il preprocessing da O(B) a O(1) mel spec.
    """
    import torch
    import numpy as np
    from QA_analysis.utils.shared_utils import (
        prepare_qwen25_omni_inputs,
        _build_qwen25_audio_messages,
    )

    B = len(full_prompts)
    if B == 0:
        return []

    audio_np = np.asarray(audio_array, dtype=np.float32).reshape(-1)
    audio_input = {"array": audio_np, "sampling_rate": int(target_sr)}

    # ── Feature-extractor cache ──────────────────────────────────────────
    # Stesso audio per tutti i B item → mel spec identico → computato 1 sola volta.
    # Fingerprint O(80 elementi): lunghezza + primi 64 + ultimi 16 campioni.
    # Robusto a np.asarray() che crea nuovi oggetti ma stesso contenuto.
    _fe_cache: dict = {}

    def _audio_fingerprint(arr: np.ndarray, sr: int) -> tuple:
        n = int(arr.shape[0]) if arr.ndim > 0 else 0
        head = arr.ravel()[:min(64, n)].tobytes()
        tail = arr.ravel()[-min(16, n):].tobytes() if n >= 16 else b""
        return (n, head + tail, int(sr))

    original_fe_call = processor.feature_extractor.__call__

    def _cached_fe(audio, sampling_rate=None, **kwargs):
        # Normalizza in np.ndarray per calcolare il fingerprint
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            arr = np.asarray(audio[0], dtype=np.float32).reshape(-1)
        elif isinstance(audio, np.ndarray):
            arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        else:
            return original_fe_call(audio, sampling_rate=sampling_rate, **kwargs)

        sr = int(sampling_rate) if sampling_rate is not None else int(target_sr)
        key = _audio_fingerprint(arr, sr)

        if key in _fe_cache:
            return _fe_cache[key]  # cache hit: mel spec gratis

        result = original_fe_call(audio, sampling_rate=sampling_rate, **kwargs)
        _fe_cache[key] = result
        return result

    processor.feature_extractor.__call__ = _cached_fe
    # ────────────────────────────────────────────────────────────────────

    try:
        per_item_inputs = []
        for fp in full_prompts:
            messages = _build_qwen25_audio_messages(audio_input, fp)
            _, inp = prepare_qwen25_omni_inputs(
                processor=processor,
                conversation=messages,
                device=None,
                dtype=None,
            )
            per_item_inputs.append({k: v.detach().clone() for k, v in inp.items()})
    finally:
        # Ripristina sempre il feature extractor originale
        processor.feature_extractor.__call__ = original_fe_call

    if B == 1:
        inp0 = {
            k: (v.to(model.device).to(model.dtype) if v.is_floating_point() else v.to(model.device))
            for k, v in per_item_inputs[0].items()
        }
        with torch.inference_mode():
            outputs = model(**inp0, use_cache=False)
        return [float(outputs.logits[0, -1, target_id].item())]

    # ── Manual tensor batching ────────────────────────────────────────────
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    all_keys = list(per_item_inputs[0].keys())
    batched = {}

    for key in all_keys:
        tensors = [inp[key] for inp in per_item_inputs]
        shapes = [t.shape for t in tensors]

        if key == "input_ids":
            max_len = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                gap = max_len - t.shape[1]
                if gap > 0:
                    padded.append(torch.cat(
                        [torch.full((1, gap), pad_id, dtype=t.dtype), t], dim=1
                    ))
                else:
                    padded.append(t)
            batched[key] = torch.cat(padded, dim=0)

        elif key == "attention_mask":
            max_len = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                gap = max_len - t.shape[1]
                if gap > 0:
                    padded.append(torch.cat(
                        [torch.zeros((1, gap), dtype=t.dtype), t], dim=1
                    ))
                else:
                    padded.append(t)
            batched[key] = torch.cat(padded, dim=0)

        elif all(s == shapes[0] for s in shapes):
            batched[key] = torch.cat(tensors, dim=0)

        else:
            max_last = max(t.shape[-1] for t in tensors)
            padded = []
            for t in tensors:
                gap = max_last - t.shape[-1]
                if gap > 0:
                    pad_shape = list(t.shape)
                    pad_shape[-1] = gap
                    padded.append(torch.cat(
                        [t, torch.zeros(pad_shape, dtype=t.dtype)], dim=-1
                    ))
                else:
                    padded.append(t)
            try:
                batched[key] = torch.cat(padded, dim=0)
            except RuntimeError:
                batched[key] = tensors[0].expand(B, *tensors[0].shape[1:]).clone()

    batched = {
        k: (v.to(model.device).to(model.dtype) if v.is_floating_point() else v.to(model.device))
        for k, v in batched.items()
    }

    with torch.inference_mode():
        outputs = model(**batched, use_cache=False)

    logits_last = outputs.logits[:, -1, :]
    vals = [float(logits_last[i, target_id].item()) for i in range(B)]

    del outputs, batched
    return vals


def _batch_dime_logit_same_prompt_inner(
    model,
    processor,
    audio_arrays: List["np.ndarray"],
    target_sr: int,
    full_prompt: str,
    target_id: int,
) -> List[float]:
    """
    Batch forward pass: N audio diversi, stesso prompt.

    Stesso meccanismo di _batch_dime_logit_same_audio_inner:
    prepare_qwen25_omni_inputs per item, poi batching manuale dei tensori.
    """
    import torch
    import numpy as np
    from QA_analysis.utils.shared_utils import (
        prepare_qwen25_omni_inputs,
        _build_qwen25_audio_messages,
    )

    B = len(audio_arrays)
    if B == 0:
        return []

    per_item_inputs = []
    for arr in audio_arrays:
        audio_input = {
            "array": np.asarray(arr, dtype=np.float32).reshape(-1),
            "sampling_rate": int(target_sr),
        }
        messages = _build_qwen25_audio_messages(audio_input, full_prompt)
        _, inp = prepare_qwen25_omni_inputs(
            processor=processor,
            conversation=messages,
            device=None,
            dtype=None,
        )
        per_item_inputs.append({k: v.detach().clone() for k, v in inp.items()})

    if B == 1:
        inp0 = {
            k: (v.to(model.device).to(model.dtype) if v.is_floating_point() else v.to(model.device))
            for k, v in per_item_inputs[0].items()
        }
        with torch.inference_mode():
            outputs = model(**inp0, use_cache=False)
        return [float(outputs.logits[0, -1, target_id].item())]

    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    all_keys = list(per_item_inputs[0].keys())
    batched = {}

    for key in all_keys:
        tensors = [inp[key] for inp in per_item_inputs]
        shapes = [t.shape for t in tensors]

        if key == "input_ids":
            max_len = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                gap = max_len - t.shape[1]
                if gap > 0:
                    padded.append(torch.cat(
                        [torch.full((1, gap), pad_id, dtype=t.dtype), t], dim=1
                    ))
                else:
                    padded.append(t)
            batched[key] = torch.cat(padded, dim=0)

        elif key == "attention_mask":
            max_len = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                gap = max_len - t.shape[1]
                if gap > 0:
                    padded.append(torch.cat(
                        [torch.zeros((1, gap), dtype=t.dtype), t], dim=1
                    ))
                else:
                    padded.append(t)
            batched[key] = torch.cat(padded, dim=0)

        elif all(s == shapes[0] for s in shapes):
            batched[key] = torch.cat(tensors, dim=0)

        else:
            max_last = max(t.shape[-1] for t in tensors)
            padded = []
            for t in tensors:
                gap = max_last - t.shape[-1]
                if gap > 0:
                    pad_shape = list(t.shape)
                    pad_shape[-1] = gap
                    padded.append(torch.cat(
                        [t, torch.zeros(pad_shape, dtype=t.dtype)], dim=-1
                    ))
                else:
                    padded.append(t)
            try:
                batched[key] = torch.cat(padded, dim=0)
            except RuntimeError:
                batched[key] = tensors[0].expand(B, *tensors[0].shape[1:]).clone()

    batched = {
        k: (v.to(model.device).to(model.dtype) if v.is_floating_point() else v.to(model.device))
        for k, v in batched.items()
    }

    with torch.inference_mode():
        outputs = model(**batched, use_cache=False)

    logits_last = outputs.logits[:, -1, :]
    vals = [float(logits_last[i, target_id].item()) for i in range(B)]

    del outputs, batched
    return vals

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
            attn_implementation="flash_attention_2",
        ).eval().to("cuda:0")

        logger.info(f"[worker gpu={gpu_id}] Modello caricato correttamente.")
    except Exception as e:
        result_q.put(("__worker_init_error__", {"gpu": gpu_id, "error": str(e)}))
        return

    torch.set_grad_enabled(False)

    from QA_analysis.utils.analysis_2 import dime_token_value
    from QA_analysis.utils.shared_utils import prepare_qwen25_omni_inputs

    CACHE_EVERY_N_TASKS = int(os.environ.get("DIME_WORKER_EMPTYCACHE_EVERY", "0"))
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

            elif task.kind == "dime_probe_audio_io":
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])
                file_audio_path = task.payload["file_audio_path"]
                inline_audio = task.payload["inline_audio"]
                prompt = task.payload["prompt"]
                inline_audio_input, inline_handle = _materialize_worker_audio_input(inline_audio)
                try:
                    with torch.inference_mode():
                        v_file = dime_token_value(
                            model=model,
                            processor=processor,
                            audio_path=file_audio_path,
                            prompt=prompt,
                            caption_ids=caption_ids,
                            token_index=token_index,
                        )
                    with torch.inference_mode():
                        v_inline = dime_token_value(
                            model=model,
                            processor=processor,
                            audio_path=inline_audio_input,
                            prompt=prompt,
                            caption_ids=caption_ids,
                            token_index=token_index,
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
                caption_ids = task.payload["caption_ids"]
                token_index = int(task.payload["token_index"])
                audio_paths = task.payload["audio_paths"]
                prompts = task.payload["prompts"]
                ij_list = task.payload["ij_list"]
                batch_id = int(task.payload["batch_id"])

                vals = []
                for (i, j) in ij_list:
                    ai_raw = audio_paths[int(i)]
                    tj = prompts[int(j)]

                    ai, ai_handle = _materialize_worker_audio_input(ai_raw)
                    try:
                        with torch.inference_mode():
                            v = dime_token_value(
                                model=model,
                                processor=processor,
                                audio_path=ai,
                                prompt=tj,
                                caption_ids=caption_ids,
                                token_index=token_index,
                            )
                    finally:
                        _close_worker_audio_handle(ai_handle)

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
                audio_paths = task.payload["audio_paths"]
                prompts = task.payload["prompts"]
                batch_id = int(task.payload["batch_id"])
                tok = processor.tokenizer
                target_id = int(caption_ids[token_index])
                prefix_ids = caption_ids[:token_index]
                prefix_text = ""
                if prefix_ids:
                    prefix_text = tok.convert_tokens_to_string(
                        tok.convert_ids_to_tokens(prefix_ids)
                    )
                # full_prompts: uguale per TUTTI gli audio perturbati del batch
                full_prompts = [
                    (p + " " + prefix_text).strip() if prefix_text else p
                    for p in prompts
                ]
                target_sr = processor.feature_extractor.sampling_rate
                inner_bs = int(os.environ.get("DIME_WORKER_INNER_BATCH_SIZE", "4"))
                use_pkv_cache = os.environ.get("DIME_TEXT_PKV_CACHE", "1").lower() in ("1", "true", "yes")
                rows = []
                for ap_raw in audio_paths:
                    ap, ap_handle = _materialize_worker_audio_input(ap_raw)
                    try:
                        if isinstance(ap, dict) and "array" in ap:
                            audio_arr = np.asarray(ap["array"], dtype=np.float32).reshape(-1)
                            arr_sr = int(ap.get("sampling_rate", target_sr))
                            if arr_sr != target_sr:
                                import librosa as _librosa
                                audio_arr = _librosa.resample(
                                    audio_arr, orig_sr=arr_sr, target_sr=target_sr
                                )
                                audio_arr = np.asarray(audio_arr, dtype=np.float32)
                        else:
                            import librosa as _librosa
                            audio_arr, _ = _librosa.load(str(ap), sr=target_sr, mono=True)
                            audio_arr = np.asarray(audio_arr, dtype=np.float32)
                        if use_pkv_cache:
                            # PKV cache: passa TUTTI i prompts in una sola chiamata.
                            # Il prefisso comune (audio tokens + scaffolding, ~1547 token)
                            # viene computato UNA volta; i suffix (~50-100 token ciascuno)
                            # vengono processati in serial ma sono 10-20× più corti.
                            # NON usare inner_bs qui: il costo è dominato dal prefix forward,
                            # non dal numero di suffix.
                            row = _batch_dime_logit_same_audio_with_prefix_kv_inner(
                                model=model,
                                processor=processor,
                                audio_array=audio_arr,
                                target_sr=target_sr,
                                full_prompts=full_prompts,
                                target_id=target_id,
                            )
                        else:
                            # Fallback: batching manuale con mel spec cache
                            row = []
                            for b0 in range(0, len(full_prompts), inner_bs):
                                batch_fps = full_prompts[b0:b0 + inner_bs]
                                vals = _batch_dime_logit_same_audio_inner(
                                    model=model,
                                    processor=processor,
                                    audio_array=audio_arr,
                                    target_sr=target_sr,
                                    full_prompts=batch_fps,
                                    target_id=target_id,
                                )
                                row.extend(vals)
                        rows.append([float(x) for x in row])
                    finally:
                        _close_worker_audio_handle(ap_handle)
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
                    prefix_text = tok.convert_tokens_to_string(
                        tok.convert_ids_to_tokens(prefix_ids)
                    )
                full_prompts = [
                    (p + " " + prefix_text).strip() if prefix_text else p
                    for p in prompts
                ]
                target_sr = processor.feature_extractor.sampling_rate
                use_pkv_cache = os.environ.get("DIME_TEXT_PKV_CACHE", "1").lower() in ("1", "true", "yes")
                audio_arrays: List[np.ndarray] = []
                audio_handles = []
                try:
                    for ap_raw in audio_paths:
                        ap, ap_handle = _materialize_worker_audio_input(ap_raw)
                        audio_handles.append(ap_handle)
                        if isinstance(ap, dict) and "array" in ap:
                            arr = np.asarray(ap["array"], dtype=np.float32).reshape(-1)
                            arr_sr = int(ap.get("sampling_rate", target_sr))
                            if arr_sr != target_sr:
                                import librosa as _librosa
                                arr = _librosa.resample(arr, orig_sr=arr_sr, target_sr=target_sr)
                                arr = np.asarray(arr, dtype=np.float32)
                        else:
                            import librosa as _librosa
                            arr, _ = _librosa.load(str(ap), sr=target_sr, mono=True)
                            arr = np.asarray(arr, dtype=np.float32)
                        audio_arrays.append(arr)
                    rows_by_audio: List[List[float]] = []
                    for audio_arr in audio_arrays:
                        if use_pkv_cache:
                            row_vals = _batch_dime_logit_same_audio_with_prefix_kv_inner(
                                model=model,
                                processor=processor,
                                audio_array=audio_arr,
                                target_sr=target_sr,
                                full_prompts=full_prompts,
                                target_id=target_id,
                            )
                        else:
                            row_vals = _batch_dime_logit_same_audio_inner(
                                model=model,
                                processor=processor,
                                audio_array=audio_arr,
                                target_sr=target_sr,
                                full_prompts=full_prompts,
                                target_id=target_id,
                            )
                        rows_by_audio.append([float(x) for x in row_vals])
                    num_prompts = len(full_prompts)
                    num_audios = len(audio_arrays)
                    cols: List[List[float]] = []
                    for j in range(num_prompts):
                        col_j = [float(rows_by_audio[i][j]) for i in range(num_audios)]
                        cols.append(col_j)

                finally:
                    for h in audio_handles:
                        _close_worker_audio_handle(h)
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
                    messages = _build_messages(audio_path, prompt)
                    _text, inputs = prepare_qwen25_omni_inputs(
                        processor=processor,
                        conversation=messages,
                        device=model.device,
                        dtype=getattr(model, "dtype", None),
                        use_audio_in_video=False,
                    )
                    T = len(target_ids)
                    vals = []
                    if T == 1:
                        # ── Fast path: single target token → 1 forward pass ──────────
                        # Nessuna generazione autoregressiva: solo il logit al last token.
                        # 3-5× più veloce di model.generate() per risposte mono-token
                        # (A/B/C/D in HumMusQA).
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
                        # ── Multi-token path: usa generate() come prima ───────────────
                        # Necessario per risposte con più token (raro in HumMusQA).
                        with torch.inference_mode():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max(1, T + 2),
                                return_dict_in_generate=True,
                                output_logits=True,
                                use_cache=False,
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

    def probe_step4a_audio_equivalence(
        self,
        file_audio_path: str,
        inline_audio: Dict[str, Any],
        prompt: str,
        caption_ids: List[int],
        token_index: int,
    ) -> Dict[str, float]:
        req_id = self._new_req_id()

        self._put_task(
            "dime_probe_audio_io",
            {
                "file_audio_path": file_audio_path,
                "inline_audio": inline_audio,
                "prompt": prompt,
                "caption_ids": list(caption_ids),
                "token_index": int(token_index),
            },
            req_id=req_id,
        )

        kind, payload = self._get_for_req(req_id)
        if kind != "dime_probe_audio_io":
            raise RuntimeError(f"Unexpected response kind for probe_step4a_audio_equivalence: {kind}")

        return {
            "file_value": float(payload["file_value"]),
            "inline_value": float(payload["inline_value"]),
        }

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
        audio_paths: List[Any],
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
    max_gpus: int = 8,
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