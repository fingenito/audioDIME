import os
import json
import tempfile
import logging
import librosa
import soundfile as sf
import numpy as np
import random
import sklearn.metrics
from typing import Any, Dict, List, Tuple, Optional

from .visualization import plot_dime_step6_paper_aligned

from .shared_utils import (
    tokenize_caption_for_mmshap,
    get_token_logit_autoregressive_id,  # Step0: pre-softmax logit (paper-aligned)
    build_word_groups_from_token_ids,
    aggregate_vector_by_groups,
    aggregate_matrix_rows_by_groups,
)

from .background_utils import (
    build_audio_background_set,
    build_text_background_set,
    build_paired_background_set,
)

from .masking_utils import mask_text_token_ids

from .audioLIME import (
    build_demucs_factorization_for_dime,
    make_audiolime_binary_masks,
    compose_from_binary_mask,
    get_factorization_metadata,
    LimeAudioExplainer,
)

logger = logging.getLogger("DIME-Analysis")

# ==========================
# DIME CONFIG
# ==========================
NUM_EXPECTATION_SAMPLES = int(os.environ.get("DIME_NUM_EXPECTATION_SAMPLES", "10"))
WINDOW_SIZE_SEC = 0.5

# Step 0: value function mode (paper-aligned: logits)
DIME_VALUE_MODE = os.environ.get("DIME_VALUE_MODE", "logit").strip().lower()

_FAST_DEBUG = os.environ.get("DIME_FAST_DEBUG", "0") in ("1", "true", "True")
if _FAST_DEBUG:
    NUM_LIME_SAMPLES = int(os.environ.get("DIME_NUM_LIME_SAMPLES", "64"))
    NUM_EXPECTATION_SAMPLES = int(os.environ.get("DIME_NUM_EXPECTATION_SAMPLES", "4"))

# Step 2 options
_DEFAULT_L_BATCH_SIZE = int(os.environ.get("DIME_L_BATCH_SIZE", "8"))
_SAVE_L_TABLE = os.environ.get("DIME_SAVE_L_TABLE", "1").lower() in ("1", "true", "yes")
_L_TABLE_FORMAT = os.environ.get("DIME_L_TABLE_FORMAT", "npy").strip().lower()  # "npy" or "json"

# Step 4/5 options (batching)
_LIME_BATCH_SIZE = int(os.environ.get("DIME_LIME_BATCH_SIZE", "16"))
_LIME_KERNEL_WIDTH = float(os.environ.get("DIME_LIME_KERNEL_WIDTH", "0.25"))
_LIME_L2 = float(os.environ.get("DIME_LIME_L2", "1e-2"))
_LIME_NUM_FEATURES_AUDIO = int(os.environ.get("DIME_LIME_NUM_FEATURES_AUDIO", "10"))
_LIME_NUM_FEATURES_TEXT = int(os.environ.get("DIME_LIME_NUM_FEATURES_TEXT", "10"))

# Step 5 practical: audio tempfile batching to reduce /dev/shm + disk pressure
_STEP5_AUDIO_PERTURB_BATCH = int(os.environ.get("DIME_STEP5_AUDIO_PERTURB_BATCH", str(_LIME_BATCH_SIZE)))
_STEP5_TEXT_PERTURB_BATCH = int(os.environ.get("DIME_STEP5_TEXT_PERTURB_BATCH", str(_LIME_BATCH_SIZE)))
_DIME_AUDIO_FEATURE_MODE = os.environ.get("DIME_AUDIO_FEATURE_MODE", "time").strip().lower()


# ======================================
# JSON SAFE DUMP
# ======================================
def _to_str_token(t: Any) -> str:
    if isinstance(t, (bytes, bytearray)):
        return t.decode("utf-8", errors="replace")
    if isinstance(t, np.generic):
        t = t.item()
    return str(t)


def _make_json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            safe_key = _to_str_token(k)
            safe[safe_key] = _make_json_safe(v)
        return safe
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]
    return str(obj)


def _atomic_json_dump(data: Any, path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safe_data = _make_json_safe(data)

    tmp_dir = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_dime_", suffix=".json", dir=tmp_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(safe_data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


# ==========================
# PATH NORMALIZATION (target-first)
# ==========================
def _abspath(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return str(p)


def _target_first_paths(target_path: str, paths: List[str]) -> List[str]:
    t_abs = _abspath(target_path)
    out: List[str] = [target_path]
    seen = {t_abs}

    for p in (paths or []):
        if not p:
            continue
        ap = _abspath(p)
        if ap == t_abs:
            continue
        if ap in seen:
            continue
        seen.add(ap)
        out.append(p)
    return out


def _target_first_prompts(target_prompt: str, prompts: List[str]) -> List[str]:
    out: List[str] = [target_prompt]
    seen = {target_prompt}

    for p in (prompts or []):
        if not isinstance(p, str) or not p.strip():
            continue
        if p == target_prompt:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


# ==========================
# SEEDS
# ==========================
def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        pass


# ==========================
# INVARIANTS (UC = Ea + Et - Eat; MI = base - UC)
# ==========================
def _check_dime_invariants(
    token: str,
    uc: float,
    mi: float,
    base: float,
    Ea: float,
    Et: float,
    Eat: float,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    raise_on_fail: bool = False,
) -> Dict[str, float]:
    uc_from_terms = Ea + Et - Eat
    mi_from_terms = base - uc
    base_from_terms = uc + mi

    res_uc = float(abs(uc - uc_from_terms))
    res_mi = float(abs(mi - mi_from_terms))
    res_base = float(abs(base - base_from_terms))

    thr_uc = float(atol + rtol * max(1.0, abs(uc_from_terms)))
    thr_mi = float(atol + rtol * max(1.0, abs(mi_from_terms)))
    thr_base = float(atol + rtol * max(1.0, abs(base)))

    ok = (res_uc <= thr_uc) and (res_mi <= thr_mi) and (res_base <= thr_base)

    if not ok:
        msg = (
            f"[DIME INVARIANTS FAIL] token={repr(token)} | "
            f"res_uc={res_uc:.3e} (thr {thr_uc:.3e}) | "
            f"res_mi={res_mi:.3e} (thr {thr_mi:.3e}) | "
            f"res_base={res_base:.3e} (thr {thr_base:.3e}) | "
            f"uc={uc:.6f}, mi={mi:.6f}, base={base:.6f}, Ea={Ea:.6f}, Et={Et:.6f}, Eat={Eat:.6f}"
        )
        if raise_on_fail:
            raise RuntimeError(msg)
        logger.warning(msg)

    return {
        "res_uc": res_uc,
        "res_mi": res_mi,
        "res_base": res_base,
        "thr_uc": thr_uc,
        "thr_mi": thr_mi,
        "thr_base": thr_base,
    }


# ==========================
# STEP 0 VALUE FUNCTION (paper-aligned: pre-softmax logit)
# ==========================
def dime_token_value(
    model,
    tokenizer,
    audio_path: str,
    prompt: str,
    caption_ids: List[int],
    token_index: int,
) -> float:
    """
    Paper-aligned: M returns pre-softmax logits.
    Here: scalar score for a fixed caption token t_k:
      M_k(audio,prompt) = logit(target_token_id | audio,prompt,prefix)
    """
    if DIME_VALUE_MODE != "logit":
        raise ValueError(f"DIME_VALUE_MODE must be 'logit' for paper alignment. Got: {DIME_VALUE_MODE}")

    target_id = int(caption_ids[token_index])
    prefix_ids = caption_ids[:token_index]

    return float(
        get_token_logit_autoregressive_id(
            model=model,
            tokenizer=tokenizer,
            audio_path=audio_path,
            prompt=prompt,
            prefix_ids=prefix_ids,
            target_id=target_id,
        )
    )


# ==========================
# STEP 2 — PRECOMPUTE L TABLE (NxN)
# ==========================
def _ensure_same_N(bg_audio_paths: List[str], bg_prompts: List[str]) -> Tuple[List[str], List[str], int]:
    na = len(bg_audio_paths or [])
    nt = len(bg_prompts or [])
    N = int(min(na, nt))
    if N <= 0:
        raise RuntimeError("Background sets are empty: cannot build L table.")
    if na != nt:
        logger.warning(f"[DIME Step2] Mismatch sizes: |A|={na}, |T|={nt}. Truncating both to N={N}.")
    return bg_audio_paths[:N], bg_prompts[:N], N


def compute_L_table_for_token_serial(
    model,
    tokenizer,
    caption_ids: List[int],
    token_index: int,
    bg_audio_paths: List[str],
    bg_prompts: List[str],
) -> np.ndarray:
    bg_audio_paths, bg_prompts, N = _ensure_same_N(bg_audio_paths, bg_prompts)
    L = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        ai = bg_audio_paths[i]
        for j in range(N):
            tj = bg_prompts[j]
            L[i, j] = float(dime_token_value(model, tokenizer, ai, tj, caption_ids, token_index))
    return L


def compute_L_table_for_token_parallel(
    runner,
    caption_ids: List[int],
    token_index: int,
    bg_audio_paths: List[str],
    bg_prompts: List[str],
    batch_size: int = _DEFAULT_L_BATCH_SIZE,
) -> np.ndarray:
    bg_audio_paths, bg_prompts, N = _ensure_same_N(bg_audio_paths, bg_prompts)
    L_list2d = runner.run_dime_L_table(
        bg_audio_paths=bg_audio_paths,
        bg_prompts=bg_prompts,
        caption_ids=caption_ids,
        token_index=int(token_index),
        batch_size=int(batch_size),
    )
    L = np.asarray(L_list2d, dtype=np.float32)
    if L.shape != (N, N):
        raise RuntimeError(f"Bad L shape from runner: got {L.shape}, expected {(N, N)}")
    return L


def reduce_L_to_ucmi(L: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Paper/repo aligned reduction with AVG including index 0.
    For target-first, explained datapoint index is k=0.
    """
    L = np.asarray(L, dtype=float)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square 2D, got {L.shape}")
    N = int(L.shape[0])
    if N < 1:
        raise ValueError("L is empty")

    k = 0
    base = float(L[k, k])
    avg_row = float(np.mean(L[k, :]))
    avg_col = float(np.mean(L[:, k]))
    avg_all = float(np.mean(L[:, :]))

    uc = float(avg_row + avg_col - avg_all)
    mi = float(base - uc)

    Ea = avg_col
    Et = avg_row
    Eat = avg_all
    return uc, mi, base, Ea, Et, Eat


def _save_L_table(results_dir: str, token_index: int, L: np.ndarray) -> Dict[str, Any]:
    out = {}
    ldir = os.path.join(results_dir, "L_tables")
    os.makedirs(ldir, exist_ok=True)

    if _L_TABLE_FORMAT == "json":
        path = os.path.join(ldir, f"L_token_{token_index:04d}.json")
        _atomic_json_dump({"token_index": int(token_index), "L": np.asarray(L, dtype=float)}, path, indent=2)
        out["format"] = "json"
        out["path"] = path
    else:
        path = os.path.join(ldir, f"L_token_{token_index:04d}.npy")
        np.save(path, np.asarray(L, dtype=np.float32))
        out["format"] = "npy"
        out["path"] = path
    return out


# ==========================
# LIME helpers (kernel + ridge)
# ==========================
def _cosine_distance_to_x0(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    x0 = X[0].astype(float)
    xv = X @ x0
    xnorm = np.linalg.norm(X, axis=1) + 1e-12
    x0norm = np.linalg.norm(x0) + 1e-12
    cos_sim = xv / (xnorm * x0norm)
    return 1.0 - cos_sim


def _lime_kernel(distances: np.ndarray, kernel_width: float = 0.25) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    return np.exp(-(d ** 2) / (kernel_width ** 2))


def _fit_weighted_ridge_intercept(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)

    X1 = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    w = np.clip(w, 0.0, np.inf)
    sw = np.sqrt(w + 1e-12)

    Xw = X1 * sw[:, None]
    yw = y * sw

    A = Xw.T @ Xw + l2 * np.eye(X1.shape[1])
    b = Xw.T @ yw
    return np.linalg.solve(A, b)


def _select_topk_features_weighted_corr(X: np.ndarray, y: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)

    _, F = X.shape
    if F == 0:
        return np.array([], dtype=int)
    k = int(max(1, min(k, F)))

    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 1e-12:
        w = np.ones_like(w, dtype=float)
        wsum = float(np.sum(w))

    y_mean = float(np.sum(w * y) / wsum)
    X_mean = np.sum(w[:, None] * X, axis=0) / wsum

    yc = y - y_mean
    Xc = X - X_mean[None, :]

    cov = np.sum((w * yc)[:, None] * Xc, axis=0)
    var_y = float(np.sum(w * (yc ** 2)) / wsum) + 1e-12
    var_X = (np.sum(w[:, None] * (Xc ** 2), axis=0) / wsum) + 1e-12

    corr = cov / (np.sqrt(var_X) * np.sqrt(var_y))
    score = np.abs(corr)

    idx = np.argpartition(score, -k)[-k:]
    idx = idx[np.argsort(score[idx])[::-1]]
    return idx.astype(int)

def _audiolime_kernel(distances: np.ndarray, kernel_width: float = 0.25) -> np.ndarray:
    """
    Kernel identico alla repo audioLIME:
        sqrt(exp(-(d^2) / kernel_width^2))
    """
    d = np.asarray(distances, dtype=float)
    return np.sqrt(np.exp(-(d ** 2) / (kernel_width ** 2)))


def _fit_audiolime_surrogate_from_binary_data(
    X: np.ndarray,
    y: np.ndarray,
    kernel_width: float,
    num_features: int,
    seed: int,
    absolute_feature_sort: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    Fit del surrogate locale audio EXACT-style audioLIME repo:
    - distanza cosine con sklearn
    - kernel sqrt(exp(...))
    - feature_selection='auto'
    - Ridge(alpha=1)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"Bad y shape {y.shape} for X shape {X.shape}")

    # audioLIME usa labels matriciale [num_samples, num_targets]
    y_mat = y.reshape(-1, 1)

    distances = sklearn.metrics.pairwise_distances(
        X,
        X[0].reshape(1, -1),
        metric="cosine",
    ).ravel()

    explainer = LimeAudioExplainer(
        kernel_width=float(kernel_width),
        verbose=False,
        feature_selection="auto",
        absolute_feature_sort=bool(absolute_feature_sort),
        random_state=int(seed),
    )

    intercept, local_exp, score, local_pred = explainer.base.explain_instance_with_data(
        neighborhood_data=X,
        neighborhood_labels=y_mat,
        distances=distances,
        label=0,
        num_features=int(num_features),
        feature_selection="auto",
        model_regressor=None,   # -> Ridge(alpha=1) come repo
        fit_intercept=True,
    )

    full_weights = np.zeros((X.shape[1],), dtype=float)
    selected_indices: List[int] = []

    for feat_idx, weight in local_exp:
        fi = int(feat_idx)
        full_weights[fi] = float(weight)
        selected_indices.append(fi)

    return (
        full_weights,
        np.asarray(selected_indices, dtype=int),
        float(score),
        np.asarray(local_pred, dtype=float).reshape(-1),
        float(intercept),
    )

# ==========================
# Audio masking (for Step4A)
# ==========================
def _masked_audio_to_tempfile(y_masked: np.ndarray, sr: int) -> str:
    tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    fd, path = tempfile.mkstemp(prefix="dime_", suffix=".wav", dir=tmp_dir)
    os.close(fd)
    sf.write(path, y_masked, sr)
    return path


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _apply_audio_mask_replace_with_background(
    y: np.ndarray,
    sr: int,
    am: np.ndarray,
    window_size: float,
    bg_audio_paths: List[str],
    seed: int = 0,
) -> np.ndarray:
    mode = os.environ.get("DIME_AUDIO_MASK_MODE", "bg_random_energy").strip().lower()
    silence_rms = float(os.environ.get("DIME_SILENCE_RMS", "1e-3"))

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y_masked = y.copy()

    am = np.asarray(am, dtype=int).reshape(-1)
    nwin = len(am)

    bg_candidates = [p for p in (bg_audio_paths or []) if p]
    rng = random.Random(int(seed))

    def _load_bg_once(path: str) -> np.ndarray:
        yb, _ = librosa.load(path, sr=sr)
        return np.asarray(yb, dtype=np.float32).reshape(-1)

    bg_audio = None
    if mode.startswith("bg") and len(bg_candidates) > 1:
        pool = bg_candidates[1:] if len(bg_candidates) > 1 else bg_candidates
        bg_path = rng.choice(pool) if pool else None
        if bg_path:
            try:
                bg_audio = _load_bg_once(bg_path)
            except Exception:
                bg_audio = None

    def _get_bg_segment(seg_len: int) -> np.ndarray:
        if bg_audio is None or bg_audio.size == 0:
            return np.zeros(seg_len, dtype=np.float32)
        if bg_audio.size <= seg_len:
            out = np.zeros(seg_len, dtype=np.float32)
            out[: bg_audio.size] = bg_audio
            return out
        start = rng.randrange(0, bg_audio.size - seg_len + 1)
        return bg_audio[start:start + seg_len].copy()

    for j in range(nwin):
        if int(am[j]) == 1:
            continue

        s0 = int(j * float(window_size) * sr)
        s1 = int(min((j + 1) * float(window_size) * sr, y_masked.size))
        if s0 >= s1 or s0 >= y_masked.size:
            continue

        seg_len = s1 - s0
        orig_seg = y[s0:s1]
        orig_rms = _rms(orig_seg)

        if mode == "zero":
            y_masked[s0:s1] = 0.0
            continue

        if mode == "noise_energy":
            rs = np.random.RandomState(int(seed) + 1009 * j)
            noise = rs.randn(seg_len).astype(np.float32)
            noise_rms = _rms(noise)
            if orig_rms <= silence_rms:
                y_masked[s0:s1] = 0.0
            else:
                scale = orig_rms / max(1e-12, noise_rms)
                y_masked[s0:s1] = noise * float(scale)
            continue

        if mode in ("bg_random", "bg_random_energy"):
            rep = _get_bg_segment(seg_len)
            if mode == "bg_random":
                y_masked[s0:s1] = rep
                continue
            rep_rms = _rms(rep)
            if orig_rms <= silence_rms:
                y_masked[s0:s1] = 0.0
            else:
                scale = orig_rms / max(1e-12, rep_rms)
                y_masked[s0:s1] = rep * float(scale)
            continue

        y_masked[s0:s1] = 0.0

    if os.environ.get("DIME_CROSSFADE", "1").lower() in ("1", "true", "yes"):
        xf_ms = float(os.environ.get("DIME_CROSSFADE_MS", "5.0"))
        xf = int(max(0, min(int(sr * xf_ms / 1000.0), int(0.1 * sr))))
        if xf > 0 and y_masked.size > 2 * xf:
            w = np.ones_like(y_masked, dtype=np.float32)
            ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
            w[:xf] *= ramp
            w[-xf:] *= ramp[::-1]
            y_masked = y_masked * w + y * (1.0 - w)

    return y_masked


# ==========================
# Step4: build modality-specific LIME masks
# ==========================
def _make_audio_masks(
    audio_path: str,
    token_index: int,
    seed: int,
    window_size: float,
    num_samples: int,
) -> Tuple[np.ndarray, List[List[int]], List[int], int, float, Optional[Dict[str, Any]]]:
    """
    Costruisce le maschere LIME audio per DIME.

    Modalità:
    - time: comportamento originale DIME
    - audiolime_demucs: feature interpretabili = source x temporal segment
    """
    mode = _DIME_AUDIO_FEATURE_MODE

    if mode == "audiolime_demucs":
        factorization = build_demucs_factorization_for_dime(audio_path=audio_path, sr=16000)
        n_audio = int(factorization.get_number_components())
        duration = float(len(load_audio_for_duration(audio_path)) / 16000.0)

        A, am_list, sample_idx_list = make_audiolime_binary_masks(
            n_components=n_audio,
            num_samples=int(num_samples),
            seed=int(seed),
            token_index=int(token_index),
        )

        meta = get_factorization_metadata(factorization)
        meta["mode"] = "audiolime_demucs"
        meta["factorization_obj"] = factorization
        return A, am_list, sample_idx_list, n_audio, duration, meta

    # fallback: time-only originale
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    n_audio = max(1, int(np.ceil(duration / float(window_size))))

    S = int(num_samples)
    rng = np.random.RandomState(int(seed) + 997 * int(token_index) + 11)

    A = rng.randint(0, 2, size=(S, n_audio)).astype(int)
    A[0, :] = 1
    am_list = [A[s, :].tolist() for s in range(S)]
    sample_idx_list = list(range(S))

    meta = {
        "mode": "time",
        "factorization_obj": None,
        "factorization_type": "TimeOnlyWindows",
        "n_components": int(n_audio),
        "component_names": [f"T{i}" for i in range(n_audio)],
        "n_temporal_segments": int(n_audio),
        "temporal_segments_samples": [],
        "target_sr": 16000,
    }
    return A, am_list, sample_idx_list, n_audio, float(duration), meta


def load_audio_for_duration(audio_path: str) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=16000)
    return np.asarray(y, dtype=np.float32)


def _make_text_masks(
    tokenizer,
    prompt: str,
    token_index: int,
    seed: int,
    num_samples: int,
) -> Tuple[np.ndarray, List[List[int]], List[int], List[str]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    raw_prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    prompt_tokens = [_to_str_token(t) for t in raw_prompt_tokens]
    n_text = max(1, len(prompt_tokens))

    S = int(num_samples)
    rng = np.random.RandomState(int(seed) + 997 * int(token_index) + 29)

    T = rng.randint(0, 2, size=(S, n_text)).astype(int)
    T[0, :] = 1  # x0 all-ones
    tm_list = [T[s, :].tolist() for s in range(S)]
    sample_idx_list = list(range(S))

    return T, tm_list, sample_idx_list, prompt_tokens


# ==========================
# Step4/5: efficient UC/MI via row/col update WITHOUT deepcopy
# ==========================
def _ucmi_from_row_update(
    L: np.ndarray,
    row0_new: np.ndarray,
    sum_L: float,
    sum_row0_old: float,
    sum_col0_old: float,
) -> Tuple[float, float, float, float, float, float]:
    L = np.asarray(L, dtype=float)
    row0_new = np.asarray(row0_new, dtype=float).reshape(-1)
    N = int(L.shape[0])

    base = float(row0_new[0])
    avg_row = float(np.mean(row0_new))

    col0_sum_new = float(sum_col0_old - float(L[0, 0]) + base)
    avg_col = col0_sum_new / float(N)

    all_sum_new = float(sum_L - sum_row0_old + float(np.sum(row0_new)))
    avg_all = all_sum_new / float(N * N)

    uc = float(avg_row + avg_col - avg_all)
    mi = float(base - uc)

    Ea = avg_col
    Et = avg_row
    Eat = avg_all
    return uc, mi, base, Ea, Et, Eat


def _ucmi_from_col_update(
    L: np.ndarray,
    col0_new: np.ndarray,
    sum_L: float,
    sum_row0_old: float,
    sum_col0_old: float,
) -> Tuple[float, float, float, float, float, float]:
    L = np.asarray(L, dtype=float)
    col0_new = np.asarray(col0_new, dtype=float).reshape(-1)
    N = int(L.shape[0])

    base = float(col0_new[0])

    row0_sum_new = float(sum_row0_old - float(L[0, 0]) + base)
    avg_row = row0_sum_new / float(N)

    avg_col = float(np.mean(col0_new))

    all_sum_new = float(sum_L - sum_col0_old + float(np.sum(col0_new)))
    avg_all = all_sum_new / float(N * N)

    uc = float(avg_row + avg_col - avg_all)
    mi = float(base - uc)

    Ea = avg_col
    Et = avg_row
    Eat = avg_all
    return uc, mi, base, Ea, Et, Eat


# ==========================
# MAIN ENTRY
# ==========================
def analyze_dime(
    model,
    tokenizer,
    audio_path: str,
    prompt: str,
    caption: str,
    results_dir: str,
    runner=None,
    background_audio_paths: Optional[List[str]] = None,
    background_prompts: Optional[List[str]] = None,
    seed: Optional[int] = None,
    check_invariants: bool = True,
    invariants_raise: bool = False,
    invariants_atol: float = 1e-6,
    invariants_rtol: float = 1e-5,
    num_lime_samples: int = 256,
    num_features: int = 256,
) -> Dict[str, Any]:
    import torch
    import gc

    print(f"\nAnalisi DIME per: {os.path.basename(audio_path)}")
    os.makedirs(results_dir, exist_ok=True)

    if seed is None:
        seed_env = os.environ.get("DIME_SEED", None)
        seed = int(seed_env) if seed_env is not None else 0

    deterministic = os.environ.get("DIME_DETERMINISTIC", "0") in ("1", "true", "True")
    set_global_seed(seed, deterministic_torch=deterministic)

    caption_ids, caption_tokens = tokenize_caption_for_mmshap(caption, tokenizer, return_ids=True)

    # ==========================
    # STEP 1: READ DEFAULTS HERE
    # ==========================
    default_bg_audio_dir = os.environ["DIME_BG_AUDIO_DIR"]
    default_bg_prompts_file = os.environ["DIME_PROMPTS_FILE"]

    # ==========================
    # STEP 1: BACKGROUND SETS (target-first)
    # ==========================
    default_bg_dataset_name = os.environ.get("DIME_BG_AUDIO_DATASET_NAME", "").strip()

    # ==========================================================
    # STEP 1: BACKGROUND SETS (PAIRED DATASET, paper-aligned)
    # ==========================================================
    if background_audio_paths is not None and background_prompts is not None:
        bg_audio_paths = _target_first_paths(audio_path, list(background_audio_paths))
        bg_prompts = _target_first_prompts(prompt, list(background_prompts))
        bg_audio_paths, bg_prompts, Nbg = _ensure_same_N(bg_audio_paths, bg_prompts)

        logger.info(
            f"[DIME BG] explicit background provided | "
            f"N={Nbg} | target_first=True | mode=explicit_lists"
        )
    else:
        bg_pairs = build_paired_background_set(
            target_audio_path=audio_path,
            target_prompt=prompt,
            background_audio_dir=default_bg_audio_dir,
            prompts_file=default_bg_prompts_file,
            k=int(NUM_EXPECTATION_SAMPLES),
            seed=int(seed),
            include_target_first=True,
            dataset_name=default_bg_dataset_name,
        )

        bg_audio_paths = [a for a, _p in bg_pairs]
        bg_prompts = [p for _a, p in bg_pairs]
        bg_audio_paths, bg_prompts, Nbg = _ensure_same_N(bg_audio_paths, bg_prompts)

        logger.info(
            f"[DIME BG] paired background set built | "
            f"dataset={default_bg_dataset_name or 'unknown'} | "
            f"audio_dir={default_bg_audio_dir} | prompts_file={default_bg_prompts_file} | "
            f"N={Nbg} | target_first=True | mode=paired_datapoints"
        )

    def _cleanup():
        # Nel main process non c'è il modello (sta nei worker). empty_cache è inutile e può rallentare.
        gc.collect()

    # ==========================================================
    # STEP 2: PRECOMPUTE L TABLES (one per caption token)  (NxN)
    # ==========================================================
    L_tables: Dict[int, np.ndarray] = {}
    L_cache_meta: Dict[str, Any] = {}
    ucmi_base: Dict[int, Dict[str, float]] = {}

    for k_tok in range(len(caption_tokens)):
        if runner is not None:
            L = compute_L_table_for_token_parallel(
                runner=runner,
                caption_ids=caption_ids,
                token_index=k_tok,
                bg_audio_paths=bg_audio_paths,
                bg_prompts=bg_prompts,
                batch_size=int(_DEFAULT_L_BATCH_SIZE),
            )
        else:
            with torch.inference_mode():
                L = compute_L_table_for_token_serial(
                    model=model,
                    tokenizer=tokenizer,
                    caption_ids=caption_ids,
                    token_index=k_tok,
                    bg_audio_paths=bg_audio_paths,
                    bg_prompts=bg_prompts,
                )

        L_tables[k_tok] = np.asarray(L, dtype=np.float32)

        uc, mi, base, Ea, Et, Eat = reduce_L_to_ucmi(L_tables[k_tok])
        ucmi_base[k_tok] = {
            "uc": float(uc),
            "mi": float(mi),
            "base": float(base),
            "Ea": float(Ea),
            "Et": float(Et),
            "Eat": float(Eat),
        }

        if _SAVE_L_TABLE:
            try:
                L_cache_meta[str(k_tok)] = _save_L_table(results_dir, k_tok, L_tables[k_tok])
            except Exception as e:
                logger.warning(f"[DIME Step2] Could not save L table for token {k_tok}: {e}")

        _cleanup()

    # ==========================================================
    # STEP 4 + STEP 5
    # ==========================================================
    explanations: Dict[str, Any] = {}
    invariants_report: Dict[str, Any] = {}

    y0_full, sr0 = librosa.load(audio_path, sr=16000)

    for k_tok, cap_tok in enumerate(caption_tokens):
        key = str(k_tok)
        L = np.asarray(L_tables[k_tok], dtype=float)
        N = int(L.shape[0])

        sum_L = float(np.sum(L))
        sum_row0_old = float(np.sum(L[0, :]))
        sum_col0_old = float(np.sum(L[:, 0]))

        # -----------------------
        # 4A/5A) LIME on audio only
        # -----------------------
        X_a, am_list, sidx_a, n_audio, duration, audio_lime_meta = _make_audio_masks(
            audio_path=audio_path,
            token_index=k_tok,
            seed=int(seed),
            window_size=float(WINDOW_SIZE_SEC),
            num_samples=int(num_lime_samples),
        )

        audio_factorization = audio_lime_meta.get("factorization_obj", None)

        y_uc1 = np.zeros((len(sidx_a),), dtype=float)
        y_mi1 = np.zeros((len(sidx_a),), dtype=float)

        batchA = max(1, int(_STEP5_AUDIO_PERTURB_BATCH))
        for b0 in range(0, len(sidx_a), batchA):
            b1 = min(len(sidx_a), b0 + batchA)
            batch_indices = sidx_a[b0:b1]

            pert_audio_paths: List[str] = []
            try:
                for s in batch_indices:
                    am = np.asarray(am_list[s], dtype=int)

                    if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs":
                        y_masked = compose_from_binary_mask(audio_factorization, am)
                    else:
                        y_masked = _apply_audio_mask_replace_with_background(
                            y=y0_full,
                            sr=sr0,
                            am=am,
                            window_size=float(WINDOW_SIZE_SEC),
                            bg_audio_paths=bg_audio_paths,
                            seed=int(seed) + 31 * int(k_tok) + 101 * int(s),
                        )

                    pert_audio_paths.append(_masked_audio_to_tempfile(y_masked, sr0))

                if runner is not None:
                    logger.info(
                        f"[DIME Step4A] batch_indices={len(batch_indices)} pert_audio_paths={len(pert_audio_paths)} "
                        f"batchA={batchA} _STEP5_AUDIO_PERTURB_BATCH={_STEP5_AUDIO_PERTURB_BATCH} "
                        f"_LIME_BATCH_SIZE={_LIME_BATCH_SIZE}"
                    )                                                                       ##controllo gpu non in parallelo
                    rows = runner.run_dime_row_values(
                        audio_paths=pert_audio_paths,
                        prompts=bg_prompts,
                        caption_ids=caption_ids,
                        token_index=int(k_tok),
                        batch_size=int(_LIME_BATCH_SIZE),
                    )
                    rows_np = [np.asarray(r, dtype=float) for r in rows]
                else:
                    rows_np = []
                    with torch.inference_mode():
                        for ap in pert_audio_paths:
                            row = [
                                float(dime_token_value(model, tokenizer, ap, p, caption_ids, k_tok))
                                for p in bg_prompts
                            ]
                            rows_np.append(np.asarray(row, dtype=float))

                for local_i, s in enumerate(batch_indices):
                    uc_s, mi_s, _, _, _, _ = _ucmi_from_row_update(
                        L=L,
                        row0_new=rows_np[local_i],
                        sum_L=sum_L,
                        sum_row0_old=sum_row0_old,
                        sum_col0_old=sum_col0_old,
                    )
                    y_uc1[s] = float(uc_s)
                    y_mi1[s] = float(mi_s)

            finally:
                for p in pert_audio_paths:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

            _cleanup()

        k_audio = int(max(1, min(int(_LIME_NUM_FEATURES_AUDIO), int(n_audio))))

        if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs":
            (
                w_uc1_full,
                sel_uc1,
                uc1_score,
                uc1_local_pred,
                uc1_intercept,
            ) = _fit_audiolime_surrogate_from_binary_data(
                X=X_a,
                y=y_uc1,
                kernel_width=float(_LIME_KERNEL_WIDTH),
                num_features=int(k_audio),
                seed=int(seed) + 1000 * int(k_tok) + 1,
                absolute_feature_sort=False,
            )

            (
                w_mi1_full,
                sel_mi1,
                mi1_score,
                mi1_local_pred,
                mi1_intercept,
            ) = _fit_audiolime_surrogate_from_binary_data(
                X=X_a,
                y=y_mi1,
                kernel_width=float(_LIME_KERNEL_WIDTH),
                num_features=int(k_audio),
                seed=int(seed) + 1000 * int(k_tok) + 2,
                absolute_feature_sort=False,
            )

        else:
            distances_a = _cosine_distance_to_x0(X_a)
            w_a = _lime_kernel(distances_a, kernel_width=float(_LIME_KERNEL_WIDTH))

            sel_uc1 = _select_topk_features_weighted_corr(X_a, y_uc1, w_a, k=k_audio)
            sel_mi1 = _select_topk_features_weighted_corr(X_a, y_mi1, w_a, k=k_audio)

            beta_uc1 = _fit_weighted_ridge_intercept(X_a[:, sel_uc1], y_uc1, w_a, l2=float(_LIME_L2))
            beta_mi1 = _fit_weighted_ridge_intercept(X_a[:, sel_mi1], y_mi1, w_a, l2=float(_LIME_L2))

            w_uc1_full = np.zeros((n_audio,), dtype=float)
            w_mi1_full = np.zeros((n_audio,), dtype=float)
            w_uc1_full[sel_uc1] = beta_uc1[1:]
            w_mi1_full[sel_mi1] = beta_mi1[1:]

            uc1_score = None
            uc1_local_pred = None
            uc1_intercept = None
            mi1_score = None
            mi1_local_pred = None
            mi1_intercept = None

        _cleanup()

        # -----------------------
        # 4B/5B) LIME on text only
        # -----------------------
        X_t, tm_list, sidx_t, prompt_tokens = _make_text_masks(
            tokenizer=tokenizer,
            prompt=prompt,
            token_index=k_tok,
            seed=int(seed),
            num_samples=int(num_lime_samples),
        )
        n_text = int(X_t.shape[1])

        y_uc2 = np.zeros((len(sidx_t),), dtype=float)
        y_mi2 = np.zeros((len(sidx_t),), dtype=float)

        batchT = max(1, int(_STEP5_TEXT_PERTURB_BATCH))
        for b0 in range(0, len(sidx_t), batchT):
            b1 = min(len(sidx_t), b0 + batchT)
            batch_indices = sidx_t[b0:b1]

            pert_prompts: List[str] = []
            for s in batch_indices:
                tm = np.asarray(tm_list[s], dtype=int).reshape(-1)
                mask_idx = [int(i) for i in range(len(tm)) if int(tm[i]) == 0]
                masked_prompt, _orig_ids, _masked_ids = mask_text_token_ids(tokenizer, prompt, mask_idx)
                pert_prompts.append(masked_prompt)

            if runner is not None:
                cols = runner.run_dime_col_values(
                    audio_paths=bg_audio_paths,
                    prompts=pert_prompts,
                    caption_ids=caption_ids,
                    token_index=int(k_tok),
                    batch_size=int(_LIME_BATCH_SIZE),
                )
                cols_np = [np.asarray(c, dtype=float) for c in cols]
            else:
                cols_np = []
                with torch.inference_mode():
                    for tp in pert_prompts:
                        col = [
                            float(dime_token_value(model, tokenizer, ap, tp, caption_ids, k_tok))
                            for ap in bg_audio_paths
                        ]
                        cols_np.append(np.asarray(col, dtype=float))

            for local_i, s in enumerate(batch_indices):
                uc_s, mi_s, _, _, _, _ = _ucmi_from_col_update(
                    L=L,
                    col0_new=cols_np[local_i],
                    sum_L=sum_L,
                    sum_row0_old=sum_row0_old,
                    sum_col0_old=sum_col0_old,
                )
                y_uc2[s] = float(uc_s)
                y_mi2[s] = float(mi_s)

            _cleanup()

        distances_t = _cosine_distance_to_x0(X_t)
        w_t = _lime_kernel(distances_t, kernel_width=float(_LIME_KERNEL_WIDTH))

        k_text = int(max(1, min(int(_LIME_NUM_FEATURES_TEXT), int(n_text))))
        sel_uc2 = _select_topk_features_weighted_corr(X_t, y_uc2, w_t, k=k_text)
        sel_mi2 = _select_topk_features_weighted_corr(X_t, y_mi2, w_t, k=k_text)

        beta_uc2 = _fit_weighted_ridge_intercept(X_t[:, sel_uc2], y_uc2, w_t, l2=float(_LIME_L2))
        beta_mi2 = _fit_weighted_ridge_intercept(X_t[:, sel_mi2], y_mi2, w_t, l2=float(_LIME_L2))

        w_uc2_full = np.zeros((n_text,), dtype=float)
        w_mi2_full = np.zeros((n_text,), dtype=float)
        w_uc2_full[sel_uc2] = beta_uc2[1:]
        w_mi2_full[sel_mi2] = beta_mi2[1:]

        _cleanup()

        # -----------------------
        # base UC/MI (Step3) + invariants
        # -----------------------
        base_stats = ucmi_base[k_tok]
        if check_invariants:
            inv = _check_dime_invariants(
                token=str(cap_tok),
                uc=float(base_stats["uc"]),
                mi=float(base_stats["mi"]),
                base=float(base_stats["base"]),
                Ea=float(base_stats["Ea"]),
                Et=float(base_stats["Et"]),
                Eat=float(base_stats["Eat"]),
                atol=float(invariants_atol),
                rtol=float(invariants_rtol),
                raise_on_fail=bool(invariants_raise),
            )
            invariants_report[key] = {"token_index": int(k_tok), "token": str(cap_tok), **inv}

        explanations[key] = {
            "token_index": int(k_tok),
            "token": str(cap_tok),
            "base_ucmi": {
                "uc": float(base_stats["uc"]),
                "mi": float(base_stats["mi"]),
                "base": float(base_stats["base"]),
                "Ea": float(base_stats["Ea"]),
                "Et": float(base_stats["Et"]),
                "Eat": float(base_stats["Eat"]),
            },
            "uc1_audio": {
                "weights": w_uc1_full.tolist(),
                "n_audio_windows": int(n_audio),
                "window_size": float(WINDOW_SIZE_SEC),
                "duration_sec": float(duration),
                "feature_mode": str(audio_lime_meta.get("mode", "time")),
                "factorization_type": str(audio_lime_meta.get("factorization_type", "TimeOnlyWindows")),
                "component_names": [str(x) for x in audio_lime_meta.get("component_names", [])],
                "n_temporal_segments": int(audio_lime_meta.get("n_temporal_segments", 0)),
                "temporal_segments_samples": audio_lime_meta.get("temporal_segments_samples", []),
                "kernel": {
                    "type": "audiolime_repo_sqrt_exp_cosine" if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs" else "cosine_to_x0",
                    "kernel_width": float(_LIME_KERNEL_WIDTH),},
                "ridge": {
                    "alpha": 1.0 if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs" else None,
                    "l2": None if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs" else float(_LIME_L2),},
                "feature_selection": {
                    "method": "auto" if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs" else "topk_weighted_corr",
                    "k": int(k_audio),
                    "selected_indices": sel_uc1.tolist(),
                    "full_dim": int(n_audio),},
                "surrogate_fit": {
                    "score_r2": None if uc1_score is None else float(uc1_score),
                    "local_pred": None if uc1_local_pred is None else np.asarray(uc1_local_pred, dtype=float).tolist(),
                    "intercept": None if uc1_intercept is None else float(uc1_intercept),
                 },
            },
            "mi1_audio": {
                "weights": w_mi1_full.tolist(),
                "n_audio_windows": int(n_audio),
                "window_size": float(WINDOW_SIZE_SEC),
                "duration_sec": float(duration),
                "feature_mode": str(audio_lime_meta.get("mode", "time")),
                "factorization_type": str(audio_lime_meta.get("factorization_type", "TimeOnlyWindows")),
                "component_names": [str(x) for x in audio_lime_meta.get("component_names", [])],
                "n_temporal_segments": int(audio_lime_meta.get("n_temporal_segments", 0)),
                "temporal_segments_samples": audio_lime_meta.get("temporal_segments_samples", []),
                "kernel": {
                    "type": "audiolime_repo_sqrt_exp_cosine" if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs" else "cosine_to_x0",
                    "kernel_width": float(_LIME_KERNEL_WIDTH),
                },
                "ridge": {
                    "alpha": 1.0 if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs" else None,
                    "l2": None if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs" else float(_LIME_L2),
                },
                "feature_selection": {
                    "method": "auto" if _DIME_AUDIO_FEATURE_MODE == "audiolime_demucs" else "topk_weighted_corr",
                    "k": int(k_audio),
                    "selected_indices": sel_mi1.tolist(),
                    "full_dim": int(n_audio),
                },
                "surrogate_fit": {
                    "score_r2": None if mi1_score is None else float(mi1_score),
                    "local_pred": None if mi1_local_pred is None else np.asarray(mi1_local_pred, dtype=float).tolist(),
                    "intercept": None if mi1_intercept is None else float(mi1_intercept),},
            },
            "uc2_text": {
                "weights": w_uc2_full.tolist(),
                "prompt_tokens": prompt_tokens,
                "n_text_tokens": int(n_text),
                "kernel": {"type": "cosine_to_x0", "kernel_width": float(_LIME_KERNEL_WIDTH)},
                "ridge": {"l2": float(_LIME_L2)},
                "feature_selection": {
                    "method": "topk_weighted_corr",
                    "k": int(k_text),
                    "selected_indices": sel_uc2.tolist(),
                    "full_dim": int(n_text),
                },
            },
            "mi2_text": {
                "weights": w_mi2_full.tolist(),
                "prompt_tokens": prompt_tokens,
                "n_text_tokens": int(n_text),
                "kernel": {"type": "cosine_to_x0", "kernel_width": float(_LIME_KERNEL_WIDTH)},
                "ridge": {"l2": float(_LIME_L2)},
                "feature_selection": {
                    "method": "topk_weighted_corr",
                    "k": int(k_text),
                    "selected_indices": sel_mi2.tolist(),
                    "full_dim": int(n_text),
                },
            },
        }

    # ==========================================================
    # Plot (WORD-LEVEL display supported, without changing DIME core)
    # ==========================================================
    plot_path = os.path.join(
        results_dir,
        f"{os.path.splitext(os.path.basename(audio_path))[0]}_dime_step4_separate.png",
    )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    plot_info = plot_dime_step6_paper_aligned(
        audio_path=audio_path,
        caption=caption,
        caption_tokens=[str(t) for t in caption_tokens],
        prompt=prompt,
        explanations=explanations,
        output_path=plot_path,
        window_size=float(WINDOW_SIZE_SEC),
        tokenizer=tokenizer,
        caption_ids=caption_ids,
        prompt_ids=prompt_ids,
        waveform_output_path=os.path.join(
            results_dir,
            f"{os.path.splitext(os.path.basename(audio_path))[0]}_dime_step4_wave_strip.png",
        ),
    )
    wave_strip_path = str(plot_info.get("wave_strip_path", ""))

    json_path = os.path.join(
        results_dir,
        f"{os.path.splitext(os.path.basename(audio_path))[0]}_dime_results_step4.json",
    )

    # ==========================================================
    # Word-level aggregation (POSTPROCESS ONLY — does not change DIME core)
    # ==========================================================
    do_wordlevel = os.environ.get("DIME_SAVE_WORDLEVEL", "1").lower() in ("1", "true", "yes")
    wordlevel = None

    if do_wordlevel:
        caption_groups = build_word_groups_from_token_ids(
            tokenizer=tokenizer,
            token_ids=caption_ids,
            drop_empty=True,
        )
        prompt_groups = build_word_groups_from_token_ids(
            tokenizer=tokenizer,
            token_ids=prompt_ids,
            drop_empty=True,
        )

        caption_words = [g.get("label", "") for g in caption_groups]
        prompt_words = [g.get("label", "") for g in prompt_groups]

        K = len(caption_ids)
        first_key = "0" if "0" in explanations else None
        if first_key is not None:
            n_audio = int(explanations[first_key]["uc1_audio"]["n_audio_windows"])
            n_text_tok = int(explanations[first_key]["uc2_text"]["n_text_tokens"])
            prompt_tokens_tok = explanations[first_key]["uc2_text"].get("prompt_tokens", None)
        else:
            n_audio = 0
            n_text_tok = 0
            prompt_tokens_tok = None

        UC1_A = np.zeros((K, n_audio), dtype=float) if n_audio > 0 else np.zeros((K, 0), dtype=float)
        MI1_A = np.zeros((K, n_audio), dtype=float) if n_audio > 0 else np.zeros((K, 0), dtype=float)
        UC2_T = np.zeros((K, n_text_tok), dtype=float) if n_text_tok > 0 else np.zeros((K, 0), dtype=float)
        MI2_T = np.zeros((K, n_text_tok), dtype=float) if n_text_tok > 0 else np.zeros((K, 0), dtype=float)

        for k in range(K):
            dk = explanations.get(str(k), None)
            if not dk:
                continue
            if n_audio > 0:
                UC1_A[k, :] = np.asarray(dk["uc1_audio"]["weights"], dtype=float)[:n_audio]
                MI1_A[k, :] = np.asarray(dk["mi1_audio"]["weights"], dtype=float)[:n_audio]
            if n_text_tok > 0:
                UC2_T[k, :] = np.asarray(dk["uc2_text"]["weights"], dtype=float)[:n_text_tok]
                MI2_T[k, :] = np.asarray(dk["mi2_text"]["weights"], dtype=float)[:n_text_tok]

        UC1_A_word = np.asarray(aggregate_matrix_rows_by_groups(UC1_A.tolist(), caption_groups), dtype=float)
        MI1_A_word = np.asarray(aggregate_matrix_rows_by_groups(MI1_A.tolist(), caption_groups), dtype=float)

        uc2_global_tok = np.sum(UC2_T, axis=0) if UC2_T.size else np.zeros((0,), dtype=float)
        mi2_global_tok = np.sum(MI2_T, axis=0) if MI2_T.size else np.zeros((0,), dtype=float)

        uc2_global_word = aggregate_vector_by_groups(uc2_global_tok.tolist(), prompt_groups)
        mi2_global_word = aggregate_vector_by_groups(mi2_global_tok.tolist(), prompt_groups)

        wordlevel = {
            "caption": {
                "token_ids": [int(x) for x in caption_ids],
                "tokens": [str(t) for t in caption_tokens],
                "groups": caption_groups,
                "words": caption_words,
                "uc1_audio_word_matrix": UC1_A_word.tolist(),
                "mi1_audio_word_matrix": MI1_A_word.tolist(),
            },
            "prompt": {
                "token_ids": [int(x) for x in prompt_ids],
                "tokens_from_explanations": [str(x) for x in (prompt_tokens_tok or [])],
                "groups": prompt_groups,
                "words": prompt_words,
                "uc2_global_word": uc2_global_word,
                "mi2_global_word": mi2_global_word,
            },
        }

    results = {
        "audio_file": os.path.basename(audio_path),
        "caption": caption,
        "prompt": prompt,
        "all_tokens": [str(t) for t in caption_tokens],
        "plot_path": plot_path,
        "wave_strip_path": wave_strip_path,
        "json_path": json_path,

        "background": {
            "audio_dir": str(default_bg_audio_dir),
            "audio_dataset_name": str(default_bg_dataset_name),
            "prompts_file": str(default_bg_prompts_file),
            "N": int(Nbg),
            "audio_target_first": True,
            "prompt_target_first": True,
            "sampling_mode": "paired_datapoints" if (background_audio_paths is None and background_prompts is None) else "explicit_lists",
            "bg_audio_paths": bg_audio_paths,
            "bg_prompts": bg_prompts,
            "bg_pairs": [[str(a), str(p)] for a, p in zip(bg_audio_paths, bg_prompts)],
        },

        "step2_L_cache": L_cache_meta,
        "invariants_report": invariants_report,

        "explanations": explanations,  # TOKEN-LEVEL CORE (unchanged)
        "wordlevel": wordlevel,        # WORD-LEVEL VIEW (new)

        "config": {
            "DIME_VALUE_MODE": str(DIME_VALUE_MODE),

            "STEP2_L_TABLE": True,
            "L_BATCH_SIZE": int(_DEFAULT_L_BATCH_SIZE),
            "SAVE_L_TABLE": bool(_SAVE_L_TABLE),
            "L_TABLE_FORMAT": str(_L_TABLE_FORMAT),

            "STEP4_SEPARATE_LIME": True,

            "STEP5_STREAMING_BATCHED": True,
            "STEP5_AUDIO_PERTURB_BATCH": int(_STEP5_AUDIO_PERTURB_BATCH),
            "STEP5_TEXT_PERTURB_BATCH": int(_STEP5_TEXT_PERTURB_BATCH),

            "NUM_LIME_SAMPLES": int(num_lime_samples),
            "WINDOW_SIZE_SEC": float(WINDOW_SIZE_SEC),
            "LIME_KERNEL_WIDTH": float(_LIME_KERNEL_WIDTH),
            "LIME_L2": float(_LIME_L2),
            "LIME_NUM_FEATURES_AUDIO": int(_LIME_NUM_FEATURES_AUDIO),
            "LIME_NUM_FEATURES_TEXT": int(_LIME_NUM_FEATURES_TEXT),

            "NUM_EXPECTATION_SAMPLES": int(NUM_EXPECTATION_SAMPLES),
            "FAST_DEBUG": bool(_FAST_DEBUG),
            "SEED": int(seed),
            "DETERMINISTIC": bool(deterministic),
            "PARALLEL_RUNNER": bool(runner is not None),

            "DIME_AUDIO_FEATURE_MODE": str(_DIME_AUDIO_FEATURE_MODE),
            "DIME_AUDIOLIME_DEMUCS_MODEL": os.environ.get("DIME_AUDIOLIME_DEMUCS_MODEL", "htdemucs"),
            "DIME_AUDIOLIME_USE_PRECOMPUTED": os.environ.get("DIME_AUDIOLIME_USE_PRECOMPUTED", "0"),
            "DIME_AUDIOLIME_PRECOMPUTED_DIR": os.environ.get("DIME_AUDIOLIME_PRECOMPUTED_DIR", ""),
            "DIME_AUDIOLIME_RECOMPUTE": os.environ.get("DIME_AUDIOLIME_RECOMPUTE", "0"),
            "DIME_AUDIOLIME_DEMUCS_DEVICE": os.environ.get("DIME_AUDIOLIME_DEMUCS_DEVICE", ""),
            "DIME_AUDIOLIME_DEMUCS_SEGMENT": os.environ.get("DIME_AUDIOLIME_DEMUCS_SEGMENT", ""),
            "DIME_AUDIOLIME_DEMUCS_SHIFTS": os.environ.get("DIME_AUDIOLIME_DEMUCS_SHIFTS", "0"),
            "DIME_AUDIOLIME_DEMUCS_SPLIT": os.environ.get("DIME_AUDIOLIME_DEMUCS_SPLIT", "1"),
            "DIME_AUDIOLIME_DEMUCS_OVERLAP": os.environ.get("DIME_AUDIOLIME_DEMUCS_OVERLAP", "0.25"),
            "DIME_AUDIOLIME_DEMUCS_JOBS": os.environ.get("DIME_AUDIOLIME_DEMUCS_JOBS", "0"),
            "DIME_AUDIOLIME_DEMUCS_PROGRESS": os.environ.get("DIME_AUDIOLIME_DEMUCS_PROGRESS", "0"),
            "DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS": os.environ.get("DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS", ""),
        },
    }

    _atomic_json_dump(results, json_path, indent=2)

    try:
        meta_path = os.path.join(results_dir, "meta_run.json")
        meta = {
            "audio_path": str(audio_path),
            "audio_file": os.path.basename(audio_path),
            "prompt": str(prompt),
            "caption": str(caption),
            "results_dir": str(results_dir),
            "json_path": str(json_path),
            "plot_path": str(plot_path),
            "num_lime_samples": int(num_lime_samples),
            "window_size_sec": float(WINDOW_SIZE_SEC),
            "num_expectation_samples": int(NUM_EXPECTATION_SAMPLES),
            "seed": int(seed),
            "value_mode": str(DIME_VALUE_MODE),
            "bg_audio_dir": str(default_bg_audio_dir),
            "prompts_file": str(default_bg_prompts_file),
            "N_bg": int(Nbg),
        }
        _atomic_json_dump(meta, meta_path, indent=2)
    except Exception as e:
        logger.warning(f"[DIME] Could not write meta_run.json: {e}")

    return results