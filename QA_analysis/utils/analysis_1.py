import os
import json
import re
import tempfile
from typing import Dict, List, Optional, Any
import librosa
import numpy as np
import shap

import shap.utils.transformers as shap_tf_utils

def _patched_is_transformers_lm(model):
    return False

shap_tf_utils.is_transformers_lm = _patched_is_transformers_lm

import soundfile as sf
import torch

from QA_analysis.utils.visualization import plot_token_contributions
from QA_analysis.utils.shared_utils import (
    build_hummusqa_qwen25_prompt,
    extract_only_question_text_span,
)
from QA_analysis.utils.mmshap_utils import (
    compute_mm_score,
    normalize_mmshap_values_array,
)

# ======================================================================================
# CONSTANTS
# ======================================================================================

TEXT_MASK_LITERAL = "[MASK]"
TEXT_MASK_SENTINEL = -99999991  # internal sentinel, never fed directly to the model

# m = number of random permutations used to approximate Shapley values
MM_SHAP_NUM_PERMUTATIONS = 30

def _get_mmshap_parallel_window(runner) -> int:
    """
    Finestra di richieste concorrenti per MM-SHAP.
    Non cambia la logica dell'analisi: cambia solo quante inference
    lasciamo in coda contemporaneamente ai worker.
    """
    env_v = os.environ.get("MMSHAP_QUEUE_WINDOW", "").strip()
    if env_v:
        try:
            return max(1, int(env_v))
        except Exception:
            pass

    n_workers = max(1, len(getattr(runner, "procs", []) or []))
    return max(1, n_workers * 4)

# ======================================================================================
# LOW-LEVEL AUDIO MASKING
# ======================================================================================

def _tmp_wav_path(prefix: str = "mmshap_", dir_prefer: str = "/dev/shm") -> str:
    tmp_dir = dir_prefer if os.path.isdir(dir_prefer) else None
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=tmp_dir)
    os.close(fd)
    return path


# ======================================================================================
# MODEL HELPERS (Qwen2.5-Omni)
# ======================================================================================

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


def _trim_generated_ids_at_first_im_end(gen_ids_1d: torch.Tensor, processor) -> torch.Tensor:
    """
    Keep only assistant answer tokens up to the first <|im_end|>.
    """
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


def _stack_generation_logits(outputs_logits):
    """
    Convert HF generate(..., output_logits=True).logits into [T, vocab].
    """
    if isinstance(outputs_logits, tuple):
        return torch.stack([x[0] for x in outputs_logits], dim=0)
    if isinstance(outputs_logits, list):
        return torch.stack([x[0] for x in outputs_logits], dim=0)
    if hasattr(outputs_logits, "dim") and outputs_logits.dim() == 3:
        return outputs_logits[0]
    return outputs_logits


# ======================================================================================
# TEXT RECONSTRUCTION FOR LITERAL [MASK] TOKEN-LEVEL PERTURBATIONS
# ======================================================================================

def _decode_no_cleanup(tokenizer, ids: List[int]) -> str:
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=False)


def _rebuild_question_text_with_literal_masks(
    tokenizer,
    original_question_token_ids: List[int],
    masked_question_token_ids: List[int],
    text_mask_sentinel: int = TEXT_MASK_SENTINEL,
    mask_literal: str = TEXT_MASK_LITERAL,
) -> str:
    """
    Rebuild the question text token-by-token, replacing masked token positions
    with the literal string [MASK].

    Important:
    - features remain token-level
    - masked input shown to the model contains literal [MASK]
    - reconstruction uses incremental decode deltas for robustness
    """
    if len(original_question_token_ids) != len(masked_question_token_ids):
        raise ValueError(
            "original_question_token_ids and masked_question_token_ids must have the same length"
        )

    pieces: List[str] = []

    prev_text = _decode_no_cleanup(tokenizer, [])
    for i, orig_tid in enumerate(original_question_token_ids):
        cur_text = _decode_no_cleanup(tokenizer, original_question_token_ids[: i + 1])

        if cur_text.startswith(prev_text):
            delta = cur_text[len(prev_text):]
        else:
            delta = _decode_no_cleanup(tokenizer, [orig_tid])

        prev_text = cur_text

        if int(masked_question_token_ids[i]) == int(text_mask_sentinel):
            # preserve leading whitespace contributed by the original token
            m = re.match(r"^(\s*)", delta)
            leading_ws = m.group(1) if m else ""
            pieces.append(f"{leading_ws}{mask_literal}")
        else:
            pieces.append(delta)

    return "".join(pieces)


# ======================================================================================
# MM-SHAP QA CORE
# ======================================================================================

def compute_mmshap_qa_sample(
    runner,
    processor,
    audio_path: str,
    question: str,
    options: List[str],
    correct_answer: str,
    results_dir: str,
    sample_id: str,
    shared_baseline: Optional[Dict] = None,
):
    """
    MM-SHAP for HumMusQA + Qwen2.5-Omni
    Runner-only version: no local model, no local generate, no fallback.

    FIX CRITICO:
    - SHAP lavora su input 2D: [n_samples, n_features]
    - NON usiamo più X con shape [1, 1, n_features]
    - get_prediction e token_masker sono allineati alla repo ufficiale MM-SHAP
    """
    if runner is None:
        raise RuntimeError("MM-SHAP requires runner. Local model is not allowed.")

    os.makedirs(results_dir, exist_ok=True)

    prompt = build_hummusqa_qwen25_prompt(question, options)

    # --------------------------------------------------
    # 1) Baseline answer (shared, runner-only)
    # --------------------------------------------------
    if shared_baseline is None:
        raise RuntimeError("MM-SHAP requires shared_baseline from runner.")

    shared_prompt = str(shared_baseline.get("prompt", ""))
    shared_audio_path = str(shared_baseline.get("audio_path", ""))

    if shared_prompt != prompt:
        raise RuntimeError(
            "Shared baseline prompt mismatch in MM-SHAP. "
            f"expected={prompt!r} | got={shared_prompt!r}"
        )

    if os.path.abspath(shared_audio_path) != os.path.abspath(audio_path):
        raise RuntimeError(
            "Shared baseline audio_path mismatch in MM-SHAP. "
            f"expected={os.path.abspath(audio_path)!r} | got={os.path.abspath(shared_audio_path)!r}"
        )

    baseline_answer = str(shared_baseline.get("baseline_answer", "")).strip()
    if not baseline_answer:
        raise RuntimeError("Empty baseline answer in shared_baseline.")

    target_ids = processor.tokenizer.encode(
        baseline_answer,
        add_special_tokens=False,
    )
    if len(target_ids) == 0:
        raise RuntimeError("Tokenization of baseline_answer is empty.")

    # --------------------------------------------------
    # 2) Question-only text features
    # --------------------------------------------------
    question_token_ids_list = processor.tokenizer.encode(
        question,
        add_special_tokens=False,
    )
    n_question_tokens = len(question_token_ids_list)

    if n_question_tokens <= 0:
        raise RuntimeError(f"Question tokenization failed. sample_id={sample_id}")

    question_tokens = torch.tensor(question_token_ids_list, dtype=torch.long)

    # --------------------------------------------------
    # 3) Audio features: n_A = n_T
    # --------------------------------------------------
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_duration = float(len(y)) / float(sr) if sr > 0 else 0.0
    if audio_duration <= 0:
        raise RuntimeError(f"Invalid/empty audio for sample_id={sample_id}")

    n_audio_tokens = int(n_question_tokens)
    window_size = audio_duration / float(n_audio_tokens)

    # ==================================================
    # FIX CRITICO:
    # X deve essere 2D: [1, n_features]
    # NON 3D: [1, 1, n_features]
    # ==================================================
    audio_token_ids = torch.arange(
        -1, -(n_audio_tokens + 1), -1, dtype=torch.long
    ).unsqueeze(0)  # [1, n_audio_tokens]

    X = torch.cat(
        (audio_token_ids, question_tokens.unsqueeze(0).cpu().long()),
        dim=1,
    )  # [1, n_audio_tokens + n_question_tokens]

    # --------------------------------------------------
    # 4) Token masker
    # --------------------------------------------------
    def token_masker(mask, x):
        """
        SHAP passes x as [n_samples, n_features] (or sometimes [n_features] for edge cases).
        mask=True  -> keep feature
        mask=False -> hide feature

        Audio pseudo-features hidden with 0
        Text question tokens hidden with TEXT_MASK_SENTINEL
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long)
        else:
            x = x.clone().detach().to(torch.long)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        masked_X = x.clone().detach()

        mask_t = torch.tensor(mask, dtype=torch.bool)
        if mask_t.ndim == 1:
            mask_t = mask_t.unsqueeze(0)

        if mask_t.shape[0] != masked_X.shape[0]:
            if mask_t.shape[0] == 1 and masked_X.shape[0] > 1:
                mask_t = mask_t.expand(masked_X.shape[0], -1)
            else:
                raise RuntimeError(
                    f"token_masker shape mismatch: mask={tuple(mask_t.shape)} vs x={tuple(masked_X.shape)}"
                )

        # audio block
        audio_mask = mask_t.clone().detach()
        audio_mask[:, -n_question_tokens:] = True
        masked_X[~audio_mask] = 0

        # text block
        text_mask = mask_t.clone().detach()
        text_mask[:, :-n_question_tokens] = True
        masked_X[~text_mask] = TEXT_MASK_SENTINEL

        return masked_X.cpu()

    # --------------------------------------------------
    # 5) Prediction function for SHAP
    # --------------------------------------------------
    def get_prediction(x):
        """
        Input atteso da SHAP:
            x shape = [n_masks, n_features]

        Split coerente con MM-SHAP ufficiale:
            audio  = x[:, :n_audio_tokens]
            text   = x[:, n_audio_tokens:]

        FIX PARALLELISMO:
        - prima il codice faceva una richiesta runner per volta
        - ora costruisce tutte le perturbazioni del batch
        - le invia in parallelo ai worker/GPU
        - raccoglie i risultati in ordine

        Questo NON cambia la metodologia MM-SHAP.
        Cambia solo lo scheduling delle inferenze.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long)
        else:
            x = x.clone().detach().to(torch.long)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.ndim != 2:
            raise RuntimeError(
                f"MM-SHAP get_prediction expects 2D input [batch, features], got {tuple(x.shape)}"
            )

        expected_features = int(n_audio_tokens + n_question_tokens)
        if x.shape[1] != expected_features:
            raise RuntimeError(
                f"MM-SHAP feature mismatch: expected {expected_features}, got {x.shape[1]}"
            )

        masked_audio_token_ids = x[:, :n_audio_tokens]
        masked_question_tokens = x[:, n_audio_tokens:]

        batch_size = int(masked_question_tokens.shape[0])
        result = np.zeros((batch_size, len(target_ids)), dtype=float)
        audio_segment_size = max(1, len(y) // n_audio_tokens)

        q_start, q_end = extract_only_question_text_span(prompt)

        # --------------------------------------------------
        # 1) Costruzione di tutte le perturbazioni del batch
        # --------------------------------------------------
        request_items: List[Dict[str, Any]] = []
        created_tmp_paths: List[str] = []

        try:
            for i in range(batch_size):
                rebuilt_question = _rebuild_question_text_with_literal_masks(
                    tokenizer=processor.tokenizer,
                    original_question_token_ids=question_token_ids_list,
                    masked_question_token_ids=masked_question_tokens[i].detach().cpu().tolist(),
                    text_mask_sentinel=TEXT_MASK_SENTINEL,
                    mask_literal=TEXT_MASK_LITERAL,
                )

                new_prompt = prompt[:q_start] + rebuilt_question + prompt[q_end:]

                masked_audio = y.copy()
                to_mask = torch.where(masked_audio_token_ids[i] == 0)[0].detach().cpu().tolist()

                for k in to_mask:
                    start = int(k) * audio_segment_size
                    end = min((int(k) + 1) * audio_segment_size, len(masked_audio))
                    masked_audio[start:end] = 0.0

                tmp_audio = _tmp_wav_path(prefix="mmshap_qa_")
                sf.write(tmp_audio, masked_audio, sr)

                created_tmp_paths.append(tmp_audio)
                request_items.append({
                    "audio_path": tmp_audio,
                    "prompt": new_prompt,
                    "target_ids": list(target_ids),
                })

            # --------------------------------------------------
            # 2) Invio parallelo ai worker/GPU
            # --------------------------------------------------
            parallel_window = _get_mmshap_parallel_window(runner)

            batch_outputs = runner.get_mmshap_logits_batch(
                items=request_items,
                window=parallel_window,
            )

            # --------------------------------------------------
            # 3) Ricostruzione output in ordine
            # --------------------------------------------------
            for i, vals in enumerate(batch_outputs):
                row = np.zeros((len(target_ids),), dtype=float)
                T = min(len(vals), len(target_ids))
                if T > 0:
                    row[:T] = np.asarray(vals[:T], dtype=float)
                result[i] = row

        finally:
            for tmp_audio in created_tmp_paths:
                try:
                    os.remove(tmp_audio)
                except OSError:
                    pass

        return result

    # --------------------------------------------------
    # 6) SHAP
    # --------------------------------------------------
    explainer = shap.PermutationExplainer(
        model=get_prediction,
        masker=token_masker,
        feature_names=None,
    )

    n_features_total = int(n_audio_tokens + n_question_tokens)
    max_evals = int(MM_SHAP_NUM_PERMUTATIONS * n_features_total)

    shap_values = explainer(
        X,
        max_evals=max_evals,
        silent=True,
    )

    # ==================================================
    # SHAP output shape normalization
    # ==================================================
    raw_shap_values_arr = np.asarray(shap_values.values, dtype=float)
    expected_n_features_total = int(n_audio_tokens + n_question_tokens)

    shap_values_arr = normalize_mmshap_values_array(
        raw_shap_values_arr,
        expected_n_features=expected_n_features_total,
    )

    a_shap, t_shap = compute_mm_score(
        audio_length=n_audio_tokens,
        shap_values=shap_values_arr,
        method="sum",
        verbose=False,
        expected_n_features=expected_n_features_total,
    )

    # --------------------------------------------------
    # 7) Save raw artifacts
    # --------------------------------------------------
    tok = processor.tokenizer
    out_npz = os.path.join(results_dir, f"{sample_id}_info.npz")

    question_token_str = tok.convert_ids_to_tokens(question_token_ids_list)
    question_token_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in question_token_str]

    output_token_str = tok.convert_ids_to_tokens(target_ids)
    output_token_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in output_token_str]

    np.savez(
        out_npz,
        shapley_values_raw=raw_shap_values_arr,
        shapley_values=shap_values_arr,
        base_values=np.asarray(shap_values.base_values),
        input_ids=X.cpu().numpy(),
        input_tokens_str=question_token_str,
        output_ids=np.asarray(target_ids, dtype=np.int64),
        output_tokens_str=output_token_str,
        baseline_answer=baseline_answer,
        prompt=prompt,
        n_audio_tokens=np.asarray([n_audio_tokens], dtype=np.int64),
        n_question_tokens=np.asarray([n_question_tokens], dtype=np.int64),
        text_mask_literal=np.asarray([TEXT_MASK_LITERAL], dtype=object),
        text_mask_sentinel=np.asarray([TEXT_MASK_SENTINEL], dtype=np.int64),
        shap_method=np.asarray(["permutation_explicit"], dtype=object),
        shap_num_permutations=np.asarray([MM_SHAP_NUM_PERMUTATIONS], dtype=np.int64),
        shap_max_evals=np.asarray([max_evals], dtype=np.int64),
        shap_raw_shape=np.asarray(raw_shap_values_arr.shape, dtype=np.int64),
        shap_normalized_shape=np.asarray(shap_values_arr.shape, dtype=np.int64),
    )

    # --------------------------------------------------
    # 8) Token-level modality plot
    # --------------------------------------------------
    shap_arr = shap_values_arr
    token_audio_vals = []
    token_text_vals = []

    for out_idx in range(len(target_ids)):
        phi_audio = shap_arr[0, 0, :n_audio_tokens, out_idx]
        phi_text = shap_arr[0, 0, n_audio_tokens:, out_idx]

        PhiA = float(np.sum(np.abs(phi_audio)))
        PhiT = float(np.sum(np.abs(phi_text)))
        Z = PhiA + PhiT

        if Z > 0:
            token_audio_vals.append(PhiA / Z)
            token_text_vals.append(PhiT / Z)
        else:
            token_audio_vals.append(0.5)
            token_text_vals.append(0.5)

    plot_path = os.path.join(results_dir, f"{sample_id}_mmshap_analysis.png")
    plot_token_contributions(output_token_str, token_audio_vals, token_text_vals, plot_path)

    out_json = os.path.join(results_dir, f"{sample_id}_mmshap.json")
    payload = {
        "sample_id": sample_id,
        "audio_file": os.path.basename(audio_path),
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "prompt": prompt,
        "baseline_answer_raw": baseline_answer,
        "baseline_answer_used": baseline_answer,
        "A-SHAP": float(a_shap),
        "T-SHAP": float(t_shap),
        "n_audio_tokens": int(n_audio_tokens),
        "n_question_tokens": int(n_question_tokens),
        "window_size_sec": float(window_size),
        "text_scope": "question_only",
        "text_masking": {
            "mode": "literal_[MASK]",
            "literal": TEXT_MASK_LITERAL,
        },
        "audio_masking": {
            "mode": "waveform_zeroing",
        },
        "shap_explainer": "PermutationExplainer",
        "shap_settings": {
            "mode": "explicit_num_permutations",
            "num_permutations": int(MM_SHAP_NUM_PERMUTATIONS),
            "max_evals": int(max_evals),
        },
        "shap_raw_shape": list(raw_shap_values_arr.shape),
        "shap_normalized_shape": list(shap_values_arr.shape),
        "expected_n_features_total": int(expected_n_features_total),
        "npz_path": out_npz,
        "plot_path": plot_path,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


def analysis_1_start(
    model,
    processor,
    results_dir,
    runner=None,
    qa_entry: Optional[Dict] = None,
    shared_baseline: Optional[Dict] = None,
):
    if qa_entry is None:
        raise ValueError("This QA-only version of analysis_1.py requires qa_entry.")

    if runner is None:
        raise RuntimeError("analysis_1_start requires runner for MM-SHAP.")

    return compute_mmshap_qa_sample(
        runner=runner,
        processor=processor,
        audio_path=qa_entry["audio_path"],
        question=qa_entry["question"],
        options=qa_entry["options"],
        correct_answer=qa_entry["correct_answer"],
        results_dir=results_dir,
        sample_id=qa_entry["sample_id"],
        shared_baseline=shared_baseline,
    )