#analysis_1.py
import os
import tempfile
import numpy as np
import soundfile as sf

from .masking_utils import create_random_permutations, mask_text_token_ids
from .visualization import plot_token_contributions
from .shared_utils import (
    merge_word_tokens,
    filter_punctuation, get_token_logprob_autoregressive_id
)


def _tmp_wav_path(prefix: str = "mmshap_", dir_prefer: str = "/dev/shm") -> str:
    tmp_dir = dir_prefer if os.path.isdir(dir_prefer) else None
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=tmp_dir)
    os.close(fd)
    return path


def _mask_audio_segments_array(y: np.ndarray, sr: int, mask_segments):
    y_masked = y.copy()
    n = len(y_masked)
    for start_time, end_time in mask_segments:
        s0 = max(0, int(start_time * sr))
        s1 = min(n, int(end_time * sr))
        if s1 > s0:
            y_masked[s0:s1] = 0
    return y_masked


def compute_mmshap_for_token(
    model,
    tokenizer,
    audio_path: str,
    prompt: str,
    caption_ids: list,
    token_index: int,
    num_permutations: int = 10,
):
    """
    MM-SHAP autoregressivo per captioning, paper-compliant:

    - Feature set: A (finestre audio) + T (token del prompt)
    - Approssima Shapley per-feature ϕ_{j,t} con Permutation SHAP:
        ϕ_{j,t} = E_π [ f_t(S_π(j) ∪ {j}) - f_t(S_π(j)) ]
      dove f_t è la log-prob del token target (token_index) dato il prefisso della caption.
    - Modality contribution per token t:
        Φ_A,t = sum_{j in A} |ϕ_{j,t}|   ;   Φ_T,t = sum_{j in T} |ϕ_{j,t}|
    - Normalizzazione:
        A-SHAP_t = Φ_A,t / (Φ_A,t + Φ_T,t),  T-SHAP_t = 1 - A-SHAP_t

    Ritorna: (A-SHAP_t, T-SHAP_t)
    """
    # ---------
    # Target token (id) + prefix ids
    # ---------
    if token_index < 0 or token_index >= len(caption_ids):
        return 0.5, 0.5

    target_id = int(caption_ids[token_index])
    prefix_ids = [int(x) for x in caption_ids[:token_index]]

    # ---------
    # Audio load once
    # ---------
    y, sr = librosa.load(audio_path, sr=16000)
    audio_duration = float(len(y)) / float(sr) if sr > 0 else 0.0
    if audio_duration <= 0:
        return 0.5, 0.5

    # ---------
    # Text features = prompt token ids (no special tokens)
    # ---------
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    num_text_tokens = max(1, len(prompt_ids))

    # ---------
    # Audio windows: nA = nT (paper choice: same number of features per modality)
    # => window_size = duration / nT, nA = nT
    # ---------
    num_audio_windows = int(num_text_tokens)
    window_size = audio_duration / float(num_audio_windows)

    # Feature indices (paper): union set U = A ∪ T
    # we'll represent as ("audio", j) with j in [0,nA-1], ("text", k) with k in [0,nT-1]
    permutations = create_random_permutations(num_audio_windows, num_text_tokens, num_permutations)

    # Accumula deltas signed per-feature (questo è ϕ_j,t via media)
    deltas_audio = {j: [] for j in range(num_audio_windows)}
    deltas_text = {k: [] for k in range(num_text_tokens)}

    # ---------
    # v(empty): tutto mascherato (audio + testo)
    # ---------
    full_audio_mask = [
        (j * window_size, min((j + 1) * window_size, audio_duration))
        for j in range(num_audio_windows)
    ]

    tmp_init = _tmp_wav_path(prefix="mmshap_init_")
    try:
        y_masked0 = _mask_audio_segments_array(y, sr, full_audio_mask)
        sf.write(tmp_init, y_masked0, sr)

        masked_prompt0, _, _ = mask_text_token_ids(tokenizer, prompt, list(range(num_text_tokens)))

        v_empty = get_token_logprob_autoregressive_id(
            model=model,
            tokenizer=tokenizer,
            audio_path=tmp_init,
            prompt=masked_prompt0,
            prefix_ids=prefix_ids,
            target_id=target_id,
        )
    finally:
        try:
            os.remove(tmp_init)
        except OSError:
            pass

    # ---------
    # Permutation SHAP (paper Eq.1 approssimata)
    # ---------
    all_audio = set(range(num_audio_windows))
    all_text = set(range(num_text_tokens))

    for perm in permutations:
        S_audio = set()
        S_text = set()
        current_value = float(v_empty)

        for ftype, findex in perm:
            if ftype == "audio":
                new_audio = S_audio | {int(findex)}
                new_text = S_text
            else:
                new_text = S_text | {int(findex)}
                new_audio = S_audio

            # ---- mask audio: maschera tutte le finestre NON in new_audio
            mask_audio = sorted(all_audio - new_audio)
            tmp_audio = None
            audio_used = audio_path

            if mask_audio:
                segments = [
                    (j * window_size, min((j + 1) * window_size, audio_duration))
                    for j in mask_audio
                ]
                y_masked = _mask_audio_segments_array(y, sr, segments)
                tmp_audio = _tmp_wav_path(prefix="mmshap_tmp_")
                sf.write(tmp_audio, y_masked, sr)
                audio_used = tmp_audio

            # ---- mask text: maschera tutti i token NON in new_text
            mask_text = sorted(all_text - new_text)
            if mask_text:
                masked_prompt, _, _ = mask_text_token_ids(tokenizer, prompt, mask_text)
            else:
                masked_prompt = prompt

            new_value = get_token_logprob_autoregressive_id(
                model=model,
                tokenizer=tokenizer,
                audio_path=audio_used,
                prompt=masked_prompt,
                prefix_ids=prefix_ids,
                target_id=target_id,
            )

            # Marginale SIGNED (paper Eq.1): f(S∪{j}) - f(S)
            delta = float(new_value) - float(current_value)

            if ftype == "audio":
                deltas_audio[int(findex)].append(delta)
            else:
                deltas_text[int(findex)].append(delta)

            current_value = float(new_value)
            S_audio, S_text = new_audio, new_text

            if tmp_audio is not None:
                try:
                    os.remove(tmp_audio)
                except OSError:
                    pass

    # ---------
    # ϕ_j,t = mean(delta_j) ; poi Φ_A,t = sum |ϕ| (paper Eq.2) ; normalizza (Eq.3)
    # ---------
    phi_audio = []
    for j in range(num_audio_windows):
        vals = deltas_audio[j]
        phi_audio.append(float(np.mean(vals)) if vals else 0.0)

    phi_text = []
    for k in range(num_text_tokens):
        vals = deltas_text[k]
        phi_text.append(float(np.mean(vals)) if vals else 0.0)

    PhiA = float(np.sum(np.abs(np.asarray(phi_audio, dtype=float))))
    PhiT = float(np.sum(np.abs(np.asarray(phi_text, dtype=float))))
    Z = PhiA + PhiT

    if Z <= 0.0:
        return 0.5, 0.5

    a_shap = PhiA / Z
    t_shap = PhiT / Z
    return float(a_shap), float(t_shap)


import librosa
from .shared_utils import tokenize_caption_for_mmshap


def compute_caption_shapley_parallel(
    runner,
    model,
    tokenizer,
    audio_path,
    prompt,
    base_caption,
    num_permutations=10,
):
    """
    Token-wise parallel se runner disponibile, altrimenti seriale (fallback).
    Output: per_token {idx: {a_shap,t_shap}}, + global
    """
    # usa IDs (fondamentale per autoregressivo pulito)
    caption_ids, caption_tokens = tokenize_caption_for_mmshap(base_caption, tokenizer, return_ids=True)
    n = len(caption_ids)

    token_shapley_values = {
        "per_window": {},   # lasciato per future heatmap, ma non necessario per word-level
        "global_audio": {},
        "global_text": {},
        "per_token": {},
    }

    if runner is None:
        for i in range(n):
            a_shap, t_shap = compute_mmshap_for_token(
                model=model,
                tokenizer=tokenizer,
                audio_path=audio_path,
                prompt=prompt,
                caption_ids=caption_ids,
                token_index=i,
                num_permutations=num_permutations,
            )
            token_shapley_values["per_token"][i] = {"a_shap": a_shap, "t_shap": t_shap}
            token_shapley_values["global_audio"][i] = a_shap
            token_shapley_values["global_text"][i] = t_shap
    else:
        out = runner.run_mmshap_tokens(
            audio_path=audio_path,
            prompt=prompt,
            caption_ids=caption_ids,
            num_permutations=num_permutations,
        )
        for i in range(n):
            v = out[i]
            token_shapley_values["per_token"][i] = {"a_shap": float(v["a_shap"]), "t_shap": float(v["t_shap"])}
            token_shapley_values["global_audio"][i] = float(v["a_shap"])
            token_shapley_values["global_text"][i] = float(v["t_shap"])

    total_audio = sum(token_shapley_values["global_audio"].values())
    total_text = sum(token_shapley_values["global_text"].values())
    total = total_audio + total_text

    a_shap_global = total_audio / total if total > 0 else 0.5
    t_shap_global = total_text / total if total > 0 else 0.5

    y, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(y) / sr

    return {
        "token_shapley_values": token_shapley_values,
        "global_shap_values": {"A-SHAP": a_shap_global, "T-SHAP": t_shap_global},
        "base_caption": base_caption,
        "audio_duration": audio_duration,
        "num_tokens": n,
        "caption_tokens": caption_tokens,   # per merge_word_tokens/plot
    }


def analysis_1_start(model, tokenizer, audio_path, prompt, caption, results_dir, runner=None):
    print(f"\nAnalisi 1 per: {os.path.basename(audio_path)}")
    print("Calcolo valori MM-SHAP per ogni token...")

    results = compute_caption_shapley_parallel(
        runner=runner,
        model=model,
        tokenizer=tokenizer,
        audio_path=audio_path,
        prompt=prompt,
        base_caption=caption,
        num_permutations=10,
    )

    caption_tokens = results["caption_tokens"]
    token_shap = results["token_shapley_values"]["per_token"]  # idx -> {a_shap,t_shap}

    # parole & mapping
    words, mapping, token_list = merge_word_tokens(caption_tokens)

    # rimuovi punteggiatura
    filtered_mapping, filtered_words = filter_punctuation(mapping, token_list)

    # aggregate per parola
    word_shapley_values = {}
    for clean_word, token_indices in filtered_mapping:
        a_sum = 0.0
        t_sum = 0.0
        for idx in token_indices:
            val = token_shap.get(idx, {"a_shap": 0.0, "t_shap": 0.0})
            a_sum += float(val["a_shap"])
            t_sum += float(val["t_shap"])

        total = a_sum + t_sum
        if total > 0:
            a_shap = a_sum / total
            t_shap = t_sum / total
        else:
            a_shap = 0.5
            t_shap = 0.5

        word_shapley_values[clean_word] = {
            "a_shap": a_shap,
            "t_shap": t_shap,
            "original_shapley": {"a_shap": a_sum, "t_shap": t_sum},
            "tokens_indices": token_indices,
        }

    words_ordered = filtered_words
    a_shap_values = [word_shapley_values[w]["a_shap"] for w in words_ordered]
    t_shap_values = [word_shapley_values[w]["t_shap"] for w in words_ordered]

    print("Creazione visualizzazione...")
    plot_path = os.path.join(results_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_mmshap_analysis.png")
    plot_token_contributions(words_ordered, a_shap_values, t_shap_values, plot_path)
    print(f"Grafico MM-SHAP salvato in: {plot_path}")

    results_to_save = {
        "audio_file": os.path.basename(audio_path),
        "prompt": prompt,
        "caption": caption,
        "words": words_ordered,
        "word_shapley_values": {
            word: {
                "a_shap": values["a_shap"],
                "t_shap": values["t_shap"],
                "original_shapley": values["original_shapley"],
                "tokens_indices": values["tokens_indices"],
            }
            for word, values in word_shapley_values.items()
        },
        "global_shap_values": results["global_shap_values"],
        "plot_path": plot_path,
    }

    return results_to_save