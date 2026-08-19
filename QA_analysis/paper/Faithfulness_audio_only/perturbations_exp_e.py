"""
Esperimento E — Modulo perturbazioni (v4)
==========================================
Funzioni per costruire input perturbati (audio + testo) coerenti con il masking
usato da DIME nella sua pipeline di spiegazione.

Questa versione è snellita per il design MI-only:
- mantiene il ranking audio e testo separati per riuso (UC_audio / UC_text
  diagnostici);
- aggiunge `scale_normalize_ranked()` per la fusione MI scale-invariante;
- mantiene un'unica strategia di mascheramento (`bg_random_energy` per l'audio,
  `[MASK]` per il testo) coerente con DIME.

==============================================================================
COERENZA CON DIME — strategie di masking di default
==============================================================================

AUDIO
-----
- Modalità feature: `audiolime_demucs` → 32 componenti (4 stem × 8 segmenti).
- Modalità di sostituzione: `bg_random_energy`.
- Cross-fade ai bordi: 5 ms.
- Composizione: `compose_from_binary_mask()` da audioLIME.

TESTO
-----
- Funzione: `mask_structured_mcqa_prompt_words()` da masking_utils.py.
- Le parole della domanda vengono sostituite con `[MASK]`.
- Le opzioni A/B/C/D restano sempre intatte.
==============================================================================
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf


N_STEMS = 4
N_SEGMENTS = 8
N_AUDIO_FEATURES = N_STEMS * N_SEGMENTS  # 32


# =============================================================================
# Indici flat ↔ (stem, segment)
# =============================================================================
def stem_segment_to_flat(stem_idx: int, segment_idx: int,
                         n_segments: int = N_SEGMENTS) -> int:
    return int(stem_idx) * int(n_segments) + int(segment_idx)


def flat_to_stem_segment(flat_idx: int,
                         n_segments: int = N_SEGMENTS) -> Tuple[int, int]:
    return int(flat_idx) // int(n_segments), int(flat_idx) % int(n_segments)


# =============================================================================
# Ranking
# =============================================================================
def rank_audio_features(matrix_4x8: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Ordina le 32 feature audio per |valore| decrescente.
    Input: matrice 4×8 (uc1_stem_x_seg o mi1_stem_x_seg).
    """
    if not matrix_4x8:
        return []
    flat = []
    for stem_idx, row in enumerate(matrix_4x8):
        for seg_idx, val in enumerate(row):
            v = float(val)
            flat.append({
                "feature_idx": stem_segment_to_flat(stem_idx, seg_idx),
                "stem_idx": stem_idx,
                "segment_idx": seg_idx,
                "value": v,
                "abs_value": abs(v),
            })
    flat.sort(key=lambda d: d["abs_value"], reverse=True)
    return flat


def rank_text_words(uc2_or_mi2_words: List[float],
                    word_strings: List[str]) -> List[Dict[str, Any]]:
    """
    Ordina le parole della domanda per |valore| decrescente.
    """
    n = min(len(uc2_or_mi2_words), len(word_strings))
    flat = []
    for i in range(n):
        v = float(uc2_or_mi2_words[i])
        flat.append({
            "word_idx": i,
            "word": str(word_strings[i]),
            "value": v,
            "abs_value": abs(v),
        })
    flat.sort(key=lambda d: d["abs_value"], reverse=True)
    return flat


def scale_normalize_ranked(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizza i valori dividendo per il massimo |valore| del ranking.

    Serve a fondere ranking di modalità diverse (audio vs testo MI) su una scala
    comune [-1, 1]. Aggiunge i campi `value_norm` e `abs_value_norm`. Non
    modifica `value` o `abs_value` originali.
    """
    if not ranked:
        return []
    max_abs = max((float(f.get("abs_value", 0.0)) for f in ranked), default=0.0)
    if max_abs <= 1e-12:
        return [dict(f, abs_value_norm=0.0, value_norm=0.0) for f in ranked]
    out = []
    for f in ranked:
        v = float(f.get("value", 0.0))
        out.append({
            **f,
            "value_norm": v / max_abs,
            "abs_value_norm": abs(v) / max_abs,
        })
    return out


# =============================================================================
# Costruzione maschere audio
# =============================================================================
def build_audio_binary_mask(
    selected_features: List[Dict[str, Any]],
    mode: str,
    n_features: int = N_AUDIO_FEATURES,
) -> np.ndarray:
    """
    Convenzione audioLIME: am[i] = 1 → componente PRESENTE
                           am[i] = 0 → componente SOSTITUITA con background.

    mode == "sufficiency": tieni solo le feature selezionate
                           → am = 0 ovunque, 1 sulle selezionate.
    mode == "necessity"  : maschera le feature selezionate
                           → am = 1 ovunque, 0 sulle selezionate.
    """
    sel_idx = [int(f["feature_idx"]) for f in selected_features]

    if mode == "sufficiency":
        am = np.zeros(n_features, dtype=int)
        if sel_idx:
            am[sel_idx] = 1
    elif mode == "necessity":
        am = np.ones(n_features, dtype=int)
        if sel_idx:
            am[sel_idx] = 0
    else:
        raise ValueError(f"mode must be 'sufficiency' or 'necessity', got {mode}")
    return am


def materialize_perturbed_audio(
    audio_path: str,
    audio_binary_mask: np.ndarray,
    out_dir: str,
    suffix: str = "_perturbed",
) -> str:
    """
    Crea il WAV perturbato via audioLIME `compose_from_binary_mask()`.
    Riusa la factorization demucs cached da Exp A.
    """
    from QA_analysis.utils.audioLIME import (
        build_demucs_factorization_for_dime,
        compose_from_binary_mask,
    )

    factorization = build_demucs_factorization_for_dime(
        audio_path=audio_path,
        sr=16000,
    )

    n_components = int(factorization.get_number_components())
    if n_components != len(audio_binary_mask):
        raise ValueError(
            f"Mismatch tra audio_binary_mask (len={len(audio_binary_mask)}) "
            f"e n_components della factorization ({n_components}). "
            f"Verifica DIME_AUDIOLIME_NUM_TEMPORAL_SEGMENTS."
        )

    y_perturbed = compose_from_binary_mask(factorization, audio_binary_mask)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(out_dir, f"{base}{suffix}.wav")
    sf.write(out_path, y_perturbed, 16000)
    return out_path


# =============================================================================
# Costruzione prompt testuale perturbato
# =============================================================================
def build_perturbed_prompt(
    question: str,
    options: List[str],
    selected_words: List[Dict[str, Any]],
    mode: str,
    tokenizer,
    n_words_total: int,
) -> str:
    """
    Costruisce il prompt con maschera testuale.

    mode == "sufficiency": maschera tutte le parole TRANNE le selezionate.
    mode == "necessity"  : maschera SOLO le selezionate.

    Le opzioni A/B/C/D restano sempre intatte (DIME pipeline standard).
    """
    from QA_analysis.utils.masking_utils import mask_structured_mcqa_prompt_words

    sel_word_indices = set(int(w["word_idx"]) for w in selected_words)

    if mode == "sufficiency":
        mask_indices = [i for i in range(n_words_total) if i not in sel_word_indices]
    elif mode == "necessity":
        mask_indices = sorted(sel_word_indices)
    else:
        raise ValueError(f"mode must be 'sufficiency' or 'necessity', got {mode}")

    result = mask_structured_mcqa_prompt_words(
        tokenizer=tokenizer,
        question=question,
        options=options,
        mask_indices=mask_indices,
    )
    return str(result["masked_prompt"])


# =============================================================================
# Helpers target IDs A/B/C/D
# =============================================================================
def get_letter_token_ids(tokenizer, letters: Tuple[str, ...] = ("A", "B", "C", "D")) -> List[int]:
    """
    Codifica A/B/C/D come singoli token id. Usa prefix space (più frequente nel
    formato generato da Qwen) con fallback alla lettera nuda.
    """
    out = []
    for letter in letters:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids) == 1:
            out.append(int(ids[0]))
            continue
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) == 1:
            out.append(int(ids[0]))
            continue
        out.append(int(ids[0]))
    return out


def answer_letter_from_logits(
    logits_4: List[float],
    letters: Tuple[str, ...] = ("A", "B", "C", "D"),
) -> str:
    if not logits_4:
        return "A"
    return letters[int(np.argmax(np.asarray(logits_4)))]