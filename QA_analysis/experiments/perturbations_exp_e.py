"""
Esperimento E — Modulo perturbazioni
====================================
Funzioni per costruire input perturbati (audio + testo) coerenti con il masking
usato da DIME nella sua pipeline di spiegazione.

==============================================================================
COERENZA CON DIME — strategie di masking di default
==============================================================================

AUDIO
-----
- Modalità feature: `audiolime_demucs` (DIME_AUDIO_FEATURE_MODE)
  L'audio è fattorizzato in 32 componenti = 4 stems (drums, bass, other, vocals)
  × 8 segmenti temporali. Una "feature audio" è quindi una coppia (stem, segment)
  identificata da un indice piatto in [0..31].

- Modalità di sostituzione: `bg_random_energy` (DIME_AUDIO_MASK_MODE)
  Le componenti rimosse vengono sostituite con un segmento random preso da un
  altro audio del pool di background, scalato per matchare l'energia (RMS) del
  segmento originale. Se il segmento originale è silenzioso (RMS < 1e-3) viene
  messo a zero.

- Cross-fade ai bordi: 5 ms di rampa lineare per evitare click (DIME_CROSSFADE).

In Exp E NON costruiamo le perturbazioni manualmente: usiamo direttamente
`compose_from_binary_mask()` di audioLIME, che chiama internamente la stessa
funzione `_apply_audio_mask_replace_with_background()` con gli stessi default.
Questo garantisce coerenza al 100% con quello che DIME fa per spiegare.

TESTO
-----
- Funzione: `mask_structured_mcqa_prompt_words()` da masking_utils.py
  Riceve una lista di indici di parole della domanda da mascherare.
  Le sostituisce con la stringa letterale "[MASK]".
  Le opzioni A/B/C/D restano sempre intatte (non vengono mai mascherate da DIME).
  Anche il prefix istruzionale e il "Respond with ONLY..." restano intatti.

==============================================================================
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import tempfile

# I moduli del progetto vengono importati lazy — chi usa questo modulo
# si assume abbia già le env DIME_* impostate prima dell'import.

# Indice piatto delle feature audio audiolime_demucs:
#   feature_idx = stem_idx * N_SEGMENTS + segment_idx
N_STEMS = 4
N_SEGMENTS = 8
N_AUDIO_FEATURES = N_STEMS * N_SEGMENTS  # 32


# =============================================================================
# 1) Conversioni indice piatto ↔ (stem, segment)
# =============================================================================

def stem_segment_to_flat(stem_idx: int, segment_idx: int,
                         n_segments: int = N_SEGMENTS) -> int:
    """Converte (stem, segment) → indice piatto in [0..31]."""
    return int(stem_idx) * int(n_segments) + int(segment_idx)


def flat_to_stem_segment(flat_idx: int,
                         n_segments: int = N_SEGMENTS) -> Tuple[int, int]:
    """Converte indice piatto → (stem, segment)."""
    return int(flat_idx) // int(n_segments), int(flat_idx) % int(n_segments)


# =============================================================================
# 2) Estrazione ranking top-k dai JSON di Exp A
# =============================================================================

def rank_audio_features(matrix_4x8: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Ordina le 32 feature audio per |valore| decrescente.
    matrix_4x8 può essere uc1_stem_x_seg o mi1_stem_x_seg dal JSON di Exp A.
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
    word_strings è la lista delle parole originali (per riferimento).
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


def select_top_k(ranked: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Tiene solo le prime k feature."""
    return ranked[:int(k)]


# =============================================================================
# 3) Costruzione delle binary mask audio per sufficiency / necessity
# =============================================================================

def build_audio_binary_mask(
    top_k_features: List[Dict[str, Any]],
    mode: str,
    n_features: int = N_AUDIO_FEATURES,
) -> np.ndarray:
    """
    Costruisce la binary mask audio (lunghezza 32) da passare ad
    audioLIME.compose_from_binary_mask().

    Convenzione audioLIME: am[i] = 1 → componente PRESENTE
                           am[i] = 0 → componente SOSTITUITA con background

    mode == "sufficiency": tieni solo le top-k → am = 0 ovunque, 1 sulle top-k
    mode == "necessity"  : maschera le top-k    → am = 1 ovunque, 0 sulle top-k
    """
    top_indices = [int(f["feature_idx"]) for f in top_k_features]

    if mode == "sufficiency":
        am = np.zeros(n_features, dtype=int)
        am[top_indices] = 1
    elif mode == "necessity":
        am = np.ones(n_features, dtype=int)
        am[top_indices] = 0
    else:
        raise ValueError(f"mode must be 'sufficiency' or 'necessity', got {mode}")

    return am


# =============================================================================
# 4) Materializzazione audio perturbato in WAV temporaneo
# =============================================================================

def materialize_perturbed_audio(
    audio_path: str,
    audio_binary_mask: np.ndarray,
    out_dir: str,
    suffix: str = "_perturbed",
) -> str:
    """
    Usa la pipeline audioLIME di DIME per creare l'audio perturbato e salvarlo
    su disco. Il modello vorrà un path su file, non un array.

    La factorization è cached internamente da DIME: se l'audio è già stato
    fattorizzato in un run precedente (es. Exp A con DIME_AUDIOLIME_PRECOMPUTED_DIR
    impostato), la separazione demucs viene riusata senza rifare il calcolo.
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
# 5) Costruzione del prompt testuale perturbato
# =============================================================================

def build_perturbed_prompt(
    question: str,
    options: List[str],
    top_k_words: List[Dict[str, Any]],
    mode: str,
    tokenizer,
    n_words_total: int,
) -> str:
    """
    Costruisce il prompt con maschera testuale.

    Sufficiency: maschera tutte le parole della domanda TRANNE le top-k
    Necessity  : maschera SOLO le top-k parole

    Usa la stessa funzione di DIME (mask_structured_mcqa_prompt_words) per
    garantire identica gestione del prefix, options block, [MASK] literal.
    """
    from QA_analysis.utils.masking_utils import mask_structured_mcqa_prompt_words

    top_word_indices = set(int(w["word_idx"]) for w in top_k_words)

    if mode == "sufficiency":
        # Maschera tutto tranne le top-k
        mask_indices = [i for i in range(n_words_total) if i not in top_word_indices]
    elif mode == "necessity":
        # Maschera solo le top-k
        mask_indices = sorted(top_word_indices)
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
# 6) Helpers per i target ids A/B/C/D
# =============================================================================

def get_letter_token_ids(tokenizer, letters: List[str] = ("A", "B", "C", "D")) -> List[int]:
    """
    Encoda le 4 lettere come singoli token id, da passare a runner.get_mmshap_logits.
    Ritorna la lista in ordine A,B,C,D.

    Importante: usiamo lo SPAZIO prefix per allineare alla tokenizzazione
    naturale che il modello produce in output (" A", " B", ecc.). Qwen tokenizer
    è sensibile a questo. In caso di mismatch, fallback a lettera nuda.
    """
    out = []
    for letter in letters:
        # Tentativo 1: con prefix space (più frequente in output del modello)
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids) == 1:
            out.append(int(ids[0]))
            continue
        # Tentativo 2: lettera nuda
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) == 1:
            out.append(int(ids[0]))
            continue
        # Ultima spiaggia: primo token
        out.append(int(ids[0]))
    return out


def answer_letter_from_logits(logits_4: List[float],
                              letters: List[str] = ("A", "B", "C", "D")) -> str:
    """argmax sui 4 logit → lettera."""
    if not logits_4:
        return "A"
    return letters[int(np.argmax(np.asarray(logits_4)))]