"""
Utility condivise specifiche per Audio Flamingo 3 (AF3).
==========================================================
Equivalente di shared_utils.py per Qwen2.5-Omni, ma per AF3 (backbone
Qwen2.5-7B + encoder AF-Whisper, integrato nativamente in transformers
come AudioFlamingo3ForConditionalGeneration + AutoProcessor).
"""

from typing import List, Tuple


def get_letter_token_ids_af3(
    tokenizer,
    letters: Tuple[str, ...] = ("A", "B", "C", "D"),
) -> List[int]:
    """
    Codifica A/B/C/D come singoli token id per AF3.

    A differenza di Qwen2.5-Omni (che risponde col token nudo " A"), AF3
    con il prompt MCQA standard (opzioni scritte come "(A) ...") risponde
    sistematicamente col token fuso "(A" -- verificato smoke test 2026-08-25,
    8/8 sample reali HumMusQA con top-1 token = forma "(X", 0/8 forma nuda.

    Leggere solo i token nudi (come fa get_letter_token_ids per Qwen)
    sottostima p_orig/p_suf/p_nec: la maggior parte della massa di
    probabilita' reale del modello sta sui token "(A"/"(B"/"(C"/"(D", non
    su quelli nudi, distorcendo le metriche di sufficiency/necessity a
    valle.

    Prova prima "(" + lettera; se non tokenizza a un singolo id, ricade
    sulla forma nuda " " + lettera, poi sulla lettera senza spazio.
    """
    out: List[int] = []
    for letter in letters:
        ids = tokenizer.encode("(" + letter, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
            continue
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(ids) == 1:
            out.append(ids[0])
            continue
        ids = tokenizer.encode(letter, add_special_tokens=False)
        out.append(ids[0])
    return out
