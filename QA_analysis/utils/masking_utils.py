import re

# =============================================================================
# WORD-LEVEL MASKING FOR DIME — QUESTION ONLY
# =============================================================================

_WORD_RE = re.compile(r"\S+")


def _validate_mcqa(question: str, options: list):
    q = str(question).strip()
    opts = [str(x).strip() for x in (options or [])]

    if len(opts) != 4 or any(x == "" for x in opts):
        raise ValueError(f"Expected exactly 4 non-empty options, got: {opts}")

    return q, opts


def _apply_replacements_by_span(text: str, replacements):
    """
    replacements: lista di tuple (start, end, replacement_text)
    con start/end riferiti alla stringa ORIGINALE.
    """
    if not replacements:
        return text

    replacements = sorted(replacements, key=lambda x: x[0])

    merged = []
    last_end = -1
    for s, e, rep in replacements:
        if s < last_end:
            raise ValueError("Overlapping replacements are not allowed.")
        merged.append((int(s), int(e), str(rep)))
        last_end = e

    out = []
    cur = 0
    for s, e, rep in merged:
        out.append(text[cur:s])
        out.append(rep)
        cur = e
    out.append(text[cur:])
    return "".join(out)


def tokenize_structured_mcqa_dynamic_words(tokenizer, question: str, options: list):
    """
    Costruisce la parte dinamica del prompt e definisce le FEATURE TESTUALI
    perturbabili a livello di PAROLA.

    Feature perturbabili:
      - SOLO parole della domanda

    NON perturbabili:
      - scaffolding istruzionale
      - "Options:"
      - marker "(A) (B) (C) (D)"
      - contenuto semantico delle 4 opzioni
      - blocco finale "Respond with ONLY..."
    """
    q, opts = _validate_mcqa(question, options)

    options_lines = [f"({chr(65+i)}) {opt}" for i, opt in enumerate(opts)]
    options_text = "Options:\n" + "\n".join(options_lines)
    dynamic_text = f"{q}\n{options_text}"

    word_spans = []
    word_labels = []

    # -------------------------
    # QUESTION WORDS ONLY
    # -------------------------
    for m in _WORD_RE.finditer(q):
        word_spans.append({
            "start": int(m.start()),
            "end": int(m.end()),
            "text": m.group(0),
            "section": "question",
            "option_index": None,
        })
        word_labels.append(m.group(0))

    q_count = len(word_labels)

    metadata = {
        "question_text": q,
        "dynamic_text": dynamic_text,
        "question_word_span": (0, q_count),
        "options_header_word_span": (q_count, q_count),  # non perturbabile
        "options_word_span": (q_count, q_count),         # nessuna option perturbabile
        "num_dynamic_words": len(word_labels),
        "word_spans": word_spans,
        "feature_unit": "word",
        "text_scope": "question_only",
    }

    # Manteniamo l'interfaccia compatibile con analysis_2.py
    dynamic_ids = None
    dynamic_tokens = word_labels

    return dynamic_text, dynamic_ids, dynamic_tokens, metadata


def mask_structured_mcqa_prompt_words(
    tokenizer,
    question: str,
    options: list,
    mask_indices,
):
    """
    Applica masking WORD-LEVEL SOLO sulla domanda.

    Maschera:
      - parole della domanda

    NON maschera:
      - prefix/suffix istruzionali
      - "Options:"
      - marker (A)/(B)/(C)/(D)
      - contenuto delle opzioni
      - blocco finale "Respond with ONLY..."
    """
    from QA_analysis.utils.shared_utils import (
        build_hummusqa_qwen25_prompt_parts,
        build_hummusqa_qwen25_prompt_from_parts,
    )

    if mask_indices is None:
        mask_indices = []
    mask_indices = sorted(set(
        int(x) for x in mask_indices
        if isinstance(x, (int, float)) and int(x) >= 0
    ))

    parts = build_hummusqa_qwen25_prompt_parts(question, options)

    dynamic_text, _dynamic_ids, dynamic_words, metadata = tokenize_structured_mcqa_dynamic_words(
        tokenizer=tokenizer,
        question=question,
        options=options,
    )

    word_spans = metadata["word_spans"]
    valid_mask_indices = [i for i in mask_indices if 0 <= i < len(word_spans)]

    replacements = []
    for mi in valid_mask_indices:
        sp = word_spans[mi]
        replacements.append((sp["start"], sp["end"], "[MASK]"))

    # Qui masked_dynamic_text contiene:
    #   masked_question + "\nOptions:\n..." originale non toccato
    masked_dynamic_text = _apply_replacements_by_span(dynamic_text, replacements)

    split_marker = "\nOptions:\n"
    if split_marker not in masked_dynamic_text:
        raise RuntimeError(
            "Structured MCQA format corrupted during word masking: "
            f"split marker {repr(split_marker)} not found."
        )

    masked_question_text, untouched_options_text = masked_dynamic_text.split(split_marker, 1)

    new_parts = dict(parts)
    new_parts["question"] = masked_question_text.strip()

    # Le options restano IDENTICHE all'originale
    # Ricostruiamo options_block dalle parti originali, non dal testo mascherato
    new_parts["options_block"] = parts["options_block"]

    masked_full_prompt = build_hummusqa_qwen25_prompt_from_parts(new_parts)

    return {
        "masked_prompt": masked_full_prompt,
        "dynamic_text": dynamic_text,
        "dynamic_words": list(dynamic_words),
        "masked_dynamic_text": masked_dynamic_text,
        "mask_indices": valid_mask_indices,
        "metadata": metadata,
    }


# =============================================================================
# BACKWARD-COMPATIBILITY WRAPPERS
# =============================================================================

def tokenize_structured_mcqa_dynamic_text(tokenizer, question: str, options: list):
    """
    Wrapper compatibile col vecchio nome.
    Ora restituisce feature a livello di PAROLA sulla SOLA domanda.
    """
    return tokenize_structured_mcqa_dynamic_words(
        tokenizer=tokenizer,
        question=question,
        options=options,
    )


def mask_structured_mcqa_prompt_token_ids(
    tokenizer,
    question: str,
    options: list,
    mask_indices,
):
    """
    Wrapper compatibile col vecchio nome.
    Ora applica masking WORD-LEVEL SOLO sulla domanda.
    """
    return mask_structured_mcqa_prompt_words(
        tokenizer=tokenizer,
        question=question,
        options=options,
        mask_indices=mask_indices,
    )