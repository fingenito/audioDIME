import random


def mask_text_tokens(tokenizer, prompt, mask_indices):
    """
    Maschera i token del prompt basandosi sugli INDICI dei token.
    """
    if mask_indices is None:
        mask_indices = []

    mask_indices = [x for x in mask_indices if x is not None and isinstance(x, (int, float))]
    mask_indices = [int(x) for x in mask_indices if int(x) >= 0]

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)

    masked_tok = (
        tokenizer.mask_token
        if hasattr(tokenizer, "mask_token") and tokenizer.mask_token
        else "[MASK]"
    )

    for mi in mask_indices:
        if 0 <= mi < len(tokens):
            tokens[mi] = masked_tok

    masked_prompt = tokenizer.convert_tokens_to_string(tokens)
    return masked_prompt


def create_random_permutations(num_audio_windows, num_text_tokens, num_permutations=10):
    permutations = []

    audio_feats = [("audio", i) for i in range(num_audio_windows)]
    text_feats = [("text", i) for i in range(num_text_tokens)]
    all_feats = audio_feats + text_feats

    for _ in range(num_permutations):
        perm = all_feats.copy()
        random.shuffle(perm)
        permutations.append(perm)

    return permutations

def mask_text_token_ids(tokenizer, text: str, mask_indices):
    """
    Paper-compliant masking (id-level): maschera per INDICI di token della tokenizzazione di `text`
    sostituendo gli ids con mask_id (se esiste) altrimenti 0 (stile repo originale).
    Ritorna: (masked_text_str, original_ids, masked_ids)
    """
    if mask_indices is None:
        mask_indices = []
    mask_indices = [int(x) for x in mask_indices if isinstance(x, (int, float)) and int(x) >= 0]

    ids = tokenizer.encode(text, add_special_tokens=False)
    masked_ids = list(ids)

    # Preferisci mask_token_id se definito; altrimenti usa 0 (coerente con implementazioni che azzerano token)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = 0

    for mi in mask_indices:
        if 0 <= mi < len(masked_ids):
            masked_ids[mi] = int(mask_id)

    # Decodifica senza cleanup aggressivo per ridurre drift di spaziatura
    try:
        masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        # fallback per tokenizers che non supportano clean_up_tokenization_spaces
        masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)

    return masked_text, ids, masked_ids