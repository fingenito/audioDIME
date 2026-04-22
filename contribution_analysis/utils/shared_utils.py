import os
import string
import re
from typing import Any, Dict, List, Tuple, Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

_MUSICCAPS_SEGMENT_RE = re.compile(
    r"^\[[^\]]+\]-\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\.(wav|flac|mp3|ogg|m4a)$",
    re.IGNORECASE,
)


def is_musiccaps_segment_file(path: str) -> bool:
    name = os.path.basename(path or "")
    return bool(_MUSICCAPS_SEGMENT_RE.match(name))


def list_audio_files(
    audio_dir: str,
    recursive: bool = True,
    dataset_name: Optional[str] = None,
    prefer_musiccaps_segmented: bool = True,
) -> List[str]:
    """
    Lista file audio in modo consistente per tutto il progetto.

    Per MusicCaps:
    - se sono presenti file segmentati del tipo [ytid]-[start-end].wav,
      usa SOLO quelli (più coerenti col dataset ufficiale).
    - altrimenti fallback a tutti gli audio trovati.
    """
    if not audio_dir or not os.path.isdir(audio_dir):
        return []

    out: List[str] = []

    if recursive:
        for root, _dirs, files in os.walk(audio_dir):
            for fn in sorted(files):
                if fn.lower().endswith(_AUDIO_EXTS):
                    p = os.path.join(root, fn)
                    if os.path.isfile(p):
                        out.append(p)
    else:
        for fn in sorted(os.listdir(audio_dir)):
            if fn.lower().endswith(_AUDIO_EXTS):
                p = os.path.join(audio_dir, fn)
                if os.path.isfile(p):
                    out.append(p)

    out = sorted(set(out))

    if dataset_name and dataset_name.strip().lower() == "musiccaps" and prefer_musiccaps_segmented:
        segmented = [p for p in out if is_musiccaps_segment_file(p)]
        if segmented:
            return sorted(segmented)

    return out


def ask_yes_no(question):
    while True:
        reply = input(f"{question} (s/n): ").strip().lower()
        if reply in ["s", "si", "y", "yes"]:
            return True
        if reply in ["n", "no"]:
            return False
        print("Risposta non valida, inserisci 's' o 'n'.")



def generate_caption(model, tokenizer, audio_path, prompt):
    try:
        print("Generazione caption base...")
        query = tokenizer.from_list_format([
            {'audio': audio_path},
            {'text': prompt}
        ])
        audio_info = tokenizer.process_audio(query)
        response, _ = model.chat(tokenizer, query=query, audio_info=audio_info, history=None)
        print(f"Caption generata: {response}")
        return response
    except Exception as e:
        print(f"Errore nella generazione della caption: {str(e)}")
        raise


def tokenize_caption_for_mmshap(text, tokenizer, return_ids: bool = False):
    ids = tokenizer.encode(text, add_special_tokens=False)
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)

    clean = []
    for t in raw_tokens:
        if isinstance(t, bytes):
            t = t.decode("utf-8", errors="ignore")
        clean.append(str(t).replace("\n", "").replace("\t", ""))

    if return_ids:
        return ids, clean
    return clean



def get_token_logprob_autoregressive_id(
    model,
    tokenizer,
    audio_path: str,
    prompt: str,
    prefix_ids: list,
    target_id: int,
) -> float:
    import torch

    if prefix_ids:
        prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_ids)
        prefix_text = tokenizer.convert_tokens_to_string(prefix_tokens)
        full_prompt = prompt + " " + prefix_text if prefix_text else prompt
    else:
        full_prompt = prompt

    query = tokenizer.from_list_format([
        {"audio": audio_path},
        {"text": full_prompt}
    ])

    audio_info = tokenizer.process_audio(query)

    inputs = tokenizer(
        query,
        return_tensors="pt",
        audio_info=audio_info
    ).to(model.device)

    # IMPORTANT: use_cache=False riduce memoria (niente KV cache)
    with torch.inference_mode():
        outputs = model(**inputs, audio_info=audio_info, use_cache=False)

    logits = outputs.logits[0, -1]
    log_probs = torch.log_softmax(logits, dim=-1)

    vocab_size = int(log_probs.size(-1))
    if not isinstance(target_id, int) or target_id < 0 or target_id >= vocab_size:
        return float(log_probs.mean().item())

    return float(log_probs[target_id].item())


def get_token_logit_autoregressive_id(
    model,
    tokenizer,
    audio_path: str,
    prompt: str,
    prefix_ids: list,
    target_id: int,
) -> float:
    """
    DIME Step 0 (paper/repo aligned):
    ritorna il PRE-SOFTMAX LOGIT del token target al passo autoregressivo corrente.

    Nel paper DIME l'output M(x1,x2) è "pre-softmax logits".
    Qui, per captioning, l'analogo più fedele è il logit del token target t_k
    dato (audio, prompt, prefisso t_<k).
    """
    import torch

    if prefix_ids:
        prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_ids)
        prefix_text = tokenizer.convert_tokens_to_string(prefix_tokens)
        full_prompt = prompt + " " + prefix_text if prefix_text else prompt
    else:
        full_prompt = prompt

    query = tokenizer.from_list_format([
        {"audio": audio_path},
        {"text": full_prompt}
    ])

    audio_info = tokenizer.process_audio(query)

    inputs = tokenizer(
        query,
        return_tensors="pt",
        audio_info=audio_info
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, audio_info=audio_info, use_cache=False)

    logits = outputs.logits[0, -1]  # pre-softmax logits (vocab_size,)
    vocab_size = int(logits.size(-1))

    if not isinstance(target_id, int) or target_id < 0 or target_id >= vocab_size:
        # fallback robusto (evita crash)
        return float(logits.mean().item())

    return float(logits[target_id].item())


def merge_word_tokens(caption_tokens):
    token_list = []
    for t in caption_tokens:
        if isinstance(t, (bytes, bytearray)):
            s = t.decode("utf-8", errors="ignore")
        else:
            s = str(t)
        token_list.append(s)

    words = []
    mapping = []

    current_word = ""
    current_indices = []

    for idx, tok in enumerate(token_list):
        if tok.startswith(" "):
            if current_indices:
                words.append(current_word)
                mapping.append((current_word, current_indices))

            current_word = tok.lstrip()
            current_indices = [idx]
        else:
            current_word += tok
            current_indices.append(idx)

    if current_indices:
        words.append(current_word)
        mapping.append((current_word, current_indices))

    return words, mapping, token_list


def filter_punctuation(mapping, token_list):
    filtered_mapping = []
    filtered_words = []

    for word, indices in mapping:
        clean = word.translate(str.maketrans({
            "“": "", "”": "", "‘": "", "’": ""
        }))
        stripped = clean.strip(string.punctuation)
        if stripped == "":
            continue

        filtered_mapping.append((stripped, indices))
        filtered_words.append(stripped)

    return filtered_mapping, filtered_words



# ======================================================================================
# WORD-LEVEL UTILS (POSTPROCESS ONLY) — robust for byte-level BPE (Qwen2 / Qwen-Audio)
# ======================================================================================
from typing import Any, Dict, List, Optional, Tuple

def _to_int_list(xs: List[Any]) -> List[int]:
    out = []
    for x in xs or []:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out

def _clean_word_label_ws(s: str) -> str:
    # normalize whitespace for display only
    if s is None:
        return ""
    s = str(s).replace("\n", " ").replace("\t", " ").replace("\r", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()

def build_word_groups_from_token_ids(
    tokenizer,
    token_ids: List[int],
    drop_empty: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build word groups from token IDs using tokenizer.decode incrementally.
    This is robust for byte-level BPE (Qwen2/Qwen-Audio), GPT-2-like, etc.

    Idea:
      - decode prefix up to i (exclusive) and up to i+1, and take the delta
      - the delta is the new text contributed by token i
      - we split deltas by whitespace boundaries to form word groups
      - we maintain mapping word -> token_indices

    Output groups:
      [{"label": "word", "raw": "word", "token_indices": [...]}]
    """
    ids = _to_int_list(token_ids)
    if not ids:
        return []

    # decode prefix texts
    # we keep cleanup disabled where possible to preserve spacing behavior
    def _decode(prefix_ids: List[int]) -> str:
        try:
            return tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except TypeError:
            return tokenizer.decode(prefix_ids, skip_special_tokens=False)

    groups: List[Dict[str, Any]] = []
    current_raw = ""
    current_token_indices: List[int] = []

    prev_text = _decode([])

    for i in range(len(ids)):
        cur_text = _decode(ids[: i + 1])
        # delta contributed by this token
        delta = cur_text[len(prev_text):] if cur_text.startswith(prev_text) else cur_text
        prev_text = cur_text

        if delta == "":
            # sometimes special/empty behavior; still keep token mapping if needed
            if not current_token_indices:
                current_token_indices = [i]
            else:
                current_token_indices.append(i)
            continue

        # If delta contains whitespace, it may start a new word boundary
        # Example: " hello" or " world"
        parts: List[str] = []
        buf = ""
        for ch in delta:
            if ch.isspace():
                if buf != "":
                    parts.append(buf)
                    buf = ""
                # whitespace triggers boundary; represent as special marker
                parts.append(" ")
            else:
                buf += ch
        if buf != "":
            parts.append(buf)

        for p in parts:
            if p == " ":
                # boundary: flush current word
                if current_token_indices:
                    label = _clean_word_label_ws(current_raw)
                    if (not drop_empty) or (label != ""):
                        groups.append({
                            "label": label,
                            "raw": current_raw,
                            "token_indices": list(current_token_indices),
                        })
                    current_raw = ""
                    current_token_indices = []
                # else ignore multiple spaces
                continue

            # normal characters: accumulate
            if not current_token_indices:
                current_token_indices = [i]
            else:
                # token i contributes to this word too
                if current_token_indices[-1] != i:
                    current_token_indices.append(i)
            current_raw += p

        # Ensure token i belongs somewhere even if delta had only spaces
        if delta.strip() == "" and not current_token_indices:
            current_token_indices = [i]

    # flush last
    if current_token_indices:
        label = _clean_word_label_ws(current_raw)
        if (not drop_empty) or (label != ""):
            groups.append({
                "label": label,
                "raw": current_raw,
                "token_indices": list(current_token_indices),
            })

    # Post-clean: merge empty labels if any slipped through
    cleaned: List[Dict[str, Any]] = []
    for g in groups:
        lbl = _clean_word_label_ws(g.get("label", ""))
        if drop_empty and lbl == "":
            continue
        g["label"] = lbl
        cleaned.append(g)

    return cleaned

def aggregate_vector_by_groups(vec: List[float], groups: List[Dict[str, Any]]) -> List[float]:
    v = [float(x) for x in (vec or [])]
    out: List[float] = []
    for g in (groups or []):
        s = 0.0
        for idx in g.get("token_indices", []):
            try:
                ii = int(idx)
            except Exception:
                continue
            if 0 <= ii < len(v):
                s += float(v[ii])
        out.append(float(s))
    return out

def aggregate_matrix_rows_by_groups(mat: List[List[float]], row_groups: List[Dict[str, Any]]) -> List[List[float]]:
    if mat is None or len(mat) == 0:
        return []
    n_cols = len(mat[0]) if isinstance(mat[0], list) else 0
    out: List[List[float]] = []
    for g in (row_groups or []):
        acc = [0.0] * n_cols
        for idx in g.get("token_indices", []):
            try:
                ii = int(idx)
            except Exception:
                continue
            if 0 <= ii < len(mat):
                row = mat[ii]
                for c in range(n_cols):
                    acc[c] += float(row[c])
        out.append([float(x) for x in acc])
    return out