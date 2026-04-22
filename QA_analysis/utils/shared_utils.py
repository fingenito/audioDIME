import re
import os
import torch
from qwen_omni_utils import process_mm_info
from typing import Any, Dict, List, Optional, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

_MUSICCAPS_SEGMENT_RE = re.compile(
    r"^\[[^\]]+\]-\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\.(wav|flac|mp3|ogg|m4a)$",
    re.IGNORECASE,
)

def _build_qwen25_audio_messages(audio_path: str, prompt: str):
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
    Mantiene solo i token generati dell'assistente fino al primo <|im_end|>.
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


def generate_shared_hummusqa_baseline(
    model,
    processor,
    audio_path: str,
    prompt: str,
    max_new_tokens: int = 16,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Genera UNA sola baseline condivisa tra DIME e MM-SHAP.

    Restituisce:
    - baseline_answer: risposta testuale del modello
    - input_ids: input tokenizzati Qwen2.5-Omni
    - output_ids: token generati della baseline
    - prompt/audio_path: per validazione coerenza downstream

    Nota:
    - salva input_ids/output_ids su CPU per evitare retention inutile su GPU
    - NON modifica la metodologia di nessuna analisi; serve solo a condividere
      il punto di partenza tra analisi diverse.
    """
    if verbose:
        print("Generazione baseline condivisa...")

    messages = _build_qwen25_audio_messages(audio_path, prompt)

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
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
            output_logits=False,
            use_cache=False,
        )

    sequences = outputs.sequences
    input_len = int(inputs["input_ids"].shape[1])

    gen_ids = sequences[0, input_len:]
    gen_ids = _trim_generated_ids_at_first_im_end(gen_ids, processor)

    baseline_answer = processor.tokenizer.decode(
        gen_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    if verbose:
        print(f"Baseline condivisa: {baseline_answer}")

    return {
        "audio_path": str(audio_path),
        "prompt": str(prompt),
        "baseline_answer": str(baseline_answer),
        "input_ids": inputs["input_ids"].detach().cpu().clone(),
        "output_ids": gen_ids.unsqueeze(0).detach().cpu().clone(),
        "max_new_tokens": int(max_new_tokens),
    }

def prepare_qwen25_omni_inputs(
    processor,
    conversation,
    device=None,
    dtype=None,
    use_audio_in_video: bool = False,
):
    """
    Pipeline ufficiale Qwen2.5-Omni:
    apply_chat_template(tokenize=False) -> process_mm_info -> processor(...)
    """
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=use_audio_in_video,
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )

    if device is not None:
        inputs = inputs.to(device)
    if dtype is not None:
        try:
            inputs = inputs.to(dtype)
        except Exception:
            pass

    return text, inputs
def _build_full_prompt_with_prefix_ids(
    tokenizer,
    prompt: str,
    prefix_ids: list,
) -> str:
    """
    Costruisce il prompt completo per la valutazione autoregressiva
    del token target, identico alla logica single-example già usata.
    """
    if prefix_ids:
        prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_ids)
        prefix_text = tokenizer.convert_tokens_to_string(prefix_tokens)
        full_prompt = prompt + " " + prefix_text if prefix_text else prompt
    else:
        full_prompt = prompt
    return full_prompt

def _unwrap_singleton_audio_container(x):
    """
    Alcune versioni / casi di process_mm_info restituiscono l'audio
    con nesting inutile, ad esempio:
        ['/path.wav']
        [['/path.wav']]
    Questo helper rimuove SOLO contenitori singleton annidati.
    NON tocca strutture reali con più elementi.
    """
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x

def prepare_qwen25_omni_inputs_batch(
    processor,
    conversations: List[List[Dict[str, Any]]],
    device=None,
    dtype=None,
    use_audio_in_video: bool = False,
):
    """
    Versione batchata della pipeline Qwen2.5-Omni per il caso AUDIO+TEXT.

    Importantissimo:
    - ogni conversazione nel nostro setup contiene 1 solo audio
    - costruiamo un batch vero sui testi
    - costruiamo una lista piatta di audio items
    - rimuoviamo eventuali singleton annidati prodotti da process_mm_info
    """
    if not isinstance(conversations, list) or len(conversations) == 0:
        raise ValueError("conversations must be a non-empty list")

    texts: List[str] = []
    audio_items: List[Any] = []

    for conv in conversations:
        text = processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=False,
        )

        audios, images, videos = process_mm_info(
            conv,
            use_audio_in_video=use_audio_in_video,
        )

        if images not in (None, [], ()):
            raise RuntimeError(
                "prepare_qwen25_omni_inputs_batch currently supports only audio+text conversations "
                f"(found non-empty images: {type(images)})"
            )

        if videos not in (None, [], ()):
            raise RuntimeError(
                "prepare_qwen25_omni_inputs_batch currently supports only audio+text conversations "
                f"(found non-empty videos: {type(videos)})"
            )

        if audios is None:
            raise RuntimeError("process_mm_info returned audios=None for an audio conversation")

        audio_item = _unwrap_singleton_audio_container(audios)

        # Se dopo l'unwrap è ancora una lista/tupla, allora non è più il caso previsto
        # del nostro setup (1 solo audio per esempio) e vogliamo fallire in modo chiaro.
        if isinstance(audio_item, (list, tuple)):
            raise RuntimeError(
                f"Batch audio item still nested after singleton unwrap: type={type(audio_item)}, value={audio_item}"
            )

        texts.append(text)
        audio_items.append(audio_item)

    inputs = processor(
        text=texts,
        audio=audio_items,
        images=None,
        videos=None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )

    if device is not None:
        inputs = inputs.to(device)

    if dtype is not None:
        try:
            inputs = inputs.to(dtype)
        except Exception:
            pass

    return texts, inputs

def get_token_logit_autoregressive_id_batch(
    model,
    processor,
    batch_items: List[Dict[str, Any]],
) -> List[float]:
    """
    Valuta in UN SOLO forward batchato più esempi del tipo:

      M_k(audio, prompt) = logit(target_token_id | audio, prompt, prefix)

    batch_items: lista di dict con chiavi:
      - audio_path: str
      - prompt: str
      - prefix_ids: list[int]
      - target_id: int

    Ritorna una lista di float, stesso ordine di batch_items.

    Questa funzione NON cambia la metodologia DIME:
    è identica alla single-example, solo eseguita in minibatch reale.
    """
    if not batch_items:
        return []

    tok = processor.tokenizer
    conversations = []
    target_ids: List[int] = []

    for item in batch_items:
        audio_path = str(item["audio_path"])
        prompt = str(item["prompt"])
        prefix_ids = list(item.get("prefix_ids", []))
        target_id = int(item["target_id"])

        full_prompt = _build_full_prompt_with_prefix_ids(
            tokenizer=tok,
            prompt=prompt,
            prefix_ids=prefix_ids,
        )

        conv = _build_qwen25_audio_messages(audio_path, full_prompt)
        conversations.append(conv)
        target_ids.append(target_id)

    _texts, inputs = prepare_qwen25_omni_inputs_batch(
        processor=processor,
        conversations=conversations,
        device=model.device,
        dtype=getattr(model, "dtype", None),
        use_audio_in_video=False,
    )

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)

    logits = outputs.logits[:, -1, :]   # [B, V]
    vocab_size = int(logits.size(-1))

    vals: List[float] = []
    for b, target_id in enumerate(target_ids):
        if not isinstance(target_id, int) or target_id < 0 or target_id >= vocab_size:
            vals.append(float(logits[b].mean().item()))
        else:
            vals.append(float(logits[b, target_id].item()))

    return vals

def ask_yes_no(question):
    while True:
        reply = input(f"{question} (s/n): ").strip().lower()
        if reply in ["s", "si", "y", "yes"]:
            return True
        if reply in ["n", "no"]:
            return False
        print("Risposta non valida, inserisci 's' o 'n'.")



def generate_caption(model, processor, audio_path, prompt):

    try:
        print("Generazione caption base...")

        messages = _build_qwen25_audio_messages(audio_path, prompt)
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
                use_cache=False
            )

        sequences = outputs.sequences
        input_len = int(inputs["input_ids"].shape[1])
        gen_ids = sequences[0, input_len:]

        tok = processor.tokenizer
        try:
            im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            im_end_id = getattr(tok, "eos_token_id", None)

        if im_end_id is not None:
            cut = len(gen_ids)
            for i in range(len(gen_ids)):
                if int(gen_ids[i].item()) == int(im_end_id):
                    cut = i
                    break
            gen_ids = gen_ids[:cut]

        response = tok.decode(
            gen_ids.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        print(f"Caption generata: {response}")
        return response

    except Exception as e:
        print(f"Errore nella generazione della caption: {str(e)}")
        raise

def tokenize_caption_for_mmshap(text, processor, return_ids: bool = False):
    tok = processor.tokenizer
    ids = tok.encode(text, add_special_tokens=False)
    raw_tokens = tok.convert_ids_to_tokens(ids)

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
    processor,
    audio_path: str,
    prompt: str,
    prefix_ids: list,
    target_id: int,
) -> float:


    tok = processor.tokenizer

    if prefix_ids:
        prefix_tokens = tok.convert_ids_to_tokens(prefix_ids)
        prefix_text = tok.convert_tokens_to_string(prefix_tokens)
        full_prompt = prompt + " " + prefix_text if prefix_text else prompt
    else:
        full_prompt = prompt

    messages = _build_qwen25_audio_messages(audio_path, full_prompt)
    _text, inputs = prepare_qwen25_omni_inputs(
        processor=processor,
        conversation=messages,
        device=model.device,
        dtype=getattr(model, "dtype", None),
        use_audio_in_video=False,
    )

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)

    logits = outputs.logits[0, -1]
    log_probs = torch.log_softmax(logits, dim=-1)

    vocab_size = int(log_probs.size(-1))
    if not isinstance(target_id, int) or target_id < 0 or target_id >= vocab_size:
        return float(log_probs.mean().item())

    return float(log_probs[target_id].item())


def get_token_logit_autoregressive_id(
    model,
    processor,
    audio_path: str,
    prompt: str,
    prefix_ids: list,
    target_id: int,
) -> float:

    tok = processor.tokenizer

    if prefix_ids:
        prefix_tokens = tok.convert_ids_to_tokens(prefix_ids)
        prefix_text = tok.convert_tokens_to_string(prefix_tokens)
        full_prompt = prompt + " " + prefix_text if prefix_text else prompt
    else:
        full_prompt = prompt

    messages = _build_qwen25_audio_messages(audio_path, full_prompt)
    _text, inputs = prepare_qwen25_omni_inputs(
        processor=processor,
        conversation=messages,
        device=model.device,
        dtype=getattr(model, "dtype", None),
        use_audio_in_video=False,
    )

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)

    logits = outputs.logits[0, -1]
    vocab_size = int(logits.size(-1))

    if not isinstance(target_id, int) or target_id < 0 or target_id >= vocab_size:
        return float(logits.mean().item())

    return float(logits[target_id].item())



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


def build_hummusqa_qwen25_prompt_parts(question: str, options: list) -> Dict[str, Any]:
    """
    Restituisce le parti strutturate del prompt HumMusQA per Qwen2.5-Omni.

    Struttura:
      - prefix: scaffolding fisso NON perturbabile
      - question: unico contenuto semantico perturbabile nel setup ufficiale
      - options_block: contenuto SEMPRE fisso nel setup ufficiale
      - suffix: scaffolding fisso NON perturbabile
    """
    q = str(question).strip()
    opts = [str(x).strip() for x in (options or [])]

    if len(opts) != 4 or any(x == "" for x in opts):
        raise ValueError(f"Expected exactly 4 non-empty options, got: {opts}")

    prefix = (
        "You are a music audio understanding model.\n"
        "Listen carefully to the provided audio clip. Answer the following multiple-choice\n"
        "question based on what you hear.\n"
        "Question:\n"
    )

    options_header = "Options:\n"
    options_lines = [f"({chr(65+i)}) {opt}" for i, opt in enumerate(opts)]
    options_block = "\n".join(options_lines)

    suffix = (
        "\nRespond with ONLY the letter of the correct option (A, B, C, or D).\n"
        "Do not include any explanation or additional text."
    )

    return {
        "prefix": prefix,
        "question": q,
        "options_header": options_header,
        "options_lines": options_lines,
        "options_block": options_block,
        "suffix": suffix,
    }


def build_hummusqa_qwen25_prompt_from_parts(parts: Dict[str, Any]) -> str:
    """
    Ricompone il prompt completo da parti strutturate.
    """
    prefix = str(parts["prefix"])
    question = str(parts["question"])
    options_header = str(parts["options_header"])
    options_block = str(parts["options_block"])
    suffix = str(parts["suffix"])

    return f"{prefix}{question}\n{options_header}{options_block}{suffix}"


def build_hummusqa_qwen25_prompt(question: str, options: list) -> str:
    """
    Prompt completo allineato al formato HumMusQA / Qwen2.5-Omni.
    Manteniamo questa funzione per backward compatibility.
    """
    parts = build_hummusqa_qwen25_prompt_parts(question, options)
    return build_hummusqa_qwen25_prompt_from_parts(parts)




def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """
    Ritorna l'indice iniziale della prima occorrenza di needle in haystack.
    Se non trovato, ritorna -1.
    """
    if not needle or not haystack:
        return -1
    n = len(needle)
    m = len(haystack)
    if n > m:
        return -1

    for i in range(m - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def extract_hummusqa_question_only_tokens_from_input_ids_qwen25(
    processor,
    input_ids,
    question: str,
):
    """
    Estrae SOLO i token della domanda dentro l'input Qwen2.5-Omni,
    senza includere:
      - options
      - scaffolding istruzionale
      - token speciali/audio/chat

    Strategia:
      1) trova lo span utente tra <|audio_eos|> e <|im_end|>
      2) tokenizza la sola stringa 'question'
      3) cerca quella sottosequenza nello span utente
      4) restituisce i token della domanda e il loro intervallo globale

    Returns:
        question_ids_t: 1D tensor dei token della sola domanda
        n_question_tokens: int
        interval: (start_idx, end_idx) nello input_ids completo
                  con convenzione [start_idx:end_idx]
    """
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape [1, T], got {tuple(input_ids.shape)}")

    tok = processor.tokenizer
    ids = input_ids[0]
    ids_list = ids.detach().cpu().tolist()

    audio_eos_id = getattr(tok, "audio_eos_token_id", None)
    if audio_eos_id is None:
        try:
            audio_eos_id = tok.convert_tokens_to_ids("<|audio_eos|>")
        except Exception:
            audio_eos_id = 151648

    im_end_id = getattr(tok, "eos_token_id", None)
    if im_end_id is None:
        try:
            im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            im_end_id = 151645

    try:
        audio_eos_pos = ids_list.index(int(audio_eos_id))
    except ValueError:
        raise RuntimeError("audio_eos token not found in input_ids")

    im_end_pos = None
    for i in range(audio_eos_pos + 1, len(ids_list)):
        if int(ids_list[i]) == int(im_end_id):
            im_end_pos = i
            break

    if im_end_pos is None:
        raise RuntimeError("user im_end token not found after audio_eos")

    user_start = audio_eos_pos + 1
    user_end = im_end_pos

    if user_end <= user_start:
        raise RuntimeError(f"Empty user text span detected: start={user_start}, end={user_end}")

    user_span_ids = ids_list[user_start:user_end]

    question = str(question).strip()
    if not question:
        raise RuntimeError("Question string is empty")

    question_ids = tok.encode(question, add_special_tokens=False)
    if not question_ids:
        raise RuntimeError("Question tokenization returned an empty sequence")

    local_start = _find_subsequence(user_span_ids, question_ids)
    if local_start == -1:
        raise RuntimeError(
            "Could not locate question token subsequence inside user text span. "
            "This likely means the prompt formatting or tokenization changed."
        )

    local_end = local_start + len(question_ids)

    global_start = user_start + local_start
    global_end = user_start + local_end

    question_ids_t = ids[global_start:global_end].detach().clone()

    return question_ids_t, int(question_ids_t.numel()), (int(global_start), int(global_end))

def load_hummusqa_entries_parquet(dataset_root: str):
    from datasets import load_dataset, Audio

    parquet_files = []
    for root, _dirs, files in os.walk(dataset_root):
        for fn in files:
            if fn.lower().endswith(".parquet"):
                parquet_files.append(os.path.join(root, fn))

    parquet_files = sorted(parquet_files)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {dataset_root}")

    ds = load_dataset("parquet", data_files={"test": parquet_files})
    split = ds["test"]

    if "audio" in split.column_names:
        split = split.cast_column("audio", Audio(decode=False))

    entries = [dict(x) for x in split]
    return entries, parquet_files

def extract_only_question_text_span(prompt: str) -> tuple:
    """
    Estrae SOLO il testo della domanda dal prompt HumMusQA.

    Ritorna:
        (start_char, end_char) nel prompt string
    """
    q_start = prompt.find("Question:\n")
    if q_start == -1:
        raise RuntimeError("Cannot find 'Question:' in prompt")

    q_start += len("Question:\n")

    opt_start = prompt.find("Options:\n")
    if opt_start == -1:
        raise RuntimeError("Cannot find 'Options:' in prompt")

    return q_start, opt_start
