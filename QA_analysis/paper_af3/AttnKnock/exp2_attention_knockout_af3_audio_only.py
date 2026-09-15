"""
Attention Knockout for HumMusQA + Audio Flamingo 3 — variante audio_only
===============================================================================
Copia adattata di paper_af3/AttnKnock/exp2_attention_knockout_af3.py (AF3,
caso complete, gia' validato con run completo su 320 campioni), STESSO
IDENTICO ESPERIMENTO applicato alla variante audio_only.

Unica differenza funzionale: la domanda testuale reale di ogni campione viene
sostituita dal placeholder NULL_QUESTION = "[MASK]" prima di costruire il
prompt (vedi prepare_inputs) -- l'audio resta quello vero del campione.
Stesso identico placeholder gia' usato per Qwen
(paper/AttnKnock/exp2_attention_knockout_audio_only.py) ed Exp E
(paper/Faithfulness_audio_only, paper_af3/Faithfulness_audio_only).

Tutta la parte di caricamento/hook/moduli interni resta quella gia'
debuggata e validata per il caso complete (vedi sotto per i dettagli
originali di quel porting) -- nessuna modifica architetturale qui.

Copia adattata di experiments/Exp_Net/exp2_attention_knockout_hummusqa.py
(Qwen2.5-Omni, caso complete audio+testo), STESSO IDENTICO ESPERIMENTO
(stesso meccanismo di hook, stesse regole, stesse componenti, stessa
metodologia decision-logits/generation) applicato ad Audio Flamingo 3.

Feasibility verificata (2026-09, af3_knockout_feasibility_probe.py):
  - model.language_model.layers[i] e' un Qwen2DecoderLayer standard con
    sottomodulo .self_attn -- stesso meccanismo di hook applicabile.
  - config.audio_token_id (non audio_token_index come in Qwen2.5-Omni)
    conferma che l'audio e' concatenato nella sequenza di token, non
    gestito via cross-attention dedicata.
  - Confermato anche dalla documentazione ufficiale: model card HF
    ("MLP-based audio adaptor, Decoder-only LLM backbone Qwen2.5-7B") e
    paper arXiv 2507.08128 Sez. 3.1 ("adapted embeddings serve as prompts
    to the LLM, alongside the textual instruction").
  - Unica differenza reale: i kwargs ricevuti da self_attn usano
    past_key_values (plurale, oggetto Cache) e position_ids invece di
    cache_position/past_key_value (singolare) -- vedi il fallback esteso
    in DecisionAttentionKnockoutHook piu' sotto.

Adaptation of AVLLM's attention knockout experiment to an audio-text MCQ task.

Original AVLLM idea
-------------------
The original repository blocks attention edges by installing forward pre-hooks
on `model.thinker.model.layers[i].self_attn`. A rule has the form:

    (source_token_type, target_token_type, start_layer, end_layer)

During a forward/generation step, the hook edits the 4D attention mask so that
queries of the source type cannot attend to keys of the target type in the
selected layer window.

HumMusQA adaptation
-------------------
This script supports two complementary paths:

Path A: generation knockout, closest to the original AVLLM experiment.
  1. Build the same HumMusQA audio+text prompt used by the project pipeline.
  2. Generate the model answer with model.generate().
  3. For each component and layer window, block attention:

         generated -> component   (literal AVLLM)
         decision  -> component   (MCQ first-token causal variant)

     and generate again.
  4. Save response flips and correctness changes.

Path B: decision-logit knockout, a stable diagnostic at the MCQ decision point.

  1. Build the same HumMusQA audio+text prompt used by the project pipeline.
  2. Classify prompt tokens into audio, instruction, question, options,
     other_text, and mark the last prompt token as `decision`.
  3. Compute baseline logits for A/B/C/D at the final prompt position.
  4. For each component and layer window, block attention:

         decision -> component

     and recompute A/B/C/D logits.
  5. Save probability/logit drops for the correct answer and baseline answer.

The generation path is the primary AVLLM-style protocol. The decision-logit path
is useful as an auxiliary, faster diagnostic over restricted A/B/C/D logits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from datasets import Audio as HFAudio
from datasets import load_dataset
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

# _build_af3_conversation: stesso helper gia' validato e usato in produzione
# per tutti i run Exp E AF3 (gpu_utils_af3.py) -- equivalente AF3 di
# process_mm_info+build_conversation per costruire l'input audio+testo.
# Import diretto (nessun sys.path hack): paper_af3/AttnKnock/ e' un package
# Python vero (__init__.py), lanciato con `python -m` dalla project root
# esattamente come tutti gli altri script del progetto.
from QA_analysis.utils.gpu_utils_af3 import _build_af3_conversation


# audio_only: placeholder che sostituisce la domanda testuale reale di ogni
# campione -- stesso identico placeholder usato per Qwen audio_only ed Exp E.
NULL_QUESTION = "[MASK]"

TEXT_COMPONENT_TYPES = ("instruction", "question", "options", "other_text")
TOKEN_TYPES = TEXT_COMPONENT_TYPES + (
    "query_text",
    "audio",
    "image",
    "video",
    "decision",
    "generated",
    "other",
)
TOKEN_TYPE_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(TOKEN_TYPES)}

# Virtual compound component: ALL text token types are blocked from attending
# to audio simultaneously within a layer window.
#
# This measures the *total* causal effect of audio on the MCQ decision — not
# just the direct decision→audio path (which is what the individual "audio"
# component measures).  When all text types are cut off from audio in a window,
# the model cannot build or refresh any audio-grounded representations there.
#
# Expected behaviour: much larger delta_prob_correct than plain "audio"
# component (which only blocks one token), concentrated in early windows
# where audio is primarily integrated into text representations.
ALL_TEXT_TO_AUDIO = "all_text_to_audio"
_ALL_TEXT_SOURCE_TYPES = ("decision", "instruction", "question", "options", "other_text")


# =============================================================================
# Generic helpers
# =============================================================================


def normalize_difficulty(x: object) -> str:
    s = str(x or "").strip()
    return "" if not s else s[:1].upper() + s[1:].lower()


def safe_identifier(sample: Dict, idx: int) -> str:
    for key in ("identifier", "id", "sample_id", "track_id", "question_id"):
        if key in sample and sample[key] is not None:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample[key]))[:180]
    return f"sample_{idx:05d}"


def get_module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        pass
    raise RuntimeError(f"Cannot infer device for module {module.__class__.__name__}")


def get_thinker_module(model):
    """
    Per Qwen2.5-Omni "thinker" e' il sotto-modulo multimodale (model.thinker).

    BUGFIX (trovato durante lo smoke test, 2026-09): AF3
    (AudioFlamingo3ForConditionalGeneration) NON ha language_model/audio_tower
    come attributi diretti del wrapper esterno -- ha un .model interno
    (il modello multimodale base) che li contiene. Verificato con
    af3_knockout_feasibility_probe.py: i path reali erano
    "model.language_model.layers" e "model.audio_tower.layers" (col prefisso
    "model." incluso), non "language_model.layers" diretti sul top-level.
    Avevo letto quell'output in modo troppo affrettato la prima volta,
    saltando il livello ".model" in mezzo. get_thinker_module ritorna quindi
    model.model (l'equivalente strutturale del "thinker" di Qwen2.5-Omni),
    non il wrapper esterno.
    """
    return model.model


def get_text_model_module(model):
    """
    Qwen2.5-Omni: model.thinker.model (il transformer testuale dentro il thinker).
    AF3: model.language_model -- ha gia' .layers direttamente (verificato con
    af3_knockout_feasibility_probe.py: path='model.language_model.layers',
    layer_class='Qwen2DecoderLayer'), niente ulteriore '.model' in mezzo.
    """
    return get_thinker_module(model).language_model


def get_thinker_layers(model):
    return get_text_model_module(model).layers


def get_thinker_config(model):
    """
    Qwen2.5-Omni annida la config testuale sotto config.thinker_config.
    AF3 non ha questa nidificazione: config.audio_token_id (non
    audio_token_index) e' gia' sulla config di primo livello (verificato
    con af3_knockout_feasibility_probe.py).
    """
    return getattr(model, "config")


def get_text_backbone_device(model) -> torch.device:
    return get_module_device(get_text_model_module(model).embed_tokens)


def get_audio_tower_device(model) -> torch.device:
    return get_module_device(get_thinker_module(model).audio_tower)


def place_af3_inputs(inputs: Dict, *, device: torch.device, dtype: torch.dtype) -> Dict:
    """
    Equivalente di place_qwen_omni_inputs, semplificato: nel nostro setup
    (launch_exp2_parallel_af3.sh passa sempre --device_map "cuda:0", un solo
    GPU per shard) audio tower e language model backbone stanno sempre sullo
    stesso device fisico -- la distinzione text_device/audio_device di Qwen
    esisteva per un eventuale device_map="auto" multi-GPU che qui non usiamo
    mai in pratica (idem per l'originale: anche li' i due device coincidono
    sempre nella configurazione di lancio realmente usata). Un solo device
    di destinazione e' quindi equivalente e piu' semplice.

    Cast del dtype per i tensori floating-point (feature audio in float32
    dal processor): stesso identico passo gia' presente in _af3_worker_loop
    (gpu_utils_af3.py, usato in produzione per tutti i run Exp E AF3) --
    senza questo cast le feature audio non combaciano col bf16 del modello.
    """
    placed = {}
    for key, value in dict(inputs).items():
        if not torch.is_tensor(value):
            placed[key] = value
            continue
        value = value.to(device)
        if value.dtype.is_floating_point:
            value = value.to(dtype)
        placed[key] = value
    return placed


def debug_input_devices(inputs: Dict) -> Dict[str, str]:
    out = {}
    for key, value in inputs.items():
        out[key] = str(value.device) if torch.is_tensor(value) else type(value).__name__
    return out


def reset_qwen_omni_generation_state(model) -> None:
    """
    Workaround per un bug di TMRoPE (rope_deltas) specifico del thinker
    multimodale di Qwen2.5-Omni. AF3 usa RoPE standard di Qwen2.5-7B (nessun
    tracking di rope_deltas multimodale) -- gli hasattr guard sotto rendono
    la funzione un no-op sicuro per AF3, la lasciamo invariata (stesso nome,
    stesse chiamate nel resto del file) invece di rimuoverla per minimizzare
    il diff con l'originale.
    """
    thinker = get_thinker_module(model)
    for obj in (thinker, getattr(thinker, "model", None), getattr(thinker, "language_model", None)):
        if obj is not None and hasattr(obj, "rope_deltas"):
            try:
                obj.rope_deltas = None
            except Exception:
                pass


# =============================================================================
# HumMusQA prompt and input preparation
# =============================================================================


def build_option_orders(identifier: str, n_orders: int, seed: int) -> List[Tuple[int, int, int, int]]:
    if not 1 <= n_orders <= 24:
        raise ValueError(f"option_permutations must be in [1, 24], got {n_orders}")
    identity = (0, 1, 2, 3)
    if n_orders == 1:
        return [identity]
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    orders = list(itertools.permutations(range(4)))
    rng.shuffle(orders)
    return [tuple(int(x) for x in order) for order in orders[:n_orders]]


def get_options_from_sample(
    sample: Dict,
    option_order: Sequence[int] = (0, 1, 2, 3),
) -> Tuple[List[str], str]:
    canonical = [str(sample["answer"]).strip()] + [
        str(sample[f"distractor_{i}"]).strip() for i in (1, 2, 3)
    ]
    if sorted(option_order) != [0, 1, 2, 3]:
        raise ValueError(f"Invalid option permutation: {option_order}")
    displayed = [canonical[int(idx)] for idx in option_order]
    correct_letter = chr(65 + list(option_order).index(0))
    return displayed, correct_letter


def build_hummusqa_prompt_parts(question: str, options: Sequence[str]) -> Dict[str, object]:
    q = str(question).strip()
    opts = [str(opt).strip() for opt in options]
    if len(opts) != 4 or any(not opt for opt in opts):
        raise ValueError(f"Expected exactly 4 non-empty options, got: {opts}")

    answer_str = "\n".join(f"({chr(65 + i)}) {opt}" for i, opt in enumerate(opts))
    prefix = (
        "You are a music audio understanding model.\n"
        "Listen carefully to the provided audio clip. Answer the following multiple-choice\n"
        "question based on what you hear.\n"
        "Question:\n"
    )
    between_question_and_options = "\nOptions:\n"
    suffix = (
        "\nRespond with ONLY the letter of the correct option (A, B, C, or D).\n"
        "Do not include any explanation or additional text."
    )

    prompt = f"{prefix}{q}{between_question_and_options}{answer_str}{suffix}"
    question_start = len(prefix)
    question_end = question_start + len(q)
    options_start = question_end + len(between_question_and_options)
    options_end = options_start + len(answer_str)

    spans = {
        "instruction": [(0, question_start), (question_end, options_start), (options_end, len(prompt))],
        "question": [(question_start, question_end)],
        "options": [(options_start, options_end)],
    }
    return {"prompt": prompt, "spans": spans}


def build_conversation(prompt_text: str, audio_path: str):
    """
    Qwen2.5-Omni si aspetta {"type": "audio", "audio": path}; il processor
    AF3 si aspetta {"type": "audio", "path": path} (vedi _build_af3_conversation
    in gpu_utils_af3.py, gia' validato in produzione per tutti i run Exp E AF3).
    Deleghiamo a quella funzione per riusare esattamente la stessa convenzione,
    invece di duplicarla qui con rischio di disallineamento futuro.
    """
    return _build_af3_conversation(audio_path, prompt_text)


def _offset_prompt_spans(
    rendered_text: str,
    prompt_text: str,
    prompt_spans: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, List[Tuple[int, int]]]:
    base = rendered_text.find(prompt_text)
    if base < 0:
        print("   WARNING: prompt not found in rendered chat template; broad instruction fallback.")
        return {"instruction": [(0, len(rendered_text))]}
    rendered_spans: Dict[str, List[Tuple[int, int]]] = {}
    for name, spans in prompt_spans.items():
        shifted = [(base + int(s), base + int(e)) for s, e in spans if int(e) > int(s)]
        if shifted:
            rendered_spans[name] = shifted
    return rendered_spans


def materialize_audio(audio_field, tmp_dir: str, identifier: str) -> str:
    if isinstance(audio_field, dict):
        path = audio_field.get("path")
        if path and os.path.exists(path):
            return path
        raw = audio_field.get("bytes")
        if raw is not None:
            out_path = os.path.join(tmp_dir, f"{identifier}.wav")
            arr, sr = sf.read(io.BytesIO(raw))
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            sf.write(out_path, arr.astype(np.float32), sr)
            return out_path
        raise ValueError(f"Audio dict has no local path and no bytes. Keys: {list(audio_field.keys())}")
    raise ValueError(f"Unsupported audio format: {type(audio_field)}")


def prepare_inputs(
    sample: Dict,
    sample_idx: int,
    processor: AutoProcessor,
    tmp_dir: str,
    *,
    option_order: Sequence[int],
) -> Tuple[Dict[str, torch.Tensor], str, str, bool, str, Dict[str, object]]:
    identifier = safe_identifier(sample, sample_idx)
    # audio_only: la domanda testuale reale viene ignorata -- unica riga
    # diversa dal caso complete (era: str(sample["question"]).strip()). Le
    # opzioni di risposta restano vere, l'audio resta quello vero del
    # campione (materialize_audio piu' sotto, invariato).
    question = NULL_QUESTION
    options, correct_letter = get_options_from_sample(sample, option_order=option_order)
    prompt_parts = build_hummusqa_prompt_parts(question, options)
    prompt_text = str(prompt_parts["prompt"])

    audio_path = materialize_audio(sample["audio"], tmp_dir, identifier)
    should_delete = audio_path.startswith(tmp_dir)
    conversation = build_conversation(prompt_text, audio_path)

    # Stessa identica logica dell'originale Qwen per tracciare gli span di
    # instruction/question/options: render del chat template SENZA tokenizzare,
    # per trovare gli offset di prompt_text nel testo finale. Feature standard
    # di apply_chat_template, non specifica di Qwen -- nessun cambiamento.
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    if isinstance(text, list):
        text = text[0]
    text_spans = _offset_prompt_spans(
        rendered_text=text,
        prompt_text=prompt_text,
        prompt_spans=prompt_parts["spans"],  # type: ignore[arg-type]
    )

    # BUGFIX/adattamento: Qwen2.5-Omni usa process_mm_info() + processor(text=,
    # audio=, ...) in due passi separati. AF3 (stesso pattern gia' validato in
    # gpu_utils_af3.py per tutti i run Exp E) tokenizza e processa l'audio in
    # un solo passo con apply_chat_template(tokenize=True, return_dict=True).
    # NOTA: nessun return_tensors="pt" qui -- questo e' esattamente il
    # signature gia' validato in gpu_utils_af3.py (_af3_worker_loop, usato in
    # produzione per tutti i run Exp E AF3). Aggiungere return_tensors non
    # testato sarebbe una deviazione rispetto al pattern provato.
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    return inputs, correct_letter, audio_path, should_delete, text, text_spans


# =============================================================================
# Token mapping
# =============================================================================


def _classify_text_offset(offset: Tuple[int, int], text_spans: Dict[str, object]) -> str:
    start, end = int(offset[0]), int(offset[1])
    if start == end:
        return "other_text"
    for component in TEXT_COMPONENT_TYPES:
        spans = text_spans.get(component, [])
        for span_start, span_end in spans:
            if start < int(span_end) and end > int(span_start):
                return component
    return "other_text"


def create_token_type_mapping(
    input_ids: torch.Tensor,
    thinker_config,
    *,
    tokenizer,
    rendered_text: str,
    text_spans: Dict[str, object],
) -> List[str]:
    ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
    text_encoding = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    text_ids = [int(x) for x in text_encoding["input_ids"]]
    text_offsets = [(int(a), int(b)) for a, b in text_encoding["offset_mapping"]]
    text_ptr = 0
    token_types: List[str] = []

    for tid in ids:
        # AF3: config.audio_token_id (non audio_token_index come Qwen2.5-Omni) --
        # verificato con af3_knockout_feasibility_probe.py (= 151669).
        if tid == thinker_config.audio_token_id:
            token_types.append("audio")
            if text_ptr < len(text_ids) and text_ids[text_ptr] == tid:
                text_ptr += 1
            continue
        if hasattr(thinker_config, "image_token_index") and tid == thinker_config.image_token_index:
            token_types.append("image")
            if text_ptr < len(text_ids) and text_ids[text_ptr] == tid:
                text_ptr += 1
            continue
        if hasattr(thinker_config, "video_token_index") and tid == thinker_config.video_token_index:
            token_types.append("video")
            if text_ptr < len(text_ids) and text_ids[text_ptr] == tid:
                text_ptr += 1
            continue

        mapped = "other_text"
        if text_ids:
            match = -1
            if text_ptr < len(text_ids) and text_ids[text_ptr] == tid:
                match = text_ptr
            else:
                search_end = min(len(text_ids), text_ptr + 256)
                for j in range(text_ptr, search_end):
                    if text_ids[j] == tid:
                        match = j
                        break
            if match >= 0:
                if match < len(text_offsets):
                    mapped = _classify_text_offset(text_offsets[match], text_spans or {})
                text_ptr = match + 1
        token_types.append(mapped)

    if token_types:
        token_types[-1] = "decision"
    return token_types


# =============================================================================
# MCQ scoring
# =============================================================================


def get_mcq_label_token_ids(tokenizer, style: str) -> Dict[str, int]:
    if style not in {"bare", "space", "paren"}:
        raise ValueError("--label_token_style must be 'bare', 'space' or 'paren'")
    out: Dict[str, int] = {}
    for letter in "ABCD":
        if style == "bare":
            text = letter
        elif style == "space":
            text = " " + letter
        else:  # "paren" -- vedi get_letter_token_ids_af3 in shared_utils_af3.py
            text = "(" + letter
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"MCQ label {text!r} is not one token: {ids}")
        out[letter] = int(ids[0])
    return out


def logits_for_labels(logits_last: torch.Tensor, label_ids: Dict[str, int]) -> Dict[str, float]:
    logits_cpu = logits_last.detach().float().cpu()
    return {label: float(logits_cpu[token_id].item()) for label, token_id in label_ids.items()}


def softmax_dict(label_logits: Dict[str, float]) -> Dict[str, float]:
    labels = list(label_logits.keys())
    arr = np.array([label_logits[label] for label in labels], dtype=np.float64)
    arr = arr - np.max(arr)
    probs = np.exp(arr)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    return {label: float(probs[i]) for i, label in enumerate(labels)}


def score_mcq_forward(model, inputs: Dict[str, torch.Tensor], label_ids: Dict[str, int]) -> Dict:
    """
    Extract A/B/C/D logits for the MCQ decision point.

    Come per Qwen2_5OmniForConditionalGeneration, anche
    AudioFlamingo3ForConditionalGeneration non espone un forward() standard
    per chiamata diretta con input prodotti dal processor.  Usiamo invece
    model.generate(max_new_tokens=1, output_scores=True) che:
      - runs the full prefill forward pass (audio tower + language model)
      - returns scores[0]: the raw logit distribution for the first generated
        token — this IS the MCQ decision (A/B/C/D), computed from the
        representation at the last prompt position (the 'decision' token)
    This is functionally identical to a forward pass at the decision point and
    uses the exact same code path that is already verified to work with generate().
    Any attention-knockout hooks registered on the thinker layers fire during
    this generate call, so knockout effects are correctly captured.
    """
    reset_qwen_omni_generation_state(model)
    generate_kwargs: Dict = {
        "max_new_tokens": 1,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "use_cache": True,
    }
    if hasattr(model, "thinker"):
        generate_kwargs["return_audio"] = False
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)
    # scores[0]: (batch, vocab_size) logits for token 0 of generation = MCQ answer
    raw_logits = outputs.scores[0][0]
    logits = logits_for_labels(raw_logits, label_ids)
    probs = softmax_dict(logits)
    pred = max(logits, key=logits.get)
    del outputs
    return {"logits": logits, "probs": probs, "pred": pred}


def trim_generated_ids_at_first_im_end(gen_ids: torch.Tensor, processor) -> torch.Tensor:
    tok = processor.tokenizer
    try:
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = getattr(tok, "eos_token_id", None)

    if im_end_id is None:
        return gen_ids

    cut = len(gen_ids)
    for i, token_id in enumerate(gen_ids):
        if int(token_id.item()) == int(im_end_id):
            cut = i
            break
    return gen_ids[:cut]


def parse_mcq_answer(response: str) -> Tuple[Optional[str], str]:
    text = str(response or "").strip()
    upper = text.upper()
    if upper in {"A", "B", "C", "D"}:
        return upper, "exact"

    match = re.search(r"(?<![A-Z])([ABCD])(?![A-Z])", upper)
    if match:
        return match.group(1), "standalone_regex"

    match = re.search(r"[ABCD]", upper)
    if match:
        return match.group(0), "loose_regex"

    return None, "none"


def generate_text_response(
    model,
    processor,
    inputs: Dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
    use_cache: bool,
) -> Dict[str, object]:
    reset_qwen_omni_generation_state(model)
    t0 = time.time()
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": False,
        "output_logits": False,
        "use_cache": use_cache,
    }
    if hasattr(model, "thinker"):
        generate_kwargs["return_audio"] = False
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generate_kwargs,
        )

    sequences = outputs.sequences
    input_len = int(inputs["input_ids"].shape[1])
    gen_ids = sequences[0, input_len:]
    gen_ids = trim_generated_ids_at_first_im_end(gen_ids, processor)

    response = processor.tokenizer.decode(
        gen_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    answer, parse_method = parse_mcq_answer(response)
    ids = [int(x) for x in gen_ids.detach().cpu().tolist()]
    del outputs
    return {
        "response": response,
        "answer": answer,
        "parse_method": parse_method,
        "generated_ids": ids,
        "elapsed_sec": round(time.time() - t0, 3),
    }


# =============================================================================
# Attention knockout hook
# =============================================================================


class DecisionAttentionKnockoutHook:
    """
    Forward pre-hook that blocks attention from the decision query to target
    token types by editing the 4D additive attention mask.
    """

    def __init__(
        self,
        *,
        numeric_rules: torch.Tensor,
        numeric_token_types: torch.Tensor,
        original_input_len: int,
        decision_type_id: int,
        generated_type_id: int,
        other_type_id: int,
        layer_idx: int = 0,
    ):
        self.numeric_rules = numeric_rules
        self.numeric_token_types = numeric_token_types
        self.original_input_len = int(original_input_len)
        self.decision_type_id = int(decision_type_id)
        self.generated_type_id = int(generated_type_id)
        self.other_type_id = int(other_type_id)
        self.layer_idx = int(layer_idx)

    def _extended_token_types(self, length: int, device: torch.device) -> torch.Tensor:
        token_types = self.numeric_token_types.to(device)
        if length <= token_types.shape[0]:
            return token_types[:length]
        pad = torch.full(
            (length - token_types.shape[0],),
            self.generated_type_id,
            dtype=token_types.dtype,
            device=device,
        )
        return torch.cat([token_types, pad], dim=0)

    def __call__(self, module, args, kwargs):
        if self.numeric_rules.numel() == 0:
            return args, kwargs

        # ── Determine hidden_states (to get q_len, dtype, device) ─────────────
        # In Qwen2.5-Omni the decoder layer calls self_attn with hidden_states
        # as the first keyword argument.  Fall back to positional args[0].
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            hidden_states = args[0] if args else None
        if hidden_states is None:
            return args, kwargs

        device = hidden_states.device
        dtype = hidden_states.dtype
        q_len = int(hidden_states.shape[1])

        # ── Determine k_len ────────────────────────────────────────────────────
        # With attn_implementation="sdpa", transformers calls
        # _update_causal_mask() which returns None when there is no padding
        # and SDPA can handle causality via is_causal=True.  In that case
        # attention_mask is None but we still need to inject a block.
        # We derive k_len from cache_position (most reliable) or past_key_value.
        #
        # ADATTAMENTO AF3: verificato con af3_knockout_feasibility_probe.py che
        # self_attn di AudioFlamingo3/Qwen2DecoderLayer riceve past_key_values
        # (PLURALE, oggetto Cache) e position_ids nei kwargs, non cache_position
        # ne' past_key_value (singolare) come nell'originale Qwen2.5-Omni.
        # Aggiunti due fallback in piu' (past_key_values plurale, poi
        # position_ids) PRIMA del default k_len=q_len, senza toccare i
        # fallback originali (restano validi se mai presenti anche qui).
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            k_len = int(attention_mask.shape[-1])
        else:
            cache_position = kwargs.get("cache_position")
            if cache_position is not None:
                # cache_position holds the absolute positions being written:
                # prefill → [0, 1, …, q_len-1]; decode → [total_seen]
                k_len = int(cache_position[-1].item()) + 1
            else:
                past_kv = kwargs.get("past_key_value")
                if past_kv is not None and hasattr(past_kv, "get_seq_length"):
                    k_len = past_kv.get_seq_length() + q_len
                elif past_kv is not None and hasattr(past_kv, "key_cache") and past_kv.key_cache:
                    k_len = int(past_kv.key_cache[0].shape[2]) + q_len
                else:
                    # AF3: past_key_values (plurale) -- stesso identico oggetto Cache,
                    # solo nome della chiave diverso nei kwargs.
                    #
                    # BUG FIX: get_seq_length() di default legge layer_idx=0.
                    # Le hook di ogni layer condividono lo STESSO oggetto Cache;
                    # durante il forward prefill i layer vengono eseguiti in
                    # ordine (0,1,2,...) e ciascuno scrive la propria cache
                    # PRIMA che il layer successivo esegua il proprio pre-hook.
                    # Senza layer_idx esplicito, get_seq_length() vedeva sempre
                    # lo stato della cache del layer 0 -- corretto (0) solo per
                    # il layer 0 stesso, ma gia' "inquinato" (= q_len) per ogni
                    # layer successivo, raddoppiando k_len e spostando
                    # query_positions fuori dai token reali (=> block_mask
                    # sempre vuoto => knockout no-op silenzioso per ogni
                    # finestra diversa da quella che include il layer 0).
                    past_kvs = kwargs.get("past_key_values")
                    if past_kvs is not None and hasattr(past_kvs, "get_seq_length"):
                        k_len = past_kvs.get_seq_length(self.layer_idx) + q_len
                    else:
                        # AF3: position_ids sempre presente (prefill: 0..q_len-1;
                        # decode: posizione assoluta del nuovo token).
                        position_ids = kwargs.get("position_ids")
                        if position_ids is not None:
                            k_len = int(position_ids.reshape(-1)[-1].item()) + 1
                        else:
                            k_len = q_len  # prefill with no past tokens

        # ── Classify query and key positions ──────────────────────────────────
        key_type_ids = self._extended_token_types(k_len, device)

        if q_len == 1:
            query_positions = torch.tensor([k_len - 1], dtype=torch.long, device=device)
        else:
            query_start = max(0, k_len - q_len)
            query_positions = torch.arange(
                query_start,
                query_start + q_len,
                dtype=torch.long,
                device=device,
            )

        full_type_ids = self._extended_token_types(max(k_len, int(query_positions[-1].item()) + 1), device)
        query_type_ids = full_type_ids[query_positions]

        # ── Build block mask ───────────────────────────────────────────────────
        block_mask = torch.zeros((q_len, k_len), dtype=torch.bool, device=device)
        for src_id, tgt_id in self.numeric_rules:
            q_mask = query_type_ids == src_id
            k_mask = key_type_ids == tgt_id
            if torch.any(q_mask) and torch.any(k_mask):
                block_mask.logical_or_(q_mask.unsqueeze(1) & k_mask.unsqueeze(0))

        if not torch.any(block_mask):
            return args, kwargs

        # ── Apply block to attention mask ──────────────────────────────────────
        mask_value = torch.finfo(dtype).min

        if attention_mask is not None:
            # Standard path (eager / explicit SDPA mask): modify in-place clone.
            modified = attention_mask.clone()
            modified = modified.masked_fill(block_mask.view(1, 1, q_len, k_len), mask_value)
            kwargs["attention_mask"] = modified
        else:
            # SDPA None path: build an equivalent causal mask from scratch, then
            # inject it so SDPA uses our explicit mask (is_causal becomes False).
            past_len = k_len - q_len  # tokens already in KV cache
            if q_len > 1:
                # Standard causal: query at absolute pos (past_len + i) can attend
                # to key positions 0 .. past_len + i  (inclusive).
                row = torch.arange(q_len, device=device, dtype=torch.long).unsqueeze(1)  # [q,1]
                col = torch.arange(k_len, device=device, dtype=torch.long).unsqueeze(0)  # [1,k]
                future_mask = col > (row + past_len)   # True → should be -inf
                causal = torch.zeros(1, 1, q_len, k_len, dtype=dtype, device=device)
                causal.masked_fill_(future_mask.unsqueeze(0).unsqueeze(0), mask_value)
            else:
                # Decode step (q_len=1): token attends to all k without causal masking.
                causal = torch.zeros(1, 1, 1, k_len, dtype=dtype, device=device)

            causal.masked_fill_(block_mask.view(1, 1, q_len, k_len), mask_value)
            kwargs["attention_mask"] = causal

        return args, kwargs


@contextmanager
def apply_attention_knockout(
    model,
    *,
    rules: List[Tuple[str, str, int, int]],
    token_types: List[str],
    original_input_len: int,
):
    handles = []
    layers = get_thinker_layers(model)
    total_layers = len(layers)

    try:
        for layer_idx, layer in enumerate(layers):
            active_pairs: List[Tuple[str, str]] = []
            for src, tgt, start, end in rules:
                if start <= layer_idx < end:
                    active_pairs.append((src, tgt))
            if not active_pairs:
                continue

            layer_device = get_module_device(layer)
            numeric_token_types = torch.tensor(
                [TOKEN_TYPE_MAP.get(t, TOKEN_TYPE_MAP["other"]) for t in token_types],
                dtype=torch.long,
                device=layer_device,
            )
            numeric_rules = torch.tensor(
                [(TOKEN_TYPE_MAP[src], TOKEN_TYPE_MAP[tgt]) for src, tgt in active_pairs],
                dtype=torch.long,
                device=layer_device,
            )
            hook = DecisionAttentionKnockoutHook(
                numeric_rules=numeric_rules,
                numeric_token_types=numeric_token_types,
                original_input_len=original_input_len,
                decision_type_id=TOKEN_TYPE_MAP["decision"],
                generated_type_id=TOKEN_TYPE_MAP["generated"],
                other_type_id=TOKEN_TYPE_MAP["other"],
                layer_idx=layer_idx,
            )
            handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))

        if handles:
            print(f"   Applied {len(handles)} knockout hooks: {rules}")
        yield
    finally:
        for handle in handles:
            handle.remove()
        if handles:
            print(f"   Removed {len(handles)} knockout hooks.")


def build_layer_windows(total_layers: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    if window_size <= 0:
        return [(0, total_layers)]
    windows = []
    start = 0
    while start < total_layers:
        end = min(total_layers, start + window_size)
        windows.append((start, end))
        if end == total_layers:
            break
        start += max(1, stride)
    return windows


# =============================================================================
# CLI and main loop
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attention knockout for HumMusQA generation or decision logits."
    )
    parser.add_argument("--model_path", default="/nas/home/fingenito/Models/audio-flamingo-3-hf")
    parser.add_argument("--output_dir", default="results/attention_knockout_hummusqa")
    parser.add_argument("--dataset_name", default="mtg-upf/HumMusQA")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--tmp_dir", default=None)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--option_permutations", type=int, default=1)
    parser.add_argument("--option_seed", type=int, default=42)
    parser.add_argument("--components", nargs="*", default=["audio", "instruction", "question", "options"])
    parser.add_argument("--window_size", type=int, default=9)
    parser.add_argument("--window_stride", type=int, default=9)
    parser.add_argument(
        "--knockout_mode",
        choices=["decision_logits", "generation"],
        default="decision_logits",
        help=(
            "decision_logits: restricted A/B/C/D forward scoring; "
            "generation: AVLLM-style model.generate baseline and generated->component knockout."
        ),
    )
    parser.add_argument("--generate_max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--generate_use_cache",
        action="store_true",
        default=True,
        help=(
            "Use KV cache during generation. Default is enabled because the original "
            "AVLLM generate() path leaves use_cache at the model default."
        ),
    )
    parser.add_argument(
        "--generate_no_cache",
        action="store_false",
        dest="generate_use_cache",
        help="Disable KV cache during generation; useful only for strict Exp_A baseline comparison.",
    )
    parser.add_argument(
        "--knockout_source",
        choices=["generated", "decision"],
        default="generated",
        help=(
            "generation mode only. 'generated' is the literal AVLLM source token type. "
            "'decision' blocks the final prompt decision query, which is the causal "
            "variant that can affect a one-token MCQ answer."
        ),
    )
    parser.add_argument(
        "--label_token_style",
        choices=["bare", "space", "paren"],
        default="paren",
        help=(
            "bare='A', space=' A', paren='(A'. Default 'paren' perche' AF3 con il "
            "prompt MCQA standard (opzioni scritte come '(A) ...') risponde "
            "sistematicamente col token fuso '(A', non con la lettera nuda -- "
            "stessa scelta e stessa verifica gia' fatta per Exp E "
            "(get_letter_token_ids_af3 in shared_utils_af3.py, smoke test 2026-08-25, "
            "8/8 campioni). Usare 'bare' qui sottostimerebbe p_orig/p_ko."
        ),
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help=(
            "Attention backend used when loading the model. sdpa e' richiesto per il "
            "meccanismo di knockout (permette di iniettare/modificare l'attention "
            "mask); flash_attention_2 non lo permette -- stessa scelta dell'originale "
            "Qwen, verificata anche per AF3 con af3_knockout_feasibility_probe.py."
        ),
    )
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(output_dir: str, rows: List[Dict], metadata: Dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "attention_knockout_results.csv"), rows)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not 1 <= args.option_permutations <= 24:
        raise ValueError("--option_permutations must be between 1 and 24")

    tmp_context = None
    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True)
        tmp_dir = args.tmp_dir
    else:
        tmp_context = tempfile.TemporaryDirectory(prefix="hummusqa_knockout_")
        tmp_dir = tmp_context.name

    print("\n" + "=" * 78)
    print("Attention Knockout | HumMusQA audio-text MCQ")
    print(f"Model     : {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    if args.knockout_mode == "generation":
        print("Mode      : AVLLM-style generation attention mask knockout")
        print("           baseline: model.generate()")
        print(f"           rules   : {args.knockout_source} -> component over layer windows")
    else:
        print("Mode      : decision-point attention mask knockout")
        print("           rules   : decision -> component over layer windows")
    print("=" * 78 + "\n")

    print("Loading model...")
    # BUGFIX (trovato durante lo smoke test, 2026-09): l'originale Qwen carica
    # con device_map={"": "cuda:X"} (streaming diretto su GPU via accelerate).
    # Per AF3, con la versione di transformers installata in env_ing_af3,
    # questo percorso di caricamento causa un picco di memoria che manda in
    # OOM anche una GPU con 22GB completamente liberi (osservato empiricamente,
    # confermato con nvidia-smi). Il loader AF3 gia' validato in produzione
    # (_load_af3_model, gpu_utils_af3.py, usato per tutti i run Exp E) usa
    # invece device_map=None + low_cpu_mem_usage=True + .to(device) esplicito
    # DOPO il caricamento -- stesso pattern replicato qui.
    target_device = str(args.device_map) if str(args.device_map).startswith("cuda") else "cuda:0"
    # AF3 non ha una distinzione thinker/full come Qwen2.5-Omni -- un'unica
    # classe modello, niente --model_class da scegliere.
    model_cls = AudioFlamingo3ForConditionalGeneration

    # BUGFIX #2 (trovato subito dopo il #1, stesso smoke test): torch_dtype="auto"
    # (ereditato dall'originale Qwen) non risolve correttamente a bf16 per AF3 con
    # questa versione di transformers -- il modello risultava ~22GB invece dei
    # ~16GB attesi per un modello bf16 da 8B parametri (confermato: OOM anche a
    # 22GB di GPU libera). Ovunque nel progetto AF3 viene caricato con
    # torch.bfloat16 esplicito (_load_af3_model, gpu_utils_af3.py) -- mai "auto".
    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model = model.to(target_device)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    model.eval()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path)

    thinker_cfg = get_thinker_config(model)
    total_layers = len(get_thinker_layers(model))
    layer_windows = build_layer_windows(total_layers, args.window_size, args.window_stride)
    text_device = get_text_backbone_device(model)
    audio_device = get_audio_tower_device(model)
    label_ids = get_mcq_label_token_ids(processor.tokenizer, args.label_token_style)
    decoded_labels = {k: processor.tokenizer.decode([v]) for k, v in label_ids.items()}

    print(f"Model layers       : {total_layers}")
    print(f"audio_token_id     : {thinker_cfg.audio_token_id}")
    print(f"loaded attention   : {args.attn_implementation}")
    print(f"model class        : {model_cls.__name__}")
    print(f"layer windows      : {layer_windows}")
    print(f"components         : {args.components}")
    print(f"text device        : {text_device}")
    print(f"audio tower device : {audio_device}")
    if args.knockout_mode == "decision_logits":
        print(f"label token style  : {args.label_token_style} {label_ids} decoded={decoded_labels}")
    else:
        print(f"generation config  : max_new_tokens={args.generate_max_new_tokens}, use_cache={args.generate_use_cache}")
    print(f"hf_device_map      : {getattr(model, 'hf_device_map', 'N/A')}")

    print("\nLoading HumMusQA...")
    if args.dataset_path:
        dataset = load_dataset("parquet", data_files={args.dataset_split: args.dataset_path}, split=args.dataset_split)
    else:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.cast_column("audio", HFAudio(decode=False))

    samples = list(dataset)
    if args.sample_start:
        samples = samples[args.sample_start :]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    eval_items = []
    for local_idx, sample in enumerate(samples):
        dataset_idx = args.sample_start + local_idx
        identifier = safe_identifier(sample, dataset_idx)
        for perm_idx, order in enumerate(build_option_orders(identifier, args.option_permutations, args.option_seed)):
            eval_items.append((dataset_idx, sample, perm_idx, order))

    print(
        f"Questions to process: {len(samples)} | option permutations: "
        f"{args.option_permutations} | evaluations: {len(eval_items)}"
    )

    rows: List[Dict] = []
    errors: List[Dict] = []

    for eval_idx, (sample_idx, sample, perm_idx, option_order) in enumerate(eval_items):
        identifier = safe_identifier(sample, sample_idx)
        order_label = "".join(chr(65 + int(i)) for i in option_order)
        print(f"\n[{eval_idx + 1:04d}/{len(eval_items):04d}] {identifier} order={order_label}")
        t0 = time.time()
        audio_path = None
        should_delete = False

        try:
            inputs, correct_letter, audio_path, should_delete, rendered_text, text_spans = prepare_inputs(
                sample,
                sample_idx,
                processor,
                tmp_dir,
                option_order=option_order,
            )
            # text_device == audio_device sempre in questo setup (un solo GPU per
            # shard, vedi place_af3_inputs); passiamo text_device come unico target.
            inputs = place_af3_inputs(inputs, device=text_device, dtype=model.dtype)
            print(f"   Input devices: {debug_input_devices(inputs)}")

            original_input_len = int(inputs["input_ids"].shape[1])
            token_types = create_token_type_mapping(
                inputs["input_ids"],
                thinker_cfg,
                tokenizer=processor.tokenizer,
                rendered_text=rendered_text,
                text_spans=text_spans,
            )
            counts = Counter(token_types)
            print(f"   Token counts: {dict(counts)} | input_len={original_input_len}")
            if counts.get("audio", 0) == 0:
                raise RuntimeError("No audio tokens found in prompt.")

            if args.knockout_mode == "generation":
                print("   Starting baseline generate...")
                baseline = generate_text_response(
                    model,
                    processor,
                    inputs,
                    max_new_tokens=args.generate_max_new_tokens,
                    use_cache=args.generate_use_cache,
                )
                baseline_pred = baseline["answer"]
                baseline_correct = int(baseline_pred == correct_letter)
                print(
                    f"   baseline response={baseline['response']!r} "
                    f"parsed={baseline_pred} method={baseline['parse_method']} "
                    f"correct={correct_letter} ok={baseline_correct} "
                    f"generate_sec={baseline['elapsed_sec']}"
                )

                for component in args.components:
                    if component == ALL_TEXT_TO_AUDIO:
                        # virtual compound component — skip only if there are no audio tokens
                        if counts.get("audio", 0) == 0:
                            print(f"   skip component={component}: no audio tokens")
                            continue
                    elif component not in TOKEN_TYPE_MAP:
                        raise ValueError(f"Unknown component {component!r}; valid={list(TOKEN_TYPE_MAP)}")
                    elif counts.get(component, 0) == 0:
                        print(f"   skip component={component}: no tokens")
                        continue
                    for layer_start, layer_end in layer_windows:
                        if component == ALL_TEXT_TO_AUDIO:
                            rules = [(src, "audio", layer_start, layer_end) for src in _ALL_TEXT_SOURCE_TYPES]
                            _gen_ko_label = f"all_text->audio"
                        else:
                            rules = [(args.knockout_source, component, layer_start, layer_end)]
                            _gen_ko_label = f"{args.knockout_source}->{component}"
                        print(
                            f"   Starting KO {_gen_ko_label} "
                            f"L{layer_start}:{layer_end} generate..."
                        )
                        with apply_attention_knockout(
                            model,
                            rules=rules,
                            token_types=token_types,
                            original_input_len=original_input_len,
                        ):
                            ko = generate_text_response(
                                model,
                                processor,
                                inputs,
                                max_new_tokens=args.generate_max_new_tokens,
                                use_cache=args.generate_use_cache,
                            )

                        ko_pred = ko["answer"]
                        row = {
                            "mode": "generation",
                            "sample_idx": sample_idx,
                            "evaluation_idx": eval_idx,
                            "identifier": identifier,
                            "option_permutation_idx": perm_idx,
                            "option_order": order_label,
                            "component": component,
                            "layer_start": layer_start,
                            "layer_end": layer_end,
                            "source_component": args.knockout_source,
                            "knockout_source": args.knockout_source,
                            "target_component": component,
                            "correct_answer": correct_letter,
                            "baseline_response_raw": baseline["response"],
                            "baseline_answer": baseline_pred,
                            "baseline_parse_method": baseline["parse_method"],
                            "baseline_generated_ids": json.dumps(baseline["generated_ids"]),
                            "baseline_generate_sec": baseline["elapsed_sec"],
                            "knockout_response_raw": ko["response"],
                            "knockout_answer": ko_pred,
                            "knockout_parse_method": ko["parse_method"],
                            "knockout_generated_ids": json.dumps(ko["generated_ids"]),
                            "knockout_generate_sec": ko["elapsed_sec"],
                            "baseline_correct": baseline_correct,
                            "knockout_correct": int(ko_pred == correct_letter),
                            "answer_changed": int(ko_pred != baseline_pred),
                            "n_audio_tokens": counts.get("audio", 0),
                            "n_instruction_tokens": counts.get("instruction", 0),
                            "n_question_tokens": counts.get("question", 0),
                            "n_options_tokens": counts.get("options", 0),
                            "n_other_text_tokens": counts.get("other_text", 0),
                            "n_input_tokens": original_input_len,
                            "elapsed_sec_so_far": round(time.time() - t0, 3),
                        }
                        if isinstance(baseline_pred, str) and baseline_pred in "ABCD":
                            row["baseline_pred_source_index"] = int(option_order[ord(str(baseline_pred)) - 65])
                        else:
                            row["baseline_pred_source_index"] = None
                        if isinstance(ko_pred, str) and ko_pred in "ABCD":
                            row["knockout_pred_source_index"] = int(option_order[ord(str(ko_pred)) - 65])
                        else:
                            row["knockout_pred_source_index"] = None
                        rows.append(row)

                        print(
                            f"   KO {_gen_ko_label} L{layer_start}:{layer_end} "
                            f"response={ko['response']!r} parsed={ko_pred} "
                            f"changed={row['answer_changed']} ok={row['knockout_correct']} "
                            f"generate_sec={ko['elapsed_sec']}"
                        )

                if args.checkpoint_every > 0 and (eval_idx + 1) % args.checkpoint_every == 0:
                    save_outputs(
                        args.output_dir,
                        rows,
                        {
                            "status": "checkpoint",
                            "knockout_mode": args.knockout_mode,
                            "errors": errors,
                            "saved_at": datetime.now().isoformat(),
                        },
                    )
                continue

            baseline = score_mcq_forward(model, inputs, label_ids)
            baseline_pred = str(baseline["pred"])
            baseline_correct = int(baseline_pred == correct_letter)
            print(
                f"   baseline pred={baseline_pred} correct={correct_letter} "
                f"ok={baseline_correct} probs={baseline['probs']}"
            )

            # decision_logits: source is always "decision" (the final prompt token)
            # regardless of --knockout_source (which only applies to generation mode).
            # Exception: ALL_TEXT_TO_AUDIO uses all text source types simultaneously.
            dl_source = "decision"
            for component in args.components:
                if component == ALL_TEXT_TO_AUDIO:
                    # virtual compound component — skip only if there are no audio tokens
                    if counts.get("audio", 0) == 0:
                        print(f"   skip component={component}: no audio tokens")
                        continue
                elif component not in TOKEN_TYPE_MAP:
                    raise ValueError(f"Unknown component {component!r}; valid={list(TOKEN_TYPE_MAP)}")
                elif counts.get(component, 0) == 0:
                    print(f"   skip component={component}: no tokens")
                    continue
                for layer_start, layer_end in layer_windows:
                    if component == ALL_TEXT_TO_AUDIO:
                        rules = [(src, "audio", layer_start, layer_end) for src in _ALL_TEXT_SOURCE_TYPES]
                    else:
                        rules = [(dl_source, component, layer_start, layer_end)]
                    with apply_attention_knockout(
                        model,
                        rules=rules,
                        token_types=token_types,
                        original_input_len=original_input_len,
                    ):
                        ko = score_mcq_forward(model, inputs, label_ids)

                    ko_pred = str(ko["pred"])
                    row = {
                        "mode": "decision_logits",
                        "sample_idx": sample_idx,
                        "evaluation_idx": eval_idx,
                        "identifier": identifier,
                        "option_permutation_idx": perm_idx,
                        "option_order": order_label,
                        "component": component,
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "correct_answer": correct_letter,
                        "baseline_pred": baseline_pred,
                        "knockout_pred": ko_pred,
                        "baseline_correct": baseline_correct,
                        "knockout_correct": int(ko_pred == correct_letter),
                        "baseline_pred_source_index": int(option_order[ord(baseline_pred) - 65]),
                        "knockout_pred_source_index": int(option_order[ord(ko_pred) - 65]),
                        "n_audio_tokens": counts.get("audio", 0),
                        "n_instruction_tokens": counts.get("instruction", 0),
                        "n_question_tokens": counts.get("question", 0),
                        "n_options_tokens": counts.get("options", 0),
                        "n_other_text_tokens": counts.get("other_text", 0),
                        "n_input_tokens": original_input_len,
                    }
                    for label in "ABCD":
                        row[f"baseline_logit_{label}"] = baseline["logits"][label]
                        row[f"knockout_logit_{label}"] = ko["logits"][label]
                        row[f"delta_logit_{label}"] = ko["logits"][label] - baseline["logits"][label]
                        row[f"baseline_prob_{label}"] = baseline["probs"][label]
                        row[f"knockout_prob_{label}"] = ko["probs"][label]
                        row[f"delta_prob_{label}"] = ko["probs"][label] - baseline["probs"][label]
                    row["delta_prob_correct"] = (
                        ko["probs"][correct_letter] - baseline["probs"][correct_letter]
                    )
                    row["delta_logit_correct"] = (
                        ko["logits"][correct_letter] - baseline["logits"][correct_letter]
                    )
                    row["delta_prob_baseline_pred"] = (
                        ko["probs"][baseline_pred] - baseline["probs"][baseline_pred]
                    )
                    row["delta_logit_baseline_pred"] = (
                        ko["logits"][baseline_pred] - baseline["logits"][baseline_pred]
                    )
                    row["pred_changed"] = int(ko_pred != baseline_pred)
                    row["elapsed_sec_so_far"] = round(time.time() - t0, 3)
                    rows.append(row)

                    _dl_ko_label = "all_text->audio" if component == ALL_TEXT_TO_AUDIO else f"decision->{component}"
                    print(
                        f"   KO {_dl_ko_label} L{layer_start}:{layer_end} "
                        f"pred={ko_pred} dP(correct)={row['delta_prob_correct']:+.4f} "
                        f"dP(base)={row['delta_prob_baseline_pred']:+.4f}"
                    )

            if args.checkpoint_every > 0 and (eval_idx + 1) % args.checkpoint_every == 0:
                save_outputs(
                    args.output_dir,
                    rows,
                    {
                        "status": "checkpoint",
                        "errors": errors,
                        "saved_at": datetime.now().isoformat(),
                    },
                )

        except Exception as exc:
            import traceback

            print(f"   ERROR: {exc}")
            traceback.print_exc()
            errors.append(
                {
                    "sample_idx": sample_idx,
                    "evaluation_idx": eval_idx,
                    "identifier": identifier,
                    "error": repr(exc),
                }
            )
            if args.fail_fast:
                raise
        finally:
            if should_delete and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    # Compute per-sample baseline accuracy from the rows we saved
    # (only one baseline per eval_item, keyed by sample_idx+perm_idx)
    _seen_baseline: set = set()
    n_baseline_correct = 0
    n_baseline_total = 0
    for r in rows:
        key = (r.get("sample_idx"), r.get("option_permutation_idx", 0))
        if key not in _seen_baseline:
            _seen_baseline.add(key)
            bc = r.get("baseline_correct") if args.knockout_mode == "generation" else r.get("baseline_correct")
            if bc is not None:
                try:
                    n_baseline_correct += int(bc)
                    n_baseline_total += 1
                except (ValueError, TypeError):
                    pass

    metadata = {
        "status": "complete",
        "variant": "audio_only",
        "null_question": NULL_QUESTION,
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "dataset_path": args.dataset_path,
        "sample_start": args.sample_start,
        "max_samples": args.max_samples,
        "option_permutations": args.option_permutations,
        "option_seed": args.option_seed,
        "components": args.components,
        "layer_windows": layer_windows,
        "knockout_mode": args.knockout_mode,
        "generate_max_new_tokens": args.generate_max_new_tokens,
        "generate_use_cache": args.generate_use_cache,
        # knockout_source is only meaningful in generation mode.
        # In decision_logits mode the source is always "decision" regardless of the CLI arg.
        "knockout_source": "decision" if args.knockout_mode == "decision_logits" else args.knockout_source,
        "label_token_style": args.label_token_style,
        "label_token_ids": label_ids,
        "label_token_decoded": decoded_labels,
        "model_class": model_cls.__name__,
        "attn_implementation": args.attn_implementation,
        # Fields for merge_exp2_shards.py
        "n_processed": n_baseline_total,
        "n_correct": n_baseline_correct,
        "baseline_accuracy": round(n_baseline_correct / n_baseline_total, 6) if n_baseline_total else None,
        "methodological_note": (
            "AVLLM-style attention knockout adapted to HumMusQA. In generation "
            "mode, baseline and knockout answers are produced with model.generate(), "
            "and hooks block either generated-token attention (literal AVLLM) or "
            "final prompt decision attention (MCQ first-token causal variant) to "
            "one prompt component. "
            "In decision_logits mode, hooks block final prompt decision attention "
            "to one component and changes are measured on restricted A/B/C/D logits."
        ),
        "errors": errors,
        "n_rows": len(rows),
        "saved_at": datetime.now().isoformat(),
    }
    save_outputs(args.output_dir, rows, metadata)

    print("\n" + "=" * 78)
    print(f"Done. rows={len(rows)} errors={len(errors)}")
    print(f"Saved: {os.path.join(args.output_dir, 'attention_knockout_results.csv')}")
    print("=" * 78)

    if tmp_context is not None:
        tmp_context.cleanup()


if __name__ == "__main__":
    main()
