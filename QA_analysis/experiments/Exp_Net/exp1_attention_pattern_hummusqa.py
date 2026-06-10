"""
Experiment 1 — Attention Pattern Analysis for HumMusQA + Qwen2.5-Omni
===============================================================================
AVLLM-aligned adaptation: captures attention from the generated answer token
towards the multimodal prompt components (audio, question, options, instruction).

Targets Qwen2.5-Omni-3B — exactly the model used in the original AVLLM paper.

Primary mode: avllm_literal_generate (default)
-----------------------------------------------
Directly mirrors the AVLLM paper (attention_knockout_experiment.py):
  1. Load Qwen2_5OmniForConditionalGeneration with attn_implementation="sdpa"
     and call model.disable_talker() to disable the audio-generation Talker.
     Qwen2_5OmniSdpaAttention automatically falls back to the eager (manual)
     implementation when output_attentions=True — used only at decode time.
  2. Build inputs once with add_generation_prompt=True, NO answer suffix.
  3. Register register_forward_hook on each self_attn layer
     (model.thinker.model.layers[i].self_attn). The hooks capture output[1]
     (non-None only when output_attentions=True reaches self_attn).
  4. Register forward_pre_hooks on each self_attn that force output_attentions=True
     ONLY for autoregressive decode steps (q_len==1). This avoids the quadratic
     attention cost during prefill while still capturing all generated-token steps.
  5. Call model.generate(max_new_tokens=N) to produce a short response.
  6. Find the first standalone A/B/C/D token in the generated sequence.
  7. Select the corresponding q_len==1 attention tensor from each layer's hook storage.

The predicted answer letter comes from the model's free generation (same protocol
as the HumMusQA paper evaluation). Accuracy should be ≈ 64 % on the full dataset.

Why Qwen2.5-Omni-3B and not 7B
--------------------------------
The 7B model exhibits a TMRoPE KV-cache asymmetry bug with audio inputs + eager
attention: K stores 2x entries compared to V after prefill, causing a RuntimeError
in torch.matmul(attn_weights, value_states). This is not present in the 3B model,
which is also the original AVLLM paper's target model.

Diagnostic / baseline modes
----------------------------
  generate_score_only — free greedy generation, flash_attention_2, Thinker-only.
                        Validates accuracy without any hooks.
  score_only          — A/B/C/D logit ranking from prompt-final logits, sdpa.
                        Fast accuracy diagnostic, no attention capture.
  generation_probe    — SDPA generate without hooks to examine response format.
  constrained_mcq     — [Diagnostic] selects answer from prompt-final logits,
                        then captures answer-token attention via eager cached decode.
  cached_decode       — [Diagnostic] SDPA generate + dynamic pre-hook that
                        activates eager attention only when A/B/C/D is queried.

Usage
-----
python exp1_attention_pattern_hummusqa.py \\
    --model_path /nas/home/fingenito/Models/Qwen2.5-Omni-3B \\
    --output_dir results/exp1_attention \\
    --max_samples 3 \\
    --fail_fast

For accuracy-only diagnostics (no attention capture, faster):
python exp1_attention_pattern_hummusqa.py \\
    --model_path /nas/home/fingenito/Models/Qwen2.5-Omni-3B \\
    --output_dir results/exp1_score \\
    --max_samples 10 \\
    --attention_mode generate_score_only \\
    --baseline_device cuda:0

python -m QA_analysis.experiments.Exp_Net.exp1_attention_pattern_hummusqa \\
      --model_path /nas/home/fingenito/Models/Qwen2.5-Omni-3B \\
      --output_dir /nas/home/fingenito/Thesis_project/QA_analysis/Results_QA/experiments/exp_Net/exp1_attention/generate_score_thinker_1x4 \\
      --attention_mode generate_score_only \\
      --baseline_device cuda:0 \\
      --option_permutations 4 \\
      --option_seed 42 \\
      --max_samples 1 \\
      --fail_fast

python -m QA_analysis.experiments.Exp_Net.exp1_attention_pattern_hummusqa \\
      --model_path /nas/home/fingenito/Models/Qwen2.5-Omni-3B \\
      --output_dir /nas/home/fingenito/Thesis_project/QA_analysis/Results_QA/experiments/exp_Net/exp1_attention/exp1_fix_test \\
      --max_samples 3 \\
      --fail_fast
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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from datasets import Audio as HFAudio
from datasets import load_dataset
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


# NOTE: No monkey-patching needed for Qwen2.5-Omni-3B.
# The TMRoPE KV-cache / causal-mask asymmetry bug (K=2×V entries after prefill)
# that required patching was specific to the 7B model with audio+eager attention.
# The 3B model does not exhibit this issue.

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

TEXT_COMPONENT_TYPES = ("instruction", "question", "options", "other_text")
TOKEN_TYPES = TEXT_COMPONENT_TYPES + ("query_text", "audio", "image", "video", "generated")
TOKEN_TYPE_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(TOKEN_TYPES)}
ATTENTION_MODES = (
    "generate_score_only",
    "score_only",
    "constrained_mcq",
    "cached_decode",
    "avllm_literal_generate",
    "generation_probe",
)


# =============================================================================
# Token type mapping — same logic as the original repo
# =============================================================================

def _classify_text_offset(offset: Tuple[int, int], text_spans: Dict[str, object]) -> str:
    """Classify a rendered-template tokenizer offset into a HumMusQA prompt component."""
    start, end = int(offset[0]), int(offset[1])
    if end <= start:
        return "other_text"

    best_name = "other_text"
    best_overlap = 0
    # Prefer semantically specific regions on ties. This matters because
    # "instruction" may be represented by multiple broad template fragments.
    for name in ("question", "options", "instruction"):
        if name == "other_text" or name not in text_spans:
            continue
        raw_spans = text_spans[name]
        if isinstance(raw_spans, tuple):
            spans = [raw_spans]
        else:
            spans = list(raw_spans or [])
        for span_start, span_end in spans:
            overlap = max(0, min(end, int(span_end)) - max(start, int(span_start)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = name
    return best_name


def create_token_type_mapping(
    input_ids: torch.Tensor,
    thinker_config,
    *,
    tokenizer=None,
    rendered_text: Optional[str] = None,
    text_spans: Optional[Dict[str, object]] = None,
) -> List[str]:
    """
    Map each input token id to a granular token type.

    If tokenizer offsets and rendered prompt spans are available, text tokens are split into:
      - instruction: system/chat scaffolding plus HumMusQA task wording
      - question: the actual HumMusQA question
      - options: the four answer option strings
      - other_text: any residual text/special-template token

    Audio/image/video tokens are still identified from model config special token indices.
    """
    text_ids: List[int] = []
    text_offsets: List[Tuple[int, int]] = []
    if tokenizer is not None and rendered_text is not None and text_spans:
        try:
            enc = tokenizer(
                rendered_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            text_ids = [int(x) for x in enc.get("input_ids", [])]
            text_offsets = [(int(a), int(b)) for a, b in enc.get("offset_mapping", [])]
        except Exception as exc:
            print(f"   ⚠ Could not build tokenizer offset map; falling back to coarse text mapping: {exc}")
            text_ids = []
            text_offsets = []

    token_types: List[str] = []
    text_ptr = 0
    for token_id in input_ids[0].detach().cpu():
        tid = int(token_id.item())
        if tid == thinker_config.audio_token_index:
            token_types.append("audio")
            if text_ptr < len(text_ids) and text_ids[text_ptr] == tid:
                text_ptr += 1
        elif hasattr(thinker_config, "image_token_index") and tid == thinker_config.image_token_index:
            token_types.append("image")
            if text_ptr < len(text_ids) and text_ids[text_ptr] == tid:
                text_ptr += 1
        elif hasattr(thinker_config, "video_token_index") and tid == thinker_config.video_token_index:
            token_types.append("video")
            if text_ptr < len(text_ids) and text_ids[text_ptr] == tid:
                text_ptr += 1
        else:
            mapped = "other_text"
            if text_ids:
                match = -1
                if text_ptr < len(text_ids) and text_ids[text_ptr] == tid:
                    match = text_ptr
                else:
                    # Multimodal processors may expand a single placeholder into many
                    # audio tokens. Search ahead to re-sync with the rendered text ids.
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
    return token_types


# =============================================================================
# Attention capture — forward hook on self_attn for the answer-token decode step
# =============================================================================

def _make_capture_hook(layer_idx: int, storage: dict):
    """
    Capture only generated-token attention tensors (q_len == 1).

    Durante una generazione di prefisso risposta:
      - prima chiamata: prefill del prompt, q_len = input_len
      - seconda chiamata: primo generated token come query, q_len = 1

    La metrica usa solo q_len == 1, quindi ignoriamo il prefill prima di
    copiarlo su CPU. In literal generate mode il prefill viene ancora
    calcolato da Transformers, ma non viene trasferito e conservato.
    """
    def hook(module, inputs, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            w = output[1]
            if torch.is_tensor(w) and w.ndim == 4 and int(w.shape[-2]) == 1:
                storage[layer_idx].append(w.detach().cpu())
    return hook


def get_thinker(model):
    """Return the text/audio Thinker from either the Omni wrapper or Thinker-only model."""
    return getattr(model, "thinker", model)


@contextmanager
def capture_first_generated_token_attention(
    model: torch.nn.Module,
    capture_layer_range: Optional[Tuple[int, int]] = None,
):
    """
    Context manager that captures attention weights from each self_attn layer
    during autoregressive decode steps (q_len==1).

    Follows the AVLLM paper's approach: post-hooks on
    model.thinker.model.layers[i].self_attn capture output[1] (the attention
    weight tensor), which is non-None only when output_attentions=True reaches
    the self_attn module.

    To avoid computing the quadratic prefill attention (expensive for long audio
    prompts), we inject output_attentions=True via a self_attn pre-hook ONLY
    when q_len==1 (decode step). The decoder layer's output_attentions remains
    False at the model level, so the thinker does not accumulate all attention
    tensors in the generate output dict.

    Yields a defaultdict(list): layer_idx → list of [1, heads, 1, k_len] tensors.

    Model MUST be loaded with attn_implementation="sdpa" (or "eager").
    With "sdpa", Qwen2_5OmniSdpaAttention.forward falls back to the eager
    (manual) path when output_attentions=True, so our pre-hook injection works.
    Plain SDPA / flash_attention_2 WITHOUT the pre-hook fallback return None.
    """
    storage: Dict[int, List[torch.Tensor]] = defaultdict(list)
    handles = []
    thinker = get_thinker(model)
    layers = thinker.model.layers
    total = len(layers)
    start = 0 if capture_layer_range is None else max(0, capture_layer_range[0])
    end = total if capture_layer_range is None else min(total, capture_layer_range[1])

    def _make_self_attn_pre_hook():
        """Inject output_attentions=True into self_attn kwargs for decode steps only."""
        def pre_hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if not (
                torch.is_tensor(hidden_states)
                and hidden_states.ndim == 3
                and int(hidden_states.shape[1]) == 1
            ):
                return args, kwargs
            # Decode step (q_len==1): force attention weights to be returned.
            kwargs["output_attentions"] = True
            return args, kwargs
        return pre_hook

    for i in range(start, end):
        attn = layers[i].self_attn
        handles.append(attn.register_forward_pre_hook(_make_self_attn_pre_hook(), with_kwargs=True))
        handles.append(attn.register_forward_hook(_make_capture_hook(i, storage)))

    n_layers_hooked = end - start
    print(
        f"   Registered {len(handles)} attention-capture hooks "
        f"({n_layers_hooked} self_attn pre-hooks + {n_layers_hooked} post-hooks) "
        f"on layers [{start}, {end})."
    )
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()
        print(f"   Removed {len(handles)} attention-capture hooks.")


@contextmanager
def capture_first_answer_letter_attention(
    model: torch.nn.Module,
    tokenizer,
    capture_layer_range: Optional[Tuple[int, int]] = None,
):
    """
    Enable attention capture only when the first generated A/B/C/D token is a query.

    Used by cached_decode mode. Pre-hooks are on self_attn (not thinker) so that
    output_attentions=True is injected only at decode steps (q_len==1) without
    triggering quadratic prefill attention.
    """
    storage: Dict[int, List[torch.Tensor]] = defaultdict(list)
    handles = []
    thinker = get_thinker(model)
    layers = thinker.model.layers
    total = len(layers)
    start = 0 if capture_layer_range is None else max(0, capture_layer_range[0])
    end = total if capture_layer_range is None else min(total, capture_layer_range[1])
    state = {"captured_token_id": None, "captured_token_text": None}

    def _make_letter_self_attn_pre_hook():
        """Force output_attentions=True inside self_attn when the current decode
        token is the first standalone A/B/C/D answer letter."""
        def pre_hook(module, args, kwargs):
            if state["captured_token_id"] is not None:
                # Already captured; no need to force output_attentions further.
                return args, kwargs
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if not (
                torch.is_tensor(hidden_states)
                and hidden_states.ndim == 3
                and int(hidden_states.shape[1]) == 1
            ):
                return args, kwargs
            # We need the current decode token ID. The cache_position kwarg
            # gives the position of the current token in the full sequence;
            # from the parent context we don't easily know the token ID here.
            # Instead, rely on the post-hook to only keep tensors captured
            # after state["captured_token_id"] has been set by the
            # thinker-level state-update hook below.
            kwargs["output_attentions"] = True
            return args, kwargs
        return pre_hook

    # State-update hook on the thinker (only reads input_ids / updates state,
    # does NOT inject output_attentions — avoids TMRoPE issue).
    def thinker_state_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if (
            state["captured_token_id"] is None
            and torch.is_tensor(input_ids)
            and input_ids.ndim == 2
            and int(input_ids.shape[1]) == 1
        ):
            token_id = int(input_ids[0, 0].item())
            token_text = tokenizer.decode([token_id])
            if extract_letter_from_token(token_text) is not None:
                state["captured_token_id"] = token_id
                state["captured_token_text"] = token_text
        return args, kwargs

    handles.append(thinker.register_forward_pre_hook(thinker_state_hook, with_kwargs=True))
    for i in range(start, end):
        attn = layers[i].self_attn
        handles.append(attn.register_forward_pre_hook(_make_letter_self_attn_pre_hook(), with_kwargs=True))
        handles.append(attn.register_forward_hook(_make_capture_hook(i, storage)))

    print(f"   Registered dynamic answer-token attention hooks on layers [{start}, {end}).")
    try:
        yield storage, state
    finally:
        for h in handles:
            h.remove()
        print(f"   Removed {len(handles)} dynamic attention hook handles.")


# =============================================================================
# HumMusQA utilities
# =============================================================================

def normalize_difficulty(x: object) -> str:
    s = str(x or "").strip()
    return "" if not s else s[:1].upper() + s[1:].lower()


def safe_identifier(sample: Dict, idx: int) -> str:
    for key in ("identifier", "id", "sample_id", "track_id"):
        if key in sample and sample[key] is not None:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample[key]))[:180]
    return f"sample_{idx:05d}"


def build_option_orders(identifier: str, n_orders: int, seed: int) -> List[Tuple[int, int, int, int]]:
    """
    Return deterministic, unique option permutations for one question.

    Canonical option index 0 is the correct answer; indices 1..3 are the
    distractors. One-order debug runs preserve the historical fixed ordering.
    """
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
    """Build the exact shared-utils HumMusQA prompt and semantic text spans."""
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

    # "instruction" intentionally includes the fixed task wording, the "Question:"
    # and "Options:" labels, and the response-format suffix. The answer strings
    # themselves are kept in the separate "options" span.
    spans = {
        "instruction": [(0, question_start), (question_end, options_start), (options_end, len(prompt))],
        "question": [(question_start, question_end)],
        "options": [(options_start, options_end)],
    }
    return {"prompt": prompt, "spans": spans}


def build_hummusqa_prompt(question: str, options: Sequence[str]) -> str:
    return str(build_hummusqa_prompt_parts(question, options)["prompt"])


def _offset_prompt_spans(
    rendered_text: str,
    prompt_text: str,
    prompt_spans: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Convert prompt-local spans to rendered-chat-template spans.

    Instruction can have multiple fragments; keeping them separate prevents it
    from swallowing question/options tokens during offset classification.
    """
    base = rendered_text.find(prompt_text)
    if base < 0:
        print("   ⚠ Could not locate exact prompt inside rendered chat template; using broad text fallback.")
        return {"instruction": [(0, len(rendered_text))]}

    rendered_spans: Dict[str, List[Tuple[int, int]]] = {}
    for name, spans in prompt_spans.items():
        shifted = [(base + int(s), base + int(e)) for s, e in spans if int(e) > int(s)]
        if not shifted:
            continue
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
        raise ValueError(
            f"Audio dict has no local path and no bytes. Keys: {list(audio_field.keys())}"
        )
    raise ValueError(f"Unsupported audio format: {type(audio_field)}")


def build_conversation(prompt_text: str, audio_path: str):
    """Match the existing HumMusQA evaluation conversation in gpu_utils/analysis_1."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def prepare_inputs(
    sample: Dict,
    sample_idx: int,
    processor: Qwen2_5OmniProcessor,
    tmp_dir: str,
    *,
    audio_path_override: Optional[str] = None,
    option_order: Sequence[int] = (0, 1, 2, 3),
) -> Tuple[Dict[str, torch.Tensor], str, str, bool, str, Dict[str, object]]:
    """
    Returns (inputs_dict, correct_letter, audio_path, should_delete).
    Inputs are NOT moved to any device here — caller handles that.
    """
    identifier = safe_identifier(sample, sample_idx)
    question = str(sample["question"]).strip()
    options, correct_letter = get_options_from_sample(sample, option_order=option_order)
    prompt_parts = build_hummusqa_prompt_parts(question, options)
    prompt_text = str(prompt_parts["prompt"])

    if audio_path_override is None:
        audio_path = materialize_audio(sample["audio"], tmp_dir, identifier)
        should_delete = audio_path.startswith(tmp_dir)
    else:
        audio_path = audio_path_override
        should_delete = False

    conversation = build_conversation(prompt_text, audio_path)

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    # apply_chat_template may return str or list[str] depending on version
    if isinstance(text, list):
        text = text[0]
    text_spans = _offset_prompt_spans(
        rendered_text=text,
        prompt_text=prompt_text,
        prompt_spans=prompt_parts["spans"],  # type: ignore[arg-type]
    )

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    return inputs, correct_letter, audio_path, should_delete, text, text_spans


# =============================================================================
# Device utilities
# =============================================================================

def get_module_device(module: torch.nn.Module) -> torch.device:
    """
    Restituisce il device reale di un modulo, guardando prima i parametri e poi i buffer.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass

    try:
        return next(module.buffers()).device
    except StopIteration:
        pass

    raise RuntimeError(f"Cannot infer device for module {module.__class__.__name__}")


def get_text_backbone_device(model) -> torch.device:
    """
    Device del backbone testuale del Thinker.

    Questo è il device corretto per:
      - input_ids
      - attention_mask
      - cache_position / rope_deltas durante generate()

    Nel tuo device_map è cuda:1.
    """
    return get_module_device(get_thinker(model).model.embed_tokens)


def get_audio_tower_device(model) -> torch.device:
    """
    Device dell'audio tower.

    Questo è il device corretto per:
      - input_features
      - feature_attention_mask
      - audio feature tensors

    Nel tuo device_map è cuda:0.
    """
    return get_module_device(get_thinker(model).audio_tower)


def move_to_device(obj, device: torch.device):
    """
    Movimento ricorsivo sicuro per tensori annidati.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def place_qwen_omni_inputs(
    inputs: Dict,
    *,
    text_device: torch.device,
    audio_device: torch.device,
) -> Dict:
    """
    Placement corretto per Qwen2.5-Omni sharded su più GPU.

    NON spostare tutto sullo stesso device.

    Regola:
      - input_ids e attention_mask vanno sul device del text backbone.
      - input_features e feature_attention_mask vanno sul device dell'audio tower.
      - eventuali tensori audio-specifici vanno sull'audio device.
      - il resto, per default, va sul text device.

    Questo evita simultaneamente:
      1. rope_deltas/cache_position mismatch, se input_ids resta su cuda:0.
      2. audio_tower IndexError / feature_lens inconsistency, se input_features
         viene spostato su cuda:1.
    """
    placed = {}

    audio_like_keys = {
        "input_features",
        "feature_attention_mask",
        "audio_feature_lengths",
        "audio_features",
        "audio_attention_mask",
    }

    text_like_keys = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "cache_position",
    }

    for k, v in dict(inputs).items():
        if not torch.is_tensor(v):
            placed[k] = v
            continue

        if k in audio_like_keys or "audio" in k.lower() or "feature" in k.lower():
            placed[k] = v.to(audio_device)
        elif k in text_like_keys:
            placed[k] = v.to(text_device)
        else:
            placed[k] = v.to(text_device)

    return placed


def debug_input_devices(inputs: Dict) -> Dict[str, str]:
    """
    Piccolo helper diagnostico: stampa su che device sono finiti gli input.
    """
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = str(v.device)
        else:
            out[k] = type(v).__name__
    return out


# =============================================================================
# MCQ letter extraction from generated token
# =============================================================================


def reset_qwen_omni_generation_state(model) -> None:
    """
    Qwen2.5-Omni salva rope_deltas sul Thinker durante forward/generate.
    Con input di lunghezze diverse o dopo errori precedenti può rimanere uno
    stato non coerente. Prima di ogni sample lo resettiamo.
    """
    objs = [
        get_thinker(model),
        getattr(get_thinker(model), "model", None),
    ]
    for obj in objs:
        if obj is not None and hasattr(obj, "rope_deltas"):
            try:
                obj.rope_deltas = None
            except Exception:
                pass


def get_mcq_label_token_ids(tokenizer) -> Dict[str, int]:
    """
    Encode MCQ labels as the single token the model actually generates.

    After Qwen's chat template the response starts with a bare letter
    (e.g. "A"), NOT a space-prefixed one (" A"). The two forms have
    different token IDs in Qwen's tiktoken-based vocabulary, so using
    the wrong one causes ~25 % (random) accuracy when comparing logits.

    We therefore try the bare label first and fall back to the
    space-prefixed form only if the tokenizer splits the bare label into
    multiple sub-tokens (which would be unusual for A/B/C/D).
    """
    label_token_ids: Dict[str, int] = {}
    for label in "ABCD":
        # Bare label first — this is what Qwen generates after "\nassistant\n"
        candidate_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(candidate_ids) != 1:
            # Fallback: space-prefixed form (some tokenizer versions)
            candidate_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(candidate_ids) != 1:
            raise RuntimeError(
                f"MCQ label {label!r} is not a single tokenizer token: {candidate_ids}"
            )
        label_token_ids[label] = int(candidate_ids[0])
    return label_token_ids


def score_constrained_mcq_label(
    model,
    inputs: Dict[str, torch.Tensor],
    tokenizer,
) -> Tuple[str, Dict[str, float], float]:
    """Select A/B/C/D from prompt-final logits without attention capture."""
    label_token_ids = get_mcq_label_token_ids(tokenizer)
    with torch.inference_mode():
        t0 = time.time()
        outputs = model(
            **inputs,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
        elapsed = time.time() - t0
    logits_last = outputs.logits[0, -1].detach().float().cpu()
    label_logits = {
        label: float(logits_last[token_id].item())
        for label, token_id in label_token_ids.items()
    }
    selected_label = max(label_logits, key=label_logits.get)
    return selected_label, label_logits, elapsed


def score_generated_mcq_answer(
    model,
    inputs: Dict[str, torch.Tensor],
    original_input_len: int,
    tokenizer,
) -> Dict:
    """Run the existing HumMusQA baseline style: greedy text generation, no hooks."""
    with torch.inference_mode():
        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=False,
        )
        elapsed = time.time() - t0
    response_ids = [
        int(x) for x in outputs.sequences[0, original_input_len:].detach().cpu().tolist()
    ]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id in response_ids:
        response_ids = response_ids[: response_ids.index(im_end_id)]
    response_text = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    normalized = response_text.upper()
    predicted_letter = normalized if normalized in {"A", "B", "C", "D"} else None
    return {
        "predicted_letter": predicted_letter,
        "response_text": response_text,
        "response_ids": response_ids,
        "first_token_text": tokenizer.decode([response_ids[0]]) if response_ids else "",
        "generate_sec": elapsed,
    }


def capture_answer_token_attention(
    model,
    inputs: Dict[str, torch.Tensor],
    original_input_len: int,
    text_device: torch.device,
    capture_range: Tuple[int, int],
    attention_mode: str,
    tokenizer,
    max_answer_search_tokens: int,
) -> Tuple[int, Dict[int, List[torch.Tensor]], Dict[str, float], List[int]]:
    """
    Generate the first answer token and capture its next-step attention.

    `avllm_literal_generate` is the validation baseline: it runs the intended
    eager generate path for enough tokens to find a standalone A/B/C/D token
    and retain the q_len == 1 attention for that answer token only. The
    Thinker prefix is necessary for the Omni wrapper.

    `cached_decode` preserves the tensor being analyzed while avoiding eager
    prompt-prefill attentions. It uses a single SDPA generate() trajectory and
    enables attention output dynamically only when the first standalone
    A/B/C/D token is fed back as a query. This preserves Qwen's official
    multimodal position/cache preparation.

    `constrained_mcq` is the robust HumMusQA path. It selects the answer by
    comparing the prompt-final logits for the four valid MCQ labels, then
    forwards that selected label once through the cached prompt while
    requesting attention. The queried token is still the selected answer
    token, but no unconstrained response formatting tokens are generated.
    """
    if attention_mode in {"generate_score_only", "score_only", "generation_probe"}:
        raise RuntimeError(f"{attention_mode} is handled in main() before attention capture.")

    if attention_mode == "constrained_mcq":
        label_token_ids = get_mcq_label_token_ids(tokenizer)

        with torch.inference_mode():
            t_prefill = time.time()
            # Full model handles multimodal encoding in the prefill step.
            prefill_outputs = model(
                **inputs,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            prefill_sec = time.time() - t_prefill
            decision_logits = prefill_outputs.logits[0, -1].detach().float().cpu()
            past_key_values = prefill_outputs.past_key_values
            del prefill_outputs

            selected_label = max(
                label_token_ids,
                key=lambda label: float(decision_logits[label_token_ids[label]].item()),
            )
            selected_id = label_token_ids[selected_label]
            answer_input_ids = torch.tensor(
                [[selected_id]], dtype=torch.long, device=text_device
            )
            answer_attention_mask = torch.cat(
                [
                    inputs["attention_mask"].to(text_device),
                    torch.ones(
                        (inputs["attention_mask"].shape[0], 1),
                        dtype=inputs["attention_mask"].dtype,
                        device=text_device,
                    ),
                ],
                dim=1,
            )
            cache_position = torch.tensor(
                [original_input_len], dtype=torch.long, device=text_device
            )

            # Decode step: only input_ids needed (audio already encoded in prefill).
            # Use the thinker directly for the single-token cached decode step.
            thinker = get_thinker(model)
            t_decode = time.time()
            with capture_first_generated_token_attention(
                model, capture_layer_range=capture_range
            ) as attention_storage:
                thinker(
                    input_ids=answer_input_ids,
                    attention_mask=answer_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
            decode_sec = time.time() - t_decode
        return selected_id, attention_storage, {
            "constrained_prefill_sec": prefill_sec,
            "eager_answer_decode_sec": decode_sec,
        }, [selected_id]

    if attention_mode == "avllm_literal_generate":
        with torch.inference_mode():
            t0 = time.time()
            with capture_first_generated_token_attention(
                model, capture_layer_range=capture_range
            ) as attention_storage:
                # Call model.generate() — the full Qwen2_5OmniForConditionalGeneration
                # (with disable_talker()) handles multimodal routing correctly and
                # produces valid text token IDs.
                # The self_attn pre-hooks inside capture_first_generated_token_attention
                # inject output_attentions=True ONLY for q_len==1 decode steps, so we
                # do NOT pass output_attentions=True here (avoids quadratic prefill cost).
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_answer_search_tokens + 1,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
        response_ids = [
            int(x) for x in generated.sequences[0, original_input_len:].detach().cpu().tolist()
        ]
        answer_offset = None
        for idx, candidate_id in enumerate(response_ids[:max_answer_search_tokens]):
            if extract_letter_from_token(tokenizer.decode([candidate_id])) is not None:
                answer_offset = idx
                break
        if answer_offset is None:
            raise RuntimeError(
                "No standalone MCQ answer token A/B/C/D appeared in the first "
                f"{max_answer_search_tokens} literal generated tokens: "
                f"{tokenizer.decode(response_ids[:max_answer_search_tokens])!r}."
            )
        selected_storage: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for layer_idx, tensors in attention_storage.items():
            if answer_offset >= len(tensors):
                raise RuntimeError(
                    f"Missing literal attention step for answer at response offset {answer_offset} "
                    f"in layer {layer_idx}; captured {len(tensors)} q_len==1 tensors."
                )
            selected_storage[layer_idx].append(tensors[answer_offset])
        token_id = response_ids[answer_offset]
        return (
            token_id,
            selected_storage,
            {"literal_generate_sec": time.time() - t0},
            response_ids[: answer_offset + 1],
        )

    if attention_mode != "cached_decode":
        raise ValueError(
            f"Unsupported attention_mode={attention_mode!r}; expected one of {ATTENTION_MODES}."
        )

    with torch.inference_mode():
        # The answer token becomes a query only on the following decode step,
        # so request one extra token past the maximum response search window.
        t_generate = time.time()
        with capture_first_answer_letter_attention(
            model, tokenizer=tokenizer, capture_layer_range=capture_range
        ) as (attention_storage, capture_state):
            generated = model.generate(
                **inputs,
                max_new_tokens=max_answer_search_tokens + 1,
                do_sample=False,
                return_dict_in_generate=True,
            )
        generate_sec = time.time() - t_generate
        response_ids = [
            int(x) for x in generated.sequences[0, original_input_len:].detach().cpu().tolist()
        ]
        answer_offset = None
        for idx, candidate_id in enumerate(response_ids[:max_answer_search_tokens]):
            if extract_letter_from_token(tokenizer.decode([candidate_id])) is not None:
                answer_offset = idx
                break
        if answer_offset is None:
            raise RuntimeError(
                "No standalone MCQ answer token A/B/C/D appeared in the first "
                f"{max_answer_search_tokens} generated tokens: "
                f"{tokenizer.decode(response_ids[:max_answer_search_tokens])!r}."
            )
        token_id = response_ids[answer_offset]
        if capture_state["captured_token_id"] != token_id:
            raise RuntimeError(
                "Dynamic capture did not activate on the located answer token: "
                f"located={tokenizer.decode([token_id])!r}, "
                f"captured={capture_state['captured_token_text']!r}."
            )
        return token_id, attention_storage, {
            "sdpa_dynamic_capture_generate_sec": generate_sec,
        }, response_ids[: answer_offset + 1]


def run_generation_probe(
    model,
    inputs: Dict[str, torch.Tensor],
    original_input_len: int,
    tokenizer,
    max_answer_search_tokens: int,
) -> Dict:
    """Generate a short response without attention hooks."""
    with torch.inference_mode():
        t0 = time.time()
        generated = model.generate(
            **inputs,
            max_new_tokens=max_answer_search_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
        elapsed = time.time() - t0
    response_ids = [
        int(x) for x in generated.sequences[0, original_input_len:].detach().cpu().tolist()
    ]
    answer_offset = None
    for idx, candidate_id in enumerate(response_ids):
        if extract_letter_from_token(tokenizer.decode([candidate_id])) is not None:
            answer_offset = idx
            break
    return {
        "response_ids": response_ids,
        "response_text": tokenizer.decode(response_ids),
        "answer_offset": answer_offset,
        "answer_letter": (
            extract_letter_from_token(tokenizer.decode([response_ids[answer_offset]]))
            if answer_offset is not None
            else None
        ),
        "generate_sec": elapsed,
    }


def extract_letter_from_token(token_text: str) -> Optional[str]:
    """
    Best-effort extraction of A/B/C/D from a decoded token string.
    The model may generate ' A', '\nA', 'A', '(A)', etc.
    """
    cleaned = token_text.strip().upper()
    match = re.fullmatch(r"[\(\[\{]?\s*([A-D])\s*[\)\]\}\.,:]?", cleaned)
    return match.group(1) if match else None


# =============================================================================
# Attention aggregation
# =============================================================================

def aggregate_attention_by_layer(
    storage: Dict[int, List[torch.Tensor]],
    token_types: List[str],
    original_input_len: int,
    total_layers: int,
) -> Dict:
    """
    Per ogni layer calcola la frazione di attenzione dal primo token generato
    verso audio/query_text.

    Durante la generazione/cached decode:
      - prefill: q_len > 1
      - generated step: q_len == 1

    L'esperimento del paper riguarda i generated tokens, quindi usiamo solo
    i tensori con q_len == 1.
    """
    component_by_layer: Dict[str, List[float]] = {
        "audio": [],
        "instruction": [],
        "question": [],
        "options": [],
        "other_text": [],
        "query_text": [],
        "other": [],
    }
    n_gen_tensors_by_layer: List[int] = []

    audio_idx = [i for i, t in enumerate(token_types[:original_input_len]) if t == "audio"]
    instruction_idx = [i for i, t in enumerate(token_types[:original_input_len]) if t == "instruction"]
    question_idx = [i for i, t in enumerate(token_types[:original_input_len]) if t == "question"]
    options_idx = [i for i, t in enumerate(token_types[:original_input_len]) if t == "options"]
    other_text_idx = [i for i, t in enumerate(token_types[:original_input_len]) if t == "other_text"]
    text_idx = instruction_idx + question_idx + options_idx + other_text_idx
    other_idx = [
        i for i, t in enumerate(token_types[:original_input_len])
        if t not in {"audio", "instruction", "question", "options", "other_text"}
    ]

    for layer_idx in range(total_layers):
        tensors = storage.get(layer_idx, [])
        vals = {name: [] for name in component_by_layer}
        n_gen = 0

        for t in tensors:
            if not (torch.is_tensor(t) and t.ndim == 4):
                continue

            q_len = int(t.shape[-2])
            k_len = int(t.shape[-1])

            # Solo step autoregressivo: query = token generato.
            if q_len != 1:
                continue

            # Deve poter attendere almeno al prompt originale.
            if k_len < original_input_len:
                continue

            weights = t[0].float().mean(dim=0)[0]  # [k_len]
            prompt_weights = weights[:original_input_len]
            denom = float(prompt_weights.sum().item())
            if denom <= 1e-12:
                continue

            def frac(indices):
                return float(prompt_weights[indices].sum().item() / denom) if indices else 0.0

            vals["audio"].append(frac(audio_idx))
            vals["instruction"].append(frac(instruction_idx))
            vals["question"].append(frac(question_idx))
            vals["options"].append(frac(options_idx))
            vals["other_text"].append(frac(other_text_idx))
            vals["query_text"].append(frac(text_idx))
            vals["other"].append(frac(other_idx))
            n_gen += 1

        for name in component_by_layer:
            component_by_layer[name].append(
                float(np.mean(vals[name])) if vals[name] else float("nan")
            )
        n_gen_tensors_by_layer.append(n_gen)

    return {
        **component_by_layer,
        "n_gen_tensors_by_layer": n_gen_tensors_by_layer,
        "n_gen_tensors_total": int(sum(n_gen_tensors_by_layer)),
    }


# =============================================================================
# Saving
# =============================================================================

def fmt_float(x) -> str:
    try:
        xf = float(x)
    except Exception:
        return "nan"
    return "nan" if np.isnan(xf) else f"{xf:.8f}"


def block_mean(values: Sequence[float], start: int, end: int) -> float:
    arr = np.array(values[start:end], dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def save_results(records: List[Dict], output_dir: str, metadata: Dict, total_layers: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    audio_cols = [f"audio_layer_{i}" for i in range(total_layers)]
    instruction_cols = [f"instruction_layer_{i}" for i in range(total_layers)]
    question_cols = [f"question_layer_{i}" for i in range(total_layers)]
    options_cols = [f"options_layer_{i}" for i in range(total_layers)]
    other_text_cols = [f"other_text_layer_{i}" for i in range(total_layers)]
    text_cols = [f"query_text_layer_{i}" for i in range(total_layers)]
    other_cols = [f"other_layer_{i}" for i in range(total_layers)]

    base_cols = [
        "sample_idx", "evaluation_idx", "identifier", "option_permutation_idx", "option_order",
        "question", "correct_answer", "predicted_answer", "predicted_option_source_index",
        "generated_token", "generated_prefix", "answer_token_position", "is_correct",
        "main_category", "difficulty",
        "n_audio_tokens", "n_instruction_tokens", "n_question_tokens", "n_options_tokens",
        "n_other_text_tokens", "n_query_text_tokens", "n_input_tokens", "elapsed_sec",
    ]

    layer_path = os.path.join(output_dir, "attention_per_layer.csv")
    with open(layer_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                base_cols
                + audio_cols
                + instruction_cols
                + question_cols
                + options_cols
                + other_text_cols
                + text_cols
                + other_cols
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in base_cols}
            row["is_correct"] = int(rec.get("is_correct", 0))
            for i in range(total_layers):
                row[f"audio_layer_{i}"] = fmt_float(rec["audio_by_layer"][i])
                row[f"instruction_layer_{i}"] = fmt_float(rec["instruction_by_layer"][i])
                row[f"question_layer_{i}"] = fmt_float(rec["question_by_layer"][i])
                row[f"options_layer_{i}"] = fmt_float(rec["options_by_layer"][i])
                row[f"other_text_layer_{i}"] = fmt_float(rec["other_text_by_layer"][i])
                row[f"query_text_layer_{i}"] = fmt_float(rec["query_text_by_layer"][i])
                row[f"other_layer_{i}"] = fmt_float(rec["other_by_layer"][i])
            writer.writerow(row)

    summary_path = os.path.join(output_dir, "summary_stats.csv")
    thirds = [
        (0, total_layers // 3),
        (total_layers // 3, 2 * total_layers // 3),
        (2 * total_layers // 3, total_layers),
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "sample_idx", "evaluation_idx", "identifier", "option_permutation_idx", "option_order",
            "correct_answer", "predicted_answer", "predicted_option_source_index",
            "main_category", "difficulty", "is_correct",
            "audio_early", "audio_middle", "audio_late",
            "instruction_early", "instruction_middle", "instruction_late",
            "question_early", "question_middle", "question_late",
            "options_early", "options_middle", "options_late",
            "text_early", "text_middle", "text_late",
            "audio_drop_early_to_late", "text_gain_early_to_late",
            "late_question_audio_ratio", "late_options_audio_ratio", "late_text_audio_ratio",
            "n_audio_tokens", "n_instruction_tokens", "n_question_tokens", "n_options_tokens",
            "n_other_text_tokens", "n_query_text_tokens",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            a = rec["audio_by_layer"]
            instr = rec["instruction_by_layer"]
            q = rec["question_by_layer"]
            opt = rec["options_by_layer"]
            t = rec["query_text_by_layer"]
            a_e, a_m, a_l = [block_mean(a, s, e) for s, e in thirds]
            i_e, i_m, i_l = [block_mean(instr, s, e) for s, e in thirds]
            q_e, q_m, q_l = [block_mean(q, s, e) for s, e in thirds]
            o_e, o_m, o_l = [block_mean(opt, s, e) for s, e in thirds]
            t_e, t_m, t_l = [block_mean(t, s, e) for s, e in thirds]
            writer.writerow({
                "sample_idx": rec["sample_idx"],
                "evaluation_idx": rec["evaluation_idx"],
                "identifier": rec["identifier"],
                "option_permutation_idx": rec["option_permutation_idx"],
                "option_order": rec["option_order"],
                "correct_answer": rec["correct_answer"],
                "predicted_answer": rec["predicted_answer"],
                "predicted_option_source_index": rec["predicted_option_source_index"],
                "main_category": rec.get("main_category", ""),
                "difficulty": rec.get("difficulty", ""),
                "is_correct": int(rec.get("is_correct", 0)),
                "audio_early": fmt_float(a_e),
                "audio_middle": fmt_float(a_m),
                "audio_late": fmt_float(a_l),
                "instruction_early": fmt_float(i_e),
                "instruction_middle": fmt_float(i_m),
                "instruction_late": fmt_float(i_l),
                "question_early": fmt_float(q_e),
                "question_middle": fmt_float(q_m),
                "question_late": fmt_float(q_l),
                "options_early": fmt_float(o_e),
                "options_middle": fmt_float(o_m),
                "options_late": fmt_float(o_l),
                "text_early": fmt_float(t_e),
                "text_middle": fmt_float(t_m),
                "text_late": fmt_float(t_l),
                "audio_drop_early_to_late": fmt_float(a_e - a_l),
                "text_gain_early_to_late": fmt_float(t_l - t_e),
                "late_question_audio_ratio": (
                    fmt_float(q_l / max(a_l, 1e-12))
                    if np.isfinite(q_l) and np.isfinite(a_l)
                    else "nan"
                ),
                "late_options_audio_ratio": (
                    fmt_float(o_l / max(a_l, 1e-12))
                    if np.isfinite(o_l) and np.isfinite(a_l)
                    else "nan"
                ),
                "late_text_audio_ratio": (
                    fmt_float(t_l / max(a_l, 1e-12))
                    if np.isfinite(t_l) and np.isfinite(a_l)
                    else "nan"
                ),
                "n_audio_tokens": rec.get("n_audio_tokens", 0),
                "n_instruction_tokens": rec.get("n_instruction_tokens", 0),
                "n_question_tokens": rec.get("n_question_tokens", 0),
                "n_options_tokens": rec.get("n_options_tokens", 0),
                "n_other_text_tokens": rec.get("n_other_text_tokens", 0),
                "n_query_text_tokens": rec.get("n_query_text_tokens", 0),
            })

    meta_path = os.path.join(output_dir, "metadata.json")
    ok = sum(int(r.get("is_correct", 0)) for r in records)
    meta = dict(metadata)
    meta.update(
        {
            "n_processed": len(records),
            "n_correct": ok,
            "accuracy_over_saved_evaluations": round(ok / len(records), 6) if records else None,
            "saved_at": datetime.now().isoformat(),
            "files": {
                "attention_per_layer": layer_path,
                "summary_stats": summary_path,
            },
        }
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"   ✅ {layer_path}")
    print(f"   ✅ {summary_path}")
    print(f"   ✅ {meta_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1 — HumMusQA attention pattern for Qwen2.5-Omni (FIXED)"
    )
    parser.add_argument("--model_path", default="/nas/home/fingenito/Models/Qwen2.5-Omni-3B")
    parser.add_argument("--output_dir", default="results/attention_patterns_hummusqa")
    parser.add_argument("--dataset_name", default="mtg-upf/HumMusQA")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--tmp_dir", default=None)
    parser.add_argument(
        "--sample_start",
        type=int,
        default=0,
        help="Zero-based dataset offset applied before --max_samples (useful for diagnostics).",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--option_permutations",
        type=int,
        default=1,
        help=(
            "Number of unique option orders evaluated per question. Use 4 for the "
            "HumMusQA protocol; default 1 preserves a fast fixed-order debug run."
        ),
    )
    parser.add_argument(
        "--option_seed",
        type=int,
        default=42,
        help="Seed for deterministic option-order permutations.",
    )
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--difficulties", nargs="*", default=None)
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--capture_layer_start", type=int, default=0)
    parser.add_argument("--capture_layer_end", type=int, default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--baseline_device",
        default="cuda:0",
        help="Single device for generate_score_only, matching the existing Thinker-only runner.",
    )
    parser.add_argument(
        "--max_answer_search_tokens",
        type=int,
        default=8,
        help=(
            "Maximum number of greedy response tokens inspected to locate a standalone "
            "A/B/C/D answer token before marking the sample invalid."
        ),
    )
    parser.add_argument(
        "--attention_mode",
        choices=ATTENTION_MODES,
        default="avllm_literal_generate",
        help=(
            "avllm_literal_generate [DEFAULT] is the primary AVLLM-aligned mode: loads "
            "Qwen2_5OmniForConditionalGeneration with eager attention + disable_talker(), "
            "calls model.generate(), and captures per-layer attention via self_attn hooks "
            "from the first generated A/B/C/D answer token towards the multimodal prompt. "
            "generate_score_only validates accuracy using free-generation without attention "
            "(flash_attention_2, Thinker-only). "
            "score_only validates A/B/C/D logit ranking without attention (sdpa). "
            "generation_probe runs generation without hooks to diagnose response format. "
            "[Diagnostic] constrained_mcq selects answer from prompt-final logits then captures "
            "answer-token attention via cached decode (eager). "
            "[Diagnostic] cached_decode uses SDPA generation with a dynamic pre-hook that "
            "activates eager attention only when the first A/B/C/D token is fed as query."
        ),
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately on first error.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.option_permutations <= 24:
        raise ValueError("--option_permutations must be between 1 and 24.")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True)
        tmp_context = None
        tmp_dir = args.tmp_dir
    else:
        tmp_context = tempfile.TemporaryDirectory(prefix="hummusqa_exp1_")
        tmp_dir = tmp_context.name

    print("\n" + "=" * 78)
    print("Experiment 1 — Attention Pattern Analysis | HumMusQA audio-text MCQ")
    print(f"Model     : {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode      : {args.attention_mode}")
    if args.attention_mode == "generate_score_only":
        print(f"           baseline model: Thinker-only on {args.baseline_device} with flash_attention_2")
    else:
        print("           Qwen2_5OmniForConditionalGeneration + disable_talker() (AVLLM-aligned)")
        if args.attention_mode not in {"score_only", "generation_probe"}:
            print("           attn: SDPA for prefill (fast), eager fallback for decode capture")
        print("           split device placement: text tensors -> embed_tokens device, audio -> audio_tower")
    if args.attention_mode == "avllm_literal_generate":
        print("           literal validation path: eager generate through answer-letter query step")
    elif args.attention_mode == "generation_probe":
        print("           diagnostic path: sequential SDPA generation without attention hooks")
    elif args.attention_mode == "generate_score_only":
        print("           baseline path: existing greedy text generation without attention capture")
    elif args.attention_mode == "score_only":
        print("           baseline path: constrained A/B/C/D scoring without attention capture")
    elif args.attention_mode == "constrained_mcq":
        print("           HumMusQA path: constrained A/B/C/D logits + cached answer-token attention")
    else:
        print("           diagnostic path: unconstrained SDPA generation + dynamic answer-token capture")
    print("=" * 78 + "\n")

    # ------------------------------------------------------------------
    # Load model following the AVLLM paper approach
    # ------------------------------------------------------------------
    print("Loading model...")
    device_map_arg = args.device_map
    if device_map_arg != "auto" and device_map_arg.startswith("cuda"):
        device_map_arg = {"": device_map_arg}

    # Attention implementation selection:
    #
    #   generate_score_only  → flash_attention_2, Thinker-only, single device
    #                          (fastest accuracy baseline, no attention capture)
    #
    #   All other modes      → sdpa, full Omni + disable_talker()
    #
    # WHY sdpa INSTEAD OF eager for capture modes (avllm_literal_generate etc.)
    # --------------------------------------------------------------------------
    # Qwen2_5OmniSdpaAttention.forward() has a built-in fallback:
    #
    #   if output_attentions:
    #       return super().forward(...)   # <-- calls Qwen2_5OmniAttention (eager)
    #
    # Our self_attn pre-hook injects output_attentions=True ONLY when q_len==1
    # (autoregressive decode step). This means:
    #   - Prefill (q_len=1619): SDPA fused kernel  → fast,  no explicit O(n²) matrix
    #   - Decode  (q_len=1):    eager fallback      → slow(ish), but returns
    #                                                  attn_weights for hook capture
    #
    # The semantics of the captured attention are IDENTICAL to loading with
    # attn_implementation="eager" because the fallback calls the exact same eager
    # forward path. Only the prefill changes — from ~3-4 min to ~30-60 s.
    # Combined with single-GPU deployment (--device_map cuda:N) this gives
    # ~5-10× per-sample speedup vs. the original eager+pipeline-parallel setup.
    if args.attention_mode == "generate_score_only":
        load_attention_impl = "flash_attention_2"
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
            attn_implementation=load_attention_impl,
        ).eval().to(args.baseline_device)
        model_class = "Qwen2_5OmniThinkerForConditionalGeneration_single_gpu"
    else:
        # sdpa for all capture and non-capture modes.
        # Capture modes rely on SDPA→eager fallback (see comment above).
        load_attention_impl = "sdpa"
        # Load the full Omni model (Thinker + Talker) then disable the Talker.
        # disable_talker() removes the audio-generation path so model.generate()
        # produces text tokens (following the AVLLM paper exactly).
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            attn_implementation=load_attention_impl,
            device_map=device_map_arg,
        )
        model.disable_talker()
        model.eval()
        model_class = "Qwen2_5OmniForConditionalGeneration_talker_disabled"

    print("Loading processor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    if args.attention_mode in {"score_only", "constrained_mcq"}:
        mcq_label_token_ids = get_mcq_label_token_ids(processor.tokenizer)
        decoded_labels = {
            label: processor.tokenizer.decode([token_id])
            for label, token_id in mcq_label_token_ids.items()
        }
        print(f"MCQ label tokens   : {mcq_label_token_ids} decoded={decoded_labels}")

    thinker = get_thinker(model)
    # For generate_score_only (Thinker-only): get_thinker(model) = model itself,
    #   thinker.config = model.config (the thinker config with audio_token_index).
    # For all other modes (full Omni + disable_talker):
    #   get_thinker(model) = model.thinker,
    #   thinker.config = Qwen2_5OmniThinker config (has audio_token_index).
    thinker_cfg = thinker.config
    total_layers = len(thinker.model.layers)
    capture_end = total_layers if args.capture_layer_end is None else args.capture_layer_end
    capture_range = (args.capture_layer_start, capture_end)
    model_device = getattr(model, "device", None)
    text_device = get_text_backbone_device(model)
    audio_device = get_audio_tower_device(model)

    print(f"Model layers       : {total_layers}")
    print(f"audio_token_index  : {thinker_cfg.audio_token_index}")
    print(f"capture layers     : [{capture_range[0]}, {capture_range[1]})")
    print(f"loaded attention   : {load_attention_impl}")
    print(f"layer-0 attention  : {type(thinker.model.layers[0].self_attn).__name__}")
    print(f"model.device       : {model_device}")
    print(f"text device        : {text_device}")
    print(f"audio tower device : {audio_device}")
    print(f"hf_device_map      : {getattr(model, 'hf_device_map', 'N/A')}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("\nLoading HumMusQA...")
    if args.dataset_path:
        dataset = load_dataset(
            "parquet",
            data_files={args.dataset_split: args.dataset_path},
            split=args.dataset_split,
        )
    else:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.cast_column("audio", HFAudio(decode=False))

    if args.categories:
        wanted = set(args.categories)
        dataset = dataset.filter(lambda x: x.get("main_category", "") in wanted)
    if args.difficulties:
        wanted_diff = {normalize_difficulty(d) for d in args.difficulties}
        dataset = dataset.filter(
            lambda x: normalize_difficulty(x.get("difficulty", "")) in wanted_diff
        )

    samples = list(dataset)
    if args.sample_start:
        samples = samples[args.sample_start :]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    evaluation_items = []
    for local_idx, sample in enumerate(samples):
        dataset_idx = args.sample_start + local_idx
        identifier = safe_identifier(sample, dataset_idx)
        for permutation_idx, option_order in enumerate(
            build_option_orders(identifier, args.option_permutations, args.option_seed)
        ):
            evaluation_items.append((dataset_idx, sample, permutation_idx, option_order))
    print(
        f"Questions to process: {len(samples)} (starting at dataset offset {args.sample_start}) | "
        f"option permutations: {args.option_permutations} | evaluations: {len(evaluation_items)}"
    )

    records: List[Dict] = []
    score_records: List[Dict] = []
    probe_records: List[Dict] = []
    errors: List[Dict] = []
    audio_path_cache: Optional[str] = None   # reuse materialized WAV within sample

    for evaluation_idx, (sample_idx, sample, option_permutation_idx, option_order) in enumerate(
        evaluation_items
    ):
        identifier = safe_identifier(sample, sample_idx)
        t0 = time.time()
        order_label = "".join(chr(65 + int(i)) for i in option_order)
        print(
            f"\n[{evaluation_idx + 1:04d}/{len(evaluation_items):04d}] {identifier} "
            f"permutation={option_permutation_idx + 1}/{args.option_permutations} "
            f"order={order_label}"
        )
        audio_path: Optional[str] = None
        should_delete = False

        try:
            # ----------------------------------------------------------
            # 1. Build processor inputs (ONCE, no suffix)
            # ----------------------------------------------------------
            inputs, correct_letter, audio_path, should_delete, rendered_text, text_spans = prepare_inputs(
                sample, sample_idx, processor, tmp_dir, option_order=option_order
            )
            inputs = place_qwen_omni_inputs(
                inputs,
                text_device=text_device,
                audio_device=audio_device,
            )
            if args.attention_mode == "generate_score_only":
                inputs = {
                    key: (
                        value.to(dtype=model.dtype)
                        if torch.is_tensor(value) and value.is_floating_point()
                        else value
                    )
                    for key, value in inputs.items()
                }

            devs = debug_input_devices(inputs)
            print(f"   Input devices: {devs}")

            if "input_ids" in devs and devs["input_ids"] != str(text_device):
                raise RuntimeError(
                    f"input_ids must be on text_device={text_device}, found {devs['input_ids']}"
                )

            if "input_features" in devs and devs["input_features"] != str(audio_device):
                raise RuntimeError(
                    f"input_features must be on audio_device={audio_device}, found {devs['input_features']}"
                )

            if "feature_attention_mask" in devs and devs["feature_attention_mask"] != str(audio_device):
                raise RuntimeError(
                    f"feature_attention_mask must be on audio_device={audio_device}, found {devs['feature_attention_mask']}"
                )

            original_input_len = int(inputs["input_ids"].shape[1])
            token_mapping = create_token_type_mapping(
                inputs["input_ids"],
                thinker_cfg,
                tokenizer=processor.tokenizer,
                rendered_text=rendered_text,
                text_spans=text_spans,
            )
            counts = Counter(token_mapping)
            print(f"   Token counts: {dict(counts)} | input_len={original_input_len}")

            if counts.get("audio", 0) == 0:
                raise RuntimeError(
                    "No audio tokens found. Check processor / process_mm_info pipeline."
                )

            # ----------------------------------------------------------
            # 2. Produce first answer token and capture its q_len == 1
            #    attention according to the requested execution mode.
            # ----------------------------------------------------------
            reset_qwen_omni_generation_state(model)
            print(f"   Running attention mode: {args.attention_mode}...")
            if args.attention_mode == "generate_score_only":
                generation_result = score_generated_mcq_answer(
                    model=model,
                    inputs=inputs,
                    original_input_len=original_input_len,
                    tokenizer=processor.tokenizer,
                )
                pred_letter = generation_result["predicted_letter"]
                is_correct = int(pred_letter == correct_letter)
                semantic_option = (
                    int(option_order[ord(pred_letter) - 65]) if pred_letter is not None else None
                )
                score_records.append(
                    {
                        "sample_idx": sample_idx,
                        "evaluation_idx": evaluation_idx,
                        "identifier": identifier,
                        "option_permutation_idx": option_permutation_idx,
                        "option_order": order_label,
                        "correct_answer": correct_letter,
                        "predicted_answer": pred_letter or "",
                        "predicted_option_source_index": semantic_option,
                        "is_correct": is_correct,
                        "elapsed_sec": round(time.time() - t0, 3),
                        "score_forward_sec": round(generation_result["generate_sec"], 3),
                        "generated_response": generation_result["response_text"],
                        "generated_response_ids": generation_result["response_ids"],
                        "first_token_text": generation_result["first_token_text"],
                    }
                )
                print(
                    f"   response={generation_result['response_text']!r} "
                    f"first_token={generation_result['first_token_text']!r} "
                    f"ids={generation_result['response_ids']} "
                    f"predicted={pred_letter!r} GT={correct_letter} correct={is_correct} "
                    f"generate_sec={generation_result['generate_sec']:.2f}s"
                )
                continue
            if args.attention_mode == "score_only":
                pred_letter, label_logits, score_sec = score_constrained_mcq_label(
                    model=model,
                    inputs=inputs,
                    tokenizer=processor.tokenizer,
                )
                is_correct = int(pred_letter == correct_letter)
                semantic_option = int(option_order[ord(pred_letter) - 65])
                score_records.append(
                    {
                        "sample_idx": sample_idx,
                        "evaluation_idx": evaluation_idx,
                        "identifier": identifier,
                        "option_permutation_idx": option_permutation_idx,
                        "option_order": order_label,
                        "correct_answer": correct_letter,
                        "predicted_answer": pred_letter,
                        "predicted_option_source_index": semantic_option,
                        "is_correct": is_correct,
                        "elapsed_sec": round(time.time() - t0, 3),
                        "score_forward_sec": round(score_sec, 3),
                        **{f"logit_{label}": value for label, value in label_logits.items()},
                    }
                )
                print(
                    f"   predicted={pred_letter!r} GT={correct_letter} correct={is_correct} "
                    f"score_forward_sec={score_sec:.2f}s logits={label_logits}"
                )
                continue
            if args.attention_mode == "generation_probe":
                probe_result = run_generation_probe(
                    model=model,
                    inputs=inputs,
                    original_input_len=original_input_len,
                    tokenizer=processor.tokenizer,
                    max_answer_search_tokens=args.max_answer_search_tokens,
                )
                probe_records.append(
                    {
                        "sample_idx": sample_idx,
                        "identifier": identifier,
                        "correct_answer": correct_letter,
                        **probe_result,
                    }
                )
                print(
                    f"   Probe: elapsed={probe_result['generate_sec']:.2f}s  "
                    f"answer={probe_result['answer_letter']!r}  "
                    f"answer_position={None if probe_result['answer_offset'] is None else probe_result['answer_offset'] + 1}  "
                    f"response={probe_result['response_text']!r}  "
                    f"ids={probe_result['response_ids']}"
                )
                continue
            new_token_id, attention_storage, mode_timings, generated_prefix_ids = capture_answer_token_attention(
                model=model,
                inputs=inputs,
                original_input_len=original_input_len,
                text_device=text_device,
                capture_range=capture_range,
                attention_mode=args.attention_mode,
                tokenizer=processor.tokenizer,
                max_answer_search_tokens=args.max_answer_search_tokens,
            )
            print(
                "   Timings: "
                + ", ".join(f"{name}={value:.2f}s" for name, value in mode_timings.items())
            )

            # ----------------------------------------------------------
            # 3. Decode the predicted answer letter
            # ----------------------------------------------------------
            new_token_text = processor.tokenizer.decode([new_token_id])
            generated_prefix_text = processor.tokenizer.decode(generated_prefix_ids)
            answer_token_position = len(generated_prefix_ids)
            pred_letter = extract_letter_from_token(new_token_text)

            if pred_letter is None:
                raise RuntimeError(
                    "Located token is not a standalone MCQ answer letter, so answer-token "
                    f"attention would be misaligned: token_id={new_token_id}, decoded={new_token_text!r}. "
                    "Skipping this sample instead of parsing a different response."
                )

            print(
                f"   predicted={pred_letter!r}  token_id={new_token_id}  "
                f"decoded={new_token_text!r}  answer_position={answer_token_position}  "
                f"response_prefix={generated_prefix_text!r}"
            )

            # ----------------------------------------------------------
            # 4. Validate captures
            # ----------------------------------------------------------
            n_captured = sum(1 for v in attention_storage.values() if v)
            if n_captured == 0:
                raise RuntimeError(
                    "No attention weights were captured. "
                    "Check output_attentions fallback/capture on the selected attention backend."
                )

            # ----------------------------------------------------------
            # 5. Aggregate attention fractions per layer
            # ----------------------------------------------------------
            agg = aggregate_attention_by_layer(
                attention_storage, token_mapping, original_input_len, total_layers
            )

            all_nan_audio = np.all(np.isnan(np.array(agg["audio"], dtype=float)))
            all_nan_text = np.all(np.isnan(np.array(agg["query_text"], dtype=float)))
            if all_nan_audio and all_nan_text:
                raise RuntimeError(
                    "Attention aggregation is all NaN. Verify capture hooks fired correctly."
                )

            is_correct = int(pred_letter == correct_letter)
            elapsed = round(time.time() - t0, 3)
            n_query_text_tokens = (
                counts.get("instruction", 0)
                + counts.get("question", 0)
                + counts.get("options", 0)
                + counts.get("other_text", 0)
            )
            records.append(
                {
                    "sample_idx": sample_idx,
                    "evaluation_idx": evaluation_idx,
                    "identifier": identifier,
                    "option_permutation_idx": option_permutation_idx,
                    "option_order": order_label,
                    "question": str(sample.get("question", "")),
                    "correct_answer": correct_letter,
                    "predicted_answer": pred_letter,
                    "predicted_option_source_index": int(option_order[ord(pred_letter) - 65]),
                    "generated_token": new_token_text,
                    "generated_prefix": generated_prefix_text,
                    "answer_token_position": answer_token_position,
                    "is_correct": is_correct,
                    "main_category": sample.get("main_category", ""),
                    "difficulty": normalize_difficulty(sample.get("difficulty", "")),
                    "n_audio_tokens": counts.get("audio", 0),
                    "n_instruction_tokens": counts.get("instruction", 0),
                    "n_question_tokens": counts.get("question", 0),
                    "n_options_tokens": counts.get("options", 0),
                    "n_other_text_tokens": counts.get("other_text", 0),
                    "n_query_text_tokens": n_query_text_tokens,
                    "n_input_tokens": original_input_len,
                    "elapsed_sec": elapsed,
                    "mode_timings": mode_timings,
                    "audio_by_layer": agg["audio"],
                    "instruction_by_layer": agg["instruction"],
                    "question_by_layer": agg["question"],
                    "options_by_layer": agg["options"],
                    "other_text_by_layer": agg["other_text"],
                    "query_text_by_layer": agg["query_text"],
                    "other_by_layer": agg["other"],
                }
            )
            print(
                f"   GT={correct_letter} Pred={pred_letter} correct={is_correct} "
                f"elapsed={elapsed:.1f}s"
            )

        except Exception as exc:
            import traceback

            print(f"   ❌ ERROR: {exc}")
            traceback.print_exc()
            errors.append(
                {
                    "sample_idx": sample_idx,
                    "evaluation_idx": evaluation_idx,
                    "identifier": identifier,
                    "option_permutation_idx": option_permutation_idx,
                    "option_order": order_label,
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

        # Checkpoint
        if args.checkpoint_every > 0 and records and (evaluation_idx + 1) % args.checkpoint_every == 0:
            print(f"\n💾 Checkpoint at evaluation {evaluation_idx + 1}")
            save_results(
                records,
                args.output_dir,
                {
                    "status": "in_progress",
                    "model_path": args.model_path,
                    "attention_mode": args.attention_mode,
                    "errors": errors,
                },
                total_layers,
            )

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    complete_consistency_groups = 0
    consistent_prediction_groups = 0
    metric_records = (
        score_records
        if args.attention_mode in {"generate_score_only", "score_only"}
        else records
    )
    if args.option_permutations > 1 and metric_records:
        records_by_question: Dict[str, List[Dict]] = defaultdict(list)
        for rec in metric_records:
            records_by_question[str(rec["identifier"])].append(rec)
        for grouped_records in records_by_question.values():
            if len(grouped_records) != args.option_permutations:
                continue
            complete_consistency_groups += 1
            semantic_predictions = {
                (
                    int(rec["predicted_option_source_index"])
                    if rec.get("predicted_option_source_index") is not None
                    else -1
                )
                for rec in grouped_records
            }
            consistent_prediction_groups += int(len(semantic_predictions) == 1)
    consistency_rate = (
        consistent_prediction_groups / complete_consistency_groups
        if complete_consistency_groups
        else None
    )

    print("\n" + "=" * 78)
    if args.attention_mode == "generation_probe":
        print(f"Done. probed={len(probe_records)} errors={len(errors)}")
    elif args.attention_mode in {"generate_score_only", "score_only"}:
        print(f"Done. scored={len(score_records)} errors={len(errors)}")
    else:
        print(f"Done. processed={len(records)} errors={len(errors)}")
    if metric_records:
        ok = sum(int(r["is_correct"]) for r in metric_records)
        if args.option_permutations == 1:
            print(f"Fixed-order accuracy diagnostic: {ok / len(metric_records):.4f} ({ok}/{len(metric_records)})")
        else:
            print(f"Permuted-option accuracy: {ok / len(metric_records):.4f} ({ok}/{len(metric_records)})")
            if consistency_rate is not None:
                print(
                    "Option-order semantic consistency: "
                    f"{consistency_rate:.4f} "
                    f"({consistent_prediction_groups}/{complete_consistency_groups})"
                )
    print("=" * 78 + "\n")

    if args.attention_mode == "generation_probe":
        probe_path = os.path.join(args.output_dir, "generation_probe.json")
        with open(probe_path, "w", encoding="utf-8") as f:
            json.dump(probe_records, f, ensure_ascii=False, indent=2)
        print(f"   Probe results: {probe_path}")
    if args.attention_mode in {"generate_score_only", "score_only"}:
        score_path = os.path.join(args.output_dir, f"{args.attention_mode}_predictions.csv")
        with open(score_path, "w", newline="", encoding="utf-8") as f:
            fields = [
                "sample_idx", "evaluation_idx", "identifier", "option_permutation_idx",
                "option_order", "correct_answer", "predicted_answer",
                "predicted_option_source_index", "is_correct", "elapsed_sec",
                "score_forward_sec", "logit_A", "logit_B", "logit_C", "logit_D",
                "generated_response", "generated_response_ids", "first_token_text",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(score_records)
        print(f"   Score-only predictions: {score_path}")

    save_results(
        records,
        args.output_dir,
        {
            "status": "complete",
            "model_path": args.model_path,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "dataset_path": args.dataset_path,
            "max_samples": args.max_samples,
            "sample_start": args.sample_start,
            "option_permutations": args.option_permutations,
            "option_seed": args.option_seed,
            "categories": args.categories,
            "difficulties": args.difficulties,
            "capture_layer_range": list(capture_range),
            "max_answer_search_tokens": args.max_answer_search_tokens,
            "attn_implementation_at_load": load_attention_impl,
            "attention_mode": args.attention_mode,
            "generation_probe_results": (
                "generation_probe.json" if args.attention_mode == "generation_probe" else None
            ),
            "score_only_results": (
                f"{args.attention_mode}_predictions.csv"
                if args.attention_mode in {"generate_score_only", "score_only"}
                else None
            ),
            "cached_decode_prefill_attn_implementation": (
                "sdpa"
                if args.attention_mode in {"constrained_mcq", "cached_decode"}
                else None
            ),
            "device_map": args.device_map,
            "model_class": model_class,
            "talker": (
                "not_loaded"
                if args.attention_mode == "generate_score_only"
                else "disabled"  # Qwen2_5OmniForConditionalGeneration + disable_talker()
            ),
            "baseline_device": args.baseline_device if args.attention_mode == "generate_score_only" else None,
            "option_order_protocol": (
                "fixed: A=answer, B=distractor_1, C=distractor_2, D=distractor_3"
                if args.option_permutations == 1
                else "deterministic unique randomized order per question; canonical option index 0 is answer"
            ),
            "complete_consistency_groups": complete_consistency_groups,
            "semantic_consistency_rate": consistency_rate,
            "n_scored_without_attention": len(score_records),
            "score_only_accuracy": (
                round(sum(int(r["is_correct"]) for r in score_records) / len(score_records), 6)
                if score_records else None
            ),
            "errors": errors,
            "methodological_note": (
                "Experiment 1 adapted for HumMusQA (audio-only), following the AVLLM paper. "
                "Model: Qwen2_5OmniForConditionalGeneration (Qwen2.5-Omni-3B) with "
                "attn_implementation='sdpa' (eager fallback via self_attn pre-hook for decode) and disable_talker(). "
                "Attention captured via register_forward_hook on each self_attn layer "
                "(model.thinker.model.layers[i].self_attn). Hooks store output[1] "
                "(attention weights) which is non-None only at decode steps where a "
                "self_attn pre-hook injects output_attentions=True for q_len==1. "
                "With attention_mode=avllm_literal_generate (primary): model.generate() "
                "produces a short response; the first standalone A/B/C/D token's attention "
                "towards the original multimodal prompt is extracted from the hook storage. "
                "With attention_mode=cached_decode: one generation trajectory is run while "
                "a dynamic hook captures attention only when the first answer letter is queried. "
                "With attention_mode=constrained_mcq: prefill selects the answer from "
                "prompt-final A/B/C/D logits; the answer token's cached-decode attention is "
                "captured via a single thinker forward step. "
                "Text tokens are split into instruction/scaffold, question, options, and "
                "other_text using tokenizer offsets over the rendered Qwen chat template."
            ),
            "important_note": (
                "Experiment 1 measures answer-token attention to audio, question text, answer options, "
                "and instruction/scaffold. "
                "It does not prove causality — see attention knockout (Exp 3) for that."
            ),
        },
        total_layers,
    )

    if tmp_context is not None:
        tmp_context.cleanup()


if __name__ == "__main__":
    main()
