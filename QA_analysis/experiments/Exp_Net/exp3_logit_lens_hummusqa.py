"""
Experiment 3 — Logit Lens for HumMusQA + Qwen2.5-Omni-7B
=============================================================================
Adapted from AVLLM logitlens_experiment.py for audio MCQ evaluation.

For each audio sample in HumMusQA:
  1. Run a single prefill forward pass through the Thinker (no generate needed)
  2. At each transformer layer, project audio token hidden states through lm_head
  3. Compute fraction of audio tokens whose top-1 prediction is music-related
  4. Score MCQ answer from final-position logits (same forward pass, no overhead)

Output: JSONL with one record per (sample × permutation):
  - MCQ metadata: category, difficulty, is_correct, predicted_letter
  - Per-layer: fraction_music_related, top1_counter (token frequency dict)

The logit lens reveals what audio tokens "want to say" at each layer, answering:
  "Are audio representations ever semantically meaningful?"
Connects with:
  - Exp1 (audio attention peaks at L2): what is IN those audio tokens at L2?
  - Exp2 (audio knockout weak): are audio tokens ever fully semantic by the end?
  - ExpE (UC_audio ≈ 3%): logit lens may show why audio stays "opaque"

Reference: documentation/avllm_interpretability-main/src/logitlens_experiment.py

Run (quick test):
  python -m QA_analysis.experiments.Exp_Net.exp3_logit_lens_hummusqa \\
      --model_path /nas/home/fingenito/Models/Qwen2.5-Omni-7B \\
      --output_dir /tmp/exp3_test \\
      --max_samples 3 --option_permutations 1 --fail_fast

Run (full, via launch script):
  bash QA_analysis/experiments/Exp_Net/launch_exp3_parallel.sh \\
      8 320 .../full_run_7b_3perm_parallel 3 /nas/home/fingenito/Models/Qwen2.5-Omni-7B
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
import random
import re
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from contextlib import contextmanager
from datasets import Audio as HFAudio
from datasets import load_dataset
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)


# =============================================================================
# Music-related vocabulary seed set
# Tokens are normalized: strip leading ▁ / space before matching.
# =============================================================================

MUSIC_RELATED_SEEDS = frozenset({
    # Core music concepts
    "music", "musical", "melody", "melodic", "melodies",
    "chord", "chords", "rhythm", "rhythmic", "rhythms",
    "song", "songs", "note", "notes", "beat", "beats",
    "audio", "sound", "sounds", "tone", "tones",
    "pitch", "pitches", "tempo", "key", "keys",
    # Instruments
    "bass", "treble", "drum", "drums", "drumming",
    "guitar", "guitars", "piano", "violin", "viola", "cello",
    "trumpet", "flute", "saxophone", "clarinet", "harp", "organ",
    "percussion", "strings", "brass", "wind",
    # Vocals
    "voice", "vocal", "vocals", "sing", "singing", "singer", "singers",
    "lyric", "lyrics",
    # Theory & harmony
    "harmony", "harmonies", "harmonic", "harmonious",
    "scale", "scales", "minor", "major", "sharp", "flat",
    "octave", "interval", "triad", "arpeggio",
    "tonic", "dominant", "subdominant", "cadence",
    # Genres & styles
    "jazz", "rock", "pop", "classical", "blues", "soul",
    "reggae", "country", "electronic", "folk", "rap",
    # Performance / notation
    "instrument", "instruments", "instrumental",
    "bar", "measure", "phrase", "crescendo", "forte",
    "bpm", "acoustic", "electric",
})

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


# =============================================================================
# Utilities (adapted from exp1_attention_pattern_hummusqa.py)
# =============================================================================

def normalize_difficulty(x: object) -> str:
    s = str(x or "").strip()
    return "" if not s else s[:1].upper() + s[1:].lower()


def safe_identifier(sample: Dict, idx: int) -> str:
    for key in ("identifier", "id", "sample_id", "track_id"):
        if key in sample and sample[key] is not None:
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample[key]))[:180]
    return f"sample_{idx:05d}"


def build_option_orders(identifier: str, n_orders: int, seed: int) -> List[Tuple[int, ...]]:
    if not 1 <= n_orders <= 24:
        raise ValueError(f"option_permutations must be in [1, 24], got {n_orders}")
    if n_orders == 1:
        return [(0, 1, 2, 3)]
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    orders = list(itertools.permutations(range(4)))
    rng.shuffle(orders)
    return [tuple(int(x) for x in o) for o in orders[:n_orders]]


def get_options_from_sample(
    sample: Dict, option_order: Sequence[int] = (0, 1, 2, 3)
) -> Tuple[List[str], str]:
    canonical = [str(sample["answer"]).strip()] + [
        str(sample[f"distractor_{i}"]).strip() for i in (1, 2, 3)
    ]
    displayed = [canonical[int(i)] for i in option_order]
    correct_letter = chr(65 + list(option_order).index(0))
    return displayed, correct_letter


def build_hummusqa_prompt(question: str, options: Sequence[str]) -> str:
    q = str(question).strip()
    opts = [str(o).strip() for o in options]
    answer_str = "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    return (
        "You are a music audio understanding model.\n"
        "Listen carefully to the provided audio clip. Answer the following multiple-choice\n"
        "question based on what you hear.\n"
        f"Question:\n{q}\nOptions:\n{answer_str}\n"
        "Respond with ONLY the letter of the correct option (A, B, C, or D).\n"
        "Do not include any explanation or additional text."
    )


def materialize_audio(audio_field: Dict, tmp_dir: str, identifier: str) -> Tuple[str, bool]:
    """Return (audio_path, should_delete)."""
    if isinstance(audio_field, dict):
        path = audio_field.get("path")
        if path and os.path.exists(path):
            return path, False
        raw = audio_field.get("bytes")
        if raw is not None:
            out = os.path.join(tmp_dir, f"{identifier}.wav")
            arr, sr = sf.read(io.BytesIO(raw))
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            sf.write(out, arr.astype(np.float32), sr)
            return out, True
    raise ValueError(f"Cannot materialize audio: {type(audio_field)}")



def build_conversation(prompt_text: str, audio_path: str) -> List[Dict]:
    return [{"role": "user", "content": [
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": prompt_text},
    ]}]


def get_thinker(model):
    return getattr(model, "thinker", model)


def get_module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        pass
    raise RuntimeError(f"Cannot infer device for {module.__class__.__name__}")


def get_text_backbone_device(model) -> torch.device:
    return get_module_device(get_thinker(model).model.embed_tokens)


def get_audio_tower_device(model) -> torch.device:
    return get_module_device(get_thinker(model).audio_tower)


def place_qwen_omni_inputs(inputs: Dict, *, text_device, audio_device) -> Dict:
    audio_like = {"input_features", "feature_attention_mask",
                  "audio_feature_lengths", "audio_features", "audio_attention_mask"}
    placed = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            placed[k] = v
        elif k in audio_like or "audio" in k.lower() or "feature" in k.lower():
            placed[k] = v.to(audio_device)
        else:
            placed[k] = v.to(text_device)
    return placed


def get_mcq_label_token_ids(tokenizer) -> Dict[str, int]:
    ids = {}
    for label in "ABCD":
        cands = tokenizer.encode(label, add_special_tokens=False)
        if len(cands) != 1:
            cands = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(cands) != 1:
            raise RuntimeError(f"MCQ label {label!r} is not a single token: {cands}")
        ids[label] = int(cands[0])
    return ids


def reset_qwen_omni_generation_state(model) -> None:
    for obj in [get_thinker(model), getattr(get_thinker(model), "model", None)]:
        if obj is not None and hasattr(obj, "rope_deltas"):
            try:
                obj.rope_deltas = None
            except Exception:
                pass


# =============================================================================
# Logit Lens — hook-based hidden state projection
# =============================================================================

def _token_is_music_related(token_text: str) -> bool:
    normalized = token_text.strip().lower().lstrip("▁ \t\n")
    return normalized in MUSIC_RELATED_SEEDS


@contextmanager
def capture_logit_lens(
    model,
    audio_positions: List[int],
    lm_head: torch.nn.Module,
    tokenizer,
    top_k: int = 10,
    layer_range: Optional[Tuple[int, int]] = None,
):
    """
    Register forward hooks on thinker transformer layers to project
    audio token hidden states through lm_head at each layer.

    For each layer:
      - Extracts output[0][:, audio_positions, :] (hidden states at audio positions)
      - Projects through lm_head → logits
      - Computes top-k predictions and fraction_music_related
      - Stores aggregated statistics (NOT raw tensors — memory efficient)

    Yields storage dict: layer_idx -> {fraction_music_related, top1_counter, n_audio_tokens}
    """
    storage: Dict[int, Dict] = {}
    handles = []
    thinker = get_thinker(model)
    layers = thinker.model.layers
    n_layers = len(layers)
    start = 0 if layer_range is None else max(0, layer_range[0])
    end = n_layers if layer_range is None else min(n_layers, layer_range[1])
    lm_head_device = get_module_device(lm_head)

    def _make_hook(layer_idx: int):
        def hook(module, inp, output):
            try:
                hs = output[0] if isinstance(output, tuple) else output
                if hs is None or not torch.is_tensor(hs):
                    return

                if not audio_positions:
                    storage[layer_idx] = {
                        "fraction_music_related": 0.0,
                        "top1_counter": {},
                        "n_audio_tokens": 0,
                    }
                    return

                # [n_audio, hidden_dim] — keep native model dtype (bfloat16 for 7B)
                # Do NOT cast to float32: lm_head weights are bfloat16 and
                # PyTorch raises RuntimeError on dtype mismatch for nn.Linear.
                audio_hs = hs[0, audio_positions, :].detach()

                with torch.no_grad():
                    logits = lm_head(audio_hs.to(lm_head_device))  # [n_audio, vocab_size]
                logits = logits.float()  # cast AFTER projection for stable topk

                k = min(top_k, logits.shape[-1])
                top_ids = torch.topk(logits, k=k, dim=-1).indices.cpu().tolist()
                del logits  # free GPU memory immediately

                n_music = 0
                top1_counter: Dict[str, int] = {}
                for row in top_ids:
                    tok = tokenizer.decode([row[0]])
                    top1_counter[tok] = top1_counter.get(tok, 0) + 1
                    if _token_is_music_related(tok):
                        n_music += 1

                n_audio = len(audio_positions)

                # Top-1 dominance: fraction of audio tokens predicting the
                # single most common token (measures representational convergence).
                top1_count = max(top1_counter.values()) if top1_counter else 0
                top1_dominance = top1_count / n_audio if n_audio > 0 else 0.0

                # Shannon entropy of top-1 distribution (bits).
                # High entropy = dispersed (noisy); low entropy = convergent (structured).
                import math as _math
                entropy_bits = 0.0
                for cnt in top1_counter.values():
                    p = cnt / n_audio
                    if p > 0:
                        entropy_bits -= p * _math.log2(p)

                storage[layer_idx] = {
                    "fraction_music_related": n_music / n_audio,
                    "top1_dominance": top1_dominance,
                    "entropy_bits": entropy_bits,
                    "top1_token": max(top1_counter, key=top1_counter.get) if top1_counter else "",
                    "top1_counter": top1_counter,
                    "n_audio_tokens": n_audio,
                }
            except Exception as exc:
                import traceback as _tb
                err_str = str(exc)
                # Print only the first occurrence to avoid log spam; subsequent layers
                # likely have the same error.
                if layer_idx == 0:
                    print(f"   ⚠ Hook error (L{layer_idx}): {err_str}")
                    _tb.print_exc()
                storage[layer_idx] = {
                    "fraction_music_related": float("nan"),
                    "top1_counter": {},
                    "n_audio_tokens": len(audio_positions),
                    "error": err_str,
                }
        return hook

    for i in range(start, end):
        handles.append(layers[i].register_forward_hook(_make_hook(i)))

    print(f"   Registered {len(handles)} logit-lens hooks on layers [{start}, {end}).")
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()
        print(f"   Removed {len(handles)} logit-lens hooks.")


# =============================================================================
# Per-sample processing
# =============================================================================

def process_sample(
    sample: Dict,
    sample_idx: int,
    permutation_idx: int,
    option_order: Tuple[int, ...],
    model,
    processor,
    thinker_cfg,
    mcq_label_ids: Dict[str, int],
    tmp_dir: str,
    top_k: int,
    total_layers: int,
    text_device: torch.device,
    audio_device: torch.device,
    audio_path_cache: Dict[str, str],
) -> Dict:
    """Run one (sample, permutation) evaluation. Returns the JSONL record."""
    identifier = safe_identifier(sample, sample_idx)
    question = str(sample["question"]).strip()
    options, correct_letter = get_options_from_sample(sample, option_order=option_order)
    prompt_text = build_hummusqa_prompt(question, options)
    order_label = "".join(chr(65 + int(i)) for i in option_order)

    # ── 1. Audio ──────────────────────────────────────────────────────────────
    if identifier in audio_path_cache:
        audio_path, should_delete = audio_path_cache[identifier], False
    else:
        audio_path, should_delete = materialize_audio(sample["audio"], tmp_dir, identifier)
        audio_path_cache[identifier] = audio_path

    # ── 2. Processor inputs ───────────────────────────────────────────────────
    conversation = build_conversation(prompt_text, audio_path)
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    if isinstance(text, list):
        text = text[0]

    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    raw_inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    )
    inputs = place_qwen_omni_inputs(raw_inputs, text_device=text_device, audio_device=audio_device)

    # ── 3. Audio token positions ───────────────────────────────────────────────
    audio_tok_id = thinker_cfg.audio_token_index
    input_ids_list = inputs["input_ids"][0].tolist()
    audio_positions = [i for i, tid in enumerate(input_ids_list) if tid == audio_tok_id]
    n_audio_tokens = len(audio_positions)
    if n_audio_tokens == 0:
        raise RuntimeError("No audio tokens found in input_ids.")

    # ── 4. Forward pass with logit-lens hooks ─────────────────────────────────
    thinker = get_thinker(model)
    lm_head = thinker.lm_head

    reset_qwen_omni_generation_state(model)
    t0 = time.time()

    with torch.inference_mode():
        with capture_logit_lens(
            model,
            audio_positions=audio_positions,
            lm_head=lm_head,
            tokenizer=processor.tokenizer,
            top_k=top_k,
            layer_range=(0, total_layers),
        ) as logit_lens_storage:
            outputs = thinker(
                **inputs,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )

    elapsed = time.time() - t0

    # ── 5. MCQ scoring from final-position logits ─────────────────────────────
    logits_last = outputs.logits[0, -1].detach().float().cpu()
    label_logits = {lbl: float(logits_last[mcq_label_ids[lbl]].item()) for lbl in "ABCD"}
    predicted_letter = max(label_logits, key=label_logits.get)
    is_correct = predicted_letter == correct_letter
    predicted_source_idx = int(option_order[ord(predicted_letter) - 65])

    # ── 6. Cleanup ────────────────────────────────────────────────────────────
    # NOTE: do NOT delete the temp WAV here — it may be needed by subsequent
    # permutations of the same sample (audio_path_cache stores the path).
    # The TemporaryDirectory created in main() calls cleanup() at the end,
    # which removes all temp files at once.

    # ── 7. Build record ───────────────────────────────────────────────────────
    per_layer = {}
    for li in range(total_layers):
        stats = logit_lens_storage.get(li, {})
        per_layer[str(li)] = {
            "fraction_music_related": float(stats.get("fraction_music_related", float("nan"))),
            "top1_dominance": float(stats.get("top1_dominance", float("nan"))),
            "entropy_bits": float(stats.get("entropy_bits", float("nan"))),
            "top1_token": str(stats.get("top1_token", "")),
            "top1_counter": dict(stats.get("top1_counter", {})),
        }

    return {
        "sample_idx": sample_idx,
        "evaluation_idx": None,   # filled by caller
        "identifier": identifier,
        "permutation_idx": permutation_idx,
        "option_order": order_label,
        "question": question[:200],
        "correct_letter": correct_letter,
        "predicted_letter": predicted_letter,
        "predicted_option_source_idx": predicted_source_idx,
        "is_correct": int(is_correct),
        "main_category": str(sample.get("main_category", "")),
        "difficulty": normalize_difficulty(sample.get("difficulty", "")),
        "n_audio_tokens": n_audio_tokens,
        "n_input_tokens": len(input_ids_list),
        "label_logits": {k: round(v, 4) for k, v in label_logits.items()},
        "elapsed_sec": round(elapsed, 3),
        "per_layer": per_layer,
    }


# =============================================================================
# Output helpers
# =============================================================================

def append_jsonl(record: Dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_metadata(
    output_dir: str,
    args,
    records: List[Dict],
    errors: List[Dict],
    total_layers: int,
    n_total_evals: int,
) -> None:
    n_correct = sum(r.get("is_correct", 0) for r in records)
    n = len(records)
    meta = {
        "status": "complete",
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "max_samples": args.max_samples,
        "sample_start": args.sample_start,
        "option_permutations": args.option_permutations,
        "option_seed": args.option_seed,
        "top_k_tokens": args.top_k_tokens,
        "total_layers": total_layers,
        "music_related_seeds_n": len(MUSIC_RELATED_SEEDS),
        "n_total_evaluations": n_total_evals,
        "n_processed": n,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n, 6) if n else None,
        "n_errors": len(errors),
        "errors": errors,
        "saved_at": datetime.now().isoformat(),
        "files": {
            "logit_lens_results": os.path.join(output_dir, "logit_lens_results.jsonl"),
        },
    }
    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"   ✅ {path}")


# =============================================================================
# Argument parsing & main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 3 — Logit Lens | HumMusQA + Qwen2.5-Omni"
    )
    p.add_argument("--model_path", default="/nas/home/fingenito/Models/Qwen2.5-Omni-7B")
    p.add_argument("--output_dir", default="results/exp3_logit_lens")
    p.add_argument("--dataset_name", default="mtg-upf/HumMusQA")
    p.add_argument("--dataset_split", default="test")
    p.add_argument("--dataset_path", default=None)
    p.add_argument("--tmp_dir", default=None)
    p.add_argument("--sample_start", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--option_permutations", type=int, default=1)
    p.add_argument("--option_seed", type=int, default=42)
    p.add_argument("--top_k_tokens", type=int, default=10,
                   help="Top-k token predictions stored per audio position per layer.")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--checkpoint_every", type=int, default=20)
    p.add_argument("--fail_fast", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tmp_context = None
    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True)
        tmp_dir = args.tmp_dir
    else:
        tmp_context = tempfile.TemporaryDirectory(prefix="hummusqa_exp3_")
        tmp_dir = tmp_context.name

    print("\n" + "=" * 78)
    print("Experiment 3 — Logit Lens | HumMusQA audio MCQ")
    print(f"Model     : {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Top-k     : {args.top_k_tokens}")
    print("=" * 78 + "\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model (sdpa, disable_talker)...")
    device_map_arg = args.device_map
    if device_map_arg != "auto" and device_map_arg.startswith("cuda"):
        device_map_arg = {"": device_map_arg}

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        attn_implementation="sdpa",
        device_map=device_map_arg,
    )
    model.disable_talker()
    model.eval()

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    thinker = get_thinker(model)
    thinker_cfg = thinker.config
    total_layers = len(thinker.model.layers)
    text_device = get_text_backbone_device(model)
    audio_device = get_audio_tower_device(model)
    lm_head_device = get_module_device(thinker.lm_head)
    mcq_label_ids = get_mcq_label_token_ids(processor.tokenizer)

    print(f"Total layers    : {total_layers}")
    print(f"audio_token_idx : {thinker_cfg.audio_token_index}")
    print(f"text device     : {text_device}")
    print(f"audio device    : {audio_device}")
    print(f"lm_head device  : {lm_head_device}")
    print()

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading HumMusQA...")
    if args.dataset_path:
        dataset = load_dataset(
            "parquet",
            data_files={args.dataset_split: args.dataset_path},
            split=args.dataset_split,
        )
    else:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.cast_column("audio", HFAudio(decode=False))

    samples = list(dataset)
    if args.sample_start:
        samples = samples[args.sample_start:]
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    evaluation_items = []
    for local_idx, sample in enumerate(samples):
        dataset_idx = args.sample_start + local_idx
        ident = safe_identifier(sample, dataset_idx)
        for perm_idx, option_order in enumerate(
            build_option_orders(ident, args.option_permutations, args.option_seed)
        ):
            evaluation_items.append((dataset_idx, sample, perm_idx, option_order))

    print(
        f"Questions : {len(samples)} (offset={args.sample_start}) | "
        f"permutations : {args.option_permutations} | "
        f"total evals : {len(evaluation_items)}"
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    jsonl_path = os.path.join(args.output_dir, "logit_lens_results.jsonl")
    records: List[Dict] = []
    errors: List[Dict] = []
    audio_path_cache: Dict[str, str] = {}

    for eval_idx, (sample_idx, sample, perm_idx, option_order) in enumerate(evaluation_items):
        ident = safe_identifier(sample, sample_idx)
        order_label = "".join(chr(65 + int(i)) for i in option_order)
        print(
            f"\n[{eval_idx + 1:04d}/{len(evaluation_items):04d}] "
            f"{ident}  perm={perm_idx + 1}/{args.option_permutations}  order={order_label}"
        )

        try:
            record = process_sample(
                sample=sample,
                sample_idx=sample_idx,
                permutation_idx=perm_idx,
                option_order=option_order,
                model=model,
                processor=processor,
                thinker_cfg=thinker_cfg,
                mcq_label_ids=mcq_label_ids,
                tmp_dir=tmp_dir,
                top_k=args.top_k_tokens,
                total_layers=total_layers,
                text_device=text_device,
                audio_device=audio_device,
                audio_path_cache=audio_path_cache,
            )
            record["evaluation_idx"] = eval_idx
            records.append(record)
            append_jsonl(record, jsonl_path)

            ok = bool(record["is_correct"])
            print(
                f"   predicted={record['predicted_letter']}  correct={record['correct_letter']}  "
                f"{'✓' if ok else '✗'}  n_audio={record['n_audio_tokens']}  "
                f"elapsed={record['elapsed_sec']:.1f}s"
            )
            # Quick sanity: entropy + dominance at L0, L_mid, L_last
            for li in [0, total_layers // 2, total_layers - 1]:
                stats = record["per_layer"].get(str(li), {})
                ent   = stats.get("entropy_bits", float("nan"))
                dom   = stats.get("top1_dominance", float("nan"))
                tok   = stats.get("top1_token", "?")
                print(f"   L{li:2d}  H={ent:.2f}bits  dom={dom:.3f}  top1={repr(tok)}")

        except Exception as exc:
            import traceback as tb
            err_msg = tb.format_exc()
            print(f"   ❌ {exc}")
            errors.append({
                "evaluation_idx": eval_idx,
                "sample_idx": sample_idx,
                "identifier": ident,
                "permutation_idx": perm_idx,
                "error": str(exc),
                "traceback": err_msg,
            })
            if args.fail_fast:
                raise

        if (eval_idx + 1) % args.checkpoint_every == 0:
            save_metadata(args.output_dir, args, records, errors, total_layers, len(evaluation_items))
            print(f"   📌 Checkpoint @ eval {eval_idx + 1}")

    # ── Final save ─────────────────────────────────────────────────────────────
    save_metadata(args.output_dir, args, records, errors, total_layers, len(evaluation_items))

    n_correct = sum(r.get("is_correct", 0) for r in records)
    print(f"\n{'=' * 78}")
    if records:
        print(
            f"Done. {len(records)} evals | "
            f"accuracy {n_correct}/{len(records)} = {n_correct/len(records):.4f}"
        )
    else:
        print("Done. 0 evals.")
    print(f"Errors : {len(errors)}")
    print(f"Results: {jsonl_path}")
    print(f"{'=' * 78}\n")

    if tmp_context is not None:
        tmp_context.cleanup()


if __name__ == "__main__":
    main()
