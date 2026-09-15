"""
Diagnostica: perche' il knockout AF3 non ha effetto oltre la finestra L0-2?
=============================================================================
NON e' un esperimento -- e' uno strumento usa-e-getta per capire perche'
knockout_logit == baseline_logit bit-per-bit per ogni finestra oltre L0-2
(verificato su full_run_af3_decision_logits_3perm_w2_merged: 960/960 righe
identiche a L14-16, contro 4/960 a L0-2).

Riusa (import diretto, nessuna duplicazione) le funzioni reali di
exp2_attention_knockout_af3.py: stesso caricamento modello, stesso
prepare_inputs, stessa DecisionAttentionKnockoutHook. Instrumenta l'hook con
delle stampe diagnostiche per un solo campione, confrontando una finestra
che sappiamo funzionare (L0-2) con una che sappiamo essere rotta (L14-16).

Uso:
    python -m QA_analysis.paper_af3.AttnKnock.debug_knockout_hook_af3
"""

import os
import sys

_T0 = __import__("time").time()


def _log(msg: str) -> None:
    print(f"[T+{__import__('time').time() - _T0:6.1f}s] {msg}", flush=True)


sys.stdout.reconfigure(line_buffering=True)
_log("Script avviato.")

from QA_analysis.utils.gpu_utils import get_available_gpus_with_memory

MIN_FREE_GB_RUNNER = 21.0
_free_gpu_ids, _ = get_available_gpus_with_memory(min_free_memory_gb=MIN_FREE_GB_RUNNER)
if not _free_gpu_ids:
    raise RuntimeError(f"Nessuna GPU con >= {MIN_FREE_GB_RUNNER} GB liberi.")
os.environ["CUDA_VISIBLE_DEVICES"] = str(_free_gpu_ids[0])
_log(f"GPU scelta: {_free_gpu_ids[0]}")

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
_log("torch importato.")

from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from datasets import Audio as HFAudio, load_dataset

from QA_analysis.paper_af3.AttnKnock.exp2_attention_knockout_af3 import (
    get_thinker_layers,
    get_thinker_config,
    get_text_backbone_device,
    get_audio_tower_device,
    place_af3_inputs,
    prepare_inputs,
    create_token_type_mapping,
    get_mcq_label_token_ids,
    score_mcq_forward,
    DecisionAttentionKnockoutHook,
    TOKEN_TYPE_MAP,
)
_log("import del modulo AF3 knockout completato.")

MODEL_PATH = "/nas/home/fingenito/Models/audio-flamingo-3-hf"


def main():
    _log("Carico modello (bf16, device_map=None, low_cpu_mem_usage=True)...")
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=None,
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).to("cuda:0").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    _log("Modello caricato.")

    thinker_cfg = get_thinker_config(model)
    text_device = get_text_backbone_device(model)
    audio_device = get_audio_tower_device(model)
    label_ids = get_mcq_label_token_ids(processor.tokenizer, "paren")

    dataset = load_dataset("mtg-upf/HumMusQA", split="test")
    dataset = dataset.cast_column("audio", HFAudio(decode=False))
    sample = dataset[0]

    tmp_dir = "/tmp/af3_debug_knockout"
    os.makedirs(tmp_dir, exist_ok=True)
    inputs, correct_letter, audio_path, should_delete, rendered_text, text_spans = prepare_inputs(
        sample, 0, processor, tmp_dir, option_order=(0, 1, 2, 3),
    )
    inputs = place_af3_inputs(inputs, device=text_device, dtype=model.dtype)
    original_input_len = int(inputs["input_ids"].shape[1])
    token_types = create_token_type_mapping(
        inputs["input_ids"], thinker_cfg, tokenizer=processor.tokenizer,
        rendered_text=rendered_text, text_spans=text_spans,
    )
    _log(f"Input pronto: input_len={original_input_len}, correct_letter={correct_letter}")

    baseline = score_mcq_forward(model, inputs, label_ids)
    _log(f"Baseline: pred={baseline['pred']} logits={baseline['logits']}")

    layers = get_thinker_layers(model)

    def _run_window_with_debug(layer_start, layer_end, tag):
        _log(f"=== Finestra {tag}: L{layer_start}-{layer_end} ===")
        rules = [("decision", "audio", layer_start, layer_end)]
        numeric_rules = torch.tensor(
            [(TOKEN_TYPE_MAP["decision"], TOKEN_TYPE_MAP["audio"])],
            dtype=torch.long, device="cuda:0",
        )
        numeric_token_types = torch.tensor(
            [TOKEN_TYPE_MAP.get(t, TOKEN_TYPE_MAP["other"]) for t in token_types],
            dtype=torch.long, device="cuda:0",
        )

        call_count = {"n": 0}
        orig_call = DecisionAttentionKnockoutHook.__call__

        def _instrumented_call(self, module, args, kwargs):
            call_count["n"] += 1
            hs = kwargs.get("hidden_states")
            if hs is None and args:
                hs = args[0]
            am_before = kwargs.get("attention_mask")
            result = orig_call(self, module, args, kwargs)
            # result e' (args, kwargs) ritornato dall'hook originale
            _, new_kwargs = result
            am_after = new_kwargs.get("attention_mask")
            print(
                f"    [hook call #{call_count['n']}] hidden_states.shape={tuple(hs.shape) if hs is not None else None} "
                f"| attention_mask PRIMA={'None' if am_before is None else tuple(am_before.shape)} "
                f"| attention_mask DOPO={'None' if am_after is None else tuple(am_after.shape)} "
                f"| kwargs_keys={sorted(kwargs.keys())}",
                flush=True,
            )
            if am_after is not None:
                n_blocked = int((am_after < -1e30).sum().item())
                print(f"    [hook call #{call_count['n']}] posizioni bloccate (-inf) nella maschera: {n_blocked}", flush=True)
            return result

        DecisionAttentionKnockoutHook.__call__ = _instrumented_call
        try:
            handles = []
            for layer_idx, layer in enumerate(layers):
                if not (layer_start <= layer_idx < layer_end):
                    continue
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
                _log(f"  Hook registrato su layer {layer_idx}")

            ko = score_mcq_forward(model, inputs, label_ids)
            identical = ko["logits"] == baseline["logits"]
            _log(f"  Risultato: pred={ko['pred']} logits={ko['logits']} | identico al baseline: {identical}")
        finally:
            for h in handles:
                h.remove()
            DecisionAttentionKnockoutHook.__call__ = orig_call

    _run_window_with_debug(0, 2, "NOTA FUNZIONANTE")
    _run_window_with_debug(14, 16, "NOTA ROTTA")


if __name__ == "__main__":
    main()
