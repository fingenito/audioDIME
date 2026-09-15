#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# launch_exp2_parallel_af3_text_only.sh — Attention Knockout Exp 2
#                                          (decision_logits), AF3, text_only
#
# Copia adattata di launch_exp2_parallel_af3.sh (caso complete, gia' validato
# con run completo su 320 campioni), STESSO IDENTICO esperimento applicato
# alla variante text_only. Vedi exp2_attention_knockout_af3_text_only.py
# per l'unica differenza funzionale (audio reale sostituito da white noise).
#
# paper_af3/AttnKnock/ e' un package Python vero (__init__.py) -- stessa
# identica invocazione `python -m` a percorso puntato, lanciata dalla
# project root.
#
# Usage:
#   bash QA_analysis/paper_af3/AttnKnock/launch_exp2_parallel_af3_text_only.sh \
#     /path/to/audio-flamingo-3-hf \
#     /path/to/output_shards_root \
#     auto            <- GPU IDs: "auto" rileva tutte le GPU libere (>=21GB),
#                        oppure lista esplicita "0,1,2,3,4,5"
#     320             <- total samples (HumMusQA test = 320)
#     auto            <- samples per shard: "auto" = TOTAL_SAMPLES / n_gpu (arrotondato per eccesso),
#                        oppure un numero esplicito
#     3               <- option_permutations (default 3; elimina il bias di posizione)
#     sdpa            <- attn_implementation: sdpa (richiesto per il knockout)
#     2               <- window_size  (default 2, come il run "complete" AF3)
#     2               <- window_stride (default == window_size)
#
# After completion:
#   python -m QA_analysis.paper_af3.AttnKnock.merge_exp2_shards \
#       --shards_root <shards_root>
# ==============================================================================

MODEL_PATH="${1:?model_path required}"
SHARDS_ROOT="${2:?shards_root required}"
GPU_IDS_CSV="${3:?comma-separated gpu ids required, e.g. 0,1,2,3,4,5 (oppure 'auto')}"
TOTAL_SAMPLES="${4:?total_samples required}"
SAMPLES_PER_SHARD_ARG="${5:?samples_per_shard required (oppure 'auto')}"
OPTION_PERMUTATIONS="${6:-3}"      # default 3 — rimuove il bias di posizione
ATTN_IMPL="${7:-sdpa}"             # default sdpa — richiesto dal meccanismo di knockout
WINDOW_SIZE="${8:-2}"              # default 2 — stesso valore usato per il run "complete" AF3
WINDOW_STRIDE="${9:-${WINDOW_SIZE}}" # default == window_size (non-overlapping)

# python -m richiede di essere lanciati dalla project root (stessa convenzione
# di tutti gli altri script del progetto).
PROJECT_ROOT="/nas/home/fingenito/Thesis_project"
cd "${PROJECT_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Rilevamento automatico GPU libere -- stessa soglia (21GB) usata da
# get_available_gpus_with_memory() in tutti gli script Python del progetto.
if [ "${GPU_IDS_CSV}" = "auto" ]; then
  echo "Rilevo GPU libere (>=21GB) automaticamente..."
  GPU_IDS_CSV="$(bash "${SCRIPT_DIR}/find_free_gpus_af3.sh" 21 | paste -sd, -)"
  if [ -z "${GPU_IDS_CSV}" ]; then
    echo "ERRORE: nessuna GPU con >=21GB liberi trovata in questo momento."
    exit 1
  fi
  echo "GPU libere rilevate: ${GPU_IDS_CSV}"
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"

if [ "${SAMPLES_PER_SHARD_ARG}" = "auto" ]; then
  SAMPLES_PER_SHARD=$(( (TOTAL_SAMPLES + ${#GPU_IDS[@]} - 1) / ${#GPU_IDS[@]} ))  # ceil division
  echo "samples_per_shard='auto' -> ${SAMPLES_PER_SHARD} (${TOTAL_SAMPLES} campioni / ${#GPU_IDS[@]} GPU)"
else
  SAMPLES_PER_SHARD="${SAMPLES_PER_SHARD_ARG}"
fi

mkdir -p "${SHARDS_ROOT}"

echo "======================================================================"
echo "Exp 2 — Attention Knockout (decision_logits mode) — Audio Flamingo 3, text_only"
echo "  model_path        : ${MODEL_PATH}"
echo "  shards_root       : ${SHARDS_ROOT}"
echo "  gpu_ids           : ${GPU_IDS_CSV}"
echo "  total_samples     : ${TOTAL_SAMPLES}"
echo "  samples_per_shard : ${SAMPLES_PER_SHARD}"
echo "  option_perms      : ${OPTION_PERMUTATIONS}  (>1 rimuove il bias di posizione della risposta corretta)"
echo "  attn_impl         : ${ATTN_IMPL}  (sdpa richiesto per il knockout)"
echo "  window_size       : ${WINDOW_SIZE}  stride=${WINDOW_STRIDE}"
echo "  mode              : decision_logits  (delta_prob_correct per finestra)"
echo "======================================================================"

pids=()
shard_idx=0
for ((start=0; start<TOTAL_SAMPLES; start+=SAMPLES_PER_SHARD)); do
  gpu="${GPU_IDS[$((shard_idx % ${#GPU_IDS[@]}))]}"
  out_dir="${SHARDS_ROOT}/shard_${shard_idx}"
  mkdir -p "${out_dir}"

  echo "Launching shard_${shard_idx}: samples ${start}..$((start + SAMPLES_PER_SHARD - 1)), GPU ${gpu}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    python -m QA_analysis.paper_af3.AttnKnock.exp2_attention_knockout_af3_text_only \
      --model_path          "${MODEL_PATH}" \
      --output_dir          "${out_dir}" \
      --knockout_mode       decision_logits \
      --attn_implementation "${ATTN_IMPL}" \
      --device_map          "cuda:0" \
      --sample_start        "${start}" \
      --max_samples         "${SAMPLES_PER_SHARD}" \
      --option_permutations "${OPTION_PERMUTATIONS}" \
      --components          audio question options instruction all_text_to_audio \
      --window_size         "${WINDOW_SIZE}" \
      --window_stride       "${WINDOW_STRIDE}" \
      --checkpoint_every    20 \
      > "${out_dir}/run.log" 2>&1
  ) &

  pids+=($!)
  echo "  → PID ${pids[-1]}, log: ${out_dir}/run.log"
  shard_idx=$((shard_idx + 1))
done

echo ""
echo "All ${#pids[@]} shards launched. Waiting..."
all_ok=true
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  if wait "$pid"; then
    echo "✅  shard_${idx}  (PID $pid)  OK"
  else
    echo "❌  shard_${idx}  (PID $pid)  FAILED — check ${SHARDS_ROOT}/shard_${idx}/run.log"
    all_ok=false
  fi
done

echo ""
if $all_ok; then
  echo "All shards completed successfully."
  echo ""
  echo "Next — merge (merge_exp2_shards.py è model-agnostico, riusato invariato):"
  echo "  python -m QA_analysis.paper_af3.AttnKnock.merge_exp2_shards \\"
  echo "      --shards_root '${SHARDS_ROOT}'"
else
  echo "⚠️  One or more shards failed."
  exit 1
fi
