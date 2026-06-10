#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# launch_exp2_parallel.sh  —  Attention Knockout Exp 2 (decision_logits mode)
#
# Faithful adaptation of the AVLLM attention knockout experiment to MCQ tasks.
#
# AVLLM original: blocks "generated → audio" during open-ended text generation
#   and measures how often the generated description changes.
#
# Our MCQ adaptation: uses decision_logits mode.
#   - For MCQ, the answer (A/B/C/D) is generated at the END of the prefill
#     (not during autoregressive decode), so "generated → audio" cannot work.
#   - Instead: we run one forward pass, block "decision → audio" in a layer
#     window, and directly measure delta_prob(correct answer).
#   - This tests the same causal question: "does audio matter in layer L-R?"
#
# Attention implementation:
#   - Default: sdpa  — works on both 3B and 7B; the hook handles the sdpa
#     None-mask case by building a causal mask from scratch (cache_position).
#   - Use eager only when sdpa causes issues AND the model is 3B (the 7B model
#     has a TMRoPE KV-cache bug that crashes with eager + audio inputs).
#
# Window size guidance:
#   - 3B (36 layers):  window_size=9  stride=9  → 4 windows
#   - 7B (80 layers):  window_size=20 stride=20 → 4 windows  (comparable)
#                   or window_size=9  stride=9  → 9 windows  (more granular)
#
# Usage:
#   bash QA_analysis/experiments/Exp_Net/launch_exp2_parallel.sh \
#     /path/to/model \
#     /path/to/output_shards_root \
#     1,2,3,4,5,6,7   <- GPU IDs (skip busy GPUs)
#     320              <- total samples (HumMusQA test = 320)
#     46               <- samples per shard  (320 / 7 GPUs = ~46)
#     3                <- option_permutations (default 3; eliminates position bias)
#     sdpa             <- attn_implementation: sdpa (default, 3B+7B) or eager (3B only)
#     9                <- window_size  (default 9; use 20 for 7B with 4 windows)
#     9                <- window_stride (default 9; usually == window_size)
#
# After completion:
#   python -m QA_analysis.experiments.Exp_Net.merge_exp2_shards \
#       --shards_root <shards_root>
#   python -m QA_analysis.experiments.Exp_Net.plot_exp2_knockout \
#       --results_dir <merged_dir>
# ==============================================================================

MODEL_PATH="${1:?model_path required}"
SHARDS_ROOT="${2:?shards_root required}"
GPU_IDS_CSV="${3:?comma-separated gpu ids required, e.g. 1,2,3,4,5,6,7}"
TOTAL_SAMPLES="${4:?total_samples required}"
SAMPLES_PER_SHARD="${5:?samples_per_shard required}"
OPTION_PERMUTATIONS="${6:-3}"      # default 3 — removes position bias
ATTN_IMPL="${7:-sdpa}"             # default sdpa — works on 3B and 7B
WINDOW_SIZE="${8:-9}"              # default 9  (use 20 for 7B to get 4 windows)
WINDOW_STRIDE="${9:-${WINDOW_SIZE}}" # default == window_size (non-overlapping)

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
mkdir -p "${SHARDS_ROOT}"

echo "======================================================================"
echo "Exp 2 — Attention Knockout (decision_logits mode)"
echo "  model_path        : ${MODEL_PATH}"
echo "  shards_root       : ${SHARDS_ROOT}"
echo "  gpu_ids           : ${GPU_IDS_CSV}"
echo "  total_samples     : ${TOTAL_SAMPLES}"
echo "  samples_per_shard : ${SAMPLES_PER_SHARD}"
echo "  option_perms      : ${OPTION_PERMUTATIONS}  (>1 removes correct-answer position bias)"
echo "  attn_impl         : ${ATTN_IMPL}  (sdpa=default/7B-safe; eager=3B-only)"
echo "  window_size       : ${WINDOW_SIZE}  stride=${WINDOW_STRIDE}"
echo "  mode              : decision_logits  (delta_prob_correct per window)"
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
    python -m QA_analysis.experiments.Exp_Net.exp2_attention_knockout_hummusqa \
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
  echo "Next — merge:"
  echo "  python -m QA_analysis.experiments.Exp_Net.merge_exp2_shards \\"
  echo "      --shards_root '${SHARDS_ROOT}'"
  echo ""
  echo "Then plot:"
  echo "  python -m QA_analysis.experiments.Exp_Net.plot_exp2_knockout \\"
  echo "      --results_dir '${SHARDS_ROOT}_merged' \\"
  echo "      --output_dir  '${SHARDS_ROOT}_merged/plots'"
else
  echo "⚠️  One or more shards failed."
  exit 1
fi
