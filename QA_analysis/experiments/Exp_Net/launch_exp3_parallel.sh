#!/usr/bin/env bash
# ==============================================================================
# launch_exp3_parallel.sh  —  Run Exp 3 (Logit Lens) in parallel across GPUs.
#
# Each process gets its own GPU (CUDA_VISIBLE_DEVICES=N + --device_map cuda:0)
# and handles a contiguous slice of HumMusQA test set.
# Outputs go to separate shard directories; run merge_exp3_shards.py afterwards.
#
# Usage:
#   cd /nas/home/fingenito/Thesis_project
#   bash QA_analysis/experiments/Exp_Net/launch_exp3_parallel.sh \
#        [N_GPUS]   [N_SAMPLES]   [OUT_ROOT]   [OPTION_PERMUTATIONS]   [MODEL_PATH]   [GPU_START]
#
# Defaults:
#   N_GPUS               = 8
#   N_SAMPLES            = 320
#   OUT_ROOT             = .../exp3_logit_lens/full_run_7b_3perm_parallel
#   OPTION_PERMUTATIONS  = 3
#   MODEL_PATH           = /nas/home/fingenito/Models/Qwen2.5-Omni-7B
#   GPU_START            = 0
#
# Examples:
#   # Full run 7B, 8 GPUs, 3 permutations (recommended)
#   bash launch_exp3_parallel.sh
#
#   # GPUs 1-7 (GPU 0 occupied by Exp 2)
#   bash launch_exp3_parallel.sh 7 320 .../full_run_7b_3perm_parallel 3 \
#       /nas/home/fingenito/Models/Qwen2.5-Omni-7B 1
#
#   # Quick test: 4 GPUs, 8 samples, 1 permutation
#   bash launch_exp3_parallel.sh 4 8 /tmp/exp3_test 1
# ==============================================================================
set -euo pipefail

N_GPUS="${1:-8}"
N_SAMPLES="${2:-320}"
OUT_ROOT="${3:-/nas/home/fingenito/Thesis_project/QA_analysis/Results_QA/experiments/exp_Net/exp3_logit_lens/full_run_7b_3perm_parallel}"
OPTION_PERMUTATIONS="${4:-3}"
MODEL="${5:-/nas/home/fingenito/Models/Qwen2.5-Omni-7B}"
GPU_START="${6:-0}"

SCRIPT="QA_analysis.experiments.Exp_Net.exp3_logit_lens_hummusqa"
CHUNK=$(( (N_SAMPLES + N_GPUS - 1) / N_GPUS ))

mkdir -p "$OUT_ROOT"

echo "======================================================================"
echo "Exp 3 — Logit Lens parallel run"
echo "  N_GPUS              = $N_GPUS"
echo "  GPU_START           = $GPU_START  (using GPUs $GPU_START .. $((GPU_START + N_GPUS - 1)))"
echo "  N_SAMPLES           = $N_SAMPLES  (chunk = $CHUNK per GPU)"
echo "  OPTION_PERMUTATIONS = $OPTION_PERMUTATIONS"
echo "  OUT_ROOT            = $OUT_ROOT"
echo "  MODEL               = $MODEL"
echo "======================================================================"

pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
    START=$(( i * CHUNK ))
    if [ "$START" -ge "$N_SAMPLES" ]; then
        echo "[GPU $i] No samples left — skipping."
        continue
    fi

    SHARD_DIR="${OUT_ROOT}/shard_${i}"
    mkdir -p "$SHARD_DIR"
    LOG="${SHARD_DIR}/run.log"

    echo "[GPU $i]  shard_${i}: samples ${START} .. $((START + CHUNK - 1))  →  $SHARD_DIR"

    CUDA_VISIBLE_DEVICES=$(( i + GPU_START )) \
    python -m "$SCRIPT" \
        --model_path          "$MODEL" \
        --output_dir          "$SHARD_DIR" \
        --device_map          "cuda:0" \
        --sample_start        "$START" \
        --max_samples         "$CHUNK" \
        --option_permutations "$OPTION_PERMUTATIONS" \
        --option_seed         42 \
        --top_k_tokens        10 \
        --checkpoint_every    20 \
        --fail_fast \
        > "$LOG" 2>&1 &

    pids+=($!)
    echo "           → PID ${pids[-1]},  log: $LOG"
done

echo ""
echo "All ${#pids[@]} shards launched. Waiting for completion..."
echo "(tail -f <shard_dir>/run.log  to follow individual progress)"
echo ""

all_ok=true
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    if wait "$pid"; then
        echo "✅  shard_${idx}  (PID $pid)  finished OK"
    else
        echo "❌  shard_${idx}  (PID $pid)  FAILED  — check ${OUT_ROOT}/shard_${idx}/run.log"
        all_ok=false
    fi
done

echo ""
if $all_ok; then
    echo "All shards completed successfully."
    echo ""
    echo "Next step — merge results:"
    echo "  python -m QA_analysis.experiments.Exp_Net.merge_exp3_shards \\"
    echo "      --shards_root '$OUT_ROOT'"
    echo ""
    echo "Then plot:"
    echo "  python -m QA_analysis.experiments.Exp_Net.plot_exp3_logit_lens \\"
    echo "      --results_dir '${OUT_ROOT}_merged' \\"
    echo "      --output_dir  '${OUT_ROOT}_merged/plots'"
else
    echo "⚠️  One or more shards failed. Fix errors before merging."
    exit 1
fi
