#!/bin/bash
# Evaluate CPR Curriculum checkpoints on GPUs 4-7
# Run inside tmux: tmux new-session -d -s eval_cpr_curriculum bash eval_cpr_curriculum_checkpoints.sh

set -e

source activate torch26
export HF_HOME=/gpudata3/hf_cache
export PYTHONPATH="/gpudata3/Wayner/VLM-R1/src/open-r1-multimodal/src:/gpudata3/Wayner/VLM-R1/mllm_evaluator:${PYTHONPATH}"

PYTHON="/scratch/miniconda3/envs/torch26/bin/python"
PROJECT="/gpudata3/Wayner/VLM-R1"
TRAIN_DIR="${PROJECT}/output/grpo-cpr-curriculum-20260221_163422"
RESULTS_DIR="${PROJECT}/GRPO_analysis/cpr_curriculum_results"
GPU_IDS="4,5,6,7"

echo "=============================================="
echo "CPR Curriculum Checkpoint Evaluation"
echo "Started: $(date)"
echo "=============================================="

# ── Step 1: Compute metrics on checkpoints that already have predictions ──
echo ""
echo ">>> [1/5] Computing metrics for checkpoint-200 (predictions exist)"
$PYTHON ${PROJECT}/compute_metrics.py \
    ${RESULTS_DIR}/checkpoint-200/predictions \
    --output-dir ${RESULTS_DIR}/checkpoint-200 \
    --num-gpus 1
echo "✓ checkpoint-200 metrics done at $(date)"

echo ""
echo ">>> [2/5] Computing metrics for checkpoint-400 (predictions exist)"
$PYTHON ${PROJECT}/compute_metrics.py \
    ${RESULTS_DIR}/checkpoint-400/predictions \
    --output-dir ${RESULTS_DIR}/checkpoint-400 \
    --num-gpus 1
echo "✓ checkpoint-400 metrics done at $(date)"

# ── Step 2: Run inference + metrics on key checkpoints ──
for CKPT in 500 1000 1500; do
    CKPT_DIR="${TRAIN_DIR}/checkpoint-${CKPT}"
    PRED_DIR="${RESULTS_DIR}/checkpoint-${CKPT}/predictions"

    echo ""
    echo "=============================================="
    echo ">>> Checkpoint-${CKPT}: Inference + Metrics"
    echo "=============================================="

    # Run multi-GPU inference
    echo "Running inference on GPUs ${GPU_IDS}..."
    bash ${PROJECT}/inference/run_multi_gpu_eval.sh \
        "${CKPT_DIR}" \
        "${PRED_DIR}" \
        "${GPU_IDS}"
    echo "✓ checkpoint-${CKPT} inference done at $(date)"

    # Compute metrics
    echo "Computing metrics..."
    $PYTHON ${PROJECT}/compute_metrics.py \
        "${PRED_DIR}" \
        --output-dir "${RESULTS_DIR}/checkpoint-${CKPT}" \
        --num-gpus 1
    echo "✓ checkpoint-${CKPT} metrics done at $(date)"
done

echo ""
echo "=============================================="
echo "ALL EVALUATIONS COMPLETE at $(date)"
echo "=============================================="
echo ""
echo "Results in: ${RESULTS_DIR}"
for CKPT in 200 400 500 1000 1500; do
    SUMMARY="${RESULTS_DIR}/checkpoint-${CKPT}/no_judge_summary.json"
    if [ -f "$SUMMARY" ]; then
        echo "--- Checkpoint-${CKPT} ---"
        cat "$SUMMARY"
        echo ""
    fi
done
