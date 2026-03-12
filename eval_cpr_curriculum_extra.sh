#!/bin/bash
# Evaluate EXTRA CPR Curriculum checkpoints to match CPR original for fair comparison
# CPR original has: 400, 500, 700, 1000, 1400, 2000, 2400, 2800
# First script covers: 200, 400, 500, 1000, 1500
# This script covers the MISSING ones: 700, 1400
# (2000, 2400, 2800 not yet available in curriculum training)

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
echo "CPR Curriculum EXTRA Checkpoints (fair comparison)"
echo "Started: $(date)"
echo "Waiting for GPUs 4-7 to be free..."
echo "=============================================="

# Wait for GPUs to be free (first eval script to finish)
while true; do
    GPU4_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4)
    if [ "$GPU4_MEM" -lt 2000 ]; then
        echo "GPUs free at $(date). Starting evaluations."
        break
    fi
    echo "  GPUs still busy (GPU4: ${GPU4_MEM} MiB). Waiting 60s..."
    sleep 60
done

for CKPT in 700 1400; do
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
echo "EXTRA EVALUATIONS COMPLETE at $(date)"
echo "=============================================="
echo ""
echo "=== FULL COMPARISON TABLE ==="
echo "CPR Curriculum checkpoints matching CPR original:"
for CKPT in 400 500 700 1000 1400 1500; do
    SUMMARY="${RESULTS_DIR}/checkpoint-${CKPT}/predictions/no_judge_summary.json"
    if [ -f "$SUMMARY" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['accuracy_metrics']['overall_accuracy']*100:.1f}%\")")
        F1=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['match_f1_metrics']['average_match_f1']:.4f}\")")
        echo "  Checkpoint-${CKPT}: Accuracy=${ACC}, Match F1=${F1}"
    fi
done
