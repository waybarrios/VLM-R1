#!/bin/bash
# Evaluate CPR Curriculum checkpoints 2000, 2400, 2800 for fair comparison with CPR original
# Waits for each checkpoint to appear before evaluating
# Run inside tmux: tmux new-session -d -s eval_fair bash eval_cpr_curriculum_fair.sh

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
echo "CPR Curriculum Fair Comparison (2000, 2400, 2800)"
echo "Started: $(date)"
echo "=============================================="

for CKPT in 2000 2400 2800; do
    CKPT_DIR="${TRAIN_DIR}/checkpoint-${CKPT}"
    PRED_DIR="${RESULTS_DIR}/checkpoint-${CKPT}/predictions"

    echo ""
    echo "=============================================="
    echo ">>> Waiting for checkpoint-${CKPT}..."
    echo "=============================================="

    # Wait for checkpoint to exist
    while [ ! -d "${CKPT_DIR}" ]; do
        echo "  checkpoint-${CKPT} not yet available. Waiting 120s... ($(date))"
        sleep 120
    done

    # Wait a bit more to ensure checkpoint is fully saved
    echo "  checkpoint-${CKPT} detected! Waiting 60s for write to finish..."
    sleep 60

    echo ">>> Checkpoint-${CKPT}: Inference + Metrics"

    # Wait for GPUs to be free
    while true; do
        GPU4_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4)
        if [ "$GPU4_MEM" -lt 2000 ]; then
            break
        fi
        echo "  GPUs busy (GPU4: ${GPU4_MEM} MiB). Waiting 60s..."
        sleep 60
    done

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
echo "FAIR COMPARISON EVALUATIONS COMPLETE at $(date)"
echo "=============================================="
echo ""
echo "=== FULL FAIR COMPARISON TABLE ==="
echo "Checkpoint | Curriculum (Acc / F1) | Original (Acc / F1)"
for CKPT in 400 500 700 1000 1400 2000 2400 2800; do
    SUMMARY="${RESULTS_DIR}/checkpoint-${CKPT}/predictions/no_judge_summary.json"
    if [ -f "$SUMMARY" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['accuracy_metrics']['overall_accuracy']*100:.2f}%\")")
        F1=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['match_f1_metrics']['average_match_f1']:.4f}\")")
        echo "  ${CKPT}: ${ACC} / ${F1}"
    else
        echo "  ${CKPT}: MISSING"
    fi
done
