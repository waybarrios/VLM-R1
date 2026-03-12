#!/bin/bash
# Evaluate all remaining CPR Curriculum checkpoints
# Checkpoints: 100, 300, 600, 800, 900, 1100, 1200, 1300, 1600, 1700, 1800, 1900, 2100, 2200, 2300, 2400

set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate torch26
export HF_HOME=/gpudata3/hf_cache
export PYTHONPATH="/gpudata3/Wayner/VLM-R1/src/open-r1-multimodal/src:/gpudata3/Wayner/VLM-R1/mllm_evaluator:${PYTHONPATH}"

PYTHON="/scratch/miniconda3/envs/torch26/bin/python"
PROJECT="/gpudata3/Wayner/VLM-R1"
TRAIN_DIR="${PROJECT}/output/grpo-cpr-curriculum-20260221_163422"
RESULTS_DIR="${PROJECT}/GRPO_analysis/cpr_curriculum_results"
GPU_IDS="4,5,6,7"

echo "=============================================="
echo "CPR Curriculum - Remaining Checkpoints Eval"
echo "Started: $(date)"
echo "=============================================="

for CKPT in 100 300 600 800 900 1100 1200 1300 1600 1700 1800 1900 2100 2200 2300 2400; do
    CKPT_DIR="${TRAIN_DIR}/checkpoint-${CKPT}"
    PRED_DIR="${RESULTS_DIR}/checkpoint-${CKPT}/predictions"

    echo ""
    echo "=============================================="
    echo ">>> checkpoint-${CKPT}"
    echo "=============================================="

    # Check if already evaluated
    if [ -f "${PRED_DIR}/no_judge_summary.json" ]; then
        echo "  Already has results, skipping."
        continue
    fi

    # Check checkpoint exists
    if [ ! -d "${CKPT_DIR}" ]; then
        echo "  checkpoint-${CKPT} does not exist yet, skipping."
        continue
    fi

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
echo "ALL REMAINING EVALUATIONS COMPLETE at $(date)"
echo "=============================================="
echo ""
echo "=== FULL TABLE ==="
for CKPT in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400; do
    SUMMARY="${RESULTS_DIR}/checkpoint-${CKPT}/predictions/no_judge_summary.json"
    if [ -f "$SUMMARY" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['accuracy_metrics']['overall_accuracy']*100:.2f}%\")")
        F1=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['match_f1_metrics']['average_match_f1']:.4f}\")")
        echo "  checkpoint-${CKPT}: Acc=${ACC}  F1=${F1}"
    else
        echo "  checkpoint-${CKPT}: MISSING"
    fi
done
