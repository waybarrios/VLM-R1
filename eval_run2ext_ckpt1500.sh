#!/bin/bash
# Evaluate Run 2 Extended checkpoint-1500 (aw=0.70, sw=0.30) on full CRYSTAL dataset
# GPUs: 4,5,6,7

set -e

export HF_HOME=/gpudata3/hf_cache
export PYTHONPATH="/gpudata3/Wayner/VLM-R1/src/open-r1-multimodal/src:/gpudata3/Wayner/VLM-R1/mllm_evaluator:${PYTHONPATH}"

PYTHON="/scratch/miniconda3/envs/torch26/bin/python"
PROJECT="/gpudata3/Wayner/VLM-R1"
CHECKPOINT="${PROJECT}/output/run2_extended_aw0.70_sw0.30/checkpoint-1500"
RESULTS_DIR="${PROJECT}/GRPO_analysis/run2_ext_aw0.70_sw0.30_ckpt1500"
PRED_DIR="${RESULTS_DIR}/predictions"
GPU_IDS="4,5,6,7"

echo "=============================================="
echo "Run 2 Extended (aw=0.70, sw=0.30) checkpoint-1500"
echo "Evaluation on CRYSTAL (6,372 samples)"
echo "GPUs: ${GPU_IDS}"
echo "Started: $(date)"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# Step 1: Multi-GPU inference
echo ""
echo ">>> Step 1: Inference..."
bash ${PROJECT}/inference/run_multi_gpu_eval.sh \
    "${CHECKPOINT}" \
    "${PRED_DIR}" \
    "${GPU_IDS}"
echo "Inference done at $(date)"

# Step 2: Compute metrics with distilroberta
echo ""
echo ">>> Step 2: Computing metrics (distilroberta, tau=0.35)..."
$PYTHON ${PROJECT}/compute_metrics.py \
    "${PRED_DIR}" \
    --output-dir "${RESULTS_DIR}" \
    --num-gpus 1
echo "Metrics done at $(date)"

# Step 3: Print results
echo ""
echo "=============================================="
echo "Run 2 Extended (aw=0.70, sw=0.30) checkpoint-1500:"
echo "=============================================="
$PYTHON -c "
import json
d = json.load(open('${PRED_DIR}/no_judge_summary.json'))
print(f\"  Accuracy: {d['accuracy_metrics']['overall_accuracy']*100:.2f}%\")
print(f\"  Match F1: {d['match_f1_metrics']['average_match_f1']:.4f}\")
print(f\"  Precision: {d['match_f1_metrics']['average_precision']:.4f}\")
print(f\"  Recall: {d['match_f1_metrics']['average_recall']:.4f}\")
"

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE at $(date)"
echo "=============================================="
