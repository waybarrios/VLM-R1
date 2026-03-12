#!/bin/bash
# Evaluate Run 2 Extended checkpoint-400 (aw=0.70, sw=0.30) on full CRYSTAL dataset
# For comparison against CPR Curriculum (aw=0.65, sw=0.35) checkpoint-400
# GPUs: 5,6 (alongside Lin's processes)

set -e

source activate torch26
export HF_HOME=/gpudata3/hf_cache
export PYTHONPATH="/gpudata3/Wayner/VLM-R1/src/open-r1-multimodal/src:/gpudata3/Wayner/VLM-R1/mllm_evaluator:${PYTHONPATH}"

PYTHON="/scratch/miniconda3/envs/torch26/bin/python"
PROJECT="/gpudata3/Wayner/VLM-R1"
CHECKPOINT="${PROJECT}/output/run2_extended_aw0.70_sw0.30/checkpoint-400"
RESULTS_DIR="${PROJECT}/GRPO_analysis/run2_ext_aw0.70_sw0.30_ckpt400"
PRED_DIR="${RESULTS_DIR}/predictions"
GPU_IDS="5,6"

echo "=============================================="
echo "Run 2 Extended (aw=0.70, sw=0.30) checkpoint-400"
echo "Evaluation on CRYSTAL (6,372 samples)"
echo "GPUs: ${GPU_IDS}"
echo "Started: $(date)"
echo "=============================================="

# Step 1: Multi-GPU inference
echo ""
echo ">>> Step 1: Inference..."
bash ${PROJECT}/inference/run_multi_gpu_eval.sh \
    "${CHECKPOINT}" \
    "${PRED_DIR}" \
    "${GPU_IDS}"
echo "✓ Inference done at $(date)"

# Step 2: Compute metrics with distilroberta
echo ""
echo ">>> Step 2: Computing metrics (distilroberta, τ=0.35)..."
$PYTHON ${PROJECT}/compute_metrics.py \
    "${PRED_DIR}" \
    --output-dir "${RESULTS_DIR}" \
    --num-gpus 1
echo "✓ Metrics done at $(date)"

# Step 3: Print comparison
echo ""
echo "=============================================="
echo "COMPARISON: Run 2 Extended vs CPR Curriculum (both at step 400)"
echo "=============================================="

CURRICULUM_SUMMARY="${PROJECT}/GRPO_analysis/cpr_curriculum_results/checkpoint-400/predictions/no_judge_summary.json"
RUN2_SUMMARY="${PRED_DIR}/no_judge_summary.json"

echo ""
echo "CPR Curriculum (aw=0.65, sw=0.35):"
$PYTHON -c "
import json
d = json.load(open('${CURRICULUM_SUMMARY}'))
print(f\"  Accuracy: {d['accuracy_metrics']['overall_accuracy']*100:.2f}%\")
print(f\"  Match F1: {d['match_f1_metrics']['average_match_f1']:.4f}\")
print(f\"  Precision: {d['match_f1_metrics']['average_precision']:.4f}\")
print(f\"  Recall: {d['match_f1_metrics']['average_recall']:.4f}\")
"

echo ""
echo "Run 2 Extended (aw=0.70, sw=0.30):"
$PYTHON -c "
import json
d = json.load(open('${RUN2_SUMMARY}'))
print(f\"  Accuracy: {d['accuracy_metrics']['overall_accuracy']*100:.2f}%\")
print(f\"  Match F1: {d['match_f1_metrics']['average_match_f1']:.4f}\")
print(f\"  Precision: {d['match_f1_metrics']['average_precision']:.4f}\")
print(f\"  Recall: {d['match_f1_metrics']['average_recall']:.4f}\")
"

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE at $(date)"
echo "=============================================="
