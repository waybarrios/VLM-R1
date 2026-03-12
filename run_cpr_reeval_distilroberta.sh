#!/bin/bash
#
# Re-evaluate ALL 8 CPR checkpoints with all-distilroberta-v1 τ=0.35
# Uses GPUs 4-7 (free). Runs 4 checkpoints in parallel, then the next 4.
#

set -e

PYTHON="/scratch/miniconda3/envs/torch26/bin/python"
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"

export PYTHONPATH="${PROJECT_ROOT}/src/open-r1-multimodal/src:${PROJECT_ROOT}/mllm_evaluator:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"

echo "========================================"
echo "CPR RE-EVALUATION: all-distilroberta-v1 τ=0.35"
echo "Started: $(date)"
echo "Using GPUs 4-7"
echo "========================================"

CHECKPOINTS=(400 500 700 1000 1400 2000 2400 2800)

# Batch 1: checkpoints 400, 500, 700, 1000 on GPUs 4, 5, 6, 7
echo ""
echo "--- Batch 1: CPR-{400,500,700,1000} ---"
PIDS=""
for i in 0 1 2 3; do
    CKPT=${CHECKPOINTS[$i]}
    GPU=$((4 + i))
    OUTPUT_DIR="evaluations/cpr-checkpoint-${CKPT}"

    echo "  Starting CPR-${CKPT} on GPU ${GPU}..."

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON compute_metrics.py \
        predictions/cpr-checkpoint-${CKPT} \
        --dataset-path "$TEST_DATASET" \
        --num-gpus 1 \
        --output-dir "$OUTPUT_DIR" \
        > "evaluations/reeval_cpr_${CKPT}.log" 2>&1 &

    PIDS="$PIDS $!"
done

echo "  Waiting for batch 1..."
wait $PIDS
echo "  Batch 1 DONE"

# Batch 2: checkpoints 1400, 2000, 2400, 2800 on GPUs 4, 5, 6, 7
echo ""
echo "--- Batch 2: CPR-{1400,2000,2400,2800} ---"
PIDS=""
for i in 4 5 6 7; do
    CKPT=${CHECKPOINTS[$i]}
    GPU=$((4 + i - 4))
    OUTPUT_DIR="evaluations/cpr-checkpoint-${CKPT}"

    echo "  Starting CPR-${CKPT} on GPU ${GPU}..."

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON compute_metrics.py \
        predictions/cpr-checkpoint-${CKPT} \
        --dataset-path "$TEST_DATASET" \
        --num-gpus 1 \
        --output-dir "$OUTPUT_DIR" \
        > "evaluations/reeval_cpr_${CKPT}.log" 2>&1 &

    PIDS="$PIDS $!"
done

echo "  Waiting for batch 2..."
wait $PIDS
echo "  Batch 2 DONE"

# Print summary
echo ""
echo "========================================"
echo "RE-EVALUATION COMPLETE: $(date)"
echo "========================================"
echo ""
echo "| Checkpoint | Accuracy | Match F1 | Precision | Recall |"
echo "|------------|----------|----------|-----------|--------|"

for CKPT in "${CHECKPOINTS[@]}"; do
    SUMMARY="evaluations/cpr-checkpoint-${CKPT}/cpr-checkpoint-${CKPT}/no_judge_summary.json"
    if [ -f "$SUMMARY" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['accuracy_metrics']['overall_accuracy']*100:.2f}%\")")
        F1=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['match_f1_metrics']['average_match_f1']:.4f}\")")
        P=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['match_f1_metrics']['average_precision']:.4f}\")")
        R=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d['match_f1_metrics']['average_recall']:.4f}\")")
        echo "| CPR-${CKPT}    | ${ACC}   | ${F1}   | ${P}   | ${R}   |"
    else
        echo "| CPR-${CKPT}    | MISSING  | MISSING  | MISSING   | MISSING|"
    fi
done

echo ""
echo "Encoder: all-distilroberta-v1, Threshold: 0.35"
echo "========================================"
