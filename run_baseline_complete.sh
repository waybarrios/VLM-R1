#!/bin/bash

# Complete baseline evaluation: predictions + metrics
# Qwen2.5-VL-3B-Instruct (base model without fine-tuning)

BASE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
TEST_DATA="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
PREDICTIONS_DIR="predictions/baseline_qwen2.5-vl-3b-instruct"
GPU_IDS="0,1,2"
BATCH_SIZE="${BATCH_SIZE:-4}"

echo "========================================="
echo "Baseline Model Evaluation"
echo "========================================="
echo "Model: ${BASE_MODEL}"
echo "GPUs: ${GPU_IDS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Dataset: ${TEST_DATA}"
echo "Output: ${PREDICTIONS_DIR}"
echo "========================================="
echo ""

# Step 1: Run inference
echo "[1/2] Running inference on baseline model..."
echo ""
python inference/run_parallel_inference.py \
    --checkpoint_dir "${BASE_MODEL}" \
    --test_dataset_path "${TEST_DATA}" \
    --predictions_dir "${PREDICTIONS_DIR}" \
    --gpu_ids "${GPU_IDS}" \
    --batch_size "${BATCH_SIZE}"

if [ $? -ne 0 ]; then
    echo "❌ Inference failed!"
    exit 1
fi

echo ""
echo "========================================="
echo ""

# Step 2: Evaluate predictions
echo "[2/2] Evaluating predictions..."
echo ""
python inference/evaluate_predictions.py \
    --predictions_dir "${PREDICTIONS_DIR}" \
    --test_dataset_path "${TEST_DATA}"

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ Baseline evaluation completed!"
echo "========================================="
echo ""
echo "Results saved to: ${PREDICTIONS_DIR}"
echo "  - Individual predictions: 0.json, 1.json, ..., N.json"
echo "  - Summary: inference_summary.json"
echo "  - Metrics: evaluation_results.json"
echo ""
echo "To view detailed metrics:"
echo "  cat ${PREDICTIONS_DIR}/evaluation_results.json"
echo ""
