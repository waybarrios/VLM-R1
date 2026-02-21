#!/bin/bash

# Sequential inference - process checkpoints one by one using a single GPU
# Usage: BATCH_SIZE=16 GPU_ID=0 bash run_baseline_inference.sh

# Define the two checkpoint folders
OUTPUT_DIRS=(
    "/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520"
    "/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133"
)

TEST_DATA="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
GPU_ID="${GPU_ID:-0}"  # Default to GPU 0, can override with: GPU_ID=1 bash script.sh
BATCH_SIZE="${BATCH_SIZE:-4}"  # Default to 4, can override with: BATCH_SIZE=8 bash script.sh
TIMEOUT="${TIMEOUT:-1800}"  # Default timeout: 30 minutes

echo "========================================="
echo "Sequential Inference Mode"
echo "========================================="
echo "Strategy: One checkpoint at a time, single GPU"
echo "GPU: ${GPU_ID}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Timeout: ${TIMEOUT}s"
echo "Dataset: ${TEST_DATA}"
echo "Checkpoints: 400 to 1500 (step 100)"
echo "========================================="
echo ""

# Function to run inference with timeout
run_inference_with_timeout() {
    local checkpoint_dir=$1
    local pred_dir=$2
    local timeout=$3

    # Run inference with timeout
    timeout ${timeout}s python inference/run_deepspeed_checkpoint_inference.py \
        --checkpoint_dir "${checkpoint_dir}" \
        --test_dataset_path "${TEST_DATA}" \
        --predictions_dir "${pred_dir}" \
        --device_ids "${GPU_ID}" \
        --batch_size "${BATCH_SIZE}"

    return $?
}

# Loop through each output directory
for OUTPUT_DIR in "${OUTPUT_DIRS[@]}"; do
    # Extract the experiment name from the path
    EXP_NAME=$(basename "${OUTPUT_DIR}")

    echo "========================================="
    echo "Processing experiment: ${EXP_NAME}"
    echo "========================================="
    echo ""

    # Loop through checkpoints from 400 to 1500 in steps of 100
    for CHECKPOINT in {400..1500..100}; do
        CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-${CHECKPOINT}"
        PRED_DIR="predictions/${EXP_NAME}/checkpoint-${CHECKPOINT}"

        # Check if checkpoint directory exists
        if [ ! -d "${CHECKPOINT_DIR}" ]; then
            echo "Warning: ${CHECKPOINT_DIR} does not exist, skipping..."
            echo ""
            continue
        fi

        # Check if predictions already exist (skip if completed)
        if [ -f "${PRED_DIR}/inference_stats.json" ]; then
            echo "✓ checkpoint-${CHECKPOINT} already completed, skipping..."
            echo ""
            continue
        fi

        echo "Running inference on checkpoint-${CHECKPOINT}..."
        echo "Start time: $(date)"

        # Run with timeout
        run_inference_with_timeout "${CHECKPOINT_DIR}" "${PRED_DIR}" "${TIMEOUT}"
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo ""
            echo "========================================="
            echo "✓ checkpoint-${CHECKPOINT} completed!"
            echo "End time: $(date)"
            echo "========================================="
            echo ""
        elif [ $exit_code -eq 124 ]; then
            echo ""
            echo "========================================="
            echo "✗ TIMEOUT: checkpoint-${CHECKPOINT} exceeded ${TIMEOUT}s"
            echo "End time: $(date)"
            echo "========================================="
            echo ""
            # Continue with next checkpoint even if one times out
        else
            echo ""
            echo "========================================="
            echo "✗ ERROR: checkpoint-${CHECKPOINT} failed with exit code $exit_code"
            echo "End time: $(date)"
            echo "========================================="
            echo ""
            # Continue with next checkpoint even if one fails
        fi

        # Small pause to ensure GPU memory is freed
        sleep 5
    done

    echo ""
    echo "========================================="
    echo "Completed all checkpoints for: ${EXP_NAME}"
    echo "========================================="
    echo ""
done

echo ""
echo "========================================="
echo "All experiments and checkpoints completed!"
echo "========================================="
