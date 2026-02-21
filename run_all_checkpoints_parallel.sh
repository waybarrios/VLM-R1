#!/bin/bash
#
# Run inference on all checkpoints in parallel - one checkpoint per GPU
#

# Configuration
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
OUTPUT_BASE="${PROJECT_ROOT}/predictions_final"
BATCH_SIZE=2

# Available GPUs
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# Checkpoint directories
CHECKPOINT_DIRS=(
    "${PROJECT_ROOT}/output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520"
    "${PROJECT_ROOT}/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133"
)

echo "========================================"
echo "Running inference on all checkpoints"
echo "========================================"
echo "Test dataset: $TEST_DATASET"
echo "Output base: $OUTPUT_BASE"
echo "Batch size: $BATCH_SIZE"
echo "Available GPUs: ${GPUS[@]}"
echo "========================================"

# Collect all checkpoints
ALL_CHECKPOINTS=()
for CHECKPOINT_DIR in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$CHECKPOINT_DIR" ]; then
        for checkpoint in "$CHECKPOINT_DIR"/checkpoint-*; do
            if [ -d "$checkpoint" ]; then
                ALL_CHECKPOINTS+=("$checkpoint")
            fi
        done
    fi
done

# Sort checkpoints by number
IFS=$'\n' ALL_CHECKPOINTS=($(sort -V <<<"${ALL_CHECKPOINTS[*]}"))
unset IFS

echo ""
echo "Found ${#ALL_CHECKPOINTS[@]} checkpoints:"
for checkpoint in "${ALL_CHECKPOINTS[@]}"; do
    echo "  - $(basename "$checkpoint")"
done
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to run inference on a checkpoint
run_inference() {
    local checkpoint=$1
    local gpu=$2
    local checkpoint_name=$(basename "$checkpoint")
    local predictions_dir="${OUTPUT_BASE}/${checkpoint_name}"

    echo "[GPU $gpu] Starting inference for $checkpoint_name..."

    CUDA_VISIBLE_DEVICES=$gpu python "${PROJECT_ROOT}/inference/run_fast_inference.py" \
        --checkpoint_dir "$checkpoint" \
        --test_dataset_path "$TEST_DATASET" \
        --predictions_dir "$predictions_dir" \
        --device cuda \
        --batch_size $BATCH_SIZE \
        2>&1 | tee "${predictions_dir}.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "[GPU $gpu] ✓ Completed: $checkpoint_name"
    else
        echo "[GPU $gpu] ✗ Failed: $checkpoint_name (exit code: $exit_code)"
    fi

    return $exit_code
}

# Export function and variables for parallel execution
export -f run_inference
export PROJECT_ROOT
export TEST_DATASET
export OUTPUT_BASE
export BATCH_SIZE

# Track running jobs
declare -A GPU_JOBS
declare -A GPU_CHECKPOINT

# Initialize GPU status
for gpu in "${GPUS[@]}"; do
    GPU_JOBS[$gpu]=""
    GPU_CHECKPOINT[$gpu]=""
done

checkpoint_idx=0
total_checkpoints=${#ALL_CHECKPOINTS[@]}

echo "========================================"
echo "Starting parallel inference..."
echo "========================================"
echo ""

# Process all checkpoints
while [ $checkpoint_idx -lt $total_checkpoints ]; do
    # Check each GPU for availability
    for gpu in "${GPUS[@]}"; do
        # If GPU has no job or job is finished, assign new checkpoint
        if [ -z "${GPU_JOBS[$gpu]}" ] || ! kill -0 ${GPU_JOBS[$gpu]} 2>/dev/null; then
            # Check if previous job succeeded
            if [ -n "${GPU_JOBS[$gpu]}" ]; then
                wait ${GPU_JOBS[$gpu]}
                exit_code=$?
                if [ $exit_code -eq 0 ]; then
                    echo "[GPU $gpu] ✓ Finished: ${GPU_CHECKPOINT[$gpu]}"
                else
                    echo "[GPU $gpu] ✗ Failed: ${GPU_CHECKPOINT[$gpu]}"
                fi
            fi

            # Assign new checkpoint if available
            if [ $checkpoint_idx -lt $total_checkpoints ]; then
                checkpoint="${ALL_CHECKPOINTS[$checkpoint_idx]}"
                checkpoint_name=$(basename "$checkpoint")

                echo "[GPU $gpu] Assigning: $checkpoint_name ($((checkpoint_idx + 1))/$total_checkpoints)"

                # Start inference in background
                run_inference "$checkpoint" "$gpu" &
                GPU_JOBS[$gpu]=$!
                GPU_CHECKPOINT[$gpu]=$checkpoint_name

                checkpoint_idx=$((checkpoint_idx + 1))
            else
                # No more checkpoints to assign
                GPU_JOBS[$gpu]=""
                GPU_CHECKPOINT[$gpu]=""
            fi
        fi
    done

    # Brief sleep to avoid busy waiting
    sleep 2
done

# Wait for all remaining jobs to complete
echo ""
echo "Waiting for remaining jobs to complete..."
for gpu in "${GPUS[@]}"; do
    if [ -n "${GPU_JOBS[$gpu]}" ] && kill -0 ${GPU_JOBS[$gpu]} 2>/dev/null; then
        echo "Waiting for GPU $gpu (${GPU_CHECKPOINT[$gpu]})..."
        wait ${GPU_JOBS[$gpu]}
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[GPU $gpu] ✓ Finished: ${GPU_CHECKPOINT[$gpu]}"
        else
            echo "[GPU $gpu] ✗ Failed: ${GPU_CHECKPOINT[$gpu]}"
        fi
    fi
done

echo ""
echo "========================================"
echo "✓ All checkpoints processed!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Summary:"
for checkpoint in "${ALL_CHECKPOINTS[@]}"; do
    checkpoint_name=$(basename "$checkpoint")
    predictions_dir="${OUTPUT_BASE}/${checkpoint_name}"

    if [ -f "${predictions_dir}/inference_stats.json" ]; then
        valid=$(grep -o '"valid_predictions": [0-9]*' "${predictions_dir}/inference_stats.json" | grep -o '[0-9]*')
        total=$(grep -o '"processed_samples": [0-9]*' "${predictions_dir}/inference_stats.json" | grep -o '[0-9]*')
        echo "  ✓ $checkpoint_name: $valid/$total valid predictions"
    else
        echo "  ✗ $checkpoint_name: No stats file found"
    fi
done
echo ""
echo "========================================"
