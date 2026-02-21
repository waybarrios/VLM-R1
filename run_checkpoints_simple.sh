#!/bin/bash
#
# Simple script to run inference on all checkpoints in parallel
# Distributes checkpoints across 4 GPUs
#

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
OUTPUT_BASE="${PROJECT_ROOT}/predictions_final"
BATCH_SIZE=16

# Add paths
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "Running inference on all checkpoints"
echo "========================================"
echo "Test dataset: $TEST_DATASET"
echo "Output: $OUTPUT_BASE"
echo "Batch size: $BATCH_SIZE"
echo "========================================"

mkdir -p "$OUTPUT_BASE"

# Checkpoint list
CHECKPOINTS=(
    # From 20251106_150520
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520/checkpoint-100"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520/checkpoint-200"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520/checkpoint-300"
    # From 20251107_235133
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-400"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-500"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-600"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-700"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-800"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-900"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1000"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1100"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1200"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1300"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1400"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1500"
)

# GPUs to use
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# Function to run inference
run_checkpoint() {
    local checkpoint_path=$1
    local gpu=$2
    local checkpoint_name=$(basename "$checkpoint_path")

    local full_checkpoint_path="${PROJECT_ROOT}/${checkpoint_path}"
    local predictions_dir="${OUTPUT_BASE}/${checkpoint_name}"
    local log_file="${OUTPUT_BASE}/${checkpoint_name}.log"

    echo "[GPU $gpu] Starting: $checkpoint_name"

    CUDA_VISIBLE_DEVICES=$gpu python "${PROJECT_ROOT}/inference/run_fast_inference.py" \
        --checkpoint_dir "$full_checkpoint_path" \
        --test_dataset_path "$TEST_DATASET" \
        --predictions_dir "$predictions_dir" \
        --device cuda \
        --batch_size $BATCH_SIZE \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "[GPU $gpu] ✓ Completed: $checkpoint_name"
    else
        echo "[GPU $gpu] ✗ Failed: $checkpoint_name (see $log_file)"
    fi
}

# Export function
export -f run_checkpoint
export PROJECT_ROOT TEST_DATASET OUTPUT_BASE BATCH_SIZE

# Distribute checkpoints across GPUs
echo ""
echo "Distributing ${#CHECKPOINTS[@]} checkpoints across $NUM_GPUS GPUs..."
echo ""

# Launch all checkpoints in parallel
for i in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$i]}"
    gpu=${GPUS[$((i % NUM_GPUS))]}

    checkpoint_name=$(basename "$checkpoint")
    echo "GPU $gpu <- $checkpoint_name"

    # Run in background
    run_checkpoint "$checkpoint" "$gpu" &

    # Brief delay to stagger startup
    sleep 1
done

echo ""
echo "========================================"
echo "All jobs launched! Waiting for completion..."
echo "========================================"
echo ""

# Wait for all background jobs
wait

echo ""
echo "========================================"
echo "✓ All checkpoints processed!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Summary:"
for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_name=$(basename "$checkpoint")
    predictions_dir="${OUTPUT_BASE}/${checkpoint_name}"

    if [ -f "${predictions_dir}/inference_stats.json" ]; then
        valid=$(python -c "import json; data=json.load(open('${predictions_dir}/inference_stats.json')); print(data.get('valid_predictions', 0))" 2>/dev/null)
        total=$(python -c "import json; data=json.load(open('${predictions_dir}/inference_stats.json')); print(data.get('processed_samples', 0))" 2>/dev/null)
        if [ -n "$valid" ] && [ -n "$total" ]; then
            echo "  ✓ $checkpoint_name: $valid/$total valid predictions"
        else
            echo "  ? $checkpoint_name: Stats file exists but couldn't parse"
        fi
    else
        echo "  ✗ $checkpoint_name: No stats file found"
    fi
done
echo ""
echo "Log files: ${OUTPUT_BASE}/*.log"
echo "========================================"
