#!/bin/bash
#
# Fast inference script - Uses HuggingFace directly (NO DeepSpeed)
# Follows the pattern from test_rec_r1.py with qwen_vl_utils
#

# Configuration
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"

# Add directories to PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default values
CHECKPOINT_DIR="${1:-}"
TEST_DATASET="${2:-/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27}"
GPU_ID="${3:-0}"
BATCH_SIZE="${4:-2}"
START_IDX="${5:-0}"
END_IDX="${6:-}"

# Validate checkpoint
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <checkpoint_dir> [test_dataset] [gpu_id] [batch_size] [start_idx] [end_idx]"
    echo ""
    echo "Examples:"
    echo "  $0 output/qwen2.5-vl-3b-vqa-deepspeed-20250110_123456/checkpoint-100"
    echo "  $0 output/checkpoint-100 /path/to/test 5 2"
    echo "  $0 output/checkpoint-100 /path/to/test 5 2 0 1000"
    echo ""
    exit 1
fi

# Extract checkpoint name
CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")

# Create predictions directory
PREDICTIONS_DIR="${PROJECT_ROOT}/predictions/${CHECKPOINT_NAME}"
mkdir -p "$PREDICTIONS_DIR"

echo "========================================"
echo "Fast Inference (HuggingFace - NO DeepSpeed)"
echo "Pattern from test_rec_r1.py"
echo "========================================"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Test dataset: $TEST_DATASET"
echo "GPU: $GPU_ID"
echo "Batch size: $BATCH_SIZE"
echo "Start index: $START_IDX"
if [ -n "$END_IDX" ]; then
    echo "End index: $END_IDX"
else
    echo "End index: all"
fi
echo "Predictions: $PREDICTIONS_DIR"
echo ""
echo "Using checkpoint defaults for pixel configuration"
echo "========================================"

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Build end_idx argument
END_IDX_ARG=""
if [ -n "$END_IDX" ]; then
    END_IDX_ARG="--end_idx $END_IDX"
fi

# Run inference
python inference/run_fast_inference.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --test_dataset_path "$TEST_DATASET" \
    --predictions_dir "$PREDICTIONS_DIR" \
    --device cuda \
    --batch_size $BATCH_SIZE \
    --start_idx $START_IDX \
    $END_IDX_ARG

echo ""
echo "========================================"
echo "✓ Inference completed!"
echo "Predictions saved to: $PREDICTIONS_DIR"
echo "========================================"
