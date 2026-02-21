#!/bin/bash
#
# Simple script - one checkpoint, one GPU, no multithreading
#

CHECKPOINT_DIR=${1:-""}
GPU=${2:-0}
BATCH_SIZE=${3:-1}

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <checkpoint_dir> [gpu] [batch_size]"
    echo ""
    echo "Example:"
    echo "  $0 output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520/checkpoint-100 0 1"
    echo ""
    exit 1
fi

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"

export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU

CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
FULL_CHECKPOINT_PATH="${PROJECT_ROOT}/${CHECKPOINT_DIR}"
PREDICTIONS_DIR="${PROJECT_ROOT}/predictions_final/${CHECKPOINT_NAME}"

echo "========================================"
echo "Simple single-GPU inference"
echo "========================================"
echo "Checkpoint: $CHECKPOINT_NAME"
echo "GPU: $GPU"
echo "Batch size: $BATCH_SIZE"
echo "Output: $PREDICTIONS_DIR"
echo "========================================"
echo ""

# Run directly - NO background, NO multithreading, NO DeepSpeed
python "${PROJECT_ROOT}/inference/run_simple_vqa.py" \
    --checkpoint_dir "$FULL_CHECKPOINT_PATH" \
    --test_dataset_path "$TEST_DATASET" \
    --predictions_dir "$PREDICTIONS_DIR" \
    --gpu $GPU \
    --batch_size $BATCH_SIZE

echo ""
echo "========================================"
echo "✓ Completed!"
echo "========================================"
