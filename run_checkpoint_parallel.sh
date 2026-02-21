#!/bin/bash
#
# Run inference on a single checkpoint using multiple GPUs in parallel
# Each GPU processes a chunk of the dataset
#

set -e

CHECKPOINT_DIR=${1:-""}
GPUS=${2:-"0,1,2"}
BATCH_SIZE=${3:-4}

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <checkpoint_dir> [gpus] [batch_size]"
    echo ""
    echo "Example:"
    echo "  $0 output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520/checkpoint-100"
    echo "  $0 output/.../checkpoint-100 \"0,1,2\" 4"
    echo ""
    exit 1
fi

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"

export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Parse GPU list
IFS="," read -ra GPULIST <<< "${GPUS}"
CHUNKS=${#GPULIST[@]}

# Get checkpoint name
CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
FULL_CHECKPOINT_PATH="${PROJECT_ROOT}/${CHECKPOINT_DIR}"

# Check if checkpoint exists
if [ ! -d "$FULL_CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found: $FULL_CHECKPOINT_PATH"
    exit 1
fi

PREDICTIONS_DIR="${PROJECT_ROOT}/predictions_final/${CHECKPOINT_NAME}"
mkdir -p "$PREDICTIONS_DIR"

echo "========================================"
echo "Parallel inference on single checkpoint"
echo "========================================"
echo "Checkpoint: $CHECKPOINT_NAME"
echo "GPUs: ${GPULIST[@]} (${CHUNKS} chunks)"
echo "Batch size: $BATCH_SIZE"
echo "Output: $PREDICTIONS_DIR"
echo "========================================"
echo ""

# Save PIDs for cleanup
PID_FILE="${PREDICTIONS_DIR}/running_pids.txt"
echo "# PIDs for $CHECKPOINT_NAME - $(date)" > "$PID_FILE"
echo "# To kill: kill -9 \$(cat $PID_FILE | grep -v '^#' | awk '{print \$1}')" >> "$PID_FILE"

# Launch one process per GPU
for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU=${GPULIST[$IDX]}
    LOG_FILE="${PREDICTIONS_DIR}/gpu${GPU}_chunk${IDX}.log"

    echo "[GPU $GPU] Processing chunk $((IDX+1))/$CHUNKS"

    CUDA_VISIBLE_DEVICES=$GPU python "${PROJECT_ROOT}/inference/run_fast_inference.py" \
        --checkpoint_dir "$FULL_CHECKPOINT_PATH" \
        --test_dataset_path "$TEST_DATASET" \
        --predictions_dir "$PREDICTIONS_DIR" \
        --device cuda \
        --batch_size $BATCH_SIZE \
        --chunk_total $CHUNKS \
        --chunk_index $IDX \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID GPU$GPU chunk$IDX/$CHUNKS" >> "$PID_FILE"

    sleep 1
done

echo ""
echo "All processes launched!"
echo "PIDs saved to: $PID_FILE"
echo ""
echo "Monitor progress:"
echo "  ./monitor_inference_progress.sh predictions_final"
echo "  tail -f $PREDICTIONS_DIR/gpu*.log"
echo ""
echo "Kill all:"
echo "  kill -9 \$(cat $PID_FILE | grep -v '^#' | awk '{print \$1}')"
echo "  Or: ./kill_inference.sh"
echo ""
echo "Waiting for all chunks to complete..."

# Wait for all background jobs
wait

echo ""
echo "========================================"
echo "✓ All chunks completed!"
echo "========================================"

# Show summary
if [ -f "${PREDICTIONS_DIR}/inference_stats.json" ]; then
    valid=$(python -c "import json; data=json.load(open('${PREDICTIONS_DIR}/inference_stats.json')); print(data.get('valid_predictions', 0))" 2>/dev/null)
    total=$(python -c "import json; data=json.load(open('${PREDICTIONS_DIR}/inference_stats.json')); print(data.get('processed_samples', 0))" 2>/dev/null)
    elapsed=$(python -c "import json; data=json.load(open('${PREDICTIONS_DIR}/inference_stats.json')); print(f\"{data.get('elapsed_time_seconds', 0)/60:.1f}\")" 2>/dev/null)
    if [ -n "$valid" ] && [ -n "$total" ]; then
        echo "Results: $valid/$total valid predictions (${elapsed}min)"
    fi
fi

# Clean up PID file
rm -f "$PID_FILE"

echo "========================================"
