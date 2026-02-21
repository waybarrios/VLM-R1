#!/bin/bash

# Parallel inference with monitoring and auto-restart
# Usage: BATCH_SIZE=16 bash run_parallel_with_monitoring.sh

OUTPUT_DIRS=(
    "/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133"
)

TEST_DATA="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TIMEOUT="${TIMEOUT:-1800}"
CHECKPOINT_RANGE="${CHECKPOINT_RANGE:-400-1500-100}"

# Time without new files before considering stuck (seconds)
STUCK_TIMEOUT="${STUCK_TIMEOUT:-600}"  # 10 minutes

echo "========================================="
echo "Parallel Checkpoint Inference with Monitoring"
echo "========================================="
echo "GPUs: ${GPU_IDS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Stuck detection: ${STUCK_TIMEOUT}s without new files"
echo "========================================="
echo ""

# Run with automatic monitoring and retry
python run_parallel_checkpoints.py \
    --output_dirs ${OUTPUT_DIRS[@]} \
    --test_dataset "${TEST_DATA}" \
    --gpu_ids "${GPU_IDS}" \
    --batch_size "${BATCH_SIZE}" \
    --timeout "${TIMEOUT}" \
    --checkpoint_range "${CHECKPOINT_RANGE}"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "========================================="
    echo "✓ All checkpoints completed!"
    echo "========================================="
else
    echo "========================================="
    echo "✗ Some checkpoints failed"
    echo "========================================="
    echo ""
    echo "Check logs: gpu*.log"
fi

exit $exit_code
