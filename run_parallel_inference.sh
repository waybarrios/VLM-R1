#!/bin/bash

# Simple parallel inference - each GPU processes a DIFFERENT checkpoint
# Usage: BATCH_SIZE=16 bash run_parallel_inference.sh

# Configuration
OUTPUT_DIRS=(
    "/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133"
)

TEST_DATA="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
GPU_IDS="${GPU_IDS:-0,1,2,3}"  # Default GPUs: 0,1,2,3
BATCH_SIZE="${BATCH_SIZE:-4}"  # Default batch size: 4
TIMEOUT="${TIMEOUT:-1800}"     # Default timeout: 30 minutes
CHECKPOINT_RANGE="${CHECKPOINT_RANGE:-400-1500-100}"  # start-end-step

echo "========================================="
echo "Parallel Checkpoint Inference"
echo "========================================="
echo "Strategy: Each GPU processes a different checkpoint"
echo "GPUs: ${GPU_IDS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Timeout: ${TIMEOUT}s per checkpoint"
echo "Dataset: ${TEST_DATA}"
echo "Checkpoints: ${CHECKPOINT_RANGE}"
echo "========================================="
echo ""

# Build output_dirs argument
OUTPUT_DIRS_ARG=""
for dir in "${OUTPUT_DIRS[@]}"; do
    OUTPUT_DIRS_ARG="${OUTPUT_DIRS_ARG} ${dir}"
done

# Run the Python script
python run_parallel_checkpoints.py \
    --output_dirs ${OUTPUT_DIRS_ARG} \
    --test_dataset "${TEST_DATA}" \
    --gpu_ids "${GPU_IDS}" \
    --batch_size "${BATCH_SIZE}" \
    --timeout "${TIMEOUT}" \
    --checkpoint_range "${CHECKPOINT_RANGE}"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "========================================="
    echo "✓ All checkpoints completed successfully!"
    echo "========================================="
else
    echo "========================================="
    echo "✗ Some checkpoints failed (see above)"
    echo "========================================="
fi

exit $exit_code
