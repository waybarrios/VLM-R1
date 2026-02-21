#!/bin/bash
# GPU 2: checkpoints 1100, 1200, 1300

OUTPUT_DIR="/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133"
TEST_DATA="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
GPU_ID=2
BATCH_SIZE="${BATCH_SIZE:-8}"  # Reduced from 16 to avoid OOM

echo "========================================="
echo "GPU 2: Processing checkpoints 1100, 1200, 1300"
echo "========================================="
echo "GPU: ${GPU_ID}"
echo "Batch Size: ${BATCH_SIZE}"
echo "========================================="
echo ""

for CHECKPOINT in 1100 1200 1300; do
    CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-${CHECKPOINT}"
    PRED_DIR="predictions/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-${CHECKPOINT}"

    echo "========================================="
    echo "Checkpoint ${CHECKPOINT}"
    echo "========================================="
    echo "Start time: $(date)"

    python inference/run_deepspeed_checkpoint_inference.py \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --test_dataset_path "${TEST_DATA}" \
        --predictions_dir "${PRED_DIR}" \
        --device_ids "${GPU_ID}" \
        --batch_size "${BATCH_SIZE}"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Checkpoint ${CHECKPOINT} completed!"
        echo "End time: $(date)"
        echo ""
    else
        echo ""
        echo "✗ Checkpoint ${CHECKPOINT} failed!"
        echo "End time: $(date)"
        echo ""
    fi

    echo ""
done

echo "========================================="
echo "GPU 2: All checkpoints completed!"
echo "========================================="
