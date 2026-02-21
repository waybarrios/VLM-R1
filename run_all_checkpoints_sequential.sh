#!/bin/bash
#
# Run all checkpoints sequentially, each using 3 GPUs in parallel
#

set -e

GPUS="0,1,2"
BATCH_SIZE=4

CHECKPOINTS=(
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520/checkpoint-100"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520/checkpoint-200"
    "output/qwen2.5-vl-3b-vqa-deepspeed-20251106_150520/checkpoint-300"
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

echo "========================================"
echo "Running all checkpoints sequentially"
echo "Each checkpoint uses GPUs: $GPUS in parallel"
echo "Batch size: $BATCH_SIZE"
echo "Total checkpoints: ${#CHECKPOINTS[@]}"
echo "========================================"
echo ""

for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_name=$(basename "$checkpoint")

    echo ""
    echo "========================================"
    echo "Processing: $checkpoint_name"
    echo "========================================"

    ./run_checkpoint_parallel.sh "$checkpoint" "$GPUS" "$BATCH_SIZE"

    echo ""
    echo "✓ Completed: $checkpoint_name"
    echo ""
done

echo ""
echo "========================================"
echo "✓ All checkpoints completed!"
echo "========================================"
