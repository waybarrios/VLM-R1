#!/bin/bash
# Multi-GPU evaluation - runs run_simple_vqa.py on separate dataset chunks per GPU
# Usage: bash inference/run_multi_gpu_eval.sh <checkpoint_dir> <predictions_dir> <gpu_ids>

set -e

CHECKPOINT_DIR="$1"
PREDICTIONS_DIR="$2"
GPU_IDS="${3:-4,5,6,7}"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
PYTHON="/scratch/miniconda3/envs/torch26/bin/python"
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"

export PYTHONPATH="${PROJECT_ROOT}/src/open-r1-multimodal/src:${PROJECT_ROOT}/mllm_evaluator:${PYTHONPATH}"

if [ -z "$CHECKPOINT_DIR" ] || [ -z "$PREDICTIONS_DIR" ]; then
    echo "Usage: $0 <checkpoint_dir> <predictions_dir> [gpu_ids]"
    exit 1
fi

echo "========================================"
echo "Multi-GPU Evaluation"
echo "========================================"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $PREDICTIONS_DIR"
echo "GPUs: $GPU_IDS"
echo "Dataset: $TEST_DATASET"
echo "========================================"

mkdir -p "$PREDICTIONS_DIR"

# Step 1: Split dataset
echo ""
echo "Step 1: Splitting dataset..."
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

$PYTHON -c "
from datasets import load_from_disk
import os, math

dataset = load_from_disk('$TEST_DATASET')
n = len(dataset)
num_splits = $NUM_GPUS
chunk = math.ceil(n / num_splits)

for i in range($NUM_GPUS):
    start = i * chunk
    end = min(start + chunk, n)
    split = dataset.select(range(start, end))
    split_dir = '${PREDICTIONS_DIR}/split_{}'.format(i)
    split.save_to_disk(split_dir)
    print(f'Split {i}: samples {start}-{end} ({len(split)} samples) -> {split_dir}')

print(f'\nTotal: {n} samples split into {num_splits} chunks')
"

# Step 2: Run inference on each GPU in parallel
echo ""
echo "Step 2: Launching inference on ${NUM_GPUS} GPUs..."

PIDS=()
for i in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[$i]}"
    SPLIT_DIR="${PREDICTIONS_DIR}/split_${i}"
    GPU_PRED_DIR="${PREDICTIONS_DIR}/gpu_${GPU_ID}"

    mkdir -p "$GPU_PRED_DIR"

    echo "  Launching GPU ${GPU_ID} (split ${i})..."

    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON ${PROJECT_ROOT}/inference/run_simple_vqa.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --test_dataset_path "$SPLIT_DIR" \
        --predictions_dir "$GPU_PRED_DIR" \
        --gpu 0 \
        --batch_size 1 \
        > "${PREDICTIONS_DIR}/gpu_${GPU_ID}.log" 2>&1 &

    PIDS+=($!)
    echo "    PID: ${PIDS[-1]}"
done

echo ""
echo "All ${NUM_GPUS} GPUs launched. Waiting for completion..."

# Wait for all processes
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    GPU_ID="${GPUS[$i]}"
    if wait "$PID"; then
        echo "  GPU ${GPU_ID} (PID ${PID}): DONE"
    else
        echo "  GPU ${GPU_ID} (PID ${PID}): FAILED"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "WARNING: $FAILED GPUs failed. Check logs in $PREDICTIONS_DIR/gpu_*.log"
fi

# Step 3: Merge results - renumber predictions with global indices
echo ""
echo "Step 3: Merging results..."

$PYTHON -c "
import os, json, math, shutil
from datasets import load_from_disk

dataset = load_from_disk('$TEST_DATASET')
n = len(dataset)
num_splits = $NUM_GPUS
chunk = math.ceil(n / num_splits)

pred_dir = '$PREDICTIONS_DIR'
gpu_ids = '$GPU_IDS'.split(',')

merged_count = 0
for i, gpu_id in enumerate(gpu_ids):
    start = i * chunk
    gpu_pred_dir = os.path.join(pred_dir, f'gpu_{gpu_id}')

    if not os.path.exists(gpu_pred_dir):
        print(f'  WARNING: GPU {gpu_id} predictions not found')
        continue

    # List all json prediction files (numbered from 0 within each split)
    json_files = sorted([
        f for f in os.listdir(gpu_pred_dir)
        if f.endswith('.json') and f not in ['inference_summary.json', 'inference_stats.json']
    ], key=lambda x: int(x.replace('.json', '')))

    for jf in json_files:
        local_idx = int(jf.replace('.json', ''))
        global_idx = start + local_idx

        src = os.path.join(gpu_pred_dir, jf)
        dst = os.path.join(pred_dir, f'{global_idx}.json')
        shutil.copy2(src, dst)
        merged_count += 1

    print(f'  GPU {gpu_id}: {len(json_files)} files (global idx {start}-{start+len(json_files)-1})')

print(f'\nTotal merged: {merged_count} predictions')
print(f'Expected: {n} samples')
print(f'Coverage: {merged_count/n*100:.1f}%')
"

# Step 4: Clean up splits and per-gpu dirs
echo ""
echo "Step 4: Cleaning up temporary files..."
for i in "${!GPUS[@]}"; do
    rm -rf "${PREDICTIONS_DIR}/split_${i}"
    # Keep gpu logs but remove prediction dirs
    rm -rf "${PREDICTIONS_DIR}/gpu_${GPUS[$i]}"
done

echo ""
echo "========================================"
echo "Inference complete!"
echo "Predictions: $PREDICTIONS_DIR"
echo "========================================"
