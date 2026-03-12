#!/bin/bash
# Evaluate InternVL3.5 CPR Curriculum checkpoint-200 on GPUs 4-7
# Run: tmux new-session -d -s eval_internvl bash eval_internvl_cpr_ckpt200.sh

set -e

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate internvl35

export HF_HOME=/gpudata3/hf_cache
export PYTHONPATH="/gpudata3/Wayner/VLM-R1/src/open-r1-multimodal/src:/gpudata3/Wayner/VLM-R1/mllm_evaluator:${PYTHONPATH}"

PYTHON="/scratch/miniconda3/envs/internvl35/bin/python"
PROJECT="/gpudata3/Wayner/VLM-R1"
CHECKPOINT="${PROJECT}/output/internvl35_cpr_curriculum_20260309_215942/checkpoint-200"
RESULTS_DIR="${PROJECT}/evaluations/internvl35_cpr_curriculum_ckpt200"
PREDICTIONS_DIR="${RESULTS_DIR}/predictions"
GPU_IDS="4,5,6,7"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"

echo "=============================================="
echo "InternVL3.5 CPR Curriculum — Checkpoint-200 Eval"
echo "Started: $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "GPUs: $GPU_IDS"
echo "=============================================="

mkdir -p "$PREDICTIONS_DIR"

# ── Step 1: Split dataset across 4 GPUs ──
echo ""
echo "Step 1: Splitting dataset..."
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

$PYTHON -c "
from datasets import load_from_disk
import os, math

dataset = load_from_disk('$TEST_DATASET')
n = len(dataset)
chunk = math.ceil(n / $NUM_GPUS)

for i in range($NUM_GPUS):
    start = i * chunk
    end = min(start + chunk, n)
    split = dataset.select(range(start, end))
    split_dir = '${PREDICTIONS_DIR}/split_{}'.format(i)
    split.save_to_disk(split_dir)
    print(f'Split {i}: samples {start}-{end} ({len(split)} samples) -> {split_dir}')

print(f'\nTotal: {n} samples split into $NUM_GPUS chunks')
"

# ── Step 2: Launch inference on each GPU ──
echo ""
echo "Step 2: Launching inference on ${NUM_GPUS} GPUs..."

PIDS=()
for i in "${!GPUS[@]}"; do
    GPU_ID="${GPUS[$i]}"
    SPLIT_DIR="${PREDICTIONS_DIR}/split_${i}"
    GPU_PRED_DIR="${PREDICTIONS_DIR}/gpu_${GPU_ID}"

    mkdir -p "$GPU_PRED_DIR"

    echo "  Launching GPU ${GPU_ID} (split ${i})..."

    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON ${PROJECT}/inference/run_simple_vqa_internvl.py \
        --checkpoint_dir "$CHECKPOINT" \
        --test_dataset_path "$SPLIT_DIR" \
        --predictions_dir "$GPU_PRED_DIR" \
        --gpu 0 \
        --max_anyres_num 12 \
        > "${PREDICTIONS_DIR}/gpu_${GPU_ID}.log" 2>&1 &

    PIDS+=($!)
    echo "    PID: ${PIDS[-1]}"
done

echo ""
echo "All ${NUM_GPUS} GPUs launched. Waiting for completion..."

# Wait for all
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

# ── Step 3: Merge results ──
echo ""
echo "Step 3: Merging results..."

$PYTHON -c "
import os, json, math, shutil
from datasets import load_from_disk

dataset = load_from_disk('$TEST_DATASET')
n = len(dataset)
chunk = math.ceil(n / $NUM_GPUS)

pred_dir = '$PREDICTIONS_DIR'
gpu_ids = '$GPU_IDS'.split(',')

merged_count = 0
for i, gpu_id in enumerate(gpu_ids):
    start = i * chunk
    gpu_pred_dir = os.path.join(pred_dir, f'gpu_{gpu_id}')

    if not os.path.exists(gpu_pred_dir):
        print(f'  WARNING: GPU {gpu_id} predictions not found')
        continue

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

# ── Step 4: Cleanup temp files ──
echo ""
echo "Step 4: Cleaning up..."
for i in "${!GPUS[@]}"; do
    rm -rf "${PREDICTIONS_DIR}/split_${i}"
    rm -rf "${PREDICTIONS_DIR}/gpu_${GPUS[$i]}"
done

# ── Step 5: Compute metrics ──
echo ""
echo "Step 5: Computing metrics (distilroberta, threshold=0.35)..."
$PYTHON ${PROJECT}/compute_metrics.py \
    "${PREDICTIONS_DIR}" \
    --output-dir "${RESULTS_DIR}" \
    --num-gpus 1

echo ""
echo "=============================================="
echo "EVALUATION COMPLETE at $(date)"
echo "=============================================="
echo ""

# Show summary
if [ -f "${RESULTS_DIR}/no_judge_summary.json" ]; then
    echo "Results:"
    python3 -c "
import json
with open('${RESULTS_DIR}/no_judge_summary.json') as f:
    d = json.load(f)
acc = d['accuracy_metrics']
mf1 = d['match_f1_metrics']
print(f\"  Accuracy:  {acc['overall_accuracy']*100:.2f}% ({acc['correct_samples']}/{acc['total_samples']})\")
print(f\"  Match F1:  {mf1['average_match_f1']:.4f}\")
print(f\"  Precision: {mf1['average_precision']:.4f}\")
print(f\"  Recall:    {mf1['average_recall']:.4f}\")
"
fi
