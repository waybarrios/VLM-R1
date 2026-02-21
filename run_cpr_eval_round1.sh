#!/bin/bash
#
# CPR Evaluation Round 1: Checkpoints 2000 and 2800
# Uses torch26 conda env. Run inside tmux.
#

set -e

PYTHON="/scratch/miniconda3/envs/torch26/bin/python"
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
CPR_TRAINING="output/grpo-cpr-20260204_235828"

export PYTHONPATH="${PROJECT_ROOT}/src/open-r1-multimodal/src:${PROJECT_ROOT}/mllm_evaluator:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"

echo "========================================"
echo "CPR EVALUATION ROUND 1"
echo "Started: $(date)"
echo "Python: $PYTHON"
echo "========================================"

# --- PHASE 1: INFERENCE ---

run_inference() {
    local CKPT_NUM=$1
    local GPU_START=$2
    local NUM_GPUS=4

    local CKPT_DIR="${PROJECT_ROOT}/${CPR_TRAINING}/checkpoint-${CKPT_NUM}"
    local PRED_DIR="${PROJECT_ROOT}/predictions/cpr-checkpoint-${CKPT_NUM}"

    echo ""
    echo "=== Inference: CPR-${CKPT_NUM} on GPUs ${GPU_START}-$((GPU_START+NUM_GPUS-1)) ==="

    mkdir -p "$PRED_DIR"

    PIDS=""
    for IDX in $(seq 0 $((NUM_GPUS-1))); do
        GPU=$((GPU_START+IDX))
        LOG="${PRED_DIR}/gpu${GPU}.log"

        CUDA_VISIBLE_DEVICES=$GPU $PYTHON inference/run_fast_inference.py \
            --checkpoint_dir "$CKPT_DIR" \
            --test_dataset_path "$TEST_DATASET" \
            --predictions_dir "$PRED_DIR" \
            --device cuda \
            --batch_size 4 \
            --chunk_total $NUM_GPUS \
            --chunk_index $IDX \
            > "$LOG" 2>&1 &

        PIDS="$PIDS $!"
        echo "  GPU $GPU (chunk $IDX): PID $!"
    done

    echo "  Waiting for CPR-${CKPT_NUM} inference to finish..."

    # Monitor progress
    while true; do
        ALL_DONE=true
        for PID in $PIDS; do
            if kill -0 "$PID" 2>/dev/null; then
                ALL_DONE=false
                break
            fi
        done

        if [ "$ALL_DONE" = true ]; then
            break
        fi

        N=$(find "$PRED_DIR" -name "*.json" ! -name "inference_*" ! -name "running_*" ! -name "gpu*" 2>/dev/null | wc -l)
        echo "  [$(date +%H:%M:%S)] CPR-${CKPT_NUM}: ${N}/6372 predictions"
        sleep 30
    done

    N=$(find "$PRED_DIR" -name "*.json" ! -name "inference_*" ! -name "running_*" ! -name "gpu*" 2>/dev/null | wc -l)
    echo "  CPR-${CKPT_NUM} inference DONE: ${N} predictions"
}

# Launch both checkpoints in parallel
echo ""
echo "--- Launching inference for CPR-2000 (GPUs 0-3) and CPR-2800 (GPUs 4-7) ---"

run_inference 2000 0 &
PID_2000=$!

run_inference 2800 4 &
PID_2800=$!

wait $PID_2000 $PID_2800

echo ""
echo "========================================"
echo "ALL INFERENCE COMPLETE - $(date)"
echo "========================================"

# --- PHASE 2: COMPUTE METRICS ---

echo ""
echo "=== Computing metrics for CPR-2000 ==="
$PYTHON compute_metrics.py predictions/cpr-checkpoint-2000 \
    --dataset-path "$TEST_DATASET" \
    --num-gpus 1 \
    --output-dir evaluations/cpr-checkpoint-2000

echo ""
echo "=== Computing metrics for CPR-2800 ==="
$PYTHON compute_metrics.py predictions/cpr-checkpoint-2800 \
    --dataset-path "$TEST_DATASET" \
    --num-gpus 1 \
    --output-dir evaluations/cpr-checkpoint-2800

# --- PHASE 3: RESULTS SUMMARY ---

echo ""
echo "========================================"
echo "========================================"
echo "       ROUND 1 RESULTS SUMMARY"
echo "========================================"
echo "========================================"
echo ""
echo "=== CPR-2000 ==="
cat evaluations/cpr-checkpoint-2000/cpr-checkpoint-2000/no_judge_summary.txt 2>/dev/null || echo "(no summary)"
echo ""
echo "=== CPR-2800 ==="
cat evaluations/cpr-checkpoint-2800/cpr-checkpoint-2800/no_judge_summary.txt 2>/dev/null || echo "(no summary)"
echo ""
echo "=== FULL CPR TREND ==="
echo "| Checkpoint | Accuracy | Match F1 | Notes           |"
echo "|------------|----------|----------|-----------------|"
echo "| CPR-400    | 39.96%   | 0.6156   | Peak F1         |"
echo "| CPR-500    | 36.30%   | 0.6016   | Dip             |"
echo "| CPR-700    | 38.31%   | 0.4084   | Soft collapse   |"
echo "| CPR-1000   | 37.71%   | 0.5205   | Recovery        |"
echo "| CPR-1400   | 40.07%   | 0.5013   | Prev best acc   |"
echo "| CPR-2000   | >>> CHECK ABOVE <<<  | NEW             |"
echo "| CPR-2800   | >>> CHECK ABOVE <<<  | NEW (final)     |"
echo ""
echo "Completed: $(date)"
echo "========================================"
