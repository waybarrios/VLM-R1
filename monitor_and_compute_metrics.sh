#!/bin/bash
#
# Monitor running inference jobs and compute metrics when done.
# Run inside tmux so it survives terminal disconnects.
#

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
export PYTHONPATH="${PROJECT_ROOT}/src/open-r1-multimodal/src:${PROJECT_ROOT}/mllm_evaluator:${PYTHONPATH}"

PIDS_2000="3358995 3358996 3358997 3358998"
PIDS_2800="3358999 3359000 3359001 3359002"

check_alive() {
    local pids="$1"
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

count_predictions() {
    local dir="$1"
    find "$dir" -name "*.json" ! -name "inference_*" ! -name "running_*" 2>/dev/null | wc -l
}

echo "========================================"
echo "CPR Checkpoint Evaluation Monitor"
echo "Started: $(date)"
echo "========================================"
echo "CPR-2000 PIDs: $PIDS_2000"
echo "CPR-2800 PIDs: $PIDS_2800"
echo "========================================"
echo ""

CPR2000_DONE=false
CPR2800_DONE=false

while true; do
    # Check CPR-2000
    if [ "$CPR2000_DONE" = false ]; then
        if check_alive "$PIDS_2000"; then
            n=$(count_predictions "${PROJECT_ROOT}/predictions/cpr-checkpoint-2000")
            echo "[$(date +%H:%M:%S)] CPR-2000: running... ($n/6372 predictions)"
        else
            n=$(count_predictions "${PROJECT_ROOT}/predictions/cpr-checkpoint-2000")
            echo ""
            echo "========================================"
            echo "[$(date +%H:%M:%S)] CPR-2000: INFERENCE DONE ($n predictions)"
            echo "Computing metrics..."
            echo "========================================"
            cd "$PROJECT_ROOT"
            python compute_metrics.py predictions/cpr-checkpoint-2000 \
                --dataset-path "$TEST_DATASET" \
                --num-gpus 1 \
                --output-dir evaluations/cpr-checkpoint-2000
            echo ""
            echo "=== CPR-2000 RESULTS ==="
            cat evaluations/cpr-checkpoint-2000/cpr-checkpoint-2000/no_judge_summary.txt 2>/dev/null || echo "(summary not found)"
            echo "========================"
            echo ""
            CPR2000_DONE=true
        fi
    fi

    # Check CPR-2800
    if [ "$CPR2800_DONE" = false ]; then
        if check_alive "$PIDS_2800"; then
            n=$(count_predictions "${PROJECT_ROOT}/predictions/cpr-checkpoint-2800")
            echo "[$(date +%H:%M:%S)] CPR-2800: running... ($n/6372 predictions)"
        else
            n=$(count_predictions "${PROJECT_ROOT}/predictions/cpr-checkpoint-2800")
            echo ""
            echo "========================================"
            echo "[$(date +%H:%M:%S)] CPR-2800: INFERENCE DONE ($n predictions)"
            echo "Computing metrics..."
            echo "========================================"
            cd "$PROJECT_ROOT"
            python compute_metrics.py predictions/cpr-checkpoint-2800 \
                --dataset-path "$TEST_DATASET" \
                --num-gpus 1 \
                --output-dir evaluations/cpr-checkpoint-2800
            echo ""
            echo "=== CPR-2800 RESULTS ==="
            cat evaluations/cpr-checkpoint-2800/cpr-checkpoint-2800/no_judge_summary.txt 2>/dev/null || echo "(summary not found)"
            echo "========================"
            echo ""
            CPR2800_DONE=true
        fi
    fi

    # Both done?
    if [ "$CPR2000_DONE" = true ] && [ "$CPR2800_DONE" = true ]; then
        echo ""
        echo "========================================"
        echo "ALL EVALUATIONS COMPLETE - $(date)"
        echo "========================================"
        echo ""
        echo "=== FULL CPR TREND ==="
        echo "| Checkpoint | Accuracy | Match F1 | (previously evaluated) |"
        echo "|------------|----------|----------|------------------------|"
        echo "| CPR-400    | 39.96%   | 0.6156   | Peak F1                |"
        echo "| CPR-500    | 36.30%   | 0.6016   | Dip                    |"
        echo "| CPR-700    | 38.31%   | 0.4084   | Soft collapse           |"
        echo "| CPR-1000   | 37.71%   | 0.5205   | Recovery               |"
        echo "| CPR-1400   | 40.07%   | 0.5013   | Best accuracy          |"
        echo "| CPR-2000   | (see above) |       | NEW                    |"
        echo "| CPR-2800   | (see above) |       | NEW                    |"
        echo ""
        echo "Results saved to:"
        echo "  evaluations/cpr-checkpoint-2000/"
        echo "  evaluations/cpr-checkpoint-2800/"
        break
    fi

    sleep 30
done
