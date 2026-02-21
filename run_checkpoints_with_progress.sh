#!/bin/bash
#
# Run inference on checkpoints - 4 at a time (one per GPU) with visible progress
#

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
TEST_DATASET="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
OUTPUT_BASE="${PROJECT_ROOT}/predictions_final"
BATCH_SIZE=4

# Add paths
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPUs to use
GPUS=(0 1 2)

# Checkpoint list
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
echo "Running inference on all checkpoints"
echo "========================================"
echo "Test dataset: $TEST_DATASET"
echo "Output: $OUTPUT_BASE"
echo "Batch size: $BATCH_SIZE"
echo "GPUs: ${GPUS[@]}"
echo "Total checkpoints: ${#CHECKPOINTS[@]}"
echo "========================================"
echo ""

mkdir -p "$OUTPUT_BASE"

# Save PID file for easy cleanup
PID_FILE="${OUTPUT_BASE}/running_pids.txt"
echo "# PIDs of running inference processes - $(date)" > "$PID_FILE"
echo "# To kill all: kill -9 \$(cat $PID_FILE | grep -v '^#' | awk '{print \$1}')" >> "$PID_FILE"
echo "# Main script PID: $$" >> "$PID_FILE"

# Process checkpoints in batches of 4 (one per GPU)
num_checkpoints=${#CHECKPOINTS[@]}
num_gpus=${#GPUS[@]}

for ((batch_start=0; batch_start<num_checkpoints; batch_start+=num_gpus)); do
    batch_end=$((batch_start + num_gpus))
    if [ $batch_end -gt $num_checkpoints ]; then
        batch_end=$num_checkpoints
    fi

    echo "========================================"
    echo "Processing batch: checkpoints $((batch_start+1)) to $batch_end of $num_checkpoints"
    echo "========================================"
    echo ""

    # Launch this batch in parallel
    pids=()
    for ((i=batch_start; i<batch_end; i++)); do
        checkpoint="${CHECKPOINTS[$i]}"
        gpu_idx=$((i - batch_start))
        gpu=${GPUS[$gpu_idx]}

        checkpoint_name=$(basename "$checkpoint")
        full_checkpoint_path="${PROJECT_ROOT}/${checkpoint}"
        predictions_dir="${OUTPUT_BASE}/${checkpoint_name}"
        log_file="${OUTPUT_BASE}/${checkpoint_name}.log"

        echo "[GPU $gpu] Starting: $checkpoint_name"

        # Run in background
        CUDA_VISIBLE_DEVICES=$gpu python "${PROJECT_ROOT}/inference/run_fast_inference.py" \
            --checkpoint_dir "$full_checkpoint_path" \
            --test_dataset_path "$TEST_DATASET" \
            --predictions_dir "$predictions_dir" \
            --device cuda \
            --batch_size $BATCH_SIZE \
            > "$log_file" 2>&1 &

        pid=$!
        pids+=($pid)

        # Save PID to file
        echo "$pid GPU$gpu $checkpoint_name" >> "$PID_FILE"

        # Brief delay to stagger startup
        sleep 2
    done

    echo ""
    echo "Waiting for batch to complete..."
    echo "Monitor progress with: tail -f ${OUTPUT_BASE}/checkpoint-*.log"
    echo "Or use: ./monitor_inference_progress.sh predictions_final"
    echo ""

    # Wait for this batch to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done

    echo ""
    echo "✓ Batch completed!"
    echo ""

    # Show summary for this batch
    for ((i=batch_start; i<batch_end; i++)); do
        checkpoint="${CHECKPOINTS[$i]}"
        checkpoint_name=$(basename "$checkpoint")
        predictions_dir="${OUTPUT_BASE}/${checkpoint_name}"

        if [ -f "${predictions_dir}/inference_stats.json" ]; then
            valid=$(python -c "import json; data=json.load(open('${predictions_dir}/inference_stats.json')); print(data.get('valid_predictions', 0))" 2>/dev/null)
            total=$(python -c "import json; data=json.load(open('${predictions_dir}/inference_stats.json')); print(data.get('processed_samples', 0))" 2>/dev/null)
            elapsed=$(python -c "import json; data=json.load(open('${predictions_dir}/inference_stats.json')); print(f\"{data.get('elapsed_time_seconds', 0)/60:.1f}\")" 2>/dev/null)
            if [ -n "$valid" ] && [ -n "$total" ]; then
                echo "  ✓ $checkpoint_name: $valid/$total valid predictions (${elapsed}min)"
            fi
        else
            echo "  ✗ $checkpoint_name: Failed or incomplete"
        fi
    done

    echo ""
done

echo ""
echo "========================================"
echo "✓ All checkpoints processed!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "Final Summary:"
total_valid=0
total_samples=0
for checkpoint in "${CHECKPOINTS[@]}"; do
    checkpoint_name=$(basename "$checkpoint")
    predictions_dir="${OUTPUT_BASE}/${checkpoint_name}"

    if [ -f "${predictions_dir}/inference_stats.json" ]; then
        valid=$(python -c "import json; data=json.load(open('${predictions_dir}/inference_stats.json')); print(data.get('valid_predictions', 0))" 2>/dev/null)
        total=$(python -c "import json; data=json.load(open('${predictions_dir}/inference_stats.json')); print(data.get('processed_samples', 0))" 2>/dev/null)
        if [ -n "$valid" ] && [ -n "$total" ]; then
            echo "  ✓ $checkpoint_name: $valid/$total valid"
            total_valid=$((total_valid + valid))
            total_samples=$((total_samples + total))
        fi
    fi
done

if [ $total_samples -gt 0 ]; then
    echo ""
    echo "Total: $total_valid/$total_samples valid predictions across all checkpoints"
fi

echo ""
echo "Log files: ${OUTPUT_BASE}/*.log"
echo ""
echo "PIDs saved to: $PID_FILE"
echo "To kill all processes: kill -9 \$(cat $PID_FILE | grep -v '^#' | awk '{print \$1}')"
echo "======================================"=="

# Clean up PID file
rm -f "$PID_FILE"
