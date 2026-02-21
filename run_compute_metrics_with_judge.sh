#!/bin/bash
# Script to compute metrics WITH judge (use_judge=True, USE LLM)
# Usage: bash run_compute_metrics_with_judge.sh [predictions_dir]
# Example: bash run_compute_metrics_with_judge.sh /gpudata3/Wayner/reasoning/outputs_testing_gemma3_4b

# Default values
DATASET_PATH="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
OUTPUT_DIR="./metrics_results"
NUM_GPUS=4
JUDGE_MODEL="gpt-oss:120b"

# Check if predictions directory is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: bash run_compute_metrics_with_judge.sh <predictions_dir>"
    echo ""
    echo "Examples:"
    echo "  bash run_compute_metrics_with_judge.sh /gpudata3/Wayner/reasoning/outputs_testing_gemma3_4b"
    echo "  bash run_compute_metrics_with_judge.sh /gpudata3/Wayner/reasoning/outputs_testing_gemma3_12b_64k"
    echo "  bash run_compute_metrics_with_judge.sh /gpudata3/Wayner/reasoning/outputs_testing_llava7b_16"
    echo ""
    echo "Available prediction directories:"
    ls -d /gpudata3/Wayner/reasoning/outputs_testing_* 2>/dev/null || echo "  (none found)"
    exit 1
fi

PREDICTIONS_DIR="$1"

# Check if predictions directory exists
if [ ! -d "$PREDICTIONS_DIR" ]; then
    echo "❌ Error: Predictions directory does not exist: $PREDICTIONS_DIR"
    exit 1
fi

# Get model name from directory
MODEL_NAME=$(basename "$PREDICTIONS_DIR")

echo "============================================================"
echo "Computing metrics WITH judge (use_judge=True, USE LLM)"
echo "============================================================"
echo "Model:        $MODEL_NAME"
echo "Predictions:  $PREDICTIONS_DIR"
echo "Dataset:      $DATASET_PATH"
echo "Output:       $OUTPUT_DIR/$MODEL_NAME/"
echo "Judge Model:  $JUDGE_MODEL"
echo "GPUs:         $NUM_GPUS (0,1,2,3)"
echo "============================================================"
echo ""

# Set GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Change to script directory
cd /gpudata3/Wayner/VLM-R1

# Run with judge (uses LLM for semantic similarity in accuracy)
python compute_metrics.py \
    "$PREDICTIONS_DIR" \
    --dataset-path "$DATASET_PATH" \
    --num-gpus $NUM_GPUS \
    --output-dir "$OUTPUT_DIR" \
    --use-judge \
    --judge-model "$JUDGE_MODEL"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Done! Results saved to:"
    echo "  $OUTPUT_DIR/$MODEL_NAME/with_judge_metrics.csv"
    echo "  $OUTPUT_DIR/$MODEL_NAME/with_judge_summary.txt"
    echo "  $OUTPUT_DIR/$MODEL_NAME/with_judge_summary.json"
    echo "============================================================"

    # Show quick summary if file exists
    if [ -f "$OUTPUT_DIR/$MODEL_NAME/with_judge_summary.txt" ]; then
        echo ""
        echo "Quick Summary:"
        echo "============================================================"
        grep -A 5 "ACCURACY METRICS" "$OUTPUT_DIR/$MODEL_NAME/with_judge_summary.txt" | head -6
        grep -A 5 "MATCH F1 METRICS" "$OUTPUT_DIR/$MODEL_NAME/with_judge_summary.txt" | head -6
    fi
else
    echo ""
    echo "❌ Error: Evaluation failed!"
    exit 1
fi
