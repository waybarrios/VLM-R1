#!/bin/bash
# Script to compute metrics WITHOUT judge (faster, no LLM)
# Processes multiple models in one run
# Usage: bash run_compute_metrics_no_judge.sh

# Default values
DATASET_PATH="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
OUTPUT_DIR="./metrics_results"
NUM_GPUS=4
REASONING_BASE="/gpudata3/Wayner/reasoning"

# List of models to evaluate
MODELS=(
    "outputs_testing_llava7b_16"
    "outputs_testing_gemma3_12b_64k"
    "outputs_testing_qwen25vl_32b_64k"
    "outputs_testing_gemma3_4b"
)

# Set GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Change to script directory
cd /gpudata3/Wayner/VLM-R1

echo "============================================================"
echo "BATCH EVALUATION - NO JUDGE MODE"
echo "============================================================"
echo "Dataset:      $DATASET_PATH"
echo "Output Dir:   $OUTPUT_DIR"
echo "GPUs:         $NUM_GPUS (0,1,2,3)"
echo "Models to evaluate: ${#MODELS[@]}"
echo "============================================================"
echo ""

# Track success/failure
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

# Process each model
for MODEL_NAME in "${MODELS[@]}"; do
    PREDICTIONS_DIR="$REASONING_BASE/$MODEL_NAME"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "[$((SUCCESS_COUNT + FAILED_COUNT + 1))/${#MODELS[@]}] Processing: $MODEL_NAME"
    echo "════════════════════════════════════════════════════════════"

    # Check if predictions directory exists
    if [ ! -d "$PREDICTIONS_DIR" ]; then
        echo "⚠️  Warning: Directory not found: $PREDICTIONS_DIR"
        echo "    Skipping..."
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME (not found)")
        continue
    fi

    echo "Predictions:  $PREDICTIONS_DIR"
    echo "Output:       $OUTPUT_DIR/$MODEL_NAME/"
    echo ""

    # Run without judge (faster, no LLM calls for accuracy)
    python compute_metrics.py \
        "$PREDICTIONS_DIR" \
        --dataset-path "$DATASET_PATH" \
        --num-gpus $NUM_GPUS \
        --output-dir "$OUTPUT_DIR"

    # Check if successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Success! Results saved to:"
        echo "  $OUTPUT_DIR/$MODEL_NAME/no_judge_metrics.csv"
        echo "  $OUTPUT_DIR/$MODEL_NAME/no_judge_summary.txt"
        echo "  $OUTPUT_DIR/$MODEL_NAME/no_judge_summary.json"

        # Show quick summary if file exists
        if [ -f "$OUTPUT_DIR/$MODEL_NAME/no_judge_summary.txt" ]; then
            echo ""
            echo "Quick Summary for $MODEL_NAME:"
            echo "------------------------------------------------------------"
            grep -A 5 "ACCURACY METRICS" "$OUTPUT_DIR/$MODEL_NAME/no_judge_summary.txt" | head -6
            grep -A 5 "MATCH F1 METRICS" "$OUTPUT_DIR/$MODEL_NAME/no_judge_summary.txt" | head -6
        fi

        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "❌ Error: Evaluation failed for $MODEL_NAME!"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME (evaluation error)")
    fi
done

# Final summary
echo ""
echo "════════════════════════════════════════════════════════════"
echo "BATCH EVALUATION COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "Total models: ${#MODELS[@]}"
echo "Successful:   $SUCCESS_COUNT"
echo "Failed:       $FAILED_COUNT"

if [ $FAILED_COUNT -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for failed_model in "${FAILED_MODELS[@]}"; do
        echo "  ❌ $failed_model"
    done
fi

echo ""
echo "Results directory: $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

# Exit with error code if any failed
if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi
