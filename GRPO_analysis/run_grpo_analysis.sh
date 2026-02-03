#!/bin/bash
# GRPO Analysis Script
# Evaluates baseline + all GRPO checkpoints with custom MatchF1 settings
# Model: all-distilroberta-v1, Threshold: 0.35
# Uses GPUs 0 and 1

# Configuration
DATASET_PATH="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
OUTPUT_DIR="./GRPO_analysis/results"
NUM_GPUS=2
MODEL_NAME="all-distilroberta-v1"
THRESHOLD=0.35

# Activate conda environment
source /scratch/miniconda3/etc/profile.d/conda.sh
conda activate reasoning

# Set GPUs to use
export CUDA_VISIBLE_DEVICES=0,1

# Change to project directory
cd /gpudata3/Wayner/VLM-R1

echo "============================================================"
echo "GRPO ANALYSIS - CHECKPOINT EVALUATION"
echo "============================================================"
echo "Dataset:      $DATASET_PATH"
echo "Output Dir:   $OUTPUT_DIR"
echo "GPUs:         $NUM_GPUS (0,1)"
echo "Model:        $MODEL_NAME"
echo "Threshold:    $THRESHOLD"
echo "============================================================"
echo ""

# Track success/failure
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

# Baseline model
BASELINE_DIR="/gpudata3/Wayner/reasoning/outputs_testing_qwen25vl_3b"

# Checkpoint directories
CHECKPOINTS_DIR="/gpudata3/Wayner/VLM-R1/predictions_final"

# Get all checkpoints sorted by step number
CHECKPOINTS=($(ls -d ${CHECKPOINTS_DIR}/checkpoint-* | sort -V))

# Add baseline to the front
ALL_MODELS=("$BASELINE_DIR" "${CHECKPOINTS[@]}")

echo "Found ${#ALL_MODELS[@]} models to evaluate:"
echo "  1. Baseline: $BASELINE_DIR"
for i in "${!CHECKPOINTS[@]}"; do
    checkpoint_name=$(basename "${CHECKPOINTS[$i]}")
    echo "  $((i+2)). $checkpoint_name"
done
echo ""
echo "============================================================"
echo ""

# Process each model
for MODEL_DIR in "${ALL_MODELS[@]}"; do
    MODEL_NAME_DISPLAY=$(basename "$MODEL_DIR")

    # Special display for baseline
    if [ "$MODEL_DIR" == "$BASELINE_DIR" ]; then
        MODEL_NAME_DISPLAY="baseline_qwen25vl_3b"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "[$((SUCCESS_COUNT + FAILED_COUNT + 1))/${#ALL_MODELS[@]}] Processing: $MODEL_NAME_DISPLAY"
    echo "════════════════════════════════════════════════════════════"

    # Check if predictions directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "⚠️  Warning: Directory not found: $MODEL_DIR"
        echo "    Skipping..."
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME_DISPLAY (not found)")
        continue
    fi

    echo "Predictions:  $MODEL_DIR"
    echo "Output:       $OUTPUT_DIR/$MODEL_NAME_DISPLAY/"
    echo ""

    # Run evaluation with custom settings
    python GRPO_analysis/compute_grpo_metrics.py \
        "$MODEL_DIR" \
        --dataset-path "$DATASET_PATH" \
        --model-name "$MODEL_NAME" \
        --threshold $THRESHOLD \
        --num-gpus $NUM_GPUS \
        --output-dir "$OUTPUT_DIR"

    # Check if successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Success! Results saved to:"
        echo "  $OUTPUT_DIR/$MODEL_NAME_DISPLAY/metrics.csv"
        echo "  $OUTPUT_DIR/$MODEL_NAME_DISPLAY/summary.txt"
        echo "  $OUTPUT_DIR/$MODEL_NAME_DISPLAY/summary.json"

        # Show quick summary if file exists
        if [ -f "$OUTPUT_DIR/$MODEL_NAME_DISPLAY/summary.txt" ]; then
            echo ""
            echo "Quick Summary for $MODEL_NAME_DISPLAY:"
            echo "------------------------------------------------------------"
            grep -A 4 "ACCURACY METRICS" "$OUTPUT_DIR/$MODEL_NAME_DISPLAY/summary.txt" | tail -4
            grep -A 4 "MATCH F1 METRICS" "$OUTPUT_DIR/$MODEL_NAME_DISPLAY/summary.txt" | tail -4
        fi

        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "❌ Error: Evaluation failed for $MODEL_NAME_DISPLAY!"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME_DISPLAY (evaluation error)")
    fi
done

# Final summary
echo ""
echo "════════════════════════════════════════════════════════════"
echo "BATCH EVALUATION COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "Total models: ${#ALL_MODELS[@]}"
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
echo ""
echo "Next step: Run consolidation script to generate comparison table"
echo "  bash GRPO_analysis/consolidate_results.sh"
echo "════════════════════════════════════════════════════════════"

# Exit with error code if any failed
if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi
