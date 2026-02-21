#!/bin/bash
# ==============================================================================
# FINAL EVALUATION SCRIPT FOR CRYSTAL BENCHMARK
# ==============================================================================
# Computes final metrics for all models using DistilRoBERTa-v1 (τ=0.35)
# Generates consolidated tables and analysis for CVPR 2026 paper
# ==============================================================================

# Configuration
DATASET_PATH="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
OUTPUT_DIR="/gpudata3/Wayner/VLM-R1/final_table"
REASONING_BASE="/gpudata3/Wayner/reasoning"
NUM_GPUS=4

# Encoder settings (from paper ablation study)
ENCODER="all-distilroberta-v1"
THRESHOLD=0.35

# Set GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Change to script directory
cd /gpudata3/Wayner/VLM-R1

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "CRYSTAL BENCHMARK - FINAL EVALUATION"
echo "============================================================"
echo "Dataset:      $DATASET_PATH"
echo "Output Dir:   $OUTPUT_DIR"
echo "Encoder:      $ENCODER (from ablation study)"
echo "Threshold:    $THRESHOLD"
echo "GPUs:         $NUM_GPUS (0,1,2,3)"
echo "============================================================"
echo ""

# Define all models with their display names
declare -A MODELS=(
    # Latest models
    ["outputs_testing_qwen3vl_8b"]="Qwen3-VL-8B"
    ["outputs_testing_internvl35_8b"]="InternVL3.5-8B"
    ["outputs_testing_internvl35_4b"]="InternVL3.5-4B"
    ["outputs_testing_internvl35_2b"]="InternVL3.5-2B"
    ["outputs_testing_internvl35_1b"]="InternVL3.5-1B"
    ["outputs_testing_internvl35_38b"]="InternVL3.5-38B"
    ["outputs_testing_qwen3vl_2b"]="Qwen3-VL-2B"

    # Qwen2.5 models
    ["outputs_testing_qwen25vl_7b"]="Qwen2.5-VL-7B"
    ["outputs_testing_qwen25vl_3b"]="Qwen2.5-VL-3B"
    ["outputs_testing_qwen25vl_32b_64k"]="Qwen2.5-VL-32B"

    # Other baseline models
    ["outputs_testing_llava7b_16"]="LLaVA-v1.6-7B"
    ["outputs_testing_gemma3_12b_64k"]="Gemma3-12B"
    ["outputs_testing_gemma3_4b"]="Gemma3-4B"
    ["outputs_testing_minicpm_v_8b"]="MiniCPMv2.6-8B"
)

# Track progress
TOTAL_MODELS=${#MODELS[@]}
CURRENT=0
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

# Create temporary results file
TEMP_RESULTS="$OUTPUT_DIR/temp_results.txt"
> "$TEMP_RESULTS"

# Process each model
for MODEL_DIR in "${!MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    MODEL_NAME="${MODELS[$MODEL_DIR]}"
    PREDICTIONS_DIR="$REASONING_BASE/$MODEL_DIR"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "[$CURRENT/$TOTAL_MODELS] Evaluating: $MODEL_NAME"
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
    echo "Model Name:   $MODEL_NAME"
    echo ""

    # Run evaluation with DistilRoBERTa-v1 and threshold 0.35
    python compute_metrics_final.py \
        "$PREDICTIONS_DIR" \
        --dataset-path "$DATASET_PATH" \
        --encoder "$ENCODER" \
        --threshold $THRESHOLD \
        --num-gpus $NUM_GPUS \
        --output-dir "$OUTPUT_DIR" \
        --model-name "$MODEL_NAME"

    # Check if successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Success! Results saved for $MODEL_NAME"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Extract summary stats and append to temp file
        SUMMARY_FILE="$OUTPUT_DIR/${MODEL_DIR}/metrics_summary.json"
        if [ -f "$SUMMARY_FILE" ]; then
            echo "$MODEL_NAME|$SUMMARY_FILE" >> "$TEMP_RESULTS"
        fi
    else
        echo ""
        echo "❌ Error: Evaluation failed for $MODEL_NAME!"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME (evaluation error)")
    fi

    # Show progress
    echo ""
    echo "Progress: $CURRENT/$TOTAL_MODELS completed ($SUCCESS_COUNT successful, $FAILED_COUNT failed)"
done

# Generate consolidated table
echo ""
echo "════════════════════════════════════════════════════════════"
echo "GENERATING CONSOLIDATED RESULTS TABLE"
echo "════════════════════════════════════════════════════════════"

python generate_consolidated_table.py \
    --results-file "$TEMP_RESULTS" \
    --output-dir "$OUTPUT_DIR"

# Generate analysis
echo ""
echo "════════════════════════════════════════════════════════════"
echo "GENERATING RESULTS ANALYSIS"
echo "════════════════════════════════════════════════════════════"

python analyze_final_results.py \
    --output-dir "$OUTPUT_DIR"

# Final summary
echo ""
echo "════════════════════════════════════════════════════════════"
echo "FINAL EVALUATION COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "Total models: $TOTAL_MODELS"
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
echo "  - Individual model results: $OUTPUT_DIR/<model_dir>/"
echo "  - Consolidated table: $OUTPUT_DIR/consolidated_results.csv"
echo "  - Analysis report: $OUTPUT_DIR/results_analysis.txt"
echo "════════════════════════════════════════════════════════════"

# Clean up
rm -f "$TEMP_RESULTS"

# Exit with error code if any failed
if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi
