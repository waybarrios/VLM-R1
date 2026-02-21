#!/bin/bash
# Run ablation experiments for Section 4.4 of the paper
# Tests different similarity thresholds and embedding encoders

# Configuration
PREDICTIONS_DIR="/gpudata3/Wayner/reasoning/outputs_testing_qwen25vl_32b_64k"  # Best model
DATASET_PATH="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
OUTPUT_FILE="ablation_results.json"
DEVICE="cuda:0"

# Encoders to test
ENCODERS=(
    "all-MiniLM-L6-v2"
    "all-MiniLM-L12-v2"
    "all-mpnet-base-v2"
    "all-distilroberta-v1"
)

# Thresholds to test
THRESHOLDS="0.30 0.35 0.40 0.45 0.50"

# Change to script directory
cd /gpudata3/Wayner/VLM-R1

echo "============================================================"
echo "ABLATION STUDY FOR CVPR PAPER - SECTION 4.4"
echo "============================================================"
echo "Model: Qwen2.5-VL-32B (best performing model)"
echo "Predictions: $PREDICTIONS_DIR"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_FILE"
echo "Device: $DEVICE"
echo ""
echo "Encoders to test (${#ENCODERS[@]}):"
for encoder in "${ENCODERS[@]}"; do
    echo "  - $encoder"
done
echo ""
echo "Thresholds to test: $THRESHOLDS"
echo ""
echo "Total experiments: $((${#ENCODERS[@]} * 5))"
echo "============================================================"
echo ""

# Make script executable
chmod +x run_ablation_experiments.py

# Run ablation experiments
python3 run_ablation_experiments.py \
    "$PREDICTIONS_DIR" \
    --dataset-path "$DATASET_PATH" \
    --output-file "$OUTPUT_FILE" \
    --device "$DEVICE" \
    --encoders "${ENCODERS[@]}" \
    --thresholds $THRESHOLDS

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Ablation experiments completed successfully!"
    echo "============================================================"
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Review results in $OUTPUT_FILE"
    echo "  2. Generate LaTeX table for paper"
    echo "  3. Write Section 4.4 based on findings"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Ablation experiments failed!"
    echo "============================================================"
    exit 1
fi
