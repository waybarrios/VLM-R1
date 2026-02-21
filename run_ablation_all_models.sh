#!/bin/bash
# Run ablation experiments for ALL 5 models
# Tests different similarity thresholds and embedding encoders across all models
# For CVPR Paper Section 4.4

# Configuration
DATASET_PATH="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
REASONING_BASE="/gpudata3/Wayner/reasoning"
OUTPUT_DIR="ablation_results"

# Models to evaluate
declare -A MODELS
MODELS["qwen25vl_32b"]="outputs_testing_qwen25vl_32b_64k"
MODELS["gemma3_12b"]="outputs_testing_gemma3_12b_64k"
MODELS["gemma3_4b"]="outputs_testing_gemma3_4b"
MODELS["llava7b"]="outputs_testing_llava7b_16"
MODELS["minicpm_v_8b"]="outputs_testing_minicpm_v_8b"

# Encoders to test
ENCODERS=(
    "all-MiniLM-L6-v2"
    "all-MiniLM-L12-v2"
    "all-mpnet-base-v2"
    "all-distilroberta-v1"
)

# Thresholds to test
THRESHOLDS="0.30 0.35 0.40 0.45 0.50"

# GPUs to use (distribute models across GPUs)
GPU_IDS=(0 1 2 3)

# Change to script directory
cd /gpudata3/Wayner/VLM-R1

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "ABLATION STUDY - ALL MODELS (CVPR Section 4.4)"
echo "============================================================"
echo "Models to evaluate: ${#MODELS[@]}"
echo "Encoders to test: ${#ENCODERS[@]}"
echo "Thresholds to test: 5 (0.30, 0.35, 0.40, 0.45, 0.50)"
echo ""
echo "Total experiments: $((${#MODELS[@]} * ${#ENCODERS[@]} * 5)) = $((5 * 4 * 5))"
echo ""
echo "Estimated time: ~3-5 hours (running sequentially)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Track progress
TOTAL_MODELS=${#MODELS[@]}
CURRENT_MODEL=0
FAILED_MODELS=()

# Process each model
for MODEL_NAME in "${!MODELS[@]}"; do
    CURRENT_MODEL=$((CURRENT_MODEL + 1))
    PREDICTIONS_DIR="$REASONING_BASE/${MODELS[$MODEL_NAME]}"
    OUTPUT_FILE="$OUTPUT_DIR/${MODEL_NAME}_ablation.json"

    # Select GPU in round-robin fashion
    GPU_ID=${GPU_IDS[$((CURRENT_MODEL % ${#GPU_IDS[@]}))]}
    DEVICE="cuda:$GPU_ID"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "[$CURRENT_MODEL/$TOTAL_MODELS] Processing: $MODEL_NAME"
    echo "════════════════════════════════════════════════════════════"
    echo "Predictions: $PREDICTIONS_DIR"
    echo "Output: $OUTPUT_FILE"
    echo "Device: $DEVICE"
    echo ""

    # Check if predictions directory exists
    if [ ! -d "$PREDICTIONS_DIR" ]; then
        echo "⚠️  Warning: Directory not found: $PREDICTIONS_DIR"
        echo "    Skipping..."
        FAILED_MODELS+=("$MODEL_NAME (not found)")
        continue
    fi

    # Count prediction files
    NUM_FILES=$(ls "$PREDICTIONS_DIR"/*.json 2>/dev/null | wc -l)
    echo "Found $NUM_FILES prediction files"

    if [ $NUM_FILES -eq 0 ]; then
        echo "⚠️  Warning: No JSON files found in $PREDICTIONS_DIR"
        echo "    Skipping..."
        FAILED_MODELS+=("$MODEL_NAME (no files)")
        continue
    fi

    echo ""
    echo "Running ablation experiments..."
    echo "  - Encoders: ${#ENCODERS[@]}"
    echo "  - Thresholds: 5"
    echo "  - Total: $((${#ENCODERS[@]} * 5)) experiments for this model"
    echo ""

    # Run ablation for this model
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
        echo "✓ Success! Results saved to: $OUTPUT_FILE"
    else
        echo ""
        echo "❌ Error: Ablation failed for $MODEL_NAME!"
        FAILED_MODELS+=("$MODEL_NAME (evaluation error)")
    fi
done

# Aggregate results across all models
echo ""
echo "════════════════════════════════════════════════════════════"
echo "AGGREGATING RESULTS ACROSS ALL MODELS"
echo "════════════════════════════════════════════════════════════"

AGGREGATE_FILE="$OUTPUT_DIR/ablation_aggregate.json"

python3 << 'PYTHON_SCRIPT'
import json
import glob
import numpy as np
from pathlib import Path

output_dir = "ablation_results"
json_files = sorted(glob.glob(f"{output_dir}/*_ablation.json"))

if not json_files:
    print("⚠️  No ablation result files found!")
    exit(1)

print(f"Found {len(json_files)} model results to aggregate")

# Load all results
all_model_results = {}
for json_file in json_files:
    model_name = Path(json_file).stem.replace("_ablation", "")
    with open(json_file, 'r') as f:
        data = json.load(f)
    all_model_results[model_name] = data['results']
    print(f"  ✓ Loaded: {model_name}")

# Aggregate by encoder and threshold
aggregated = {}

# Get unique configurations
configs = set()
for model_results in all_model_results.values():
    for result in model_results:
        config = (result['encoder'], result['threshold'])
        configs.add(config)

print(f"\nAggregating {len(configs)} unique configurations across {len(all_model_results)} models...")

# For each configuration, compute mean across models
for encoder, threshold in sorted(configs):
    key = f"{encoder}___{threshold}"

    f1_means = []
    precision_means = []
    recall_means = []

    for model_name, results in all_model_results.items():
        # Find matching config
        matching = [r for r in results if r['encoder'] == encoder and r['threshold'] == threshold]
        if matching:
            result = matching[0]
            f1_means.append(result['match_f1']['mean'])
            precision_means.append(result['precision']['mean'])
            recall_means.append(result['recall']['mean'])

    if f1_means:
        aggregated[key] = {
            'encoder': encoder,
            'threshold': threshold,
            'num_models': len(f1_means),
            'match_f1_across_models': {
                'mean': float(np.mean(f1_means)),
                'std': float(np.std(f1_means)),
                'min': float(np.min(f1_means)),
                'max': float(np.max(f1_means))
            },
            'precision_across_models': {
                'mean': float(np.mean(precision_means)),
                'std': float(np.std(precision_means))
            },
            'recall_across_models': {
                'mean': float(np.mean(recall_means)),
                'std': float(np.std(recall_means))
            }
        }

# Save aggregated results
output_data = {
    'metadata': {
        'num_models': len(all_model_results),
        'models': list(all_model_results.keys()),
        'num_configurations': len(aggregated),
        'encoders': sorted(list(set(c[0] for c in configs))),
        'thresholds': sorted(list(set(c[1] for c in configs)))
    },
    'aggregated_results': list(aggregated.values()),
    'per_model_results': all_model_results
}

with open('ablation_results/ablation_aggregate.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ Aggregated results saved to: ablation_results/ablation_aggregate.json")

# Print summary table
print("\n" + "="*110)
print("AGGREGATED SUMMARY (MEAN ACROSS 5 MODELS)")
print("="*110)
print(f"{'Encoder':<40} {'Threshold':>10} {'F1':>12} {'Precision':>12} {'Recall':>12}")
print("="*110)

for config in sorted(aggregated.values(), key=lambda x: (x['encoder'], x['threshold'])):
    print(f"{config['encoder']:<40} {config['threshold']:>10.2f} "
          f"{config['match_f1_across_models']['mean']:>12.4f} "
          f"{config['precision_across_models']['mean']:>12.4f} "
          f"{config['recall_across_models']['mean']:>12.4f}")

print("="*110)

PYTHON_SCRIPT

# Final summary
echo ""
echo "════════════════════════════════════════════════════════════"
echo "ABLATION STUDY COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "Models processed: $TOTAL_MODELS"
echo "Failed: ${#FAILED_MODELS[@]}"

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for failed_model in "${FAILED_MODELS[@]}"; do
        echo "  ❌ $failed_model"
    done
fi

echo ""
echo "Results:"
echo "  - Individual model results: $OUTPUT_DIR/*_ablation.json"
echo "  - Aggregated results: $OUTPUT_DIR/ablation_aggregate.json"
echo ""
echo "Next steps:"
echo "  1. Review ablation_aggregate.json for paper table"
echo "  2. Generate LaTeX table for Section 4.4"
echo "  3. Write ablation study text"
echo "════════════════════════════════════════════════════════════"

# Exit with error code if any failed
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    exit 1
fi
