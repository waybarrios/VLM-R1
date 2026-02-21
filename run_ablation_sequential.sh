#!/bin/bash
# Run ablation experiments SEQUENTIALLY (one model at a time)
# Each model uses 4 GPUs in parallel for faster processing
# Shows progress bars with time estimates

# Configuration
DATASET_PATH="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
REASONING_BASE="/gpudata3/Wayner/reasoning"
OUTPUT_DIR="ablation_results"
NUM_GPUS=4

# Models to evaluate (in order)
declare -a MODEL_NAMES=("qwen25vl_32b" "gemma3_12b" "gemma3_4b" "llava7b" "minicpm_v_8b")
declare -A MODEL_DIRS
MODEL_DIRS["qwen25vl_32b"]="outputs_testing_qwen25vl_32b_64k"
MODEL_DIRS["gemma3_12b"]="outputs_testing_gemma3_12b_64k"
MODEL_DIRS["gemma3_4b"]="outputs_testing_gemma3_4b"
MODEL_DIRS["llava7b"]="outputs_testing_llava7b_16"
MODEL_DIRS["minicpm_v_8b"]="outputs_testing_minicpm_v_8b"

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

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "ABLATION STUDY - SEQUENTIAL WITH MULTI-GPU (Section 4.4)"
echo "============================================================"
echo "Models: ${#MODEL_NAMES[@]}"
echo "GPUs per model: $NUM_GPUS (parallel processing within each model)"
echo "Encoders: ${#ENCODERS[@]}"
echo "Thresholds: 5"
echo ""
echo "Total configs per model: $((${#ENCODERS[@]} * 5)) = 20"
echo "Total experiments: $((${#MODEL_NAMES[@]} * ${#ENCODERS[@]} * 5)) = 100"
echo ""
echo "⚡ Each model uses 4 GPUs in parallel for ~4x speedup"
echo "Estimated time per model: ~20-30 minutes"
echo "Total estimated time: ~2-3 hours"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Track progress
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

# Process each model sequentially
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_DIR=${MODEL_DIRS[$MODEL_NAME]}
    PREDICTIONS_DIR="$REASONING_BASE/$MODEL_DIR"
    OUTPUT_FILE="$OUTPUT_DIR/${MODEL_NAME}_ablation.json"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "[$((SUCCESS_COUNT + FAILED_COUNT + 1))/${#MODEL_NAMES[@]}] Processing: $MODEL_NAME"
    echo "════════════════════════════════════════════════════════════"
    echo "Predictions: $PREDICTIONS_DIR"
    echo "Output: $OUTPUT_FILE"
    echo "Using $NUM_GPUS GPUs in parallel"
    echo ""

    # Check if predictions directory exists
    if [ ! -d "$PREDICTIONS_DIR" ]; then
        echo "⚠️  ERROR: Directory not found: $PREDICTIONS_DIR"
        echo "Skipping..."
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME (not found)")
        continue
    fi

    # Count prediction files
    NUM_FILES=$(ls "$PREDICTIONS_DIR"/*.json 2>/dev/null | wc -l)
    echo "Found $NUM_FILES prediction files"

    if [ $NUM_FILES -eq 0 ]; then
        echo "⚠️  ERROR: No JSON files found in $PREDICTIONS_DIR"
        echo "Skipping..."
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME (no files)")
        continue
    fi

    echo ""
    echo "Running ablation experiments..."
    echo "  - Configs: ${#ENCODERS[@]} encoders × 5 thresholds = $((${#ENCODERS[@]} * 5))"
    echo "  - Samples: $NUM_FILES"
    echo "  - GPUs: $NUM_GPUS (parallel)"
    echo ""

    # Run ablation for this model
    python3 run_ablation_sequential_multigpu.py \
        "$PREDICTIONS_DIR" \
        --dataset-path "$DATASET_PATH" \
        --output-file "$OUTPUT_FILE" \
        --num-gpus $NUM_GPUS \
        --encoders "${ENCODERS[@]}" \
        --thresholds $THRESHOLDS

    # Check if successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ SUCCESS! Results saved to: $OUTPUT_FILE"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "❌ FAILED! Check errors above"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME")
    fi

    echo ""
    echo "Progress: $((SUCCESS_COUNT + FAILED_COUNT))/${#MODEL_NAMES[@]} models completed"
    echo "════════════════════════════════════════════════════════════"
done

# Aggregate results
echo ""
echo "════════════════════════════════════════════════════════════"
echo "ALL MODELS COMPLETED"
echo "════════════════════════════════════════════════════════════"
echo "Success: $SUCCESS_COUNT / ${#MODEL_NAMES[@]}"
echo "Failed:  $FAILED_COUNT / ${#MODEL_NAMES[@]}"

if [ $FAILED_COUNT -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for failed_model in "${FAILED_MODELS[@]}"; do
        echo "  ❌ $failed_model"
    done
fi

if [ $SUCCESS_COUNT -eq 0 ]; then
    echo ""
    echo "❌ No successful results to aggregate!"
    exit 1
fi

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

print(f"Found {len(json_files)} model results to aggregate\n")

# Load all results
all_model_results = {}
for json_file in json_files:
    model_name = Path(json_file).stem.replace("_ablation", "")
    with open(json_file, 'r') as f:
        data = json.load(f)
    all_model_results[model_name] = data['results']
    print(f"  ✓ {model_name}: {len(data['results'])} configurations")

# Aggregate by encoder and threshold
aggregated = {}

# Get unique configurations
configs = set()
for model_results in all_model_results.values():
    for result in model_results:
        config = (result['encoder'], result['threshold'])
        configs.add(config)

print(f"\nAggregating {len(configs)} configurations across {len(all_model_results)} models...\n")

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

print(f"✓ Aggregated results saved to: ablation_results/ablation_aggregate.json\n")

# Print summary table
print("="*110)
print("AGGREGATED SUMMARY (MEAN ACROSS ALL MODELS)")
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
echo "✓ ABLATION STUDY COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "Results:"
echo "  - Individual models: $OUTPUT_DIR/*_ablation.json"
echo "  - Aggregated (all models): $OUTPUT_DIR/ablation_aggregate.json"
echo ""
echo "Next steps:"
echo "  1. Review ablation_aggregate.json"
echo "  2. Generate LaTeX table for Section 4.4"
echo "  3. Write Section 4.4 text"
echo "════════════════════════════════════════════════════════════"

# Exit with error if all failed
if [ $SUCCESS_COUNT -eq 0 ]; then
    exit 1
fi
