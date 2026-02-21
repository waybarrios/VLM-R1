#!/bin/bash
# Run ablation experiments - SIMPLE VERSION
# Processes one model at a time, one GPU, fully sequential
# More reliable, shows progress bars

DATASET_PATH="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
REASONING_BASE="/gpudata3/Wayner/reasoning"
OUTPUT_DIR="ablation_results"

# Models
declare -a MODEL_NAMES=("qwen25vl_32b" "gemma3_12b" "gemma3_4b" "llava7b" "minicpm_v_8b")
declare -A MODEL_DIRS
MODEL_DIRS["qwen25vl_32b"]="outputs_testing_qwen25vl_32b_64k"
MODEL_DIRS["gemma3_12b"]="outputs_testing_gemma3_12b_64k"
MODEL_DIRS["gemma3_4b"]="outputs_testing_gemma3_4b"
MODEL_DIRS["llava7b"]="outputs_testing_llava7b_16"
MODEL_DIRS["minicpm_v_8b"]="outputs_testing_minicpm_v_8b"

# Encoders
ENCODERS=("all-MiniLM-L6-v2" "all-MiniLM-L12-v2" "all-mpnet-base-v2" "all-distilroberta-v1")

# Thresholds
THRESHOLDS="0.30 0.35 0.40 0.45 0.50"

cd /gpudata3/Wayner/VLM-R1
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "ABLATION STUDY - SIMPLE SEQUENTIAL VERSION"
echo "============================================================"
echo "Models: ${#MODEL_NAMES[@]}"
echo "Encoders: ${#ENCODERS[@]}"
echo "Thresholds: 5"
echo "Total configs per model: 20"
echo "Total experiments: 100"
echo ""
echo "⏱️  Estimated time: ~4-6 hours (simple sequential)"
echo "Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

# Process each model
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
    echo ""

    if [ ! -d "$PREDICTIONS_DIR" ]; then
        echo "⚠️  ERROR: Directory not found"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME")
        continue
    fi

    NUM_FILES=$(ls "$PREDICTIONS_DIR"/*.json 2>/dev/null | wc -l)
    echo "Found $NUM_FILES prediction files"

    if [ $NUM_FILES -eq 0 ]; then
        echo "⚠️  ERROR: No JSON files"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME")
        continue
    fi

    echo ""
    echo "Running 20 ablation experiments (4 encoders × 5 thresholds)..."
    echo ""

    # Run ablation
    python3 run_ablation_simple.py \
        "$PREDICTIONS_DIR" \
        --dataset-path "$DATASET_PATH" \
        --output-file "$OUTPUT_FILE" \
        --device "cuda:0" \
        --encoders "${ENCODERS[@]}" \
        --thresholds $THRESHOLDS

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ SUCCESS!"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "❌ FAILED!"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME")
    fi

    echo "Progress: $((SUCCESS_COUNT + FAILED_COUNT))/${#MODEL_NAMES[@]} models"
    echo "════════════════════════════════════════════════════════════"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "ALL MODELS COMPLETED"
echo "════════════════════════════════════════════════════════════"
echo "Success: $SUCCESS_COUNT / ${#MODEL_NAMES[@]}"
echo "Failed:  $FAILED_COUNT / ${#MODEL_NAMES[@]}"

if [ $FAILED_COUNT -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for failed in "${FAILED_MODELS[@]}"; do
        echo "  ❌ $failed"
    done
fi

if [ $SUCCESS_COUNT -eq 0 ]; then
    echo "❌ No successful results!"
    exit 1
fi

# Aggregate
echo ""
echo "════════════════════════════════════════════════════════════"
echo "AGGREGATING RESULTS"
echo "════════════════════════════════════════════════════════════"

python3 << 'PYTHON_SCRIPT'
import json
import glob
import numpy as np
from pathlib import Path

output_dir = "ablation_results"
json_files = sorted(glob.glob(f"{output_dir}/*_ablation.json"))

if not json_files:
    print("⚠️  No result files!")
    exit(1)

print(f"Found {len(json_files)} model results\n")

all_model_results = {}
for json_file in json_files:
    model_name = Path(json_file).stem.replace("_ablation", "")
    with open(json_file, 'r') as f:
        data = json.load(f)
    all_model_results[model_name] = data['results']
    print(f"  ✓ {model_name}: {len(data['results'])} configs")

aggregated = {}
configs = set()
for model_results in all_model_results.values():
    for result in model_results:
        config = (result['encoder'], result['threshold'])
        configs.add(config)

print(f"\nAggregating {len(configs)} configs across {len(all_model_results)} models...\n")

for encoder, threshold in sorted(configs):
    key = f"{encoder}___{threshold}"
    f1_means = []
    precision_means = []
    recall_means = []

    for model_name, results in all_model_results.items():
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

print(f"✓ Saved: ablation_results/ablation_aggregate.json\n")

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

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✓ COMPLETE"
echo "════════════════════════════════════════════════════════════"
echo "Results:"
echo "  - Individual: $OUTPUT_DIR/*_ablation.json"
echo "  - Aggregated: $OUTPUT_DIR/ablation_aggregate.json"
echo "════════════════════════════════════════════════════════════"
