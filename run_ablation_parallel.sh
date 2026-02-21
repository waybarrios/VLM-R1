#!/bin/bash
# Run ablation experiments for ALL 5 models IN PARALLEL across 4 GPUs
# Much faster than sequential execution!
# For CVPR Paper Section 4.4

# Configuration
DATASET_PATH="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
REASONING_BASE="/gpudata3/Wayner/reasoning"
OUTPUT_DIR="ablation_results"

# Models to evaluate (array for ordering)
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

# GPUs available
GPU_IDS=(0 1 2 3)
NUM_GPUS=${#GPU_IDS[@]}

# Change to script directory
cd /gpudata3/Wayner/VLM-R1

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "ABLATION STUDY - ALL MODELS IN PARALLEL (CVPR Section 4.4)"
echo "============================================================"
echo "Models to evaluate: ${#MODEL_NAMES[@]}"
echo "GPUs available: $NUM_GPUS (${GPU_IDS[@]})"
echo "Encoders to test: ${#ENCODERS[@]}"
echo "Thresholds to test: 5 (0.30, 0.35, 0.40, 0.45, 0.50)"
echo ""
echo "Total experiments: $((${#MODEL_NAMES[@]} * ${#ENCODERS[@]} * 5))"
echo ""
echo "⚡ PARALLEL EXECUTION MODE ⚡"
echo "Estimated time: ~1-2 hours (vs 3-5 hours sequential)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Function to run ablation for a single model
run_model_ablation() {
    local MODEL_NAME=$1
    local GPU_ID=$2
    local MODEL_DIR=${MODEL_DIRS[$MODEL_NAME]}
    local PREDICTIONS_DIR="$REASONING_BASE/$MODEL_DIR"
    local OUTPUT_FILE="$OUTPUT_DIR/${MODEL_NAME}_ablation.json"
    local DEVICE="cuda:$GPU_ID"
    local LOG_FILE="$OUTPUT_DIR/${MODEL_NAME}_ablation.log"

    echo "[$MODEL_NAME] Starting on GPU $GPU_ID..." | tee -a "$LOG_FILE"
    echo "[$MODEL_NAME] Predictions: $PREDICTIONS_DIR" >> "$LOG_FILE"
    echo "[$MODEL_NAME] Device: $DEVICE" >> "$LOG_FILE"

    # Check if predictions directory exists
    if [ ! -d "$PREDICTIONS_DIR" ]; then
        echo "[$MODEL_NAME] ⚠️  ERROR: Directory not found: $PREDICTIONS_DIR" | tee -a "$LOG_FILE"
        return 1
    fi

    # Count prediction files
    NUM_FILES=$(ls "$PREDICTIONS_DIR"/*.json 2>/dev/null | wc -l)
    echo "[$MODEL_NAME] Found $NUM_FILES prediction files" >> "$LOG_FILE"

    if [ $NUM_FILES -eq 0 ]; then
        echo "[$MODEL_NAME] ⚠️  ERROR: No JSON files found" | tee -a "$LOG_FILE"
        return 1
    fi

    # Run ablation
    echo "[$MODEL_NAME] Running ablation experiments (${#ENCODERS[@]} encoders × 5 thresholds = $((${#ENCODERS[@]} * 5)) configs)..." >> "$LOG_FILE"

    python3 run_ablation_experiments.py \
        "$PREDICTIONS_DIR" \
        --dataset-path "$DATASET_PATH" \
        --output-file "$OUTPUT_FILE" \
        --device "$DEVICE" \
        --encoders "${ENCODERS[@]}" \
        --thresholds $THRESHOLDS \
        >> "$LOG_FILE" 2>&1

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "[$MODEL_NAME] ✓ SUCCESS! Results saved to: $OUTPUT_FILE" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[$MODEL_NAME] ❌ FAILED! Check log: $LOG_FILE" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Export function and variables for parallel execution
export -f run_model_ablation
export REASONING_BASE
export DATASET_PATH
export OUTPUT_DIR
export ENCODERS
export THRESHOLDS
export -A MODEL_DIRS

# Track PIDs and GPU assignments
declare -A MODEL_PIDS
declare -A MODEL_GPUS

# Launch models in parallel
echo "Launching models in parallel across $NUM_GPUS GPUs..."
echo ""

GPU_IDX=0
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    GPU_ID=${GPU_IDS[$GPU_IDX]}

    echo "🚀 Launching $MODEL_NAME on GPU $GPU_ID..."

    # Run in background
    run_model_ablation "$MODEL_NAME" "$GPU_ID" &
    PID=$!

    MODEL_PIDS[$MODEL_NAME]=$PID
    MODEL_GPUS[$MODEL_NAME]=$GPU_ID

    echo "   PID: $PID"

    # Round-robin GPU assignment
    GPU_IDX=$(((GPU_IDX + 1) % NUM_GPUS))

    # Small delay to avoid race conditions
    sleep 2
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "All models launched! Running in parallel..."
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Model assignments:"
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "  $MODEL_NAME → GPU ${MODEL_GPUS[$MODEL_NAME]} (PID: ${MODEL_PIDS[$MODEL_NAME]})"
done
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTPUT_DIR/<model_name>_ablation.log"
echo ""
echo "GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo "════════════════════════════════════════════════════════════"
echo ""

# Wait for all background jobs to complete
echo "Waiting for all models to complete..."
echo "(This may take 1-2 hours depending on GPU speed)"
echo ""

SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_MODELS=()

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    PID=${MODEL_PIDS[$MODEL_NAME]}
    GPU=${MODEL_GPUS[$MODEL_NAME]}

    echo "Waiting for $MODEL_NAME (GPU $GPU, PID $PID)..."

    # Wait for specific PID
    wait $PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  ✓ $MODEL_NAME completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ❌ $MODEL_NAME failed (exit code: $EXIT_CODE)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_MODELS+=("$MODEL_NAME")
    fi
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
    for failed_model in "${FAILED_MODELS[@]}"; do
        echo "  ❌ $failed_model"
    done
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "AGGREGATING RESULTS ACROSS ALL MODELS"
echo "════════════════════════════════════════════════════════════"

if [ $SUCCESS_COUNT -eq 0 ]; then
    echo "⚠️  No successful results to aggregate!"
    exit 1
fi

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
    print(f"  ✓ Loaded: {model_name} ({len(data['results'])} configurations)")

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
echo "  - Logs: $OUTPUT_DIR/*_ablation.log"
echo ""
echo "Next steps:"
echo "  1. Review ablation_aggregate.json"
echo "  2. Generate LaTeX table for Section 4.4"
echo "  3. Write ablation study text"
echo "════════════════════════════════════════════════════════════"

# Exit with error code if all failed
if [ $SUCCESS_COUNT -eq 0 ]; then
    echo "❌ All models failed!"
    exit 1
fi
