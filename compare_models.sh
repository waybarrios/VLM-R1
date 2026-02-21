#!/bin/bash

# Compare baseline vs fine-tuned models
TEST_DATA="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"

echo "========================================="
echo "Model Comparison Script"
echo "========================================="
echo ""

# Evaluate baseline
echo "[1/3] Evaluating baseline model..."
if [ -d "predictions/baseline_qwen2.5-vl-3b-instruct" ]; then
    python inference/evaluate_predictions.py \
        --predictions_dir "predictions/baseline_qwen2.5-vl-3b-instruct" \
        --test_dataset_path "${TEST_DATA}"
else
    echo "⚠️  Baseline predictions not found. Run ./run_baseline_inference.sh first"
fi

echo ""
echo "========================================="
echo ""

# Evaluate checkpoint-500
echo "[2/3] Evaluating checkpoint-500..."
if [ -d "predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-500" ]; then
    python inference/evaluate_predictions.py \
        --predictions_dir "predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-500" \
        --test_dataset_path "${TEST_DATA}"
else
    echo "⚠️  checkpoint-500 predictions not found. Run ./run_parallel_inference.sh first"
fi

echo ""
echo "========================================="
echo ""

# Evaluate checkpoint-1000
echo "[3/3] Evaluating checkpoint-1000..."
if [ -d "predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-1000" ]; then
    python inference/evaluate_predictions.py \
        --predictions_dir "predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-1000" \
        --test_dataset_path "${TEST_DATA}"
else
    echo "⚠️  checkpoint-1000 predictions not found. Run ./run_parallel_inference.sh first"
fi

echo ""
echo "========================================="
echo "Comparison Summary"
echo "========================================="
echo ""

# Extract and compare metrics
python -c "
import json
import os

models = [
    ('Baseline (Qwen2.5-VL-3B-Instruct)', 'predictions/baseline_qwen2.5-vl-3b-instruct/evaluation_results.json'),
    ('Fine-tuned checkpoint-500', 'predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-500/evaluation_results.json'),
    ('Fine-tuned checkpoint-1000', 'predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-1000/evaluation_results.json'),
]

print(f'{'Model':<32} {'Accuracy':<12} {'Match F1':<12} {'Precision':<12} {'Recall':<12}')
print('=' * 80)

for name, path in models:
    if os.path.exists(path):
        with open(path) as f:
            results = json.load(f)
        acc = results.get('accuracy_rate', 0) * 100
        f1 = results.get('match_f1_avg', 0)
        prec = results.get('precision_avg', 0)
        rec = results.get('recall_avg', 0)
        print(f'{name:<32} {acc:>6.2f}%      {f1:>6.4f}       {prec:>6.4f}       {rec:>6.4f}')
    else:
        print(f'{name:<32} Not evaluated yet')

print('=' * 80)
print()
print('Higher is better for all metrics.')
print('Match F1 measures reasoning quality using sentence similarity.')
print()
"

echo "Done! Check individual evaluation_results.json files for detailed metrics."
