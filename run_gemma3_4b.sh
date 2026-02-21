#!/bin/bash
# Script to compute metrics for Gemma-3-4B model

echo "============================================================"
echo "Computing metrics for Gemma-3-4B"
echo "============================================================"

cd /gpudata3/Wayner/VLM-R1

# Set GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run evaluation (both with and without judge)
python compute_metrics.py \
    /gpudata3/Wayner/reasoning/outputs_testing_gemma3_4b \
    --dataset-path /gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27 \
    --num-gpus 4 \
    --output-dir ./metrics_results \
    --both \
    --judge-model "gpt-oss:120b"

echo ""
echo "============================================================"
echo "Done! Results saved to:"
echo "  metrics_results/outputs_testing_gemma3_4b/"
echo "============================================================"

# Show quick summary
echo ""
echo "Quick summary:"
cat metrics_results/outputs_testing_gemma3_4b/no_judge_summary.txt | grep -A 20 "EVALUATION SUMMARY"
