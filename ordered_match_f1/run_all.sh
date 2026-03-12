#!/bin/bash
# Run Ordered Match F1 evaluation on all 20 MLLMs from Table 2
# Uses all-distilroberta-v1 with tau=0.35 (ablation-validated defaults)
# Alpha values: 0.0 (standard F1), 0.1, 0.2, 0.3 (recommended), 0.5

set -e

cd /gpudata3/Wayner/VLM-R1

echo "========================================"
echo "Ordered Match F1 Evaluation"
echo "20 MLLMs on CRYSTAL (6,372 samples)"
echo "Encoder: all-distilroberta-v1, tau=0.35"
echo "========================================"
echo "Start time: $(date)"
echo ""

python3 ordered_match_f1/run_ordered_eval.py \
    --alphas 0.0 0.1 0.2 0.3 0.5 \
    --device auto \
    --output-dir ordered_match_f1/results

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "Results in: ordered_match_f1/results/"
echo "========================================"
