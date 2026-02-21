#!/bin/bash

# Run Charades-STA Reasoning on GPU 3
# Temporal video grounding with human actions and objects

export CUDA_VISIBLE_DEVICES=3

echo "=========================================="
echo "Running Charades-STA Reasoning on GPU 3"
echo "=========================================="
echo ""
echo "Task: temporal_grounding_charades_reasoning"
echo "Checkpoint: checkpoint-1500"
echo "GPU: 3"
echo ""

accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1500" \
  --tasks temporal_grounding_charades_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-charades-reasoning-ckpt1500-gpu3

echo ""
echo "=========================================="
echo "Charades-STA Reasoning evaluation complete!"
echo "=========================================="
echo ""
echo "Results saved to: logs-charades-reasoning-ckpt1400-gpu3"
echo ""
echo "To evaluate the results, run:"
echo "python /gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/charades_sta/eval_tvg.py -f <path_to_results.json>"
