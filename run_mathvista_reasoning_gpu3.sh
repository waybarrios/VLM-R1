#!/bin/bash

# Run MathVista Testmini Reasoning on GPU 3
# Mathematical reasoning with visual diagrams, charts, and figures

export CUDA_VISIBLE_DEVICES=3

echo "=========================================="
echo "Running MathVista Testmini Reasoning on GPU 3"
echo "=========================================="
echo ""
echo "Task: mathvista_testmini_reasoning"
echo "Checkpoint: checkpoint-1500"
echo "GPU: 3"
echo ""

accelerate launch \
  --num_processes 1 \
  --main_process_port 29503 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1500" \
  --tasks mathvista_testmini_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-mathvista-testmini-reasoning-ckpt1500-gpu3

echo ""
echo "=========================================="
echo "MathVista Testmini Reasoning evaluation complete!"
echo "=========================================="
echo ""
echo "Results saved to: logs-mathvista-testmini-reasoning-ckpt1500-gpu3"
echo ""
