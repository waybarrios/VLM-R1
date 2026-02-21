#!/bin/bash
#
# Test script for V* Bench with reasoning format
# Uses GRPO checkpoint format: {"reasoning_steps": [], "answer": ""}
#

echo "========================================"
echo "Testing V* Bench with Reasoning Format"
echo "========================================"

# First, verify the task is available
echo ""
echo "Step 1: Checking if task is registered..."
lmms-eval --tasks list | grep -i vstar

echo ""
echo "Step 2: Running inference with reasoning format..."
echo ""

CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --main_process_port 29500 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="Qwen/Qwen2.5-VL-3B-Instruct" \
  --tasks vstar_bench_reasoning \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix vstar_reasoning_test \
  --output_path logs-vstar-reasoning-test \
  --limit 10

echo ""
echo "========================================"
echo "✓ Test completed!"
echo "Check logs-vstar-reasoning-test/ for results"
echo "========================================"
