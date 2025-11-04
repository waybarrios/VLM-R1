#!/bin/bash

# VQA Training Script with Multi-GPU Support
# This script trains Qwen2.5-VL-3B on VQA tasks with GRPO
#
# REQUIREMENTS:
# - Ollama must be running with gpt-oss:20b model
#   Start with: ollama serve
#   Pull model: ollama pull gpt-oss:20b
# - transformers==4.52.4 (for Flash Attention support)
# - datasets>=3.0.0

# Configuration
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"

# Add both directories to PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"

# Training configuration
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"
OUTPUT="${PROJECT_ROOT}/output/qwen2.5-vl-3b-vqa-multi-$(date +%Y%m%d_%H%M%S)"

# GPU Configuration
# Specify which GPUs to use (comma-separated). Leave empty to use all available GPUs.
# Examples:
#   GPU_IDS="0,1,2,3"    # Use GPUs 0, 1, 2, 3
#   GPU_IDS="0,2"        # Use only GPUs 0 and 2
#   GPU_IDS=""           # Use all available GPUs
GPU_IDS="0,1,2,3"

# Set CUDA_VISIBLE_DEVICES if GPU_IDS is specified
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    # Count number of GPUs
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    echo "Using GPUs: $GPU_IDS"
else
    # Use all available GPUs
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Using all available GPUs"
fi

# Create output directory
mkdir -p $OUTPUT

echo "========================================"
echo "Starting Multi-GPU VQA training..."
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
if [ -n "$GPU_IDS" ]; then
    echo "GPUs: $GPU_IDS (count: $NUM_GPUS)"
else
    echo "GPUs: All available (count: $NUM_GPUS)"
fi
echo "LLM Judge: ENABLED (gpt-oss:20b)"
echo "Seed: 42 (with shuffle)"
echo "========================================"

# Run training with accelerate (with explicit configuration)
accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --machine_rank 0 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    ${SRC_DIR}/open_r1/grpo_rec.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --use_llm_judge \
    --llm_judge_model "gpt-oss:20b" \
    --llm_judge_base_url "http://localhost:11434/v1" \
    --output_dir $OUTPUT \
    --seed 42 \
    --shuffle_train_dataset \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_generations 4 \
    --logging_steps 10 \
    --save_steps 500 \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    2>&1 | tee $OUTPUT/training.log

echo "========================================"
echo "Training completed!"
echo "Output saved to: $OUTPUT"
echo "========================================"
