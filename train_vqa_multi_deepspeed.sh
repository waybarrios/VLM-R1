#!/bin/bash

# VQA Training Script with DeepSpeed ZeRO-3 for Multi-GPU Support
# This script trains Qwen2.5-VL-3B on VQA tasks with GRPO using DeepSpeed
#
# REQUIREMENTS:
# - Ollama must be running with gpt-oss:20b model
#   Start with: ollama serve
#   Pull model: ollama pull gpt-oss:20b
# - transformers==4.52.4 (for Flash Attention support)
# - datasets>=3.0.0
# - deepspeed>=0.14.0
#
# DEEPSPEED OPTIMIZATIONS (ZeRO-3 for 3x A100 80GB):
# - ZeRO Stage 3: Shards optimizer states, gradients, and parameters across GPUs
# - Batch size: 2 per GPU (optimized for ZeRO-3)
# - Gradient accumulation: 4 (effective batch size = 3 GPUs × 2 batch × 4 accum = 24)
# - Max pixels: 12845056 (default) - ZeRO-3 handles memory efficiently
# - Gradient checkpointing: enabled - additional memory savings
#
# Effective batch size: 3 GPUs × 2 batch × 4 accum = 24 samples per update

# Configuration
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json"

# Add both directories to PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"

# Memory optimization - reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training configuration
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"
OUTPUT="${PROJECT_ROOT}/output/qwen2.5-vl-3b-vqa-deepspeed-$(date +%Y%m%d_%H%M%S)"

# GPU Configuration - Using 3 GPUs (1,2,3)
GPU_IDS="1,2,3"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=3

# Create output directory
mkdir -p $OUTPUT

echo "========================================"
echo "Starting Multi-GPU VQA training with DeepSpeed ZeRO-3..."
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "GPUs: $GPU_IDS (count: $NUM_GPUS)"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo "Batch Size: 2 per GPU (gradient accum: 4)"
echo "Effective Batch: $(($NUM_GPUS * 2 * 4))"
echo "Max Pixels: 12845056 (default)"
echo "Gradient Checkpointing: ENABLED"
echo "LLM Judge: ENABLED (gpt-oss:20b)"
echo "Seed: 42 (with shuffle)"
echo "========================================"

# Run training with DeepSpeed
accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --machine_rank 0 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file $DEEPSPEED_CONFIG \
    --zero3_init_flag true \
    --zero3_save_16bit_model true \
    --gradient_accumulation_steps 4 \
    --gradient_clipping 1.0 \
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
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_generations 4 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_steps 500 \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    --bf16 \
    --deepspeed $DEEPSPEED_CONFIG \
    2>&1 | tee $OUTPUT/training.log

echo "========================================"
echo "Training completed!"
echo "Output saved to: $OUTPUT"
echo "========================================"
