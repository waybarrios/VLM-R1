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
# DEEPSPEED OPTIMIZATIONS (ZeRO-3 for 4x A100 80GB):
# - ZeRO Stage 3: Shards optimizer states, gradients, and parameters across GPUs
# - Batch size: 2 per GPU (memory-safe configuration for 4 GPUs with ZeRO-3)
# - Gradient accumulation: 4 (effective batch size = 4 GPUs × 2 batch × 4 accum = 32)
# - num_generations: 4 (generates 4 completions per sample for GRPO)
# - Max pixels: 12845056 (default) - ZeRO-3 handles memory efficiently
# - Gradient checkpointing: enabled - additional memory savings
#
# IMPORTANT CONSTRAINT: Global batch size must be divisible by num_generations
# - Global batch per step = 4 GPUs × 2 batch = 8
# - 8 % 4 = 0 ✓ Valid configuration (also works with 2, 4, 8 generations)
# - NOTE: 8 % 3 = 2 ✗ Invalid! num_generations=3 does NOT work with batch=2
#
# Effective batch size: 4 GPUs × 2 batch × 4 accum = 32 samples per update
# Total completions per step: 32 × 4 = 128 completions (balanced exploration)
# Memory usage: ~45-55 GB per GPU (safe with 15-25 GB headroom)

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

# GPU Configuration - Using 4 GPUs (0,1,2,3)
GPU_IDS="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=4

# Training hyperparameters - OPTIMIZED FOR SPEED (Opción 3: Balance)
# Changed from batch=2/accum=4 to batch=4/accum=2 for 2x speed
# Effective batch size = 4 GPUs × 4 batch × 2 accum = 32 (same as original)
# Gradient checkpointing: ENABLED (memory safety)
PER_DEVICE_BATCH=4
GRADIENT_ACCUM=2
NUM_GENERATIONS=4
LEARNING_RATE=1e-5
NUM_EPOCHS=2
SAVE_STEPS=100
LOGGING_STEPS=2  # Log every 2 steps for detailed monitoring
SEED=42

# Image processing parameters - REDUCED FOR MEMORY SAFETY
# Using 602112 (default for Qwen2.5-VL-3B) to prevent OOM
# Original 12845056 caused 31GB+ memory allocation in vision encoder
MAX_PIXELS=602112
MIN_PIXELS=3136

# LLM Judge configuration
# DISABLED during training for speed (50-70% faster) and memory (saves 18GB on GPU 0)
# Use evaluate_predictions.py with --use_llm_judge for final evaluation
USE_LLM_JUDGE=false
LLM_JUDGE_MODEL="gpt-oss:20b"
LLM_JUDGE_BASE_URL="http://localhost:11434/v1"

# Calculate derived values
GLOBAL_BATCH=$(($NUM_GPUS * $PER_DEVICE_BATCH))
EFFECTIVE_BATCH=$(($NUM_GPUS * $PER_DEVICE_BATCH * $GRADIENT_ACCUM))
CONSTRAINT_CHECK=$(($GLOBAL_BATCH % $NUM_GENERATIONS))
TOTAL_COMPLETIONS=$(($EFFECTIVE_BATCH * $NUM_GENERATIONS))

# Create output directory
mkdir -p $OUTPUT

# Debug mode - enables detailed logging for rewards
# Log files will be created in OUTPUT directory:
# - ${OUTPUT}/reward_format_vqa.txt (format/JSON validation)
# - ${OUTPUT}/reward_accuracy_vqa.txt (accuracy scores)
# - ${OUTPUT}/reward_reasoning_vqa.txt (reasoning quality scores)
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "Starting Multi-GPU VQA training with DeepSpeed ZeRO-3..."
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "GPUs: $GPU_IDS (count: $NUM_GPUS)"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo ""
echo "Batch Configuration:"
echo "  - Per-device batch: $PER_DEVICE_BATCH"
echo "  - Gradient accumulation: $GRADIENT_ACCUM"
echo "  - Global batch per step: $GLOBAL_BATCH ($NUM_GPUS GPUs × $PER_DEVICE_BATCH batch)"
echo "  - Effective batch size: $EFFECTIVE_BATCH ($NUM_GPUS GPUs × $PER_DEVICE_BATCH batch × $GRADIENT_ACCUM accum)"
echo "  - num_generations: $NUM_GENERATIONS"
echo "  - Total completions per step: $TOTAL_COMPLETIONS ($EFFECTIVE_BATCH samples × $NUM_GENERATIONS generations)"
if [ $CONSTRAINT_CHECK -eq 0 ]; then
    echo "  - Constraint check: $GLOBAL_BATCH % $NUM_GENERATIONS = $CONSTRAINT_CHECK ✅ VALID"
else
    echo "  - Constraint check: $GLOBAL_BATCH % $NUM_GENERATIONS = $CONSTRAINT_CHECK ❌ INVALID - Training will fail!"
    echo ""
    echo "ERROR: Global batch size must be divisible by num_generations!"
    echo "Please adjust PER_DEVICE_BATCH or NUM_GENERATIONS."
    exit 1
fi
echo ""
echo "Memory Configuration:"
echo "  - Max Pixels: $MAX_PIXELS (memory-safe: 602k instead of 12.8M)"
echo "  - Min Pixels: $MIN_PIXELS"
echo "  - Gradient Checkpointing: ENABLED"
echo "  - ZeRO Stage: 3 (parameter sharding)"
echo ""
echo "Training Configuration:"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Save steps: $SAVE_STEPS"
echo "  - Logging steps: $LOGGING_STEPS"
echo "  - Seed: $SEED (with shuffle)"
echo "  - LLM Judge: $USE_LLM_JUDGE  ($LLM_JUDGE_MODEL)"
echo "  - Debug Mode: ENABLED"
echo ""
echo "Debug Logs:"
echo "  - Format rewards: ${OUTPUT}/reward_format_vqa.txt"
echo "  - Accuracy rewards: ${OUTPUT}/reward_accuracy_vqa.txt"
echo "  - Reasoning rewards: ${OUTPUT}/reward_reasoning_vqa.txt"
echo "========================================"

# Build LLM judge arguments conditionally
LLM_JUDGE_ARGS=""
if [ "$USE_LLM_JUDGE" = "true" ]; then
    LLM_JUDGE_ARGS="--use_llm_judge --llm_judge_model $LLM_JUDGE_MODEL --llm_judge_base_url $LLM_JUDGE_BASE_URL"
fi

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
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --gradient_clipping 1.0 \
    ${SRC_DIR}/open_r1/grpo_rec.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --reward_weights 3.0 1.0 3.0 \
    $LLM_JUDGE_ARGS \
    --output_dir $OUTPUT \
    --seed $SEED \
    --shuffle_train_dataset \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --learning_rate $LEARNING_RATE \
    --num_generations $NUM_GENERATIONS \
    --gradient_checkpointing \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --max_pixels $MAX_PIXELS \
    --min_pixels $MIN_PIXELS \
    --bf16 \
    --deepspeed $DEEPSPEED_CONFIG \
    2>&1 | tee $OUTPUT/training.log

echo "========================================"
echo "Training completed!"
echo "Output saved to: $OUTPUT"
echo ""
echo "Debug reward logs saved to:"
echo "  - Format: ${OUTPUT}/reward_format_vqa.txt"
echo "  - Accuracy: ${OUTPUT}/reward_accuracy_vqa.txt"
echo "  - Reasoning: ${OUTPUT}/reward_reasoning_vqa.txt"
echo "========================================"
