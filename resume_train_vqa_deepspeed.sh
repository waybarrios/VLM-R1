#!/bin/bash

# Resume VQA Training Script - STABILITY OPTIMIZED v3
# This script resumes training from checkpoint-1100 with optimized hyperparameters
#
# STABILITY OPTIMIZATIONS v3 (BALANCED ACCURACY + REASONING):
# - Resume from checkpoint-1100 (best balance: 39.66% accuracy + F1=0.502)
# - BALANCED reward weights: (2.0, 2.5, 1.5) - reduced accuracy dominance, boosted reasoning
# - Reduced learning rate to 2e-6 (more conservative to prevent over-optimization)
# - INCREASED num_generations from 2 to 4 (larger groups = smoother gradients, better baseline)
# - Adjusted batch config: 2×4 accum for memory efficiency with 4 generations
# - Total completions per step: 32 × 4 = 128 (better gradient signal)

# Configuration
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json"

# CHECKPOINT TO RESUME FROM
# Update this path to point to your checkpoint directory
CHECKPOINT_PATH="${PROJECT_ROOT}/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1100"

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_PATH"
    echo "Please update CHECKPOINT_PATH in this script to point to your checkpoint."
    echo ""
    echo "Available checkpoints:"
    find ${PROJECT_ROOT}/output -name "checkpoint-*" -type d 2>/dev/null
    exit 1
fi

# Add both directories to PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"

# Memory optimization - reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training configuration
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"
# New output dir with timestamp to avoid overwriting previous experiment
OUTPUT="${PROJECT_ROOT}/output/qwen2.5-vl-3b-vqa-deepspeed-$(date +%Y%m%d_%H%M%S)"

# GPU Configuration - Using 4 GPUs (0,1,2,3)
GPU_IDS="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=4

# OPTIMIZED Training hyperparameters (v3: Balance accuracy + reasoning quality)
# Batch=2 with gradient checkpointing to accommodate 4 generations
PER_DEVICE_BATCH=2
# Gradient accumulation increased to 4 to maintain effective batch size
GRADIENT_ACCUM=4
# NUM_GENERATIONS=4 for better gradient signal (literature recommends 4-8 for VLMs)
NUM_GENERATIONS=4
# REDUCED to 2e-6 for more conservative updates, prevent over-optimization
LEARNING_RATE=2e-6
NUM_EPOCHS=3
SAVE_STEPS=100
LOGGING_STEPS=2  # Log every 2 steps for detailed monitoring
SEED=42

# Image processing parameters - REDUCED TO PREVENT OOM
# Original: 12845056 caused 31GB+ memory allocation in vision encoder
# Reduced to 602112 (default for Qwen2.5-VL-3B) - 95% reduction
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

# Debug mode - enables detailed logging for rewards
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "Resuming Multi-GPU VQA training with DeepSpeed ZeRO-3..."
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "Resume from: $CHECKPOINT_PATH"
echo "GPUs: $GPU_IDS (count: $NUM_GPUS)"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo ""
echo "OPTIMIZED Batch Configuration (v3: Balanced Training):"
echo "  - Per-device batch: $PER_DEVICE_BATCH"
echo "  - Gradient accumulation: $GRADIENT_ACCUM"
echo "  - Global batch per step: $GLOBAL_BATCH ($NUM_GPUS GPUs × $PER_DEVICE_BATCH batch)"
echo "  - Effective batch size: $EFFECTIVE_BATCH ($NUM_GPUS GPUs × $PER_DEVICE_BATCH batch × $GRADIENT_ACCUM accum)"
echo "  - num_generations: $NUM_GENERATIONS (INCREASED from 2 to 4 for better gradient signal)"
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
echo "  - Max Pixels: $MAX_PIXELS (REDUCED from 12845056 to prevent vision encoder OOM)"
echo "  - Min Pixels: $MIN_PIXELS"
echo "  - Gradient Checkpointing: ENABLED"
echo "  - ZeRO Stage: 3 (parameter sharding)"
echo "  - Memory optimization: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  - CUDA cache clearing: ENABLED after generation"
echo ""
echo "Training Configuration:"
echo "  - Learning rate: $LEARNING_RATE (Conservative: 2e-6 to prevent over-optimization)"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Save steps: $SAVE_STEPS"
echo "  - Logging steps: $LOGGING_STEPS"
echo "  - Seed: $SEED (with shuffle)"
echo "  - Reward weights: 2.0 (format) 2.5 (accuracy) 1.5 (reasoning) - BALANCED"
echo "  - LLM Judge: $USE_LLM_JUDGE ($LLM_JUDGE_MODEL)"
echo "  - Debug Mode: ENABLED"
echo "========================================"

# Build LLM judge arguments conditionally
LLM_JUDGE_ARGS=""
if [ "$USE_LLM_JUDGE" = "true" ]; then
    LLM_JUDGE_ARGS="--use_llm_judge --llm_judge_model $LLM_JUDGE_MODEL --llm_judge_base_url $LLM_JUDGE_BASE_URL"
fi

# Run training with DeepSpeed - RESUME FROM CHECKPOINT
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
    --reward_weights 2.0 2.5 1.5 \
    $LLM_JUDGE_ARGS \
    --output_dir $OUTPUT \
    --resume_from_checkpoint $CHECKPOINT_PATH \
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
    2>&1 | tee -a $OUTPUT/training_resume.log

echo "========================================"
echo "Training resumed and completed!"
echo "Output saved to: $OUTPUT"
echo ""
echo "Debug reward logs saved to:"
echo "  - Format: ${OUTPUT}/reward_format_vqa.txt"
echo "  - Accuracy: ${OUTPUT}/reward_accuracy_vqa.txt"
echo "  - Reasoning: ${OUTPUT}/reward_reasoning_vqa.txt"
echo "========================================"
