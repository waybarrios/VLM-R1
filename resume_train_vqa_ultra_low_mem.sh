#!/bin/bash

# Resume VQA Training Script - ULTRA LOW MEMORY MODE
# This script resumes training with maximum memory savings
#
# AGGRESSIVE MEMORY OPTIMIZATIONS:
# - Per-device batch: 1 (same as regular resume)
# - Gradient accumulation: 8 (same as regular resume)
# - NUM_GENERATIONS: 2 (REDUCED from 4 - saves 50% memory on generations)
# - MAX_PIXELS: 6422528 (REDUCED from 12845056 - saves ~40% image memory)
# - Effective batch size: 32 (maintained)
# - Total completions per step: 32 × 2 = 64 (reduced from 128)

# Configuration
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json"

# CHECKPOINT TO RESUME FROM
CHECKPOINT_PATH="${PROJECT_ROOT}/output/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-1000"

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_PATH"
    echo "Please update CHECKPOINT_PATH in this script to point to your checkpoint."
    exit 1
fi

# Add both directories to PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"

# Memory optimization - reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training configuration
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"
OUTPUT="${PROJECT_ROOT}/output/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040"

# GPU Configuration - Using 4 GPUs (0,1,2,3)
GPU_IDS="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=4

# ULTRA LOW MEMORY Training hyperparameters
PER_DEVICE_BATCH=1
GRADIENT_ACCUM=8
NUM_GENERATIONS=2  # REDUCED from 4 to save memory
LEARNING_RATE=1e-5
NUM_EPOCHS=3
SAVE_STEPS=500
LOGGING_STEPS=10
SEED=42

# REDUCED Image processing parameters
MAX_PIXELS=6422528    # REDUCED from 12845056 (half)
MIN_PIXELS=3136

# LLM Judge configuration
USE_LLM_JUDGE=true
LLM_JUDGE_MODEL="gpt-oss:20b"
LLM_JUDGE_BASE_URL="http://localhost:11434/v1"

# Calculate derived values
GLOBAL_BATCH=$(($NUM_GPUS * $PER_DEVICE_BATCH))
EFFECTIVE_BATCH=$(($NUM_GPUS * $PER_DEVICE_BATCH * $GRADIENT_ACCUM))
CONSTRAINT_CHECK=$(($GLOBAL_BATCH % $NUM_GENERATIONS))
TOTAL_COMPLETIONS=$(($EFFECTIVE_BATCH * $NUM_GENERATIONS))

# Debug mode
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "Resuming with ULTRA LOW MEMORY MODE..."
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "Resume from: $CHECKPOINT_PATH"
echo "GPUs: $GPU_IDS (count: $NUM_GPUS)"
echo ""
echo "ULTRA LOW MEMORY Batch Configuration:"
echo "  - Per-device batch: $PER_DEVICE_BATCH"
echo "  - Gradient accumulation: $GRADIENT_ACCUM"
echo "  - Global batch per step: $GLOBAL_BATCH"
echo "  - Effective batch size: $EFFECTIVE_BATCH"
echo "  - num_generations: $NUM_GENERATIONS (REDUCED from 4)"
echo "  - Total completions per step: $TOTAL_COMPLETIONS"
if [ $CONSTRAINT_CHECK -eq 0 ]; then
    echo "  - Constraint check: $GLOBAL_BATCH % $NUM_GENERATIONS = $CONSTRAINT_CHECK ✅ VALID"
else
    echo "  - Constraint check: ❌ INVALID"
    exit 1
fi
echo ""
echo "Memory Configuration:"
echo "  - Max Pixels: $MAX_PIXELS (REDUCED from 12845056)"
echo "  - Min Pixels: $MIN_PIXELS"
echo "  - Gradient Checkpointing: ENABLED"
echo "  - ZeRO Stage: 3"
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
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --gradient_clipping 1.0 \
    ${SRC_DIR}/open_r1/grpo_rec.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --use_llm_judge \
    --llm_judge_model "$LLM_JUDGE_MODEL" \
    --llm_judge_base_url "$LLM_JUDGE_BASE_URL" \
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
    2>&1 | tee -a $OUTPUT/training_resume_ultra_low_mem.log

echo "========================================"
echo "Training completed!"
echo "========================================"
