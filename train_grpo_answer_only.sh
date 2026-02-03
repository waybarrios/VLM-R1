#!/bin/bash

# GRPO Answer-Only Baseline Experiment
# Purpose: Compare GRPO with composite reward vs answer-only reward
# Reference: TODO_GRPO_ANSWER_ONLY_EXPERIMENT.md
#
# KEY CHANGE: Removed "reasoning" from reward_funcs
# - Original: --reward_funcs "format" "accuracy" "reasoning" --reward_weights 2.0 3.0 1.0
# - Answer-only: --reward_funcs "format" "accuracy" --reward_weights 2.0 3.0
#
# HYPERPARAMETERS (from checkpoint-1400 config - stable training):
# - learning_rate: 3e-6
# - num_generations: 2
# - max_steps: 1500
# - save_steps: 100

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
# Resume from previous run
OUTPUT="${PROJECT_ROOT}/output/GRPO_answer_only_baseline_20260129_202515"
RESUME_CHECKPOINT="${OUTPUT}/checkpoint-150"

# GPU Configuration - Using GPUs 4,5,6,7
GPU_IDS="4,5,6,7"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=4

# Training hyperparameters - FROM CHECKPOINT-1400 CONFIG (stable)
# DO NOT use checkpoint-1800/2800 config (caused instability)
PER_DEVICE_BATCH=4
GRADIENT_ACCUM=2
NUM_GENERATIONS=2  # Changed from 4 to 2 per checkpoint-1400 config
LEARNING_RATE=3e-6  # Changed from 1e-5 to 3e-6 per checkpoint-1400 config
MAX_STEPS=1500      # Use steps instead of epochs per TODO
SAVE_STEPS=50       # More frequent checkpoints for faster preliminary results
LOGGING_STEPS=2
SEED=42

# Image processing parameters
MAX_PIXELS=602112
MIN_PIXELS=3136

# LLM Judge configuration - DISABLED for speed
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
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "GRPO Answer-Only Baseline Experiment"
echo "========================================"
echo "Purpose: Prove step-level supervision is necessary for reasoning improvement"
echo ""
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "GPUs: $GPU_IDS (count: $NUM_GPUS)"
echo "DeepSpeed Config: $DEEPSPEED_CONFIG"
echo ""
echo "KEY EXPERIMENT CHANGE:"
echo "  - Removed 'reasoning' reward function"
echo "  - Using only 'format' and 'accuracy' rewards"
echo ""
echo "Batch Configuration:"
echo "  - Per-device batch: $PER_DEVICE_BATCH"
echo "  - Gradient accumulation: $GRADIENT_ACCUM"
echo "  - Global batch per step: $GLOBAL_BATCH ($NUM_GPUS GPUs x $PER_DEVICE_BATCH batch)"
echo "  - Effective batch size: $EFFECTIVE_BATCH ($NUM_GPUS GPUs x $PER_DEVICE_BATCH batch x $GRADIENT_ACCUM accum)"
echo "  - num_generations: $NUM_GENERATIONS (from checkpoint-1400 config)"
echo "  - Total completions per step: $TOTAL_COMPLETIONS ($EFFECTIVE_BATCH samples x $NUM_GENERATIONS generations)"
if [ $CONSTRAINT_CHECK -eq 0 ]; then
    echo "  - Constraint check: $GLOBAL_BATCH % $NUM_GENERATIONS = $CONSTRAINT_CHECK VALID"
else
    echo "  - Constraint check: $GLOBAL_BATCH % $NUM_GENERATIONS = $CONSTRAINT_CHECK INVALID - Training will fail!"
    echo ""
    echo "ERROR: Global batch size must be divisible by num_generations!"
    echo "Please adjust PER_DEVICE_BATCH or NUM_GENERATIONS."
    exit 1
fi
echo ""
echo "Training Configuration (from checkpoint-1400 - stable):"
echo "  - Learning rate: $LEARNING_RATE (conservative)"
echo "  - Max steps: $MAX_STEPS"
echo "  - Save steps: $SAVE_STEPS (frequent checkpoints for fast preliminary results)"
echo "  - Logging steps: $LOGGING_STEPS"
echo "  - Seed: $SEED"
echo "  - Reward functions: format, accuracy (NO reasoning)"
echo "  - Reward weights: 2.0 (format), 3.0 (accuracy)"
echo ""
echo "Expected Results (from TODO):"
echo "  - Accuracy: ~45% (similar to composite)"
echo "  - Match-F1: ~0.43? (should NOT improve without reasoning reward)"
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
    --main_process_port 29501 \
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
    --reward_funcs "format" "accuracy" \
    --reward_weights 2.0 3.0 \
    $LLM_JUDGE_ARGS \
    --output_dir $OUTPUT \
    --seed $SEED \
    --shuffle_train_dataset \
    --max_steps $MAX_STEPS \
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
    --resume_from_checkpoint $RESUME_CHECKPOINT \
    2>&1 | tee -a $OUTPUT/training.log

echo "========================================"
echo "Training completed!"
echo "Output saved to: $OUTPUT"
echo ""
echo "Next steps (from TODO):"
echo "1. Evaluate checkpoints:"
echo "   python compute_grpo_metrics.py --checkpoint ${OUTPUT}/checkpoint-300/"
echo "   python compute_grpo_metrics.py --checkpoint ${OUTPUT}/checkpoint-1400/"
echo ""
echo "2. Update files after experiment:"
echo "   - tables/4_grpo_results_table.tex"
echo "   - rebuttal_final_sj.tex"
echo "   - sec/4_grpo_section.tex"
echo "========================================"
