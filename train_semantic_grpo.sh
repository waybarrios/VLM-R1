#!/bin/bash

# SEMANTIC PROCESS REWARD (SPR) Training Script
# =============================================
# Uses SentenceTransformer semantic similarity instead of word overlap for reasoning reward.
#
# KEY DIFFERENCE FROM STANDARD GRPO:
# - Standard: word overlap F1 (threshold 0.45) - misses semantic equivalence
# - Semantic: cosine similarity of embeddings (threshold 0.70) - captures meaning
#
# EXPECTED BENEFITS:
# - Lower training variance (σ < 4% vs 6.8% with word overlap)
# - Better reasoning F1 (>0.50 vs 0.43)
# - Potentially better accuracy (>46% vs 44.9%)
# - More stable gradient signal
#
# DEEPSPEED COMPATIBILITY:
# - SentenceTransformer runs on CPU to avoid multi-process CUDA errors
# - Minimal overhead (~50ms per batch)

# Configuration
PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json"

# Add both directories to PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training configuration
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"
OUTPUT="${PROJECT_ROOT}/output/qwen2.5-vl-3b-semantic-grpo-$(date +%Y%m%d_%H%M%S)"

# GPU Configuration
GPU_IDS="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=4

# Training hyperparameters
PER_DEVICE_BATCH=4
GRADIENT_ACCUM=2
NUM_GENERATIONS=4
LEARNING_RATE=1e-5
NUM_EPOCHS=2
SAVE_STEPS=100
LOGGING_STEPS=2
SEED=42

# Image processing parameters
MAX_PIXELS=602112
MIN_PIXELS=3136

# ========================================
# SEMANTIC PROCESS REWARD CONFIGURATION
# ========================================
USE_SEMANTIC_REWARD=true
SEMANTIC_THRESHOLD=0.70  # Cosine similarity threshold (0.60-0.80 recommended)

# LLM Judge (disabled for training speed)
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

# Debug mode
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "Starting SEMANTIC PROCESS REWARD Training"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "GPUs: $GPU_IDS (count: $NUM_GPUS)"
echo ""
echo "SEMANTIC REWARD CONFIGURATION:"
echo "  - Semantic Reward: $USE_SEMANTIC_REWARD"
echo "  - Similarity Threshold: $SEMANTIC_THRESHOLD"
echo "  - Method: SentenceTransformer (all-MiniLM-L6-v2) on CPU"
echo ""
echo "Batch Configuration:"
echo "  - Per-device batch: $PER_DEVICE_BATCH"
echo "  - Gradient accumulation: $GRADIENT_ACCUM"
echo "  - Effective batch size: $EFFECTIVE_BATCH"
echo "  - num_generations: $NUM_GENERATIONS"
if [ $CONSTRAINT_CHECK -eq 0 ]; then
    echo "  - Constraint check: VALID"
else
    echo "  - Constraint check: INVALID"
    exit 1
fi
echo ""
echo "Training Configuration:"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Save steps: $SAVE_STEPS"
echo "  - Seed: $SEED"
echo "========================================"

# Build LLM judge arguments
LLM_JUDGE_ARGS=""
if [ "$USE_LLM_JUDGE" = "true" ]; then
    LLM_JUDGE_ARGS="--use_llm_judge --llm_judge_model $LLM_JUDGE_MODEL --llm_judge_base_url $LLM_JUDGE_BASE_URL"
fi

# Build semantic reward arguments
SEMANTIC_ARGS=""
if [ "$USE_SEMANTIC_REWARD" = "true" ]; then
    SEMANTIC_ARGS="--use_semantic_reasoning_reward --semantic_similarity_threshold $SEMANTIC_THRESHOLD"
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
    $SEMANTIC_ARGS \
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
echo "Semantic GRPO Training completed!"
echo "Output saved to: $OUTPUT"
echo ""
echo "Debug logs:"
echo "  - Semantic rewards: ${OUTPUT}/reward_reasoning_semantic.txt"
echo "  - Format rewards: ${OUTPUT}/reward_format_vqa.txt"
echo "  - Accuracy rewards: ${OUTPUT}/reward_accuracy_vqa.txt"
echo "========================================"
