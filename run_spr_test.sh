#!/bin/bash
# Quick validation run for Semantic Process Reward (10 steps)

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json"

export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"
OUTPUT="${PROJECT_ROOT}/output/spr-test-$(date +%Y%m%d_%H%M%S)"

# Use 4 GPUs for test
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Small test config
PER_DEVICE_BATCH=2
GRADIENT_ACCUM=1
NUM_GENERATIONS=4
MAX_STEPS=10

# Semantic Process Reward settings
USE_SEMANTIC_REWARD=true
SEMANTIC_THRESHOLD=0.70

mkdir -p $OUTPUT
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "SPR Validation Run (10 steps)"
echo "========================================"
echo "Semantic Reward: $USE_SEMANTIC_REWARD"
echo "Threshold: $SEMANTIC_THRESHOLD"
echo "Output: $OUTPUT"
echo "========================================"

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file $DEEPSPEED_CONFIG \
    --zero3_init_flag true \
    ${SRC_DIR}/open_r1/grpo_rec.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --reward_weights 3.0 1.0 3.0 \
    --use_semantic_reasoning_reward \
    --semantic_similarity_threshold $SEMANTIC_THRESHOLD \
    --output_dir $OUTPUT \
    --seed 42 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $PER_DEVICE_BATCH \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --learning_rate 1e-5 \
    --num_generations $NUM_GENERATIONS \
    --gradient_checkpointing \
    --logging_steps 1 \
    --max_pixels 602112 \
    --min_pixels 3136 \
    --bf16 \
    --deepspeed $DEEPSPEED_CONFIG \
    2>&1 | tee $OUTPUT/training.log

echo "========================================"
echo "Test completed! Check: $OUTPUT"
echo "========================================"
