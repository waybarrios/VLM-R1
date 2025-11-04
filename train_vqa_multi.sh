#!/bin/bash

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
NUM_GPUS=4

# Create output directory
mkdir -p $OUTPUT

echo "========================================"
echo "Starting Multi-GPU VQA training..."
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "Number of GPUs: $NUM_GPUS"
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
    --output_dir $OUTPUT \
    --seed 42 \
    --shuffle_train_dataset \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --save_steps 500 \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    2>&1 | tee $OUTPUT/training.log

echo "========================================"
echo "Training completed!"
echo "Output saved to: $OUTPUT"
echo "========================================"
