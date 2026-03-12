#!/bin/bash
# InternVL3.5-4B Answer-Only GRPO Training (Phase 1 of CPR Curriculum)
# Goal: Train InternVL3.5-4B to produce correct answers (no reasoning reward)
# Next: Use best checkpoint as base for Phase 2 (CPR training)

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate internvl35

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3_internvl.json"

export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/gpudata3/hf_cache

MODEL="OpenGVLab/InternVL3_5-4B"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"

# Resume from last checkpoint of collapsed run
RESUME_CHECKPOINT="${PROJECT_ROOT}/output/internvl35_answer_only_20260307_124744/checkpoint-100"
OUTPUT="${PROJECT_ROOT}/output/internvl35_answer_only_resumed_$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Hyperparameters (80GB GPUs - use more memory for faster training)
PER_DEVICE_BATCH=5
GRADIENT_ACCUM=2
NUM_GENERATIONS=5
LEARNING_RATE=5e-6
NUM_EPOCHS=2
SAVE_STEPS=100
LOGGING_STEPS=2
SEED=42
MAX_GRAD_NORM=1.0
WARMUP_RATIO=0.05

# InternVL-specific: max_anyres_num controls image patches
MAX_ANYRES_NUM=12

mkdir -p $OUTPUT
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "InternVL3.5-4B Answer-Only (Phase 1)"
echo "========================================"
echo "Model: $MODEL"
echo "Reward: format + accuracy (NO reasoning)"
echo "Output: $OUTPUT"
echo "========================================"

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file $DEEPSPEED_CONFIG \
    --zero3_init_flag true \
    --zero3_save_16bit_model true \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    ${SRC_DIR}/open_r1/grpo_rec.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" \
    --reward_weights 2.0 2.0 \
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
    --max_anyres_num $MAX_ANYRES_NUM \
    --max_grad_norm $MAX_GRAD_NORM \
    --warmup_ratio $WARMUP_RATIO \
    --bf16 \
    --deepspeed $DEEPSPEED_CONFIG \
    --resume_from_checkpoint $RESUME_CHECKPOINT \
    2>&1 | tee $OUTPUT/training.log

echo "========================================"
echo "Phase 1 (Answer-Only) completed!"
echo "Next: Run Phase 2 (CPR) on best checkpoint"
echo "  1. Evaluate checkpoints to find best accuracy"
echo "  2. Update train_grpo_cpr_internvl.sh with best checkpoint path"
echo "  3. bash train_grpo_cpr_internvl.sh"
echo "========================================"
