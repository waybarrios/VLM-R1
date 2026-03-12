#!/bin/bash
# InternVL3.5-4B CPR Curriculum Training - RESUMED from checkpoint-100
# CHANGE: max_anyres_num=4 (was 12) for ~3-4x speedup, comparable to Qwen
# Everything else identical to original run

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate internvl35

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3_internvl.json"

export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/gpudata3/hf_cache

# Resume from checkpoint-100 of the current run
MODEL="${PROJECT_ROOT}/output/internvl35_cpr_curriculum_20260309_215942/checkpoint-100"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"
OUTPUT="${PROJECT_ROOT}/output/internvl35_cpr_curriculum_20260309_215942"

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Hyperparameters - same as original
PER_DEVICE_BATCH=5
GRADIENT_ACCUM=2
NUM_GENERATIONS=5
LEARNING_RATE=5e-6
NUM_EPOCHS=2
SAVE_STEPS=100
LOGGING_STEPS=2
SEED=42

# KEY CHANGE: 4 patches instead of 12 (~1,024 vs ~3,072 visual tokens)
MAX_ANYRES_NUM=4

# CPR Settings - same as original
USE_CAUSAL_REWARD=true
CAUSAL_ANSWER_WEIGHT=0.65
CAUSAL_STEP_WEIGHT=0.35
USE_PCGRAD=true

mkdir -p $OUTPUT
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "InternVL3.5-4B CPR RESUMED (anyres=4)"
echo "========================================"
echo "Resume from: $MODEL"
echo "KEY CHANGE: max_anyres_num=4 (was 12)"
echo "  - Expected speedup: ~3-4x"
echo "  - Visual tokens: ~1,024 (was ~3,072)"
echo "  - CPR weights: aw=$CAUSAL_ANSWER_WEIGHT sw=$CAUSAL_STEP_WEIGHT"
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
    --resume_from_checkpoint $MODEL \
    --dataset_name $DATASET \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --reward_weights 2.0 2.0 2.0 \
    --use_causal_reasoning_reward $USE_CAUSAL_REWARD \
    --causal_answer_weight $CAUSAL_ANSWER_WEIGHT \
    --causal_step_weight $CAUSAL_STEP_WEIGHT \
    --use_pcgrad $USE_PCGRAD \
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
    --bf16 \
    --deepspeed $DEEPSPEED_CONFIG \
    2>&1 | tee -a $OUTPUT/training.log

echo "========================================"
echo "CPR Curriculum RESUMED completed!"
echo "========================================"
