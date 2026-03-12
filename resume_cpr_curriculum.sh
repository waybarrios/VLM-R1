#!/bin/bash
# Resume CPR Curriculum Training from checkpoint-2800
# Continues from step 2800 → 7578 (4,778 remaining steps)
# All optimizer states, scheduler, and RNG restored

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"

# Activate conda env (same as original training: torch26 / Python 3.13)
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate torch26

SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json"

export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Same model as original run (Answer-Only checkpoint-1400)
MODEL="${PROJECT_ROOT}/output/GRPO_answer_only_baseline_20260129_202515/checkpoint-1400"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"

# CRITICAL: Point to the SAME output dir to find checkpoint-2800
OUTPUT="${PROJECT_ROOT}/output/grpo-cpr-curriculum-20260221_163422"

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Same hyperparameters as original run
PER_DEVICE_BATCH=5
GRADIENT_ACCUM=2
NUM_GENERATIONS=5
LEARNING_RATE=5e-6
NUM_EPOCHS=2
SAVE_STEPS=100
LOGGING_STEPS=2
SEED=42

MAX_PIXELS=602112
MIN_PIXELS=3136

# CPR Settings (same as original)
USE_CAUSAL_REWARD=true
CAUSAL_ANSWER_WEIGHT=0.65
CAUSAL_STEP_WEIGHT=0.35
USE_PCGRAD=true

export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "CPR Curriculum Training — RESUME from step 2800"
echo "========================================"
echo "Checkpoint: ${OUTPUT}/checkpoint-2800"
echo "Remaining: steps 2800 → 7578 (~4,778 steps)"
echo "GPUs: 0,1,2,3"
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
    --max_pixels $MAX_PIXELS \
    --min_pixels $MIN_PIXELS \
    --bf16 \
    --deepspeed $DEEPSPEED_CONFIG \
    --resume_from_checkpoint "${OUTPUT}/checkpoint-2800" \
    2>&1 | tee -a $OUTPUT/training.log

echo "========================================"
echo "CPR Curriculum Training completed!"
echo "========================================"
