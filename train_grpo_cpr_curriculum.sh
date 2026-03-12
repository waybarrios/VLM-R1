#!/bin/bash
# CPR Curriculum Training (Phase 2)
# Base: Answer-Only checkpoint-1400 (acc 44.90%, stable)
# Goal: Add reasoning via CPR WITHOUT losing accuracy
# Strategy: Lower LR (5e-6) + save frequently to catch best balance

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json"

export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# KEY CHANGE: Start from Answer-Only best checkpoint instead of base model
MODEL="${PROJECT_ROOT}/output/GRPO_answer_only_baseline_20260129_202515/checkpoint-1400"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"
OUTPUT="${PROJECT_ROOT}/output/grpo-cpr-curriculum-$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

# Hyperparameters - lower LR to preserve accuracy from phase 1
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

# CPR Settings - slightly higher answer weight to protect accuracy
USE_CAUSAL_REWARD=true
CAUSAL_ANSWER_WEIGHT=0.65
CAUSAL_STEP_WEIGHT=0.35
USE_PCGRAD=true

mkdir -p $OUTPUT
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "CPR Curriculum Training (Phase 2)"
echo "========================================"
echo "Base model: Answer-Only checkpoint-1400 (acc 44.90%)"
echo "Strategy: CPR + PCGrad on top of pre-trained accuracy"
echo "  - Learning Rate: $LEARNING_RATE (half of original)"
echo "  - Causal Answer Weight: $CAUSAL_ANSWER_WEIGHT"
echo "  - Causal Step Weight: $CAUSAL_STEP_WEIGHT"
echo "  - PCGrad: $USE_PCGRAD"
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
    2>&1 | tee $OUTPUT/training.log

echo "========================================"
echo "CPR Curriculum Training completed!"
echo "Next: Evaluate checkpoints"
echo "  python inference/evaluate_predictions.py --predictions_dir ... --test_dataset_path ..."
echo "========================================"
