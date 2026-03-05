#!/bin/bash
# Resume Run 2 Extended (aw=0.70, sw=0.30) from checkpoint-2000
# Crashed at step 2056 due to OOM (another process took 15.85 GiB on GPU 1)
# Previous resumes: checkpoint-700 (crash@788), checkpoint-1500 (crash@2056)

set -e

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json"
PYTHON="/scratch/miniconda3/envs/torch26/bin/python"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate torch26

export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/gpudata3/hf_cache
export DEBUG_MODE="true"

RUN2_EXT_DIR="${PROJECT_ROOT}/output/run2_extended_aw0.70_sw0.30"
export LOG_PATH="${RUN2_EXT_DIR}/reward.txt"

RUN2_EXT_DIR="${PROJECT_ROOT}/output/run2_extended_aw0.70_sw0.30"
RUN2_RESUME="${PROJECT_ROOT}/ablation_grpo/run_2_aw0.70_sw0.30/checkpoint-400"
NUM_GPUS=4

echo "╔══════════════════════════════════════════════════════╗"
echo "║  RESUME Run 2 Extended (aw=0.70, sw=0.30)           ║"
echo "║  From checkpoint-2000 → 2800 steps                   ║"
echo "║  Started: $(date)                                    ║"
echo "╚══════════════════════════════════════════════════════╝"

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file $DEEPSPEED_CONFIG \
    --zero3_init_flag true \
    --zero3_save_16bit_model true \
    --gradient_accumulation_steps 2 \
    ${SRC_DIR}/open_r1/grpo_rec.py \
    --model_name_or_path $RUN2_RESUME \
    --dataset_name /gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --reward_weights 2.0 2.0 2.0 \
    --use_causal_reasoning_reward true \
    --causal_answer_weight 0.70 \
    --causal_step_weight 0.30 \
    --use_pcgrad true \
    --output_dir $RUN2_EXT_DIR \
    --seed 42 \
    --shuffle_train_dataset \
    --max_steps 2800 \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --num_generations 5 \
    --gradient_checkpointing \
    --logging_steps 2 \
    --save_steps 100 \
    --max_pixels 602112 \
    --min_pixels 3136 \
    --bf16 \
    --deepspeed $DEEPSPEED_CONFIG \
    --resume_from_checkpoint "${RUN2_EXT_DIR}/checkpoint-2000" \
    2>&1 | tee -a ${RUN2_EXT_DIR}/training.log

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Run 2 Extended COMPLETE (2800 steps)                ║"
echo "║  Finished: $(date)                                   ║"
echo "╚══════════════════════════════════════════════════════╝"
