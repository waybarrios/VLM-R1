#!/bin/bash
# Resume Gemini 2.5 Flash evaluation — continues from existing 2404 predictions
# Run in tmux: tmux new -s gemini 'bash inference/run_gemini_flash_resume.sh'

set -e

PROJECT="/gpudata3/Wayner/VLM-R1"
cd "$PROJECT"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate torch26
export PYTHONPATH="${PROJECT}/src/open-r1-multimodal/src:${PROJECT}/mllm_evaluator:${PYTHONPATH}"

echo "========================================"
echo "Gemini 2.5 Flash — Resume Evaluation"
echo "========================================"

# Step 1: Resume inference (skips existing predictions)
echo "Step 1: Resuming inference..."
python inference/run_dartmouth_eval.py \
    --model vertex_ai.gemini-2.5-flash \
    --key_file chart_dt2.txt \
    --output_dir final_table/outputs_testing_gemini_2_5_flash \
    --max_tokens 4096 \
    --workers 4 \
    --resume \
    --no_few_shot \
    --disable_thinking \
    --delay 0.2

# Step 2: Run evaluation with distilroberta τ=0.35
echo ""
echo "Step 2: Running Match F1 evaluation..."
python compute_metrics.py \
    final_table/outputs_testing_gemini_2_5_flash/predictions \
    --dataset-path /gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27 \
    --num-gpus 1 \
    --output-dir final_table/outputs_testing_gemini_2_5_flash

echo ""
echo "========================================"
echo "Gemini 2.5 Flash evaluation complete!"
echo "========================================"
cat final_table/outputs_testing_gemini_2_5_flash/predictions/no_judge_summary.json 2>/dev/null || echo "Check results manually"
