#!/bin/bash
# GPT-5.2 Instant Batch API — Full pipeline with multi-part support
# Run in tmux: tmux new -s gpt52 'bash inference/run_gpt52_batch.sh'

set -e

PROJECT="/gpudata3/Wayner/VLM-R1"
cd "$PROJECT"

export OPENAI_API_KEY=$(cat "$PROJECT/.openai.txt")
export HF_HOME=/gpudata3/hf_cache

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate torch26
export PYTHONPATH="${PROJECT}/src/open-r1-multimodal/src:${PROJECT}/mllm_evaluator:${PYTHONPATH}"

MODEL_DISPLAY="GPT-5.2 Instant"
OUTPUT_DIR="final_table/outputs_testing_gpt52_instant"
PART0="inference/gpt52_instant_batch_requests.jsonl"
PART1="inference/gpt52_instant_batch_requests_part1.jsonl"
RESULTS0="inference/gpt52_instant_batch_results_part0.jsonl"
RESULTS1="inference/gpt52_instant_batch_results_part1.jsonl"

echo "========================================"
echo "$MODEL_DISPLAY Batch Evaluation Pipeline"
echo "========================================"
echo "2 batch files to submit"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Step 1: Submit part 0
echo ""
echo "Step 1a: Submitting part 0 (2824 requests)..."
python inference/gpt5_batch_submit.py \
    --input "$PART0" \
    --output "$RESULTS0" \
    --poll_interval 60

# Step 2: Submit part 1
echo ""
echo "Step 1b: Submitting part 1 (3548 requests)..."
python inference/gpt5_batch_submit.py \
    --input "$PART1" \
    --output "$RESULTS1" \
    --poll_interval 60

# Step 3: Parse both result files and evaluate with distilroberta τ=0.35
echo ""
echo "Step 2: Parsing results and evaluating..."
python inference/gpt5_batch_parse.py \
    --input "$RESULTS0" "$RESULTS1" \
    --output_dir "$OUTPUT_DIR" \
    --encoder "all-distilroberta-v1" \
    --threshold 0.35 \
    --model_name "$MODEL_DISPLAY"

echo ""
echo "========================================"
echo "$MODEL_DISPLAY evaluation complete!"
echo "========================================"
cat "$OUTPUT_DIR/metrics_summary.json"
