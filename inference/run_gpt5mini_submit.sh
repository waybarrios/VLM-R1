#!/bin/bash
# GPT-5-mini: Submit + Parse (JSONL already generated)
#
# Usage:
#   tmux new -s gpt5mini
#   export OPENAI_API_KEY="sk-..."
#   conda activate reasoning
#   bash inference/run_gpt5mini_submit.sh

set -e

cd /gpudata3/Wayner/VLM-R1

# Load API key from file if not already set
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f /gpudata3/Wayner/VLM-R1/.new.txt ]; then
        export OPENAI_API_KEY=$(cat /gpudata3/Wayner/VLM-R1/.new.txt | tr -d '[:space:]')
        echo "Loaded API key from .new.txt"
    else
        echo "ERROR: OPENAI_API_KEY not set and .new.txt not found!"
        exit 1
    fi
fi

echo "=============================================="
echo "GPT-5-mini — Submit & Evaluate"
echo "API key: ${OPENAI_API_KEY:0:8}..."
echo "=============================================="
date

# Step 2: Submit each part
echo ""
echo "[Step 1/2] Submitting batch job(s)..."

RESULT_FILES=""

echo "--- Submitting part 0 (2826 requests, 190 MB) ---"
python inference/gpt5_batch_submit.py \
    --input inference/gpt5mini_batch_requests.jsonl \
    --output inference/gpt5mini_batch_results.jsonl

RESULT_FILES="inference/gpt5mini_batch_results.jsonl"

echo ""
echo "--- Submitting part 1 (3546 requests, 125 MB) ---"
python inference/gpt5_batch_submit.py \
    --input inference/gpt5mini_batch_requests_part1.jsonl \
    --output inference/gpt5mini_batch_results_part1.jsonl

RESULT_FILES="$RESULT_FILES inference/gpt5mini_batch_results_part1.jsonl"

# Step 3: Parse all results + evaluate
echo ""
echo "[Step 2/2] Parsing results + running evaluation..."
python inference/gpt5_batch_parse.py \
    --input $RESULT_FILES \
    --output_dir final_table/outputs_testing_gpt5mini \
    --model_name "GPT-5-mini"

echo ""
echo "=============================================="
echo "DONE! Results in: final_table/outputs_testing_gpt5mini/"
echo "=============================================="
date
