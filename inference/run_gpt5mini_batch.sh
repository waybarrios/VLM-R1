#!/bin/bash
# GPT-5-mini batch evaluation on CRYSTAL (6,372 samples)
# Batch API = 50% off (~$2 total)
#
# Usage: run inside tmux with conda reasoning env
#   tmux new -s gpt5mini
#   conda activate reasoning
#   bash inference/run_gpt5mini_batch.sh

set -e

cd /gpudata3/Wayner/VLM-R1

echo "=============================================="
echo "GPT-5-mini Batch Evaluation — CRYSTAL"
echo "Model: gpt-5-mini | Tokens: 16384 | Batch API"
echo "=============================================="
date

# Step 1: Prepare JSONL
echo ""
echo "[Step 1/3] Generating batch JSONL..."
python inference/gpt5_batch_prepare.py \
    --model gpt-5-mini \
    --max_tokens 16384 \
    --output inference/gpt5mini_batch_requests.jsonl

# Step 2: Submit each part (prepare may split at 190MB)
echo ""
echo "[Step 2/3] Submitting batch job(s)..."

RESULT_FILES=""

# Part 0 (always exists)
if [ -f inference/gpt5mini_batch_requests.jsonl ]; then
    echo "--- Submitting part 0 ---"
    python inference/gpt5_batch_submit.py \
        --input inference/gpt5mini_batch_requests.jsonl \
        --output inference/gpt5mini_batch_results.jsonl
    RESULT_FILES="inference/gpt5mini_batch_results.jsonl"
fi

# Part 1 (exists if dataset > 190MB)
if [ -f inference/gpt5mini_batch_requests_part1.jsonl ]; then
    echo "--- Submitting part 1 ---"
    python inference/gpt5_batch_submit.py \
        --input inference/gpt5mini_batch_requests_part1.jsonl \
        --output inference/gpt5mini_batch_results_part1.jsonl
    RESULT_FILES="$RESULT_FILES inference/gpt5mini_batch_results_part1.jsonl"
fi

# Part 2 (unlikely but just in case)
if [ -f inference/gpt5mini_batch_requests_part2.jsonl ]; then
    echo "--- Submitting part 2 ---"
    python inference/gpt5_batch_submit.py \
        --input inference/gpt5mini_batch_requests_part2.jsonl \
        --output inference/gpt5mini_batch_results_part2.jsonl
    RESULT_FILES="$RESULT_FILES inference/gpt5mini_batch_results_part2.jsonl"
fi

# Step 3: Parse all results + evaluate
echo ""
echo "[Step 3/3] Parsing results + running evaluation..."
python inference/gpt5_batch_parse.py \
    --input $RESULT_FILES \
    --output_dir final_table/outputs_testing_gpt5mini \
    --model_name "GPT-5-mini"

echo ""
echo "=============================================="
echo "DONE! Results in: final_table/outputs_testing_gpt5mini/"
echo "=============================================="
date
