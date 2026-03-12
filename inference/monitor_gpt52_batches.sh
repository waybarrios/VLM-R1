#!/bin/bash
# Monitor GPT-5.2 batch jobs, download results, parse predictions, and evaluate
# Run in tmux: tmux new -s gpt52 'bash inference/monitor_gpt52_batches.sh'

set -e

PROJECT="/gpudata3/Wayner/VLM-R1"
cd "$PROJECT"

export OPENAI_API_KEY=$(cat "$PROJECT/.openai.txt")
export HF_HOME=/gpudata3/hf_cache

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate torch26
export PYTHONPATH="${PROJECT}/src/open-r1-multimodal/src:${PROJECT}/mllm_evaluator:${PYTHONPATH}"

BATCH0="batch_69a0985aea308190b57d8b26c9df19d6"
BATCH1="batch_69a0985bda808190b94e60ea727a2499"
RESULTS0="inference/gpt52_instant_batch_results_part0.jsonl"
RESULTS1="inference/gpt52_instant_batch_results_part1.jsonl"
OUTPUT_DIR="final_table/outputs_testing_gpt52_instant"

echo "========================================"
echo "GPT-5.2 Batch Monitor"
echo "========================================"
echo "Batch 0: $BATCH0 (2824 requests)"
echo "Batch 1: $BATCH1 (3548 requests)"
echo "========================================"

# Step 1: Poll both batches until complete
python3 << 'PYEOF'
import os, time, json
from openai import OpenAI

client = OpenAI()

batches = {
    "part0": {
        "id": "batch_69a0985aea308190b57d8b26c9df19d6",
        "output": "inference/gpt52_instant_batch_results_part0.jsonl",
        "done": False,
    },
    "part1": {
        "id": "batch_69a0985bda808190b94e60ea727a2499",
        "output": "inference/gpt52_instant_batch_results_part1.jsonl",
        "done": False,
    },
}

terminal_states = {"completed", "failed", "expired", "cancelled"}

print(f"[{time.strftime('%H:%M:%S')}] Starting monitor loop (polling every 60s)...\n")

while not all(b["done"] for b in batches.values()):
    for name, info in batches.items():
        if info["done"]:
            continue

        batch = client.batches.retrieve(info["id"])
        counts = batch.request_counts
        completed = counts.completed if counts else 0
        total = counts.total if counts else 0
        failed = counts.failed if counts else 0

        print(f"[{time.strftime('%H:%M:%S')}] {name}: {batch.status} — {completed}/{total} done, {failed} failed")

        if batch.status in terminal_states:
            info["done"] = True
            if batch.status == "completed":
                # Download results
                output_file_id = batch.output_file_id
                if output_file_id:
                    print(f"  Downloading results to {info['output']}...")
                    content = client.files.content(output_file_id)
                    content.write_to_file(info["output"])
                    with open(info["output"]) as f:
                        n = sum(1 for _ in f)
                    print(f"  Saved {n} results!")

                # Download errors if any
                if batch.error_file_id:
                    err_path = info["output"].replace(".jsonl", "_errors.jsonl")
                    err_content = client.files.content(batch.error_file_id)
                    err_content.write_to_file(err_path)
                    print(f"  Errors saved to {err_path}")
            else:
                print(f"  BATCH FAILED: {batch.status}")
                if batch.errors and batch.errors.data:
                    for err in batch.errors.data[:5]:
                        print(f"    {err.code}: {err.message}")

    if not all(b["done"] for b in batches.values()):
        time.sleep(60)

print(f"\n[{time.strftime('%H:%M:%S')}] All batches complete!")
PYEOF

echo ""
echo "========================================"
echo "Step 2: Parsing results into predictions"
echo "========================================"

# Parse and evaluate
python inference/gpt5_batch_parse.py \
    --input "$RESULTS0" "$RESULTS1" \
    --output_dir "$OUTPUT_DIR" \
    --encoder "all-distilroberta-v1" \
    --threshold 0.35 \
    --model_name "GPT-5.2 Instant"

echo ""
echo "========================================"
echo "GPT-5.2 Evaluation Complete!"
echo "========================================"
cat "$OUTPUT_DIR/metrics_summary.json" 2>/dev/null
echo ""
echo "Predictions: $OUTPUT_DIR/predictions/"
echo "Metrics: $OUTPUT_DIR/metrics_summary.json"
