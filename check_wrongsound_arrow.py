#!/usr/bin/env python3
"""Check details for top Wrong+Sound candidates using pyarrow directly"""

import pyarrow.parquet as pq
import json
import glob

# Load all arrow files
dataset_path = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
arrow_files = sorted(glob.glob(f"{dataset_path}/*.arrow"))

print(f"Loading {len(arrow_files)} arrow files...\n")

# Read all data
import pyarrow as pa
tables = []
for arrow_file in arrow_files:
    with pa.memory_map(arrow_file, 'r') as source:
        tables.append(pa.ipc.open_file(source).read_all())

# Concatenate
full_table = pa.concat_tables(tables)
print(f"Total samples: {len(full_table)}\n")

# Top candidates (5-6 steps, F1=1.0, Wrong answer)
candidates = [32, 15, 3911, 4075]

for idx in candidates:
    row = full_table.slice(idx, 1).to_pydict()

    # Extract single values
    question = row['question'][0]
    answer = row['answer'][0]
    reference_steps = row['reference_steps'][0]

    print(f"\n{'='*100}")
    print(f"SAMPLE {idx} - {len(reference_steps)} reference steps")
    print(f"{'='*100}")
    print(f"Question: {question[:300]}")
    if len(question) > 300:
        print("  ...")
    print(f"\nCorrect Answer: {answer}")

    # Load model prediction
    pred_path = f"/gpudata3/Wayner/reasoning/outputs_testing_qwen3vl_32b/{idx}.json"
    with open(pred_path) as f:
        pred = json.load(f)

    print(f"Model Answer: {pred['answer']} ❌ WRONG")
    print(f"\nModel Reasoning ({len(pred['reasoning_steps'])} steps):")
    for i, step in enumerate(pred['reasoning_steps'], 1):
        print(f"  {i}. {step}")

    print(f"\nReference Steps ({len(reference_steps)} steps):")
    for i, step in enumerate(reference_steps, 1):
        display_step = step if len(step) <= 100 else step[:97] + "..."
        print(f"  {i}. {display_step}")

print("\n" + "="*100)
print("RECOMMENDATION: Pick sample with shortest, clearest visual/perceptual error")
print("="*100)
