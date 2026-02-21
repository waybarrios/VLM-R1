#!/usr/bin/env python3
"""Check details for top Wrong+Sound candidates - simple version"""

from datasets import load_dataset
import json

# Load dataset using Arrow format
dataset_path = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
dataset = load_dataset("arrow", data_files={'test': dataset_path + '/data-00000-of-00001.arrow'})['test']

print(f"Dataset loaded: {len(dataset)} samples\n")

# Top candidates from our search (5-6 steps, F1=1.0, Wrong answer)
candidates = [32, 15, 3911, 4075]

for idx in candidates:
    sample = dataset[idx]

    print(f"\n{'='*100}")
    print(f"SAMPLE {idx} - {len(sample['reference_steps'])} reference steps")
    print(f"{'='*100}")
    print(f"Question: {sample['question'][:300]}")
    print(f"\nCorrect Answer: {sample['answer']}")

    # Load model prediction
    pred_path = f"/gpudata3/Wayner/reasoning/outputs_testing_qwen3vl_32b/{idx}.json"
    with open(pred_path) as f:
        pred = json.load(f)

    print(f"Model Answer: {pred['answer']} (WRONG)")
    print(f"\nModel Reasoning ({len(pred['reasoning_steps'])} steps):")
    for i, step in enumerate(pred['reasoning_steps'], 1):
        print(f"  {i}. {step}")

    print(f"\nReference Steps ({len(sample['reference_steps'])} steps):")
    for i, step in enumerate(sample['reference_steps'][:8], 1):
        print(f"  {i}. {step[:100]}...")

print("\n" + "="*100)
print("BEST CHOICE: Sample with shortest, clearest visual error")
print("="*100)
