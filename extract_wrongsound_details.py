#!/usr/bin/env python3
"""Extract details for top Wrong+Sound candidates"""

from datasets import load_from_disk
import json

# Load dataset correctly
save_path = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
test = load_from_disk(save_path)

print(f"Dataset loaded: {len(test)} samples\n")

# Top candidates (5-6 steps, F1=1.0, Wrong answer)
candidates = [32, 15, 3911, 4075]

for idx in candidates:
    sample = test[idx]

    print(f"\n{'='*100}")
    print(f"SAMPLE {idx} - {len(sample['reference_steps'])} reference steps")
    print(f"{'='*100}")
    print(f"Question: {sample['question'][:400]}")
    if len(sample['question']) > 400:
        print("  ...")
    print(f"\nCorrect Answer: {sample['answer']}")

    # Load model prediction
    pred_path = f"/gpudata3/Wayner/reasoning/outputs_testing_qwen3vl_32b/{idx}.json"
    with open(pred_path) as f:
        pred = json.load(f)

    print(f"Model Answer: {pred['answer']} ❌ (WRONG)")
    print(f"\nModel Reasoning ({len(pred['reasoning_steps'])} steps):")
    for i, step in enumerate(pred['reasoning_steps'], 1):
        print(f"  {i}. {step}")

    print(f"\nReference Steps ({len(sample['reference_steps'])} steps):")
    for i, step in enumerate(sample['reference_steps'], 1):
        display_step = step if len(step) <= 150 else step[:147] + "..."
        print(f"  {i}. {display_step}")

    # Check if there are choices
    if 'choices' in sample and sample['choices']:
        print(f"\nChoices: {sample['choices']}")

print("\n" + "="*100)
print("✓ BEST CHOICE: Sample 32 (palm trees) - clear visual, only 5 steps")
print("="*100)
