#!/usr/bin/env python3
"""Check details for top Wrong+Sound candidates"""

from datasets import load_from_disk
import json

# Load dataset
dataset_path = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
dataset = load_from_disk(dataset_path)

# Top candidates from our search
candidates = [32, 15, 3911, 4075, 4096, 4760]

print("=" * 100)
print("CHECKING TOP WRONG+SOUND CANDIDATES (5-6 steps)")
print("=" * 100)

for idx in candidates:
    sample = dataset[idx]

    print(f"\n{'='*100}")
    print(f"SAMPLE {idx}")
    print(f"{'='*100}")
    print(f"Question: {sample['question'][:200]}...")
    print(f"\nGround Truth Answer: {sample['answer']}")
    print(f"Model Predicted: {sample.get('predicted_answer', 'N/A')}")
    print(f"\nReference Steps ({len(sample['reference_steps'])}):")
    for i, step in enumerate(sample['reference_steps'][:8], 1):
        print(f"  {i}. {step[:150]}...")
    if len(sample['reference_steps']) > 8:
        print(f"  ... [{len(sample['reference_steps'])-8} more steps]")

    print(f"\nSource: {sample.get('dataset_source', 'Unknown')}")
    print(f"Complexity: {sample.get('complexity_level', 'Unknown')}")

    # Load model prediction
    pred_path = f"/gpudata3/Wayner/reasoning/outputs_testing_qwen3vl_32b/{idx}.json"
    try:
        with open(pred_path) as f:
            pred = json.load(f)
        print(f"\nModel Steps ({len(pred['reasoning_steps'])}):")
        for i, step in enumerate(pred['reasoning_steps'], 1):
            print(f"  {i}. {step}")
        print(f"Model Answer: {pred['answer']}")
    except:
        print(f"\nCouldn't load prediction file")

print("\n\n" + "="*100)
print("RECOMMENDATION: Pick the one with:")
print("  1. Clear visual problem (not just text/biology)")
print("  2. Obvious perceptual error")
print("  3. Clean formatting (5-6 steps)")
print("="*100)
