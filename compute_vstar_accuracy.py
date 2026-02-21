#!/usr/bin/env python3
"""
Compute accuracy from vstar_bench_reasoning samples.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

def compute_accuracy(samples_file):
    """Compute accuracy from samples JSONL file."""

    # Read all samples
    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Aggregate by category
    category_scores = defaultdict(list)
    all_scores = []

    for sample in samples:
        # Get overall accuracy info
        if 'vstar_overall_acc' in sample:
            acc_info = sample['vstar_overall_acc']
            category = acc_info['category']
            score = acc_info['score']

            category_scores[category].append(score)
            all_scores.append(score)

            # Log incorrect predictions
            if score == 0:
                print(f"❌ Q{acc_info['question_id']} ({category}): pred={acc_info['prediction']}, gt={acc_info['ground_truth']}")
                print(f"   Answer: {acc_info['answer_text']}")

    # Print results
    print("\n" + "="*60)
    print("V* Bench Reasoning - Accuracy Results")
    print("="*60)

    for category, scores in category_scores.items():
        if scores:
            acc = sum(scores) / len(scores) * 100.0
            print(f"{category:30s}: {acc:6.2f}% (n={len(scores)})")

    if all_scores:
        overall_acc = sum(all_scores) / len(all_scores) * 100.0
        print(f"{'Overall':30s}: {overall_acc:6.2f}% (n={len(all_scores)})")
        print("="*60)

        return overall_acc

    return 0.0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_vstar_accuracy.py <samples.jsonl>")
        sys.exit(1)

    samples_file = sys.argv[1]

    if not Path(samples_file).exists():
        print(f"Error: File not found: {samples_file}")
        sys.exit(1)

    accuracy = compute_accuracy(samples_file)
    print(f"\n✓ Final Accuracy: {accuracy:.2f}%")
