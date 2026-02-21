#!/usr/bin/env python3
"""
Find examples demonstrating learned conservatism:
- Baseline: generates exploratory/low-confidence steps (many steps, lower precision)
- GRPO Ckpt-1400: suppresses uncertain steps, only high-confidence (fewer steps, higher precision)
- Both get correct answer
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def load_predictions(directory):
    """Load all prediction files from a directory."""
    predictions = {}
    for file in sorted(Path(directory).glob("*.json")):
        idx = int(file.stem)
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                predictions[idx] = data
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return predictions

def count_steps(pred):
    """Count reasoning steps in a prediction."""
    if 'reasoning_steps' in pred:
        return len(pred['reasoning_steps'])
    return 0

def get_answer(pred):
    """Extract final answer."""
    return pred.get('predicted_answer', pred.get('answer', ''))

def analyze_example(baseline_pred, grpo_pred, idx):
    """Analyze a single example for conservatism behavior."""
    baseline_steps = count_steps(baseline_pred)
    grpo_steps = count_steps(grpo_pred)

    baseline_ans = get_answer(baseline_pred)
    grpo_ans = get_answer(grpo_pred)

    # We want: baseline has more steps (exploratory), GRPO has fewer (conservative)
    # Both should have valid answers
    if baseline_ans and grpo_ans:
        step_reduction = baseline_steps - grpo_steps
        if step_reduction > 0:  # GRPO reduced steps
            return {
                'idx': idx,
                'baseline_steps': baseline_steps,
                'grpo_steps': grpo_steps,
                'step_reduction': step_reduction,
                'baseline_answer': baseline_ans,
                'grpo_answer': grpo_ans,
                'baseline_reasoning': baseline_pred.get('reasoning_steps', []),
                'grpo_reasoning': grpo_pred.get('reasoning_steps', []),
                'question': baseline_pred.get('question', ''),
                'ground_truth': baseline_pred.get('ground_truth_answer', baseline_pred.get('answer', '')),
            }
    return None

def main():
    baseline_dir = "/gpudata3/Wayner/reasoning/outputs_testing_qwen25vl_3b"
    grpo_dir = "/gpudata3/Wayner/VLM-R1/predictions_final/checkpoint-1400"

    print("Loading baseline predictions...")
    baseline_preds = load_predictions(baseline_dir)
    print(f"Loaded {len(baseline_preds)} baseline predictions")

    print("\nLoading GRPO checkpoint-1400 predictions...")
    grpo_preds = load_predictions(grpo_dir)
    print(f"Loaded {len(grpo_preds)} GRPO predictions")

    # Find common indices
    common_indices = set(baseline_preds.keys()) & set(grpo_preds.keys())
    print(f"\nFound {len(common_indices)} common examples")

    # Analyze all examples
    conservatism_examples = []
    for idx in sorted(common_indices):
        result = analyze_example(baseline_preds[idx], grpo_preds[idx], idx)
        if result:
            conservatism_examples.append(result)

    print(f"\nFound {len(conservatism_examples)} examples showing step reduction")

    # Sort by step reduction (most dramatic reduction first)
    conservatism_examples.sort(key=lambda x: x['step_reduction'], reverse=True)

    # Print top 20 examples
    print("\n" + "="*80)
    print("TOP 20 EXAMPLES SHOWING LEARNED CONSERVATISM")
    print("="*80)

    for i, ex in enumerate(conservatism_examples[:20], 1):
        print(f"\n{i}. Example {ex['idx']}")
        print(f"   Step Reduction: {ex['baseline_steps']} → {ex['grpo_steps']} ({-ex['step_reduction']} steps)")
        print(f"   Question: {ex['question'][:100]}...")
        print(f"   Ground Truth: {ex['ground_truth']}")
        print(f"   Baseline Answer: {ex['baseline_answer']}")
        print(f"   GRPO Answer: {ex['grpo_answer']}")
        print(f"   \n   Baseline Steps ({ex['baseline_steps']}):")
        for j, step in enumerate(ex['baseline_reasoning'][:5], 1):
            print(f"      {j}. {step[:80]}...")
        if ex['baseline_steps'] > 5:
            print(f"      ... and {ex['baseline_steps']-5} more")
        print(f"   \n   GRPO Steps ({ex['grpo_steps']}):")
        for j, step in enumerate(ex['grpo_reasoning'], 1):
            print(f"      {j}. {step[:80]}...")

    # Save detailed results for top 10
    output = {
        'summary': {
            'total_examples': len(conservatism_examples),
            'avg_step_reduction': sum(ex['step_reduction'] for ex in conservatism_examples) / len(conservatism_examples) if conservatism_examples else 0,
        },
        'top_examples': conservatism_examples[:10]
    }

    output_file = "/gpudata3/Wayner/VLM-R1/conservatism_examples.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
