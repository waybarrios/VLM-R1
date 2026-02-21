#!/usr/bin/env python3
"""
Find examples where:
- Baseline has vague/uncertain/low-relevance steps (exploratory)
- GRPO has only high-confidence, relevant steps
- Both get correct answer
"""

import json
from pathlib import Path
from datasets import load_from_disk

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
            pass
    return predictions

def has_vague_patterns(steps):
    """Check if steps contain vague/uncertain patterns."""
    vague_keywords = [
        'noted', 'observed', 'recognized', 'inferred', 'assumed',
        'likely', 'probably', 'seems', 'appears', 'might',
        'possibly', 'perhaps', 'could be', 'may be'
    ]

    vague_count = 0
    for step in steps:
        step_lower = step.lower()
        if any(kw in step_lower for kw in vague_keywords):
            vague_count += 1

    return vague_count, len(steps), vague_count / max(len(steps), 1)

def has_repetition(steps):
    """Check if steps have repetitive patterns."""
    if len(steps) < 2:
        return False

    # Check for near-identical consecutive steps
    repetitive = 0
    for i in range(len(steps) - 1):
        # Simple word overlap check
        words_a = set(steps[i].lower().split())
        words_b = set(steps[i+1].lower().split())
        overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
        if overlap > 0.7:
            repetitive += 1

    return repetitive > len(steps) * 0.3

def main():
    baseline_dir = "/gpudata3/Wayner/reasoning/outputs_testing_qwen25vl_3b"
    grpo_dir = "/gpudata3/Wayner/VLM-R1/predictions_final/checkpoint-1400"
    dataset_path = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v12"

    print("Loading predictions...")
    baseline_preds = load_predictions(baseline_dir)
    grpo_preds = load_predictions(grpo_dir)
    dataset = load_from_disk(dataset_path)

    print("Analyzing examples for vague baseline steps vs. confident GRPO steps...")

    candidates = []

    for idx in sorted(set(baseline_preds.keys()) & set(grpo_preds.keys())):
        baseline = baseline_preds[idx]
        grpo = grpo_preds[idx]

        baseline_steps = baseline.get('reasoning_steps', [])
        grpo_steps = grpo.get('reasoning_steps', [])

        baseline_ans = baseline.get('answer', baseline.get('predicted_answer', ''))
        grpo_ans = grpo.get('answer', grpo.get('predicted_answer', ''))

        if not baseline_ans or not grpo_ans:
            continue

        # Get ground truth
        try:
            gt_ans = dataset[idx]['answer']
        except:
            continue

        # Check if both correct (normalize strings)
        baseline_correct = str(baseline_ans).strip().lower() in str(gt_ans).strip().lower() or \
                          str(gt_ans).strip().lower() in str(baseline_ans).strip().lower()
        grpo_correct = str(grpo_ans).strip().lower() in str(gt_ans).strip().lower() or \
                      str(gt_ans).strip().lower() in str(grpo_ans).strip().lower()

        if not (baseline_correct and grpo_correct):
            continue

        # Analyze baseline for vague patterns
        baseline_vague_count, baseline_total, baseline_vague_ratio = has_vague_patterns(baseline_steps)
        grpo_vague_count, grpo_total, grpo_vague_ratio = has_vague_patterns(grpo_steps)

        # Check for repetition in baseline
        baseline_repetitive = has_repetition(baseline_steps)

        # We want: high vagueness in baseline, low in GRPO
        if baseline_vague_ratio > 0.5 and grpo_vague_ratio < 0.3 and baseline_total >= 5:
            candidates.append({
                'idx': idx,
                'baseline_steps': baseline_total,
                'grpo_steps': grpo_total,
                'baseline_vague_ratio': baseline_vague_ratio,
                'grpo_vague_ratio': grpo_vague_ratio,
                'baseline_vague_count': baseline_vague_count,
                'baseline_repetitive': baseline_repetitive,
                'baseline_reasoning': baseline_steps,
                'grpo_reasoning': grpo_steps,
                'question': dataset[idx]['question'],
                'ground_truth': gt_ans,
                'baseline_answer': baseline_ans,
                'grpo_answer': grpo_ans,
            })

    # Sort by vagueness difference
    candidates.sort(key=lambda x: x['baseline_vague_ratio'] - x['grpo_vague_ratio'], reverse=True)

    print(f"\nFound {len(candidates)} examples with vague baseline steps")
    print("\n" + "="*80)
    print("TOP 15 EXAMPLES WITH VAGUE/UNCERTAIN BASELINE STEPS")
    print("="*80)

    for i, ex in enumerate(candidates[:15], 1):
        print(f"\n{i}. Example {ex['idx']}")
        print(f"   Baseline: {ex['baseline_steps']} steps, {ex['baseline_vague_count']} vague ({ex['baseline_vague_ratio']:.1%})")
        print(f"   GRPO: {ex['grpo_steps']} steps, vague ratio: {ex['grpo_vague_ratio']:.1%}")
        print(f"   Repetitive baseline: {ex['baseline_repetitive']}")
        print(f"   Question: {ex['question'][:100]}...")
        print(f"   Ground Truth: {ex['ground_truth']}")
        print(f"\n   Baseline Steps (showing vague/uncertain reasoning):")
        for j, step in enumerate(ex['baseline_reasoning'][:8], 1):
            marker = "🟡" if any(kw in step.lower() for kw in ['noted', 'observed', 'inferred', 'assumed', 'likely']) else "  "
            print(f"      {marker} {j}. {step[:90]}...")
        if ex['baseline_steps'] > 8:
            print(f"      ... and {ex['baseline_steps']-8} more")

        print(f"\n   GRPO Steps (high-confidence, direct):")
        for j, step in enumerate(ex['grpo_reasoning'][:8], 1):
            print(f"      {j}. {step[:90]}...")
        if ex['grpo_steps'] > 8:
            print(f"      ... and {ex['grpo_steps']-8} more")

    # Save results
    output = {
        'summary': {
            'total_candidates': len(candidates),
            'avg_baseline_vague_ratio': sum(x['baseline_vague_ratio'] for x in candidates) / len(candidates) if candidates else 0,
            'avg_grpo_vague_ratio': sum(x['grpo_vague_ratio'] for x in candidates) / len(candidates) if candidates else 0,
        },
        'top_examples': candidates[:15]
    }

    output_file = "/gpudata3/Wayner/VLM-R1/vague_steps_examples.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
