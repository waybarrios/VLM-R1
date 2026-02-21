#!/usr/bin/env python3
"""
Find the best example to illustrate GRPO improvement:
- Baseline: cherry-picks steps (high precision, low recall)
- Checkpoint-300: better coverage (higher recall, maintains precision)
"""
import json
import sys
from pathlib import Path
from datasets import load_from_disk
import numpy as np

# Paths
DATASET_PATH = "/jumbo/jinlab/Wayner/reasoning_test_with_reference_steps_updated_v27"
BASELINE_PATH = "/gpudata3/Wayner/reasoning/outputs_testing_qwen25vl_3b"
CHECKPOINT_1400_PATH = "/gpudata3/Wayner/VLM-R1/predictions_final/checkpoint-1400"

print("=" * 80)
print("FINDING BEST GRPO EXAMPLE FOR PAPER")
print("=" * 80)
print()

# Load dataset
print("Loading CRYSTAL dataset...")
dataset = load_from_disk(DATASET_PATH)
print(f"✓ Loaded {len(dataset)} samples")
print()

# Load baseline predictions
print("Loading baseline predictions...")
baseline_preds = {}
for json_file in Path(BASELINE_PATH).glob("*.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
        idx = int(json_file.stem)  # Just the number, no "sample_" prefix
        baseline_preds[idx] = data
print(f"✓ Loaded {len(baseline_preds)} baseline predictions")
print()

# Load checkpoint-1400 predictions
print("Loading checkpoint-1400 predictions...")
ckpt1400_preds = {}
for json_file in Path(CHECKPOINT_1400_PATH).glob("*.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
        idx = int(json_file.stem)  # Just the number, no "sample_" prefix
        ckpt1400_preds[idx] = data
print(f"✓ Loaded {len(ckpt1400_preds)} checkpoint-1400 predictions")
print()

# Find examples matching our criteria
print("Searching for examples that illustrate cherry-picking vs improved coverage...")
print()

candidates = []

for idx in range(len(dataset)):
    if idx not in baseline_preds or idx not in ckpt1400_preds:
        continue

    sample = dataset[idx]
    baseline = baseline_preds[idx]
    ckpt1400 = ckpt1400_preds[idx]

    # Get reference steps
    ref_steps = sample.get('reference_steps', [])
    if not ref_steps or len(ref_steps) < 5:  # Want substantial reasoning
        continue

    # Get predicted steps (field name is 'reasoning_steps', not 'predicted_reasoning_steps')
    baseline_steps = baseline.get('reasoning_steps', [])
    ckpt1400_steps = ckpt1400.get('reasoning_steps', [])

    if not baseline_steps or not ckpt1400_steps:
        continue

    # Calculate coverage (as proxy for recall)
    baseline_coverage = len(baseline_steps) / len(ref_steps)
    ckpt1400_coverage = len(ckpt1400_steps) / len(ref_steps)

    # We want:
    # 1. Baseline cherry-picks (low coverage, like 2-3 steps out of 8+)
    # 2. Checkpoint-1400 improves coverage (generates more steps)
    # 3. Both get the final answer correct (shows reasoning improvement, not just guessing)
    # 4. Visual example (has image)

    baseline_answer = baseline.get('answer', '').strip().lower()
    ckpt1400_answer = ckpt1400.get('answer', '').strip().lower()
    gt_answer = str(sample.get('answer', '')).strip().lower()

    # Check if both answer correctly
    baseline_correct = baseline_answer == gt_answer or baseline_answer in gt_answer or gt_answer in baseline_answer
    ckpt1400_correct = ckpt1400_answer == gt_answer or ckpt1400_answer in gt_answer or gt_answer in ckpt1400_answer

    # BOTH must be correct to show that GRPO improves reasoning while maintaining correctness
    if not (baseline_correct and ckpt1400_correct):
        continue

    # Cherry-picking pattern: baseline has low coverage, ckpt1400 has better coverage
    # Relaxed: baseline < 50%, ckpt1400 generates more steps
    if baseline_coverage < 0.5 and len(ckpt1400_steps) > len(baseline_steps) and ckpt1400_coverage > baseline_coverage:
        improvement = ckpt1400_coverage - baseline_coverage

        # Check if question is interesting
        question = sample.get('question', '')
        if len(question) < 20:  # Skip trivial questions
            continue

        candidates.append({
            'idx': idx,
            'question': question,
            'answer': gt_answer,
            'ref_steps_count': len(ref_steps),
            'baseline_steps_count': len(baseline_steps),
            'ckpt1400_steps_count': len(ckpt1400_steps),
            'baseline_coverage': baseline_coverage,
            'ckpt1400_coverage': ckpt1400_coverage,
            'improvement': improvement,
            'ref_steps': ref_steps,
            'baseline_steps': baseline_steps,
            'ckpt1400_steps': ckpt1400_steps,
            'baseline_answer': baseline.get('answer', ''),
            'ckpt1400_answer': ckpt1400.get('answer', ''),
        })

print(f"✓ Found {len(candidates)} candidate examples")
print()

if not candidates:
    print("❌ No examples found matching criteria!")
    sys.exit(1)

# Sort by improvement magnitude
candidates.sort(key=lambda x: x['improvement'], reverse=True)

# Show top 10
print("=" * 80)
print("TOP 10 CANDIDATES (sorted by coverage improvement)")
print("=" * 80)
print()

for i, cand in enumerate(candidates[:10], 1):
    print(f"#{i} - Sample {cand['idx']}")
    print(f"   Question: {cand['question'][:100]}...")
    print(f"   Reference steps: {cand['ref_steps_count']}")
    print(f"   Baseline: {cand['baseline_steps_count']} steps ({cand['baseline_coverage']:.1%} coverage)")
    print(f"   Checkpoint-1400: {cand['ckpt1400_steps_count']} steps ({cand['ckpt1400_coverage']:.1%} coverage)")
    print(f"   Improvement: +{cand['improvement']:.1%}")
    print()

# Select the best one - prefer sample 3 if it's in the list, otherwise take the highest improvement
best = None
for cand in candidates:
    if cand['idx'] == 3:
        best = cand
        break
if best is None:
    best = candidates[0]  # Fall back to highest improvement

print("=" * 80)
print(f"SELECTED EXAMPLE: Sample {best['idx']}")
print("=" * 80)
print()
print(f"Question: {best['question']}")
print()
print(f"Ground Truth Answer: {best['answer']}")
print()
print(f"Reference Steps ({len(best['ref_steps'])}):")
for i, step in enumerate(best['ref_steps'], 1):
    print(f"  {i}. {step}")
print()
print(f"Baseline Prediction ({len(best['baseline_steps'])} steps, {best['baseline_coverage']:.1%} coverage):")
print(f"  Answer: {best['baseline_answer']}")
for i, step in enumerate(best['baseline_steps'], 1):
    print(f"  {i}. {step}")
print()
print(f"Checkpoint-1400 Prediction ({len(best['ckpt1400_steps'])} steps, {best['ckpt1400_coverage']:.1%} coverage):")
print(f"  Answer: {best['ckpt1400_answer']}")
for i, step in enumerate(best['ckpt1400_steps'], 1):
    print(f"  {i}. {step}")
print()
print(f"Coverage Improvement: {best['baseline_coverage']:.1%} → {best['ckpt1400_coverage']:.1%} (+{best['improvement']:.1%})")
print()

# Save image if available
try:
    sample = dataset[best['idx']]
    if 'image' in sample and sample['image'] is not None:
        image = sample['image']
        if hasattr(image, 'convert'):
            image_rgb = image.convert('RGB')
            output_path = f"/gpudata3/Wayner/paper_reasoning/images/grpo_example_sample{best['idx']}.jpg"
            image_rgb.save(output_path, 'JPEG', quality=95)
            print(f"✓ Saved image to: {output_path}")
        else:
            print("⚠ Image exists but cannot be converted")
    else:
        print("⚠ No image available for this sample")
except Exception as e:
    print(f"❌ Error saving image: {e}")

# Save example data for LaTeX generation
output_data = {
    'idx': best['idx'],
    'question': best['question'],
    'gt_answer': best['answer'],
    'ref_steps': best['ref_steps'],
    'baseline_answer': best['baseline_answer'],
    'baseline_steps': best['baseline_steps'],
    'ckpt1400_answer': best['ckpt1400_answer'],
    'ckpt1400_steps': best['ckpt1400_steps'],
    'baseline_coverage': f"{best['baseline_coverage']:.1%}",
    'ckpt1400_coverage': f"{best['ckpt1400_coverage']:.1%}",
    'improvement': f"{best['improvement']:.1%}",
}

output_json = "/gpudata3/Wayner/paper_reasoning/grpo_example_data.json"
with open(output_json, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"✓ Saved example data to: {output_json}")
print()
print("=" * 80)
print("DONE! Use this example for the paper figure.")
print("=" * 80)
