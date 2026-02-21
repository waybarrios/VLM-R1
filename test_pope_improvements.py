#!/usr/bin/env python3
"""
Test the improved POPE extraction on existing samples.
This shows what the improvement would be if we re-run inference.
"""
import sys
sys.path.insert(0, '/gpudata3/Wayner/original/lmms-eval')

import json
from lmms_eval.tasks.pope.utils_reasoning import extract_answer_from_json, normalize_pope_answer

samples_path = "/gpudata3/Wayner/original/logs-pope-final-grpo/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133__checkpoint-1500/20251113_004646_samples_pope_reasoning.jsonl"

print("="*80)
print("TESTING IMPROVED POPE EXTRACTION")
print("="*80)
print()

# Test extraction on existing samples
original_correct = 0
improved_correct = 0
total = 0
improvements = []

with open(samples_path, 'r') as f:
    for line in f:
        sample = json.loads(line)
        total += 1

        response = sample['filtered_resps'][0]
        target = sample['target'].lower()

        # Original prediction (from saved results)
        original_pred = sample['pope_accuracy']['prediction'].lower()

        # New extraction with normalization
        improved_pred = extract_answer_from_json(response)

        # Check accuracy
        original_is_correct = (original_pred == target)
        improved_is_correct = (improved_pred == target)

        if original_is_correct:
            original_correct += 1
        if improved_is_correct:
            improved_correct += 1

        # Track improvements
        if improved_is_correct and not original_is_correct:
            if len(improvements) < 20:
                improvements.append({
                    'question': sample['input'][:80],
                    'target': target,
                    'original': original_pred,
                    'improved': improved_pred,
                    'answer_text': sample['pope_accuracy']['answer_text'][:80]
                })

print(f"📊 Results:")
print(f"   Total samples: {total}")
print()
print(f"   Original accuracy: {original_correct}/{total} ({100*original_correct/total:.2f}%)")
print(f"   Improved accuracy: {improved_correct}/{total} ({100*improved_correct/total:.2f}%)")
print(f"   Difference: {improved_correct - original_correct} samples ({100*(improved_correct-original_correct)/total:+.2f}pp)")
print()

if improvements:
    print(f"✅ Examples of fixes (first 10):")
    for i, imp in enumerate(improvements[:10], 1):
        print(f"\n   Fix {i}:")
        print(f"      Question: {imp['question']}...")
        print(f"      Target: {imp['target']}")
        print(f"      Original: '{imp['original']}' ❌")
        print(f"      Improved: '{imp['improved']}' ✅")
        print(f"      Full text: {imp['answer_text']}...")

print()
print("="*80)
print("CONCLUSION:")
if improved_correct > original_correct:
    print(f"✅ Improved extraction fixes {improved_correct - original_correct} samples!")
    print(f"   New accuracy would be: {100*improved_correct/total:.2f}%")
    print(f"   Improvement: {100*(improved_correct-original_correct)/total:+.2f}pp")
elif improved_correct == original_correct:
    print("⚠️  No change - extraction was already working correctly")
else:
    print(f"❌ Extraction got worse by {original_correct - improved_correct} samples")
print("="*80)
