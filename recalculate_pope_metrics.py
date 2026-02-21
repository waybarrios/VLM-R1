#!/usr/bin/env python3
"""
Recalculate POPE metrics using the improved extraction on existing samples.
"""
import sys
sys.path.insert(0, '/gpudata3/Wayner/original/lmms-eval')

import json
from lmms_eval.tasks.pope.utils_reasoning import extract_answer_from_json

samples_path = "/gpudata3/Wayner/original/logs-pope-final-grpo/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133__checkpoint-1500/20251113_004646_samples_pope_reasoning.jsonl"

print("="*80)
print("RECALCULATING POPE METRICS WITH IMPROVED EXTRACTION")
print("="*80)
print()

# Calculate all metrics
total = 0
true_positives = 0  # Model says "yes" and it IS there
true_negatives = 0  # Model says "no" and it's NOT there
false_positives = 0  # Model says "yes" but it's NOT there (hallucination)
false_negatives = 0  # Model says "no" but it IS there (missed object)

yes_predictions = 0
no_predictions = 0

yes_ground_truth = 0
no_ground_truth = 0

with open(samples_path, 'r') as f:
    for line in f:
        sample = json.loads(line)
        total += 1

        # Get raw response
        response = sample['filtered_resps'][0]

        # Ground truth
        target = sample['target'].lower()

        # Apply improved extraction
        prediction = extract_answer_from_json(response)

        # Count predictions
        if prediction == "yes":
            yes_predictions += 1
        elif prediction == "no":
            no_predictions += 1

        # Count ground truth
        if target == "yes":
            yes_ground_truth += 1
        elif target == "no":
            no_ground_truth += 1

        # Calculate confusion matrix
        if target == "yes" and prediction == "yes":
            true_positives += 1
        elif target == "no" and prediction == "no":
            true_negatives += 1
        elif target == "no" and prediction == "yes":
            false_positives += 1
        elif target == "yes" and prediction == "no":
            false_negatives += 1

# Calculate metrics
accuracy = (true_positives + true_negatives) / total if total > 0 else 0
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
yes_ratio = yes_predictions / total if total > 0 else 0

# Original metrics (from results.json)
original_accuracy = 0.7898
original_precision = 0.8698
original_recall = 0.6816
original_f1 = 0.7643
original_yes_ratio = 0.5000

print(f"📊 RECALCULATED METRICS (with improved extraction):")
print()
print(f"{'Metric':<20} {'Original':>12} {'Improved':>12} {'Difference':>12}")
print("-"*60)
print(f"{'Accuracy':<20} {original_accuracy*100:>11.2f}% {accuracy*100:>11.2f}% {(accuracy-original_accuracy)*100:>+11.2f}pp")
print(f"{'Precision':<20} {original_precision*100:>11.2f}% {precision*100:>11.2f}% {(precision-original_precision)*100:>+11.2f}pp")
print(f"{'Recall':<20} {original_recall*100:>11.2f}% {recall*100:>11.2f}% {(recall-original_recall)*100:>+11.2f}pp")
print(f"{'F1 Score':<20} {original_f1*100:>11.2f}% {f1*100:>11.2f}% {(f1-original_f1)*100:>+11.2f}pp")
print(f"{'Yes Ratio':<20} {original_yes_ratio*100:>11.2f}% {yes_ratio*100:>11.2f}% {(yes_ratio-original_yes_ratio)*100:>+11.2f}pp")
print()

print(f"📈 DETAILED BREAKDOWN:")
print()
print(f"Total samples: {total}")
print(f"   Yes in ground truth: {yes_ground_truth} ({100*yes_ground_truth/total:.1f}%)")
print(f"   No in ground truth: {no_ground_truth} ({100*no_ground_truth/total:.1f}%)")
print()
print(f"Predictions:")
print(f"   Yes predictions: {yes_predictions} ({100*yes_predictions/total:.1f}%)")
print(f"   No predictions: {no_predictions} ({100*no_predictions/total:.1f}%)")
print()
print(f"Confusion Matrix:")
print(f"   True Positives (said YES, IS there):  {true_positives:4d} ({100*true_positives/total:.1f}%)")
print(f"   True Negatives (said NO, NOT there):  {true_negatives:4d} ({100*true_negatives/total:.1f}%)")
print(f"   False Positives (said YES, NOT there): {false_positives:4d} ({100*false_positives/total:.1f}%) ← Hallucinations")
print(f"   False Negatives (said NO, IS there):   {false_negatives:4d} ({100*false_negatives/total:.1f}%) ← Missed objects")
print()
print(f"Correct predictions: {true_positives + true_negatives}/{total} ({100*(true_positives+true_negatives)/total:.2f}%)")
print()

print("="*80)
print("COMPARISON WITH BASELINE:")
print("="*80)
print()
print(f"{'Model':<30} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
print("-"*80)
print(f"{'Baseline (Qwen2.5-VL-3B)':<30} {87.77:>11.2f}% {98.21:>11.2f}% {89.19:>11.2f}% {93.48:>11.2f}%")
print(f"{'GRPO (original extraction)':<30} {78.98:>11.2f}% {86.98:>11.2f}% {68.16:>11.2f}% {76.43:>11.2f}%")
print(f"{'GRPO (improved extraction)':<30} {accuracy*100:>11.2f}% {precision*100:>11.2f}% {recall*100:>11.2f}% {f1*100:>11.2f}%")
print()
print(f"Gap vs Baseline:")
print(f"   Original: {78.98 - 87.77:+.2f}pp")
print(f"   Improved: {(accuracy*100) - 87.77:+.2f}pp")
print(f"   Improvement: {(accuracy - 0.7898)*100:+.2f}pp")
print()

print("="*80)
if accuracy > original_accuracy:
    improvement = (accuracy - original_accuracy) * 100
    print(f"✅ SUCCESS! Improved extraction gains +{improvement:.2f}pp accuracy")
    print(f"   New accuracy: {accuracy*100:.2f}% (was {original_accuracy*100:.2f}%)")
elif accuracy == original_accuracy:
    print(f"⚠️  No change. Extraction was already working correctly.")
    print(f"   Accuracy remains: {accuracy*100:.2f}%")
else:
    print(f"❌ Something went wrong. Accuracy decreased.")
print("="*80)
