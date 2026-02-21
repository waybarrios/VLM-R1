#!/usr/bin/env python3
"""
Analiza por qué el confidence es bajo
"""

import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python analyze_confidence.py <path_to_metrics.csv>")
    print("Example: python analyze_confidence.py metrics_results/outputs_testing_llava7b_16_no_judge_metrics.csv")
    sys.exit(1)

csv_path = sys.argv[1]

print("="*60)
print("CONFIDENCE ANALYSIS")
print("="*60)

df = pd.read_csv(csv_path)

print(f"\nTotal samples: {len(df)}")
print(f"Average Confidence: {df['confidence'].mean():.4f}")
print(f"Median Confidence: {df['confidence'].median():.4f}")

print("\n" + "="*60)
print("CONFIDENCE DISTRIBUTION")
print("="*60)
print(f"Perfect (1.0):        {(df['confidence'] == 1.0).sum()} samples ({(df['confidence'] == 1.0).sum()/len(df)*100:.1f}%)")
print(f"Very High (0.9-1.0):  {((df['confidence'] >= 0.9) & (df['confidence'] < 1.0)).sum()} samples ({((df['confidence'] >= 0.9) & (df['confidence'] < 1.0)).sum()/len(df)*100:.1f}%)")
print(f"High (0.8-0.9):       {((df['confidence'] >= 0.8) & (df['confidence'] < 0.9)).sum()} samples ({((df['confidence'] >= 0.8) & (df['confidence'] < 0.9)).sum()/len(df)*100:.1f}%)")
print(f"Medium (0.5-0.8):     {((df['confidence'] >= 0.5) & (df['confidence'] < 0.8)).sum()} samples ({((df['confidence'] >= 0.5) & (df['confidence'] < 0.8)).sum()/len(df)*100:.1f}%)")
print(f"Low (0.3-0.5):        {((df['confidence'] >= 0.3) & (df['confidence'] < 0.5)).sum()} samples ({((df['confidence'] >= 0.3) & (df['confidence'] < 0.5)).sum()/len(df)*100:.1f}%)")
print(f"Very Low (0.0-0.3):   {(df['confidence'] < 0.3).sum()} samples ({(df['confidence'] < 0.3).sum()/len(df)*100:.1f}%)")

print("\n" + "="*60)
print("CONFIDENCE BY MATCH TYPE")
print("="*60)
match_type_conf = df.groupby('match_type').agg({
    'confidence': ['mean', 'count'],
    'accuracy_correct': 'mean'
})
print(match_type_conf.to_string())

print("\n" + "="*60)
print("CONFIDENCE BY CORRECTNESS")
print("="*60)
correct_samples = df[df['accuracy_correct'] == True]
incorrect_samples = df[df['accuracy_correct'] == False]

print(f"Correct samples:   {len(correct_samples)} → avg confidence: {correct_samples['confidence'].mean():.4f}")
print(f"Incorrect samples: {len(incorrect_samples)} → avg confidence: {incorrect_samples['confidence'].mean():.4f}")

print("\n" + "="*60)
print("SAMPLES WITH LOW CONFIDENCE (<0.7)")
print("="*60)
low_conf = df[df['confidence'] < 0.7].sort_values('confidence')
print(f"Total: {len(low_conf)} samples")
print("\nFirst 10 examples:")
for idx, row in low_conf.head(10).iterrows():
    print(f"\nSample {row['sample_idx']}:")
    print(f"  Predicted: {row['predicted_answer'][:100]}")
    print(f"  Ground Truth: {row['ground_truth_answer'][:100]}")
    print(f"  Match Type: {row['match_type']}")
    print(f"  Correct: {row['accuracy_correct']}")
    print(f"  Confidence: {row['confidence']:.3f}")

print("\n" + "="*60)
print("RECOMMENDATIONS TO INCREASE CONFIDENCE")
print("="*60)

# Analyze what's causing low confidence
total = len(df)
placeholders = (df['match_type'] == 'placeholder').sum()
llm_verified = (df['match_type'] == 'llm_verified').sum()
incorrect = (df['accuracy_correct'] == False).sum()
numeric_rounded = (df['match_type'].str.contains('numeric', na=False)).sum()

recommendations = []

if placeholders > total * 0.05:
    recommendations.append(f"• {placeholders} placeholders ({placeholders/total*100:.1f}%) → Improve model to generate predictions for all samples")

if incorrect > total * 0.2:
    recommendations.append(f"• {incorrect} incorrect answers ({incorrect/total*100:.1f}%) → Improve model accuracy")

if llm_verified > total * 0.1:
    recommendations.append(f"• {llm_verified} LLM-verified matches ({llm_verified/total*100:.1f}%) → LLM judges give variable confidence. Consider using use_judge=False for exact matches")

if numeric_rounded > total * 0.1:
    recommendations.append(f"• {numeric_rounded} numeric rounded matches ({numeric_rounded/total*100:.1f}%) → Model gives rounded numbers. Train to give exact numbers")

# Check if many correct answers have low confidence
correct_low_conf = len(df[(df['accuracy_correct'] == True) & (df['confidence'] < 0.9)])
if correct_low_conf > total * 0.2:
    recommendations.append(f"• {correct_low_conf} correct answers with confidence < 0.9 → Model gives correct but not exact answers (e.g., 'The answer is A' instead of 'A')")

if recommendations:
    for rec in recommendations:
        print(rec)
else:
    print("• Confidence is already high! No major issues detected.")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
perfect_matches = (df['confidence'] == 1.0).sum()
print(f"To increase average confidence from {df['confidence'].mean():.4f} to >0.9:")
print(f"  Current perfect matches: {perfect_matches}/{total} ({perfect_matches/total*100:.1f}%)")
print(f"  Target: >{total*0.9:.0f} perfect matches (>90%)")
print(f"  Need to improve: {max(0, int(total*0.9 - perfect_matches))} samples")
print(f"\nMain action: Train/improve the prediction model to generate more exact answers!")
