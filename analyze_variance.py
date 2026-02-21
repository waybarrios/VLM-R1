#!/usr/bin/env python3
"""
Analiza por qué la desviación estándar del Match F1 es alta
"""

import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python analyze_variance.py <path_to_metrics.csv>")
    print("Example: python analyze_variance.py metrics_results/outputs_testing_gemma3_12b_64k/no_judge_metrics.csv")
    sys.exit(1)

csv_path = sys.argv[1]

print("="*60)
print("MATCH F1 VARIANCE ANALYSIS")
print("="*60)

df = pd.read_csv(csv_path)

print(f"\nTotal samples: {len(df)}")
print(f"Average Match F1: {df['match_f1'].mean():.4f}")
print(f"Std Dev Match F1: {df['match_f1'].std():.4f}")
print(f"Median Match F1: {df['match_f1'].median():.4f}")

# Calculate coefficient of variation (CV)
cv = (df['match_f1'].std() / df['match_f1'].mean()) * 100
print(f"Coefficient of Variation: {cv:.2f}%")
print(f"  → {'HIGH variance (inconsistent)' if cv > 40 else 'MODERATE variance' if cv > 25 else 'LOW variance (consistent)'}")

print("\n" + "="*60)
print("MATCH F1 DISTRIBUTION")
print("="*60)
print(f"Min:        {df['match_f1'].min():.4f}")
print(f"25th %ile:  {df['match_f1'].quantile(0.25):.4f}")
print(f"Median:     {df['match_f1'].median():.4f}")
print(f"75th %ile:  {df['match_f1'].quantile(0.75):.4f}")
print(f"Max:        {df['match_f1'].max():.4f}")

print("\n" + "="*60)
print("MATCH F1 BUCKETS")
print("="*60)
buckets = [
    ("Perfect (0.95-1.0)", 0.95, 1.0),
    ("Excellent (0.8-0.95)", 0.8, 0.95),
    ("Good (0.6-0.8)", 0.6, 0.8),
    ("Medium (0.4-0.6)", 0.4, 0.6),
    ("Poor (0.2-0.4)", 0.2, 0.4),
    ("Very Poor (0.0-0.2)", 0.0, 0.2),
    ("Zero (0.0)", 0.0, 0.0)
]

for name, low, high in buckets:
    if low == high:
        count = (df['match_f1'] == low).sum()
    else:
        count = ((df['match_f1'] >= low) & (df['match_f1'] < high)).sum()
    pct = count / len(df) * 100
    print(f"{name:25s}: {count:5d} samples ({pct:5.1f}%)")

print("\n" + "="*60)
print("PLACEHOLDER IMPACT")
print("="*60)
placeholders = df[df['match_type'] == 'placeholder']
non_placeholders = df[df['match_type'] != 'placeholder']

print(f"Placeholders: {len(placeholders)} ({len(placeholders)/len(df)*100:.1f}%)")
print(f"  Match F1: {placeholders['match_f1'].mean():.4f} (always 0.0)")
print(f"\nNon-placeholders: {len(non_placeholders)} ({len(non_placeholders)/len(df)*100:.1f}%)")
print(f"  Match F1: {non_placeholders['match_f1'].mean():.4f} ± {non_placeholders['match_f1'].std():.4f}")
print(f"  Median F1: {non_placeholders['match_f1'].median():.4f}")

# Calculate CV for non-placeholders
if len(non_placeholders) > 0 and non_placeholders['match_f1'].mean() > 0:
    cv_no_ph = (non_placeholders['match_f1'].std() / non_placeholders['match_f1'].mean()) * 100
    print(f"  Coefficient of Variation (excluding placeholders): {cv_no_ph:.2f}%")
    print(f"    → {'Still HIGH' if cv_no_ph > 40 else 'Better, but MODERATE' if cv_no_ph > 25 else 'GOOD, low variance'}")

print("\n" + "="*60)
print("PRECISION vs RECALL ANALYSIS")
print("="*60)
print(f"Average Precision: {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
print(f"Average Recall:    {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")

# Identify pattern
if df['precision'].mean() > df['recall'].mean() + 0.05:
    print("\n⚠️  PATTERN: Precision > Recall")
    print("    → Model generates FEW steps but they match well")
    print("    → Missing many reference steps (low recall)")
elif df['recall'].mean() > df['precision'].mean() + 0.05:
    print("\n⚠️  PATTERN: Recall > Precision")
    print("    → Model generates MANY steps but some don't match")
    print("    → Generating irrelevant/incorrect steps (low precision)")
else:
    print("\n✓ Precision and Recall are balanced")

print("\n" + "="*60)
print("STEP COUNT ANALYSIS")
print("="*60)
print(f"Avg predicted steps: {df['num_predicted_steps'].mean():.2f} ± {df['num_predicted_steps'].std():.2f}")
print(f"Avg reference steps: {df['num_reference_steps'].mean():.2f} ± {df['num_reference_steps'].std():.2f}")

step_diff = df['num_predicted_steps'].mean() - df['num_reference_steps'].mean()
if abs(step_diff) > 1:
    if step_diff > 0:
        print(f"\n⚠️  Model generates {abs(step_diff):.1f} MORE steps than reference")
        print("    → May be generating redundant or irrelevant steps")
    else:
        print(f"\n⚠️  Model generates {abs(step_diff):.1f} FEWER steps than reference")
        print("    → May be missing important reasoning steps")

print("\n" + "="*60)
print("TOP 10 WORST SAMPLES (Lowest Match F1)")
print("="*60)
worst = df.nsmallest(10, 'match_f1')
for idx, row in worst.iterrows():
    print(f"\nSample {row['sample_idx']}:")
    print(f"  Match F1: {row['match_f1']:.3f}")
    print(f"  Precision: {row['precision']:.3f}, Recall: {row['recall']:.3f}")
    print(f"  Predicted steps: {row['num_predicted_steps']}, Reference steps: {row['num_reference_steps']}")
    print(f"  Answer correct: {row['accuracy_correct']}")

print("\n" + "="*60)
print("TOP 10 BEST SAMPLES (Highest Match F1)")
print("="*60)
best = df.nlargest(10, 'match_f1')
for idx, row in best.iterrows():
    print(f"\nSample {row['sample_idx']}:")
    print(f"  Match F1: {row['match_f1']:.3f}")
    print(f"  Precision: {row['precision']:.3f}, Recall: {row['recall']:.3f}")
    print(f"  Predicted steps: {row['num_predicted_steps']}, Reference steps: {row['num_reference_steps']}")
    print(f"  Answer correct: {row['accuracy_correct']}")

print("\n" + "="*60)
print("RECOMMENDATIONS TO REDUCE VARIANCE")
print("="*60)

recommendations = []

# Check placeholders
if len(placeholders) / len(df) > 0.1:
    recommendations.append(
        f"1. REDUCE PLACEHOLDERS: {len(placeholders)} samples ({len(placeholders)/len(df)*100:.1f}%) have no predictions.\n"
        f"   → Train model to generate predictions for all samples"
    )

# Check CV
if cv > 50:
    recommendations.append(
        f"2. HIGH INCONSISTENCY: CV = {cv:.1f}% (very high)\n"
        f"   → Model quality varies drastically across samples\n"
        f"   → Improve model training/consistency"
    )

# Check step count mismatch
if abs(step_diff) > 2:
    if step_diff > 0:
        recommendations.append(
            f"3. TOO MANY STEPS: Model generates {abs(step_diff):.1f} more steps than needed\n"
            f"   → Encourage concise reasoning\n"
            f"   → Filter out redundant steps"
        )
    else:
        recommendations.append(
            f"3. TOO FEW STEPS: Model generates {abs(step_diff):.1f} fewer steps than needed\n"
            f"   → Encourage more detailed reasoning\n"
            f"   → Train to generate complete step-by-step solutions"
        )

# Check precision vs recall imbalance
if abs(df['precision'].mean() - df['recall'].mean()) > 0.1:
    if df['precision'].mean() > df['recall'].mean():
        recommendations.append(
            f"4. LOW RECALL: Precision ({df['precision'].mean():.3f}) >> Recall ({df['recall'].mean():.3f})\n"
            f"   → Model generates few but accurate steps\n"
            f"   → Encourage more comprehensive reasoning"
        )
    else:
        recommendations.append(
            f"4. LOW PRECISION: Recall ({df['recall'].mean():.3f}) >> Precision ({df['precision'].mean():.3f})\n"
            f"   → Model generates many steps but some are irrelevant\n"
            f"   → Improve step quality/relevance"
        )

# Check low similarity
if df['avg_similarity'].mean() < 0.4:
    recommendations.append(
        f"5. LOW SIMILARITY: Avg similarity = {df['avg_similarity'].mean():.3f}\n"
        f"   → Predicted steps are semantically different from reference\n"
        f"   → Improve alignment with reference reasoning style"
    )

if recommendations:
    for rec in recommendations:
        print(f"\n{rec}")
else:
    print("\n✓ Variance is acceptable. Model is reasonably consistent.")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Overall Match F1: {df['match_f1'].mean():.4f} ± {df['match_f1'].std():.4f}")
print(f"Coefficient of Variation: {cv:.2f}%")
print(f"\nInterpretation:")
if cv < 25:
    print("  ✓ LOW variance - Model is CONSISTENT")
elif cv < 40:
    print("  ⚠️  MODERATE variance - Model has some inconsistency")
else:
    print("  ❌ HIGH variance - Model is VERY INCONSISTENT")
    print("     → Quality varies dramatically across samples")
    print("     → Focus on improving model consistency")

print("\n" + "="*60)
