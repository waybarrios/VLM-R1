#!/usr/bin/env python3
"""
Identifica qué hace que una pregunta sea difícil o fácil
Analiza patrones en los datos para clasificar por dificultad
"""

import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from datasets import load_from_disk

if len(sys.argv) < 3:
    print("Usage: python identify_difficulty.py <path_to_metrics.csv> <path_to_dataset>")
    print("Example: python identify_difficulty.py metrics_results/outputs_testing_gemma3_12b_64k/no_judge_metrics.csv /gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27")
    sys.exit(1)

csv_path = sys.argv[1]
dataset_path = sys.argv[2]

print("="*60)
print("DIFFICULTY ANALYSIS")
print("="*60)

# Load metrics
print("\nLoading metrics...")
df = pd.read_csv(csv_path)

# Load dataset to get questions
print("Loading dataset...")
dataset = load_from_disk(dataset_path)

# Create mapping of idx to questions
print("Creating question mapping...")
questions = {}
for idx in range(len(dataset)):
    sample = dataset[idx]
    questions[idx] = {
        'question': sample.get('question', ''),
        'answer': sample.get('answer', ''),
        'source': sample.get('source', ''),
        'num_reference_steps': len(sample.get('reference_steps', []))
    }

# Add question info to dataframe
df['question'] = df['sample_idx'].map(lambda idx: questions.get(idx, {}).get('question', ''))
df['source'] = df['sample_idx'].map(lambda idx: questions.get(idx, {}).get('source', ''))
df['question_length'] = df['question'].str.len()
df['question_word_count'] = df['question'].str.split().str.len()

print(f"Total samples: {len(df)}")

# Define difficulty levels based on Match F1
print("\n" + "="*60)
print("DIFFICULTY CLASSIFICATION (by Match F1)")
print("="*60)

# Classify by percentiles
q25 = df['match_f1'].quantile(0.25)
q50 = df['match_f1'].quantile(0.50)
q75 = df['match_f1'].quantile(0.75)

def classify_difficulty(f1):
    if f1 >= q75:
        return 'Easy'
    elif f1 >= q50:
        return 'Medium'
    elif f1 >= q25:
        return 'Hard'
    else:
        return 'Very Hard'

df['difficulty'] = df['match_f1'].apply(classify_difficulty)

print(f"Easy (F1 >= {q75:.3f}):       {(df['difficulty'] == 'Easy').sum()} samples")
print(f"Medium ({q50:.3f} <= F1 < {q75:.3f}): {(df['difficulty'] == 'Medium').sum()} samples")
print(f"Hard ({q25:.3f} <= F1 < {q50:.3f}):   {(df['difficulty'] == 'Hard').sum()} samples")
print(f"Very Hard (F1 < {q25:.3f}):  {(df['difficulty'] == 'Very Hard').sum()} samples")

# Analyze patterns
print("\n" + "="*60)
print("PATTERNS BY DIFFICULTY")
print("="*60)

for difficulty in ['Easy', 'Medium', 'Hard', 'Very Hard']:
    subset = df[df['difficulty'] == difficulty]
    if len(subset) == 0:
        continue

    print(f"\n{difficulty} Questions:")
    print(f"  Count: {len(subset)}")
    print(f"  Match F1: {subset['match_f1'].mean():.4f} ± {subset['match_f1'].std():.4f}")
    print(f"  Accuracy: {subset['accuracy_correct'].mean():.4f}")
    print(f"  Avg question length: {subset['question_length'].mean():.1f} chars")
    print(f"  Avg question words: {subset['question_word_count'].mean():.1f} words")
    print(f"  Avg predicted steps: {subset['num_predicted_steps'].mean():.2f}")
    print(f"  Avg reference steps: {subset['num_reference_steps'].mean():.2f}")
    print(f"  Avg precision: {subset['precision'].mean():.4f}")
    print(f"  Avg recall: {subset['recall'].mean():.4f}")

# Analyze by source (if available)
if 'source' in df.columns and df['source'].notna().any():
    print("\n" + "="*60)
    print("DIFFICULTY BY SOURCE")
    print("="*60)

    source_difficulty = df.groupby('source').agg({
        'match_f1': ['mean', 'std', 'count'],
        'accuracy_correct': 'mean',
        'difficulty': lambda x: (x == 'Easy').sum() / len(x)
    }).round(4)

    print(source_difficulty.to_string())

# Find key differences
print("\n" + "="*60)
print("KEY DIFFERENCES: Easy vs Very Hard")
print("="*60)

easy = df[df['difficulty'] == 'Easy']
very_hard = df[df['difficulty'] == 'Very Hard']

if len(easy) > 0 and len(very_hard) > 0:
    print(f"\nQuestion Length:")
    print(f"  Easy:      {easy['question_length'].mean():.1f} chars")
    print(f"  Very Hard: {very_hard['question_length'].mean():.1f} chars")
    print(f"  Difference: {abs(easy['question_length'].mean() - very_hard['question_length'].mean()):.1f} chars")

    print(f"\nReference Steps:")
    print(f"  Easy:      {easy['num_reference_steps'].mean():.2f} steps")
    print(f"  Very Hard: {very_hard['num_reference_steps'].mean():.2f} steps")
    print(f"  Difference: {abs(easy['num_reference_steps'].mean() - very_hard['num_reference_steps'].mean()):.2f} steps")

    print(f"\nPredicted Steps:")
    print(f"  Easy:      {easy['num_predicted_steps'].mean():.2f} steps")
    print(f"  Very Hard: {very_hard['num_predicted_steps'].mean():.2f} steps")

    print(f"\nPrecision vs Recall:")
    print(f"  Easy:      P={easy['precision'].mean():.3f}, R={easy['recall'].mean():.3f}")
    print(f"  Very Hard: P={very_hard['precision'].mean():.3f}, R={very_hard['recall'].mean():.3f}")

# Show example questions
print("\n" + "="*60)
print("EXAMPLE EASY QUESTIONS (Top 10 by Match F1)")
print("="*60)

easy_samples = df.nlargest(10, 'match_f1')
for idx, row in easy_samples.iterrows():
    print(f"\nSample {row['sample_idx']} (F1: {row['match_f1']:.3f}):")
    print(f"  Question: {row['question'][:200]}...")
    print(f"  Answer: {row['predicted_answer']}")
    print(f"  Steps: {row['num_predicted_steps']} predicted, {row['num_reference_steps']} reference")
    print(f"  P={row['precision']:.3f}, R={row['recall']:.3f}")

print("\n" + "="*60)
print("EXAMPLE HARD QUESTIONS (Bottom 10 by Match F1)")
print("="*60)

# Exclude placeholders for hard examples
hard_samples = df[df['match_type'] != 'placeholder'].nsmallest(10, 'match_f1')
for idx, row in hard_samples.iterrows():
    print(f"\nSample {row['sample_idx']} (F1: {row['match_f1']:.3f}):")
    print(f"  Question: {row['question'][:200]}...")
    print(f"  Answer: {row['predicted_answer']}")
    print(f"  Steps: {row['num_predicted_steps']} predicted, {row['num_reference_steps']} reference")
    print(f"  P={row['precision']:.3f}, R={row['recall']:.3f}")

# Correlation analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

# Calculate correlations
correlations = {
    'Question Length': df[['match_f1', 'question_length']].corr().iloc[0, 1],
    'Question Words': df[['match_f1', 'question_word_count']].corr().iloc[0, 1],
    'Reference Steps': df[['match_f1', 'num_reference_steps']].corr().iloc[0, 1],
    'Predicted Steps': df[['match_f1', 'num_predicted_steps']].corr().iloc[0, 1],
}

print("\nCorrelation with Match F1:")
for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑" if corr > 0 else "↓"
    strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
    print(f"  {feature:20s}: {corr:+.3f} {direction} ({strength})")

# Identify patterns
print("\n" + "="*60)
print("IDENTIFIED PATTERNS")
print("="*60)

patterns = []

# Pattern 1: More reference steps = harder
if correlations['Reference Steps'] < -0.2:
    patterns.append(
        f"• Questions with MORE reference steps are HARDER\n"
        f"  → Easy: {easy['num_reference_steps'].mean():.1f} steps\n"
        f"  → Hard: {very_hard['num_reference_steps'].mean():.1f} steps"
    )

# Pattern 2: Longer questions
if abs(correlations['Question Length']) > 0.1:
    if correlations['Question Length'] < 0:
        patterns.append(
            f"• LONGER questions are HARDER\n"
            f"  → Easy: {easy['question_length'].mean():.0f} chars\n"
            f"  → Hard: {very_hard['question_length'].mean():.0f} chars"
        )
    else:
        patterns.append(
            f"• SHORTER questions are HARDER\n"
            f"  → Easy: {easy['question_length'].mean():.0f} chars\n"
            f"  → Hard: {very_hard['question_length'].mean():.0f} chars"
        )

# Pattern 3: Precision vs Recall
easy_prec_recall_diff = easy['precision'].mean() - easy['recall'].mean()
hard_prec_recall_diff = very_hard['precision'].mean() - very_hard['recall'].mean()

if abs(easy_prec_recall_diff - hard_prec_recall_diff) > 0.1:
    patterns.append(
        f"• Easy questions: P={easy['precision'].mean():.3f}, R={easy['recall'].mean():.3f}\n"
        f"• Hard questions: P={very_hard['precision'].mean():.3f}, R={very_hard['recall'].mean():.3f}\n"
        f"  → Different precision/recall patterns"
    )

if patterns:
    for pattern in patterns:
        print(f"\n{pattern}")
else:
    print("\n• No clear patterns detected. May need more features to analyze.")

# Save difficulty classification
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

output_file = csv_path.replace('.csv', '_with_difficulty.csv')
df.to_csv(output_file, index=False)
print(f"✓ Saved classification to: {output_file}")

# Save summary
summary = {
    'difficulty_thresholds': {
        'easy_min_f1': float(q75),
        'medium_min_f1': float(q50),
        'hard_min_f1': float(q25)
    },
    'difficulty_counts': {
        'easy': int((df['difficulty'] == 'Easy').sum()),
        'medium': int((df['difficulty'] == 'Medium').sum()),
        'hard': int((df['difficulty'] == 'Hard').sum()),
        'very_hard': int((df['difficulty'] == 'Very Hard').sum())
    },
    'difficulty_stats': {
        'easy': {
            'match_f1': float(easy['match_f1'].mean()) if len(easy) > 0 else 0,
            'accuracy': float(easy['accuracy_correct'].mean()) if len(easy) > 0 else 0,
            'avg_steps': float(easy['num_reference_steps'].mean()) if len(easy) > 0 else 0
        },
        'very_hard': {
            'match_f1': float(very_hard['match_f1'].mean()) if len(very_hard) > 0 else 0,
            'accuracy': float(very_hard['accuracy_correct'].mean()) if len(very_hard) > 0 else 0,
            'avg_steps': float(very_hard['num_reference_steps'].mean()) if len(very_hard) > 0 else 0
        }
    },
    'correlations': correlations
}

summary_file = csv_path.replace('.csv', '_difficulty_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved summary to: {summary_file}")

print("\n" + "="*60)
print("RECOMMENDATIONS FOR PAPER")
print("="*60)

print(f"""
In your paper, you can now report:

1. Performance by difficulty:
   - Easy (top 25%, F1 >= {q75:.2f}): {easy['match_f1'].mean():.3f} ± {easy['match_f1'].std():.3f}
   - Medium (25-50%): {df[df['difficulty'] == 'Medium']['match_f1'].mean():.3f}
   - Hard (50-75%): {df[df['difficulty'] == 'Hard']['match_f1'].mean():.3f}
   - Very Hard (bottom 25%, F1 < {q25:.2f}): {very_hard['match_f1'].mean():.3f} ± {very_hard['match_f1'].std():.3f}

2. This explains the high std dev:
   "The standard deviation of ±{df['match_f1'].std():.3f} reflects performance variation
   across question difficulties, ranging from {easy['match_f1'].mean():.3f} on easy questions
   to {very_hard['match_f1'].mean():.3f} on very hard questions."

3. Key characteristics of difficult questions:
   - Longer reasoning chains ({very_hard['num_reference_steps'].mean():.1f} vs {easy['num_reference_steps'].mean():.1f} steps)
   {f"- Longer question text ({very_hard['question_length'].mean():.0f} vs {easy['question_length'].mean():.0f} chars)" if abs(easy['question_length'].mean() - very_hard['question_length'].mean()) > 50 else ""}
""")
