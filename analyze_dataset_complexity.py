#!/usr/bin/env python3
"""
Analyze dataset complexity WITHOUT using model predictions
This ensures the complexity scores are unbiased and model-agnostic

Features analyzed (NO model bias):
- Reference step count (from ground truth)
- Question length and word count
- Linguistic complexity (conditionals, causals)
- Answer type
- Question structure
"""

import json
import sys
import re
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Usage: python analyze_dataset_complexity.py <dataset_path> [output_file]")
    print("Example: python analyze_dataset_complexity.py /gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27")
    sys.exit(1)

dataset_path = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else "dataset_complexity_scores.json"

print("="*60)
print("DATASET COMPLEXITY ANALYSIS (Model-Independent)")
print("="*60)
print(f"Dataset: {dataset_path}")
print(f"Output: {output_file}")
print("\nThis analysis uses ONLY ground truth features - NO model predictions!")
print("="*60)

# Load dataset
print("\nLoading dataset...")

try:
    # Try standard load first
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset)} samples")
except (TypeError, ValueError) as e:
    # If there's a schema error, load without validation
    print(f"Warning: Schema validation error, loading with alternative method...")
    import pyarrow as pa
    from glob import glob

    # Load Arrow files directly
    arrow_files = glob(f"{dataset_path}/data-*.arrow")
    if not arrow_files:
        raise ValueError(f"No Arrow files found in {dataset_path}")

    # Read all arrow files (streaming format)
    tables = []
    for arrow_file in sorted(arrow_files):
        stream = pa.ipc.open_stream(arrow_file)
        table = stream.read_all()
        tables.append(table)

    # Concatenate all tables
    full_table = pa.concat_tables(tables)
    print(f"Loaded {len(full_table)} samples using Arrow files")

    # Create a simple Dataset-like object
    class SimpleDataset:
        def __init__(self, table):
            self.table = table
            self.length = len(table)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return {
                'question': self.table['question'][idx].as_py() if 'question' in self.table.column_names else '',
                'answer': self.table['answer'][idx].as_py() if 'answer' in self.table.column_names else '',
                'reference_steps': self.table['reference_steps'][idx].as_py() if 'reference_steps' in self.table.column_names else []
            }

    dataset = SimpleDataset(full_table)

# Analyze each sample
print("\nAnalyzing complexity features...")

complexity_scores = []

for idx in tqdm(range(len(dataset)), desc="Analyzing samples"):
    sample = dataset[idx]

    # Extract features
    question = sample.get('question', '')
    answer = sample.get('answer', '')
    reference_steps = sample.get('reference_steps', [])

    # Feature 1: Reference step count (strong indicator)
    num_ref_steps = len(reference_steps)

    # Feature 2: Question length
    question_length = len(question)
    question_words = len(question.split())

    # Feature 3: Linguistic complexity
    has_conditional = bool(re.search(r'\b(if|when|unless|provided|given that)\b', question.lower()))
    has_causal = bool(re.search(r'\b(because|since|therefore|thus|consequently|as a result)\b', question.lower()))
    has_comparison = bool(re.search(r'\b(more|less|than|compared to|relative to)\b', question.lower()))
    has_negation = bool(re.search(r'\b(not|no|never|neither|without)\b', question.lower()))

    # Feature 4: Question type indicators
    is_counting = bool(re.search(r'\b(how many|count|number of)\b', question.lower()))
    is_yes_no = bool(re.search(r'\b(is there|are there|does|do|can|will)\b', question.lower()) and
                     question.strip().endswith('?'))
    is_why = bool(re.search(r'\b(why|explain|reason)\b', question.lower()))
    is_how = bool(re.search(r'\b(how|in what way)\b', question.lower()))
    is_what_if = bool(re.search(r'\b(what if|suppose|imagine)\b', question.lower()))

    # Feature 5: Answer type (from ground truth)
    answer_length = len(str(answer))
    is_multiple_choice = bool(re.match(r'^[A-D]$', str(answer).strip()))
    is_numeric = bool(re.match(r'^\d+\.?\d*$', str(answer).strip()))

    # Feature 6: Number of sentences in question
    sentences = [s.strip() for s in re.split(r'[.!?]+', question) if s.strip()]
    num_sentences = len(sentences)

    # Calculate complexity score (0-1 scale)
    # Each feature contributes to complexity

    complexity_features = {
        # Step count (0-1 normalized, assume max 20 steps)
        'step_complexity': min(num_ref_steps / 20.0, 1.0),

        # Question length (0-1 normalized, assume max 1000 chars)
        'length_complexity': min(question_length / 1000.0, 1.0),

        # Word count (0-1 normalized, assume max 150 words)
        'word_complexity': min(question_words / 150.0, 1.0),

        # Linguistic features (binary, each adds complexity)
        'conditional': 0.15 if has_conditional else 0.0,
        'causal': 0.15 if has_causal else 0.0,
        'comparison': 0.10 if has_comparison else 0.0,
        'negation': 0.10 if has_negation else 0.0,

        # Question type (affects complexity)
        'counting': -0.15 if is_counting else 0.0,  # Counting is easier
        'yes_no': -0.10 if is_yes_no else 0.0,      # Yes/No is easier
        'why': 0.20 if is_why else 0.0,              # Why questions are harder
        'how': 0.15 if is_how else 0.0,              # How questions are harder
        'what_if': 0.25 if is_what_if else 0.0,      # Hypothetical is hardest

        # Answer type
        'multiple_choice': -0.10 if is_multiple_choice else 0.0,  # MC easier
        'numeric': -0.05 if is_numeric else 0.0,                   # Numeric easier
        'free_text': 0.10 if not is_multiple_choice and not is_numeric else 0.0,  # Free text harder

        # Multi-sentence questions
        'multi_sentence': 0.10 if num_sentences > 1 else 0.0,
    }

    # Weighted average with emphasis on step count
    step_weight = 0.4
    length_weight = 0.2
    linguistic_weight = 0.2
    other_weight = 0.2

    step_score = complexity_features['step_complexity']
    length_score = (complexity_features['length_complexity'] + complexity_features['word_complexity']) / 2
    linguistic_score = (complexity_features['conditional'] + complexity_features['causal'] +
                       complexity_features['comparison'] + complexity_features['negation'])
    other_score = (complexity_features['counting'] + complexity_features['yes_no'] +
                   complexity_features['why'] + complexity_features['how'] +
                   complexity_features['what_if'] + complexity_features['multiple_choice'] +
                   complexity_features['numeric'] + complexity_features['free_text'] +
                   complexity_features['multi_sentence'])

    raw_score = (step_weight * step_score +
                 length_weight * length_score +
                 linguistic_weight * linguistic_score +
                 other_weight * other_score)

    # Normalize to 0-1 and clip
    complexity_score = np.clip(raw_score, 0.0, 1.0)

    # Classify into levels with balanced thresholds
    # Thresholds chosen to create meaningful stratification
    # Using 3-tier classification: easy, medium, hard
    if complexity_score >= 0.42:
        complexity_level = 'hard'
    elif complexity_score >= 0.27:
        complexity_level = 'medium'
    else:
        complexity_level = 'easy'

    # Store result
    result = {
        'sample_idx': idx,
        'complexity_score': float(complexity_score),
        'complexity_level': complexity_level,
        'features': {
            'num_reference_steps': num_ref_steps,
            'question_length': question_length,
            'question_words': question_words,
            'num_sentences': num_sentences,
            'has_conditional': has_conditional,
            'has_causal': has_causal,
            'has_comparison': has_comparison,
            'has_negation': has_negation,
            'is_counting': is_counting,
            'is_yes_no': is_yes_no,
            'is_why': is_why,
            'is_how': is_how,
            'is_what_if': is_what_if,
            'is_multiple_choice': is_multiple_choice,
            'is_numeric': is_numeric,
            'answer_length': answer_length
        },
        'question': question[:200] + '...' if len(question) > 200 else question,
        'answer': str(answer)
    }

    complexity_scores.append(result)

# Calculate statistics
print("\n" + "="*60)
print("COMPLEXITY STATISTICS")
print("="*60)

scores = [s['complexity_score'] for s in complexity_scores]
levels = [s['complexity_level'] for s in complexity_scores]

print(f"\nComplexity Score Distribution:")
print(f"  Mean:   {np.mean(scores):.3f}")
print(f"  Median: {np.median(scores):.3f}")
print(f"  Std:    {np.std(scores):.3f}")
print(f"  Min:    {np.min(scores):.3f}")
print(f"  Max:    {np.max(scores):.3f}")
print(f"  Q1:     {np.percentile(scores, 25):.3f}")
print(f"  Q3:     {np.percentile(scores, 75):.3f}")

print(f"\nComplexity Level Counts:")
easy_count = levels.count('easy')
medium_count = levels.count('medium')
hard_count = levels.count('hard')
total = len(levels)

print(f"  Easy:   {easy_count:5d} ({easy_count/total*100:5.1f}%) - score < 0.27")
print(f"  Medium: {medium_count:5d} ({medium_count/total*100:5.1f}%) - score 0.27-0.42")
print(f"  Hard:   {hard_count:5d} ({hard_count/total*100:5.1f}%) - score >= 0.42")

# Feature correlations
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

ref_steps = [s['features']['num_reference_steps'] for s in complexity_scores]
q_length = [s['features']['question_length'] for s in complexity_scores]
q_words = [s['features']['question_words'] for s in complexity_scores]

corr_steps = np.corrcoef(scores, ref_steps)[0, 1]
corr_length = np.corrcoef(scores, q_length)[0, 1]
corr_words = np.corrcoef(scores, q_words)[0, 1]

print(f"\nCorrelation with complexity score:")
print(f"  Reference steps: {corr_steps:+.3f} (strong)" if abs(corr_steps) > 0.5 else f"  Reference steps: {corr_steps:+.3f}")
print(f"  Question length: {corr_length:+.3f} (strong)" if abs(corr_length) > 0.5 else f"  Question length: {corr_length:+.3f}")
print(f"  Question words:  {corr_words:+.3f} (strong)" if abs(corr_words) > 0.5 else f"  Question words:  {corr_words:+.3f}")

# Examples
print("\n" + "="*60)
print("EXAMPLE EASY QUESTIONS (Lowest 5 Complexity Scores)")
print("="*60)

sorted_by_complexity = sorted(complexity_scores, key=lambda x: x['complexity_score'])
for sample in sorted_by_complexity[:5]:
    print(f"\nSample {sample['sample_idx']} (Score: {sample['complexity_score']:.3f}):")
    print(f"  Level: {sample['complexity_level']}")
    print(f"  Steps: {sample['features']['num_reference_steps']}")
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answer']}")

print("\n" + "="*60)
print("EXAMPLE HARD QUESTIONS (Highest 5 Complexity Scores)")
print("="*60)

for sample in sorted_by_complexity[-5:]:
    print(f"\nSample {sample['sample_idx']} (Score: {sample['complexity_score']:.3f}):")
    print(f"  Level: {sample['complexity_level']}")
    print(f"  Steps: {sample['features']['num_reference_steps']}")
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answer']}")

# Save results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

output_path = Path(output_file)
with open(output_path, 'w') as f:
    json.dump({
        'metadata': {
            'dataset_path': dataset_path,
            'total_samples': len(complexity_scores),
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_median': float(np.median(scores))
        },
        'level_counts': {
            'easy': easy_count,
            'medium': medium_count,
            'hard': hard_count
        },
        'samples': complexity_scores
    }, f, indent=2)

print(f"✓ Saved complexity scores to: {output_path}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("""
Complexity scores are based on GROUND TRUTH features only:
  ✓ Reference step count (from annotations)
  ✓ Question length and structure
  ✓ Linguistic complexity (conditionals, causals, etc.)
  ✓ Question and answer types
  ✗ NO model predictions (unbiased!)

This ensures complexity scores are:
  - Model-agnostic (same for all models)
  - Unbiased (no model predictions)
  - Reproducible (deterministic)
  - Fair for comparison

Use these scores to:
  1. Report performance by difficulty tier
  2. Explain high variance in results
  3. Compare models fairly on same-difficulty questions
  4. Identify challenging question types
""")

print("="*60)
print(f"Complexity scores saved to: {output_file}")
print("="*60)
