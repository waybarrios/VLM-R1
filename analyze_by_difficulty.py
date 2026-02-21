#!/usr/bin/env python3
"""
Analyze model performance by question difficulty
Shows how to identify easy vs hard questions and compare performance
"""
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def load_complexity_scores(complexity_file: str = "dataset_complexity_scores.json"):
    """Load complexity scores"""
    with open(complexity_file) as f:
        return json.load(f)

def load_model_metrics(model_name: str, results_dir: str = "metrics_results"):
    """Load model evaluation metrics"""
    metrics_file = Path(results_dir) / model_name / "no_judge_metrics.csv"

    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return None

    return pd.read_csv(metrics_file)

def show_examples_by_difficulty(complexity_data, level: str, num_examples: int = 3):
    """Show example questions of a specific difficulty level"""
    samples = [s for s in complexity_data['samples'] if s['complexity_level'] == level]

    if not samples:
        print(f"  No {level} questions found!")
        return

    print(f"\n{'='*80}")
    print(f"EXAMPLE {level.upper()} QUESTIONS")
    print(f"{'='*80}")

    for i, sample in enumerate(samples[:num_examples], 1):
        print(f"\nExample {i}:")
        print(f"  Sample Index: {sample['sample_idx']}")
        print(f"  Complexity Score: {sample['complexity_score']:.3f}")
        print(f"  Reference Steps: {sample['features']['num_reference_steps']}")
        print(f"  Question Length: {sample['features']['question_length']} chars")
        print(f"  Question Words: {sample['features']['question_words']}")
        print(f"  Question: {sample['question'][:150]}...")
        print(f"  Answer: {sample['answer']}")

def analyze_performance_by_difficulty(metrics_df, complexity_map):
    """Analyze model performance stratified by difficulty"""

    # Add complexity info to metrics
    metrics_df['complexity_score'] = metrics_df['sample_idx'].map(
        lambda x: complexity_map.get(x, {}).get('complexity_score', np.nan)
    )
    metrics_df['complexity_level'] = metrics_df['sample_idx'].map(
        lambda x: complexity_map.get(x, {}).get('complexity_level', 'unknown')
    )
    metrics_df['num_reference_steps'] = metrics_df['sample_idx'].map(
        lambda x: complexity_map.get(x, {}).get('num_reference_steps', 0)
    )

    # Remove unknown complexity
    metrics_df = metrics_df[metrics_df['complexity_level'] != 'unknown']

    print(f"\n{'='*80}")
    print("PERFORMANCE BY DIFFICULTY TIER")
    print(f"{'='*80}")

    # Define order
    difficulty_order = ['easy', 'medium', 'hard', 'very_hard']

    # Group by difficulty
    grouped = metrics_df.groupby('complexity_level').agg({
        'accuracy_correct': ['mean', 'std', 'count'],
        'match_f1': ['mean', 'std', 'median'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'num_reference_steps': 'mean'
    }).round(4)

    # Reorder
    grouped = grouped.reindex([d for d in difficulty_order if d in grouped.index])

    print("\n" + str(grouped))

    # Calculate correlation
    corr_score = metrics_df['complexity_score'].corr(metrics_df['match_f1'])
    corr_steps = metrics_df['num_reference_steps'].corr(metrics_df['match_f1'])

    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Complexity Score vs Match F1: {corr_score:+.3f}")
    print(f"Reference Steps vs Match F1:  {corr_steps:+.3f}")

    if corr_score < -0.3:
        print("\n⚠️  NEGATIVE CORRELATION: Model struggles with complex questions")
    elif corr_score < 0.1:
        print("\n📊 WEAK CORRELATION: Performance not strongly tied to complexity")
    else:
        print("\n✅ POSITIVE CORRELATION: Model handles complexity well")

    return grouped

def show_hardest_questions(metrics_df, complexity_map, num_examples: int = 5):
    """Show the hardest questions where model failed"""

    # Add complexity
    metrics_df['complexity_score'] = metrics_df['sample_idx'].map(
        lambda x: complexity_map.get(x, {}).get('complexity_score', 0)
    )
    metrics_df['complexity_level'] = metrics_df['sample_idx'].map(
        lambda x: complexity_map.get(x, {}).get('complexity_level', 'unknown')
    )

    # Get failed hard questions
    hard_failed = metrics_df[
        (metrics_df['complexity_level'].isin(['hard', 'very_hard'])) &
        (metrics_df['accuracy_correct'] == False)
    ].sort_values('complexity_score', ascending=False)

    print(f"\n{'='*80}")
    print(f"HARDEST QUESTIONS WHERE MODEL FAILED (Top {num_examples})")
    print(f"{'='*80}")

    for i, (idx, row) in enumerate(hard_failed.head(num_examples).iterrows(), 1):
        sample_info = complexity_map.get(row['sample_idx'], {})
        print(f"\n{i}. Sample {row['sample_idx']}:")
        print(f"   Complexity Score: {row['complexity_score']:.3f}")
        print(f"   Match F1: {row['match_f1']:.3f}")
        print(f"   Correct: {row['accuracy_correct']}")
        print(f"   Reference Steps: {sample_info.get('num_reference_steps', 'N/A')}")

def show_easiest_questions_failed(metrics_df, complexity_map, num_examples: int = 5):
    """Show the easiest questions where model still failed"""

    # Add complexity
    metrics_df['complexity_score'] = metrics_df['sample_idx'].map(
        lambda x: complexity_map.get(x, {}).get('complexity_score', 0)
    )
    metrics_df['complexity_level'] = metrics_df['sample_idx'].map(
        lambda x: complexity_map.get(x, {}).get('complexity_level', 'unknown')
    )

    # Get failed easy questions
    easy_failed = metrics_df[
        (metrics_df['complexity_level'] == 'easy') &
        (metrics_df['accuracy_correct'] == False)
    ].sort_values('complexity_score', ascending=True)

    print(f"\n{'='*80}")
    print(f"EASIEST QUESTIONS WHERE MODEL FAILED (Bottom {num_examples})")
    print(f"{'='*80}")
    print("⚠️  These are concerning - model should handle these easily!")

    for i, (idx, row) in enumerate(easy_failed.head(num_examples).iterrows(), 1):
        sample_info = complexity_map.get(row['sample_idx'], {})
        print(f"\n{i}. Sample {row['sample_idx']}:")
        print(f"   Complexity Score: {row['complexity_score']:.3f}")
        print(f"   Match F1: {row['match_f1']:.3f}")
        print(f"   Correct: {row['accuracy_correct']}")
        print(f"   Reference Steps: {sample_info.get('num_reference_steps', 'N/A')}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_by_difficulty.py <model_name>")
        print("\nExample:")
        print("  python analyze_by_difficulty.py outputs_testing_qwen25vl_32b_64k")
        print("\nAvailable models:")
        results_dir = Path("metrics_results")
        if results_dir.exists():
            for model_dir in sorted(results_dir.iterdir()):
                if model_dir.is_dir():
                    print(f"  - {model_dir.name}")
        sys.exit(1)

    model_name = sys.argv[1]

    print("="*80)
    print("DIFFICULTY-STRATIFIED ANALYSIS")
    print("="*80)
    print(f"Model: {model_name}")
    print("="*80)

    # Load complexity scores
    print("\n📂 Loading complexity scores...")
    complexity_data = load_complexity_scores()

    # Create complexity map
    complexity_map = {
        s['sample_idx']: {
            'complexity_score': s['complexity_score'],
            'complexity_level': s['complexity_level'],
            'num_reference_steps': s['features']['num_reference_steps']
        }
        for s in complexity_data['samples']
    }

    print(f"✓ Loaded complexity for {len(complexity_map)} questions")

    # Show distribution
    level_counts = complexity_data['level_counts']
    total = complexity_data['metadata']['total_samples']

    print(f"\n📊 Dataset Complexity Distribution:")
    for level in ['easy', 'medium', 'hard', 'very_hard']:
        count = level_counts.get(level, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {level.capitalize():<12} {count:>5} ({pct:>5.1f}%)")

    # Show examples of each difficulty
    for level in ['easy', 'medium', 'hard']:
        show_examples_by_difficulty(complexity_data, level, num_examples=2)

    # Load model metrics
    print(f"\n📂 Loading model metrics...")
    metrics_df = load_model_metrics(model_name)

    if metrics_df is None:
        sys.exit(1)

    print(f"✓ Loaded metrics for {len(metrics_df)} samples")

    # Analyze performance by difficulty
    grouped = analyze_performance_by_difficulty(metrics_df, complexity_map)

    # Show hardest failures
    show_hardest_questions(metrics_df, complexity_map, num_examples=5)

    # Show easiest failures
    show_easiest_questions_failed(metrics_df, complexity_map, num_examples=5)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\n✓ Analysis complete!")
    print("\nKey Insights:")
    print("  1. Check if performance correlates negatively with complexity")
    print("  2. High variance in same difficulty tier indicates other factors")
    print("  3. Failures on easy questions indicate fundamental issues")
    print("  4. Use difficulty-stratified results in your paper")
    print("\nNext Steps:")
    print("  - Generate visualizations with visualize_for_paper.py")
    print("  - Report results by difficulty tier in your paper")
    print("  - Investigate why model fails on easy questions")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
