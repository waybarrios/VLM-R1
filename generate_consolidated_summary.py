#!/usr/bin/env python3
"""
Generate consolidated summary of all model evaluations
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

def load_model_summary(model_name: str, results_dir: str = "metrics_results") -> Dict:
    """Load summary JSON for a model"""
    summary_path = Path(results_dir) / model_name / "no_judge_summary.json"

    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        return json.load(f)

def format_percentage(value: float) -> str:
    """Format as percentage"""
    return f"{value * 100:.2f}%"

def format_with_std(mean: float, std: float) -> str:
    """Format value with std deviation"""
    return f"{mean:.4f} ± {std:.4f}"

def print_model_comparison_table(models_data: Dict[str, Dict]):
    """Print comparison table"""
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE")
    print("="*100)
    print(f"{'Model':<30} {'Accuracy':<12} {'Match F1':<20} {'Precision':<20} {'Recall':<20}")
    print("-"*100)

    for model_name, data in sorted(models_data.items()):
        if data is None:
            print(f"{model_name:<30} {'NO DATA':<12} {'NO DATA':<20} {'NO DATA':<20} {'NO DATA':<20}")
            continue

        acc = data['accuracy_metrics']
        f1 = data['match_f1_metrics']

        accuracy_str = format_percentage(acc['overall_accuracy'])
        f1_str = format_with_std(f1['average_match_f1'], f1['std_match_f1'])
        precision_str = format_with_std(f1['average_precision'], f1['std_precision'])
        recall_str = format_with_std(f1['average_recall'], f1['std_recall'])

        print(f"{model_name:<30} {accuracy_str:<12} {f1_str:<20} {precision_str:<20} {recall_str:<20}")

    print("="*100)

def print_detailed_analysis(model_name: str, data: Dict):
    """Print detailed analysis for one model"""
    if data is None:
        print(f"\n❌ No data available for {model_name}")
        return

    acc = data['accuracy_metrics']
    f1 = data['match_f1_metrics']
    steps = data['step_matching_details']
    match_types = data['match_type_breakdown']

    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")

    print(f"\n📊 ACCURACY METRICS:")
    print(f"  Overall Accuracy:     {format_percentage(acc['overall_accuracy'])}")
    print(f"  Correct samples:      {acc['correct_samples']}/{acc['total_samples']}")
    print(f"  Average Confidence:   {acc['average_confidence']:.4f}")
    print(f"  Median Confidence:    {acc['median_confidence']:.4f}")

    print(f"\n📊 MATCH F1 METRICS:")
    print(f"  Average Match F1:     {format_with_std(f1['average_match_f1'], f1['std_match_f1'])}")
    print(f"  Median Match F1:      {f1['median_match_f1']:.4f}")
    print(f"  Average Precision:    {format_with_std(f1['average_precision'], f1['std_precision'])}")
    print(f"  Average Recall:       {format_with_std(f1['average_recall'], f1['std_recall'])}")

    print(f"\n📊 STEP MATCHING:")
    print(f"  Avg predicted steps:  {steps['avg_predicted_steps']:.2f}")
    print(f"  Avg reference steps:  {steps['avg_reference_steps']:.2f}")
    print(f"  Avg matched (pred):   {steps['avg_matched_predictions']:.2f}")
    print(f"  Avg matched (ref):    {steps['avg_matched_references']:.2f}")
    print(f"  Avg similarity:       {steps['avg_similarity']:.4f}")
    print(f"  Avg max similarity:   {steps['avg_max_similarity']:.4f}")

    print(f"\n📊 MATCH TYPE BREAKDOWN:")
    total = sum(match_types.values())
    for match_type, count in sorted(match_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {match_type:<20} {count:>5} ({percentage:>5.1f}%)")

    # Analysis
    print(f"\n💡 ANALYSIS:")

    # Precision vs Recall
    prec = f1['average_precision']
    rec = f1['average_recall']

    if prec > rec + 0.15:
        print("  ⚠️  HIGH PRECISION, LOW RECALL")
        print("      → Model generates few but accurate steps")
        print("      → Missing important reasoning steps")
        print("      → Recommendation: Encourage more comprehensive reasoning")
    elif rec > prec + 0.15:
        print("  ⚠️  HIGH RECALL, LOW PRECISION")
        print("      → Model generates many steps, some irrelevant")
        print("      → Verbose or redundant reasoning")
        print("      → Recommendation: Encourage concise, focused reasoning")
    else:
        print("  ✅ BALANCED PRECISION AND RECALL")
        print("      → Model generates appropriate number of relevant steps")

    # Standard deviation
    std_f1 = f1['std_match_f1']
    cv = (std_f1 / f1['average_match_f1'] * 100) if f1['average_match_f1'] > 0 else 0

    if cv > 40:
        print(f"\n  ⚠️  HIGH VARIANCE (CV = {cv:.1f}%)")
        print("      → Inconsistent performance across questions")
        print("      → Likely reflects dataset diversity (EXPECTED)")
        print("      → Recommendation: Report performance by difficulty tier")
    elif cv > 25:
        print(f"\n  📊 MODERATE VARIANCE (CV = {cv:.1f}%)")
        print("      → Some inconsistency in performance")
    else:
        print(f"\n  ✅ LOW VARIANCE (CV = {cv:.1f}%)")
        print("      → Consistent performance")

    # Placeholders
    placeholder_rate = (match_types.get('placeholder', 0) / total * 100) if total > 0 else 0
    if placeholder_rate > 20:
        print(f"\n  ⚠️  HIGH PLACEHOLDER RATE ({placeholder_rate:.1f}%)")
        print("      → Many samples without predictions")
        print("      → Recommendation: Investigate why model fails to predict")

def main():
    models = [
        "outputs_testing_llava7b_16",
        "outputs_testing_gemma3_4b",
        "outputs_testing_minicpm_v_8b",
        "outputs_testing_gemma3_12b_64k",
        "outputs_testing_qwen25vl_32b_64k"
    ]

    print("="*100)
    print("CONSOLIDATED EVALUATION SUMMARY - ALL MODELS")
    print("="*100)
    print("Dataset: reasoning_test_with_reference_steps_updated_v27")
    print("Total Samples: 6,372")
    print("Evaluation Mode: No Judge (Rule-based, no LLM)")
    print("="*100)

    # Load all data
    models_data = {}
    for model in models:
        models_data[model] = load_model_summary(model)

    # Print comparison table
    print_model_comparison_table(models_data)

    # Print detailed analysis for each model
    for model in models:
        print_detailed_analysis(model, models_data[model])

    # Dataset complexity summary
    complexity_file = Path("dataset_complexity_scores.json")
    if complexity_file.exists():
        print(f"\n{'='*80}")
        print("DATASET COMPLEXITY ANALYSIS")
        print(f"{'='*80}")

        with open(complexity_file) as f:
            complexity_data = json.load(f)

        metadata = complexity_data['metadata']
        level_counts = complexity_data['level_counts']

        print(f"\nComplexity Score Statistics:")
        print(f"  Mean:   {metadata['score_mean']:.3f}")
        print(f"  Std:    {metadata['score_std']:.3f}")
        print(f"  Median: {metadata['score_median']:.3f}")

        print(f"\nDifficulty Distribution:")
        total = metadata['total_samples']
        for level, count in level_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {level.capitalize():<12} {count:>5} ({percentage:>5.1f}%)")

        print(f"\nℹ️  Complexity scores are based on GROUND TRUTH only (no model bias)")
        print(f"    - Reference step count (r = +0.910)")
        print(f"    - Question length (r = +0.665)")
        print(f"    - Linguistic features")
        print(f"    - Question and answer types")

    print(f"\n{'='*100}")
    print("SUMMARY COMPLETE")
    print(f"{'='*100}")
    print(f"\nFor detailed metrics, see:")
    for model in models:
        print(f"  - metrics_results/{model}/no_judge_summary.txt")

    print(f"\nFor complexity analysis:")
    print(f"  - dataset_complexity_scores.json")

    print(f"\nFor metrics explanation:")
    print(f"  - METRICS_GUIDE.md")

    print(f"\n{'='*100}\n")

if __name__ == "__main__":
    main()
