#!/usr/bin/env python3
"""
Consolidate GRPO Analysis Results
Creates comprehensive tables comparing baseline vs all checkpoints
"""

import json
import pandas as pd
from pathlib import Path
import sys


def extract_step_number(checkpoint_name):
    """Extract step number from checkpoint name"""
    # Baseline is outputs_testing_qwen25vl_3b (not a training step)
    if 'qwen25vl_3b' in checkpoint_name.lower() or 'baseline' in checkpoint_name.lower():
        return -1  # Use -1 to indicate baseline (not a training checkpoint)
    # Check for checkpoint-XXX pattern
    if 'checkpoint-' in checkpoint_name:
        try:
            return int(checkpoint_name.split('checkpoint-')[-1])
        except:
            return -1
    # Default for unknown patterns
    return -1


def load_all_results(results_dir):
    """Load all result summaries from the results directory"""
    results_dir = Path(results_dir)

    all_results = []

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        summary_file = model_dir / "summary.json"

        if not summary_file.exists():
            print(f"Warning: No summary.json found in {model_dir}")
            continue

        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)

            model_name = model_dir.name
            step = extract_step_number(model_name)

            # Extract key metrics
            result = {
                'model_name': model_name,
                'step': step,
                'accuracy': data['accuracy_metrics']['overall_accuracy'],
                'accuracy_correct': data['accuracy_metrics']['correct_samples'],
                'accuracy_total': data['accuracy_metrics']['total_samples'],
                'match_f1_mean': data['match_f1_metrics']['average_match_f1'],
                'match_f1_std': data['match_f1_metrics']['std_match_f1'],
                'match_f1_median': data['match_f1_metrics']['median_match_f1'],
                'precision_mean': data['match_f1_metrics']['average_precision'],
                'precision_std': data['match_f1_metrics']['std_precision'],
                'recall_mean': data['match_f1_metrics']['average_recall'],
                'recall_std': data['match_f1_metrics']['std_recall'],
                'avg_pred_steps': data['step_matching_details']['avg_predicted_steps'],
                'avg_ref_steps': data['step_matching_details']['avg_reference_steps'],
                'avg_similarity': data['step_matching_details']['avg_similarity'],
            }

            all_results.append(result)
            print(f"✓ Loaded: {model_name}")

        except Exception as e:
            print(f"Error loading {summary_file}: {e}")
            continue

    return all_results


def create_comparison_table(results):
    """Create comprehensive comparison table"""
    df = pd.DataFrame(results)

    # Sort by step number
    df = df.sort_values('step').reset_index(drop=True)

    return df


def save_results(df, output_dir):
    """Save consolidated results in multiple formats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. CSV with all metrics
    csv_file = output_dir / "consolidated_results.csv"
    df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"\n✓ Full CSV saved to: {csv_file}")

    # 2. Summary table (key metrics only)
    summary_df = df[[
        'model_name', 'step', 'accuracy',
        'match_f1_mean', 'precision_mean', 'recall_mean',
        'avg_pred_steps'
    ]].copy()

    summary_csv = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_csv, index=False, float_format='%.4f')
    print(f"✓ Summary CSV saved to: {summary_csv}")

    # 3. Markdown table for reports
    markdown_file = output_dir / "results_table.md"
    with open(markdown_file, 'w') as f:
        f.write("# GRPO Analysis Results\n\n")
        f.write("## Settings\n")
        f.write("- **Encoder**: all-distilroberta-v1\n")
        f.write("- **Threshold**: 0.35\n")
        f.write("- **Dataset**: reasoning_test_with_reference_steps_updated_v27\n\n")

        f.write("## Full Results Table\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Models**: {len(df)}\n")

        # Baseline is step=-1 (outputs_testing_qwen25vl_3b)
        baseline_df = df[df['step'] == -1]
        if len(baseline_df) > 0:
            f.write(f"- **Baseline Accuracy**: {baseline_df['accuracy'].values[0]:.4f}\n")
            f.write(f"- **Baseline Match F1**: {baseline_df['match_f1_mean'].values[0]:.4f}\n")

        if len(df) > 1:
            best_acc_idx = df['accuracy'].idxmax()
            best_f1_idx = df['match_f1_mean'].idxmax()

            f.write(f"- **Best Accuracy**: {df.loc[best_acc_idx, 'accuracy']:.4f} ({df.loc[best_acc_idx, 'model_name']})\n")
            f.write(f"- **Best Match F1**: {df.loc[best_f1_idx, 'match_f1_mean']:.4f} ({df.loc[best_f1_idx, 'model_name']})\n")

        f.write("\n")

    print(f"✓ Markdown table saved to: {markdown_file}")

    # 4. Human-readable text report
    txt_file = output_dir / "CONSOLIDATED_RESULTS.txt"
    with open(txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GRPO ANALYSIS - CONSOLIDATED RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write("Configuration:\n")
        f.write("  Encoder Model: all-distilroberta-v1\n")
        f.write("  Threshold: 0.35\n")
        f.write("  Dataset: reasoning_test_with_reference_steps_updated_v27\n")
        f.write(f"  Total Models Evaluated: {len(df)}\n\n")

        f.write("="*80 + "\n")
        f.write("RESULTS BY CHECKPOINT\n")
        f.write("="*80 + "\n\n")

        for idx, row in df.iterrows():
            f.write(f"{'─'*80}\n")
            f.write(f"Model: {row['model_name']}\n")
            step_display = 'Baseline (pre-GRPO)' if row['step'] == -1 else f"Step {row['step']}"
            f.write(f"Training: {step_display}\n")
            f.write(f"{'─'*80}\n")
            f.write(f"  Accuracy:        {row['accuracy']:.4f} ({row['accuracy_correct']}/{row['accuracy_total']})\n")
            f.write(f"  Match F1:        {row['match_f1_mean']:.4f} (±{row['match_f1_std']:.4f})\n")
            f.write(f"  Precision:       {row['precision_mean']:.4f} (±{row['precision_std']:.4f})\n")
            f.write(f"  Recall:          {row['recall_mean']:.4f} (±{row['recall_std']:.4f})\n")
            f.write(f"  Avg Pred Steps:  {row['avg_pred_steps']:.2f}\n")
            f.write(f"  Avg Ref Steps:   {row['avg_ref_steps']:.2f}\n")
            f.write(f"  Avg Similarity:  {row['avg_similarity']:.4f}\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")

        # Baseline is step=-1 (outputs_testing_qwen25vl_3b)
        baseline_row = df[df['step'] == -1].iloc[0] if len(df[df['step'] == -1]) > 0 else None

        if baseline_row is not None:
            f.write("Baseline Performance:\n")
            f.write(f"  Accuracy:  {baseline_row['accuracy']:.4f}\n")
            f.write(f"  Match F1:  {baseline_row['match_f1_mean']:.4f}\n\n")

        if len(df) > 1:
            best_acc_idx = df['accuracy'].idxmax()
            best_f1_idx = df['match_f1_mean'].idxmax()

            f.write("Best Performance:\n")
            f.write(f"  Best Accuracy:  {df.loc[best_acc_idx, 'accuracy']:.4f} ({df.loc[best_acc_idx, 'model_name']})\n")
            f.write(f"  Best Match F1:  {df.loc[best_f1_idx, 'match_f1_mean']:.4f} ({df.loc[best_f1_idx, 'model_name']})\n\n")

            if baseline_row is not None:
                f.write("Improvement over Baseline:\n")
                acc_improvement = df.loc[best_acc_idx, 'accuracy'] - baseline_row['accuracy']
                f1_improvement = df.loc[best_f1_idx, 'match_f1_mean'] - baseline_row['match_f1_mean']
                f.write(f"  Accuracy:  {acc_improvement:+.4f} ({acc_improvement/baseline_row['accuracy']*100:+.2f}%)\n")
                f.write(f"  Match F1:  {f1_improvement:+.4f} ({f1_improvement/baseline_row['match_f1_mean']*100:+.2f}%)\n\n")

        f.write("="*80 + "\n")

    print(f"✓ Text report saved to: {txt_file}")

    # 5. JSON summary
    json_file = output_dir / "consolidated_summary.json"
    summary_data = {
        'config': {
            'encoder': 'all-distilroberta-v1',
            'threshold': 0.35,
            'dataset': 'reasoning_test_with_reference_steps_updated_v27'
        },
        'results': df.to_dict(orient='records')
    }

    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"✓ JSON summary saved to: {json_file}")


def print_summary_table(df):
    """Print summary table to console"""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()

    # Create formatted table
    print(f"{'Model':<30} {'Step':<8} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("─" * 80)

    for idx, row in df.iterrows():
        model_display = row['model_name'][:28] if len(row['model_name']) > 28 else row['model_name']
        step_display = 'Baseline' if row['step'] == -1 else str(row['step'])
        print(f"{model_display:<30} {step_display:<8} "
              f"{row['accuracy']:>8.4f} {row['match_f1_mean']:>8.4f} "
              f"{row['precision_mean']:>8.4f} {row['recall_mean']:>8.4f}")

    print("="*80)
    print()


def main():
    results_dir = Path("./GRPO_analysis/results")
    output_dir = Path("./GRPO_analysis")

    print("="*80)
    print("GRPO ANALYSIS - RESULTS CONSOLIDATION")
    print("="*80)
    print()
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Check if results directory exists
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run the evaluation script first:")
        print("  bash GRPO_analysis/run_grpo_analysis.sh")
        sys.exit(1)

    # Load all results
    print("Loading results...")
    results = load_all_results(results_dir)

    if not results:
        print("Error: No results found!")
        sys.exit(1)

    print(f"\nLoaded {len(results)} model results")
    print()

    # Create comparison table
    print("Creating comparison table...")
    df = create_comparison_table(results)

    # Print to console
    print_summary_table(df)

    # Save in multiple formats
    print("Saving consolidated results...")
    save_results(df, output_dir)

    print()
    print("="*80)
    print("CONSOLIDATION COMPLETE!")
    print("="*80)
    print()
    print("Generated files:")
    print(f"  1. {output_dir}/consolidated_results.csv (full data)")
    print(f"  2. {output_dir}/summary_table.csv (key metrics)")
    print(f"  3. {output_dir}/results_table.md (markdown format)")
    print(f"  4. {output_dir}/CONSOLIDATED_RESULTS.txt (detailed report)")
    print(f"  5. {output_dir}/consolidated_summary.json (JSON format)")
    print()


if __name__ == "__main__":
    main()
