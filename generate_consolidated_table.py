#!/usr/bin/env python3
"""
Generate consolidated results table from individual model evaluations
Combines metrics from all models into unified CSV and LaTeX tables
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List


def load_summary(summary_path: str) -> Dict:
    """Load metrics summary JSON"""
    with open(summary_path, 'r') as f:
        return json.load(f)


def parse_model_name(model_name: str) -> tuple:
    """
    Extract model family and size from model name
    Returns: (family, size_str, size_value)
    """
    model_name = model_name.lower()

    # InternVL3.5
    if 'internvl' in model_name:
        family = 'InternVL3.5'
        if '38b' in model_name:
            return family, '38B', 38.0
        elif '8b' in model_name:
            return family, '8B', 8.0
        elif '4b' in model_name:
            return family, '4B', 4.0
        elif '2b' in model_name:
            return family, '2B', 2.0
        elif '1b' in model_name:
            return family, '1B', 1.0

    # Qwen3-VL
    elif 'qwen3' in model_name:
        family = 'Qwen3-VL'
        if '235b' in model_name:
            return family, '235B', 235.0
        elif '32b' in model_name:
            return family, '32B', 32.0
        elif '8b' in model_name:
            return family, '8B', 8.0
        elif '2b' in model_name:
            return family, '2B', 2.0

    # Qwen2.5-VL
    elif 'qwen2.5' in model_name or 'qwen25' in model_name:
        family = 'Qwen2.5-VL'
        if '32b' in model_name:
            return family, '32B', 32.0
        elif '7b' in model_name:
            return family, '7B', 7.0
        elif '3b' in model_name:
            return family, '3B', 3.0

    # Gemma3
    elif 'gemma3' in model_name:
        family = 'Gemma3'
        if '12b' in model_name:
            return family, '12B', 12.0
        elif '4b' in model_name:
            return family, '4B', 4.0

    # LLaVA
    elif 'llava' in model_name:
        family = 'LLaVA'
        if '7b' in model_name:
            return family, '7B', 7.0

    # MiniCPM
    elif 'minicpm' in model_name:
        family = 'MiniCPM'
        if '8b' in model_name:
            return family, '8B', 8.0

    # Llama4
    elif 'llama4' in model_name:
        family = 'Llama4'
        if '16x17b' in model_name:
            return family, '16×17B', 272.0  # Total params

    # Default
    return 'Unknown', 'N/A', 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Generate consolidated results table'
    )
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Temporary results file with model|path pairs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for consolidated results'
    )

    args = parser.parse_args()

    # Read results file
    results = []
    with open(args.results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            model_name, summary_path = line.split('|')

            # Load summary
            summary = load_summary(summary_path)

            # Parse model info
            family, size_str, size_value = parse_model_name(model_name)

            # Extract metrics (handle nested structure)
            result = {
                'Model': model_name,
                'Family': family,
                'Size': size_str,
                'Params (B)': size_value,
                'Accuracy (%)': summary['accuracy']['mean'] * 100,
                'Match F1': summary['match_f1']['mean'],
                'Precision': summary['precision']['mean'],
                'Recall': summary['recall']['mean'],
                'F1 Std': summary['match_f1']['std'],
                'Avg Steps (Pred)': summary['steps']['predicted_mean'],
                'Avg Steps (Ref)': summary['steps']['reference_mean'],
                'Avg Similarity': summary['steps']['avg_similarity'],
                'Samples': summary['total_samples']
            }

            results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by family and then by size (descending)
    df = df.sort_values(['Family', 'Params (B)'], ascending=[True, False])

    # Save CSV
    csv_path = Path(args.output_dir) / 'consolidated_results.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n✓ Consolidated CSV saved: {csv_path}")

    # Generate LaTeX table
    latex_path = Path(args.output_dir) / 'consolidated_results.tex'
    generate_latex_table(df, latex_path)
    print(f"✓ LaTeX table saved: {latex_path}")

    # Generate summary statistics by family
    summary_path = Path(args.output_dir) / 'family_summary.txt'
    generate_family_summary(df, summary_path)
    print(f"✓ Family summary saved: {summary_path}")

    # Print to console
    print("\n" + "="*80)
    print("CONSOLIDATED RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print("="*80)

    # Print top performers
    print("\n" + "="*80)
    print("TOP PERFORMERS")
    print("="*80)

    print("\n📊 Best Accuracy:")
    top_acc = df.nlargest(3, 'Accuracy (%)')
    for idx, row in top_acc.iterrows():
        print(f"  {row['Model']:30s} {row['Accuracy (%)']:6.2f}%")

    print("\n🎯 Best Match F1:")
    top_f1 = df.nlargest(3, 'Match F1')
    for idx, row in top_f1.iterrows():
        print(f"  {row['Model']:30s} {row['Match F1']:6.4f}")

    print("\n⚖️  Best Precision:")
    top_prec = df.nlargest(3, 'Precision')
    for idx, row in top_prec.iterrows():
        print(f"  {row['Model']:30s} {row['Precision']:6.4f}")

    print("\n🔍 Best Recall:")
    top_rec = df.nlargest(3, 'Recall')
    for idx, row in top_rec.iterrows():
        print(f"  {row['Model']:30s} {row['Recall']:6.4f}")

    print("\n📈 Most Steps Generated:")
    top_steps = df.nlargest(3, 'Avg Steps (Pred)')
    for idx, row in top_steps.iterrows():
        print(f"  {row['Model']:30s} {row['Avg Steps (Pred)']:6.2f} steps")

    print("="*80 + "\n")


def generate_latex_table(df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table with booktabs formatting"""

    # Select key columns for paper
    df_tex = df[[
        'Model', 'Accuracy (%)', 'Match F1', 'Precision', 'Recall',
        'Avg Steps (Pred)', 'F1 Std'
    ]].copy()

    # Find best values for bold formatting
    best_acc = df_tex['Accuracy (%)'].max()
    best_f1 = df_tex['Match F1'].max()
    best_prec = df_tex['Precision'].max()
    best_rec = df_tex['Recall'].max()

    # Format cells
    def format_cell(row, col, value):
        if col == 'Model':
            return value

        formatted = f"{value:.2f}" if col == 'Accuracy (%)' else f"{value:.4f}" if col == 'Match F1' else f"{value:.3f}"

        # Bold best values
        if col == 'Accuracy (%)' and abs(value - best_acc) < 0.01:
            return f"\\textbf{{{formatted}}}"
        elif col == 'Match F1' and abs(value - best_f1) < 0.001:
            return f"\\textbf{{{formatted}}}"
        elif col == 'Precision' and abs(value - best_prec) < 0.001:
            return f"\\textbf{{{formatted}}}"
        elif col == 'Recall' and abs(value - best_rec) < 0.001:
            return f"\\textbf{{{formatted}}}"

        return formatted

    # Generate LaTeX
    with open(output_path, 'w') as f:
        f.write("% Auto-generated consolidated results table\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comprehensive evaluation of state-of-the-art MLLMs on CRYSTAL benchmark using all-distilroberta-v1 encoder ($\\tau=0.35$). \\textbf{Bold} indicates best performance.}\n")
        f.write("\\label{tab:consolidated_results}\n")
        f.write("\\resizebox{\\columnwidth}{!}{%\n")
        f.write("\\begin{tabular}{l c c c c c c}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Acc (\\%)} & \\textbf{F1} & \\textbf{Prec} & \\textbf{Rec} & \\textbf{Steps} & \\textbf{Std} \\\\\n")
        f.write("\\midrule\n")

        # Group by family
        for family in df['Family'].unique():
            family_df = df_tex[df['Family'] == family]

            # Family separator
            if family != df['Family'].unique()[0]:
                f.write("\\midrule\n")

            # Write rows
            for idx, row in family_df.iterrows():
                cells = [format_cell(row, col, row[col]) for col in df_tex.columns]
                f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}%\n")
        f.write("\\end{table}\n")


def generate_family_summary(df: pd.DataFrame, output_path: Path):
    """Generate summary statistics by model family"""

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL FAMILY SUMMARY\n")
        f.write("="*80 + "\n\n")

        for family in sorted(df['Family'].unique()):
            family_df = df[df['Family'] == family]

            f.write(f"{family}\n")
            f.write("-" * len(family) + "\n")
            f.write(f"  Models: {len(family_df)}\n")

            # Sort by actual parameter count for correct size range
            sorted_sizes = family_df.sort_values('Params (B)')
            min_size = sorted_sizes.iloc[0]['Size']
            max_size = sorted_sizes.iloc[-1]['Size']
            f.write(f"  Size range: {min_size} - {max_size}\n")
            f.write(f"  Accuracy range: {family_df['Accuracy (%)'].min():.2f}% - {family_df['Accuracy (%)'].max():.2f}%\n")
            f.write(f"  F1 range: {family_df['Match F1'].min():.4f} - {family_df['Match F1'].max():.4f}\n")
            f.write(f"  Avg steps range: {family_df['Avg Steps (Pred)'].min():.2f} - {family_df['Avg Steps (Pred)'].max():.2f}\n")

            # Compute scaling efficiency (F1 improvement per billion params)
            if len(family_df) > 1:
                sorted_family = family_df.sort_values('Params (B)')
                smallest = sorted_family.iloc[0]
                largest = sorted_family.iloc[-1]

                param_diff = largest['Params (B)'] - smallest['Params (B)']
                f1_diff = largest['Match F1'] - smallest['Match F1']

                if param_diff > 0:
                    efficiency = f1_diff / param_diff
                    f.write(f"  Scaling efficiency: {efficiency:.6f} F1 per B params\n")

            f.write("\n")


if __name__ == '__main__':
    main()
