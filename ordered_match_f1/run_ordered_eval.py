#!/usr/bin/env python3
"""
Ordered Match F1 Evaluation for all 20 MLLMs on CRYSTAL.

Computes standard Match F1 and Ordered Match F1 (with Kendall's Tau order penalty)
for all models that have individual prediction files.

Uses all-distilroberta-v1 with tau=0.35 (ablation-validated defaults).
"""

import json
import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict
import argparse
from tqdm import tqdm

# Add mllm_evaluator to path
sys.path.insert(0, '/gpudata3/Wayner/VLM-R1/mllm_evaluator')
from mllm_evaluator import MLLMReasoningEvaluator

import pyarrow as pa
from glob import glob as pyglob

# ============================================================================
# Model prediction directories (all 20 from Table 2)
# ============================================================================
MODELS = {
    # Commercial models (predictions in VLM-R1/final_table)
    "GPT-5": "/gpudata3/Wayner/VLM-R1/final_table/outputs_testing_gpt5/predictions",
    "GPT-5-mini": "/gpudata3/Wayner/VLM-R1/final_table/outputs_testing_gpt5mini/predictions",
    "GPT-5.2 Instant": "/gpudata3/Wayner/VLM-R1/final_table/outputs_testing_gpt52_instant/predictions",
    "Gemini 2.5 Flash": "/gpudata3/Wayner/VLM-R1/final_table/outputs_testing_gemini_2_5_flash/predictions",
    # Qwen family (predictions in /gpudata3/Wayner/reasoning/)
    "Qwen3-VL-8B": "/gpudata3/Wayner/reasoning/outputs_testing_qwen3vl_8b",
    "Qwen3-VL-32B": "/gpudata3/Wayner/reasoning/outputs_testing_qwen3vl_32b",
    "Qwen2.5-VL-32B": "/gpudata3/Wayner/reasoning/outputs_testing_qwen25vl_32b_64k",
    "Qwen3-VL-2B": "/gpudata3/Wayner/reasoning/outputs_testing_qwen3vl_2b",
    "Qwen2.5-VL-7B": "/gpudata3/Wayner/reasoning/outputs_testing_qwen25vl_7b",
    "Qwen2.5-VL-3B": "/gpudata3/Wayner/reasoning/outputs_testing_qwen25vl_3b",
    # InternVL family
    "InternVL3.5-38B": "/gpudata3/Wayner/reasoning/outputs_testing_internvl35_38b",
    "InternVL3.5-8B": "/gpudata3/Wayner/reasoning/outputs_testing_internvl35_8b",
    "InternVL3.5-4B": "/gpudata3/Wayner/reasoning/outputs_testing_internvl35_4b",
    "InternVL3.5-2B": "/gpudata3/Wayner/reasoning/outputs_testing_internvl35_2b",
    "InternVL3.5-1B": "/gpudata3/Wayner/reasoning/outputs_testing_internvl35_1b",
    # Other open-source
    "Gemma3-12B": "/gpudata3/Wayner/reasoning/outputs_testing_gemma3_12b_64k",
    "Gemma3-4B": "/gpudata3/Wayner/reasoning/outputs_testing_gemma3_4b",
    "Llama 3.2-11B": "/gpudata3/Wayner/VLM-R1/final_table/outputs_testing_llama3_2_11b/predictions",
    "LLaVA-v1.6-7B": "/gpudata3/Wayner/reasoning/outputs_testing_llava7b_16",
    "MiniCPMv2.6-8B": "/gpudata3/Wayner/reasoning/outputs_testing_minicpm_v_8b",
}

DATASET_PATH = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
OUTPUT_DIR = "/gpudata3/Wayner/VLM-R1/ordered_match_f1/results"


def load_predictions(pred_dir: str) -> Dict[int, Dict]:
    """Load prediction JSON files from a directory."""
    predictions = {}
    pred_path = Path(pred_dir)

    if not pred_path.exists():
        print(f"  WARNING: Directory not found: {pred_dir}")
        return predictions

    for json_file in pred_path.glob("*.json"):
        # Skip non-numeric files (summary, metrics, etc.)
        try:
            idx = int(json_file.stem)
        except ValueError:
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Handle raw_responses format (Gemini, Llama)
            if 'raw_response' in data and 'reasoning_steps' not in data:
                try:
                    parsed = json.loads(data['raw_response'])
                    data['reasoning_steps'] = parsed.get('reasoning_steps', [])
                    data['answer'] = parsed.get('answer', '')
                except (json.JSONDecodeError, TypeError):
                    data['reasoning_steps'] = []
                    data['answer'] = ''

            predictions[idx] = data
        except (json.JSONDecodeError, Exception) as e:
            continue

    return predictions


def evaluate_model(
    model_name: str,
    pred_dir: str,
    dataset,
    evaluator: MLLMReasoningEvaluator,
    alphas: List[float],
) -> Dict:
    """Evaluate a single model with multiple alpha values, computing both Kendall's Tau and LIS."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Predictions: {pred_dir}")
    print(f"{'='*60}")

    predictions = load_predictions(pred_dir)
    if not predictions:
        print(f"  No predictions found, skipping.")
        return None

    print(f"  Loaded {len(predictions)} predictions")

    detailed = []
    evaluated = 0
    total_samples = len(dataset)

    for idx in tqdm(range(total_samples), desc=f"  {model_name}", leave=True):
        ref_steps = dataset[idx].get('reference_steps', [])
        if not ref_steps:
            continue

        pred = predictions.get(idx, {})
        pred_steps = pred.get('reasoning_steps', [])

        if not pred_steps:
            # Missing or empty prediction: count as zeros (consistent with Table 2)
            detailed.append({
                'sample_idx': idx,
                'match_f1': 0.0,
                'kendall_tau': 1.0,
                'tau_normalized': 1.0,
                'lis_ratio': 1.0,
                'precision': 0.0,
                'recall': 0.0,
                'num_predicted_steps': 0,
                'num_reference_steps': len(ref_steps),
                'num_matched': 0,
            })
            evaluated += 1
            continue

        # Evaluate once: gets both tau and lis
        metrics = evaluator.evaluate_single(pred_steps, ref_steps, alpha=0.3, order_metric="kendall_tau")
        evaluated += 1

        detailed.append({
            'sample_idx': idx,
            'match_f1': metrics.match_f1,
            'kendall_tau': metrics.kendall_tau,
            'tau_normalized': metrics.tau_normalized,
            'lis_ratio': metrics.lis_ratio,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'num_predicted_steps': metrics.num_predicted_steps,
            'num_reference_steps': metrics.num_reference_steps,
            'num_matched': metrics.num_matched_predictions,
        })

    if not detailed:
        print(f"  No valid samples, skipping.")
        return None

    # Compute summary for both metrics and all alphas
    import numpy as np
    n = len(detailed)
    avg_f1 = np.mean([d['match_f1'] for d in detailed])
    avg_tau = np.mean([d['kendall_tau'] for d in detailed])
    avg_tau_norm = np.mean([d['tau_normalized'] for d in detailed])
    avg_lis = np.mean([d['lis_ratio'] for d in detailed])
    avg_prec = np.mean([d['precision'] for d in detailed])
    avg_rec = np.mean([d['recall'] for d in detailed])
    avg_steps = np.mean([d['num_predicted_steps'] for d in detailed])
    avg_matched = np.mean([d['num_matched'] for d in detailed])

    summary = {
        'model': model_name,
        'total_evaluated': evaluated,
        'match_f1': float(avg_f1),
        'kendall_tau': float(avg_tau),
        'tau_norm': float(avg_tau_norm),
        'lis_ratio': float(avg_lis),
        'precision': float(avg_prec),
        'recall': float(avg_rec),
        'avg_steps': float(avg_steps),
        'avg_matched': float(avg_matched),
    }

    # Compute ordered F1 for each alpha x each order metric
    for alpha in alphas:
        # Kendall Tau version
        kt_vals = [d['match_f1'] * ((1 - alpha) + alpha * d['tau_normalized']) for d in detailed]
        summary[f'ordered_f1_kt_a{alpha:.2f}'] = float(np.mean(kt_vals))
        # LIS version
        lis_vals = [d['match_f1'] * ((1 - alpha) + alpha * d['lis_ratio']) for d in detailed]
        summary[f'ordered_f1_lis_a{alpha:.2f}'] = float(np.mean(lis_vals))

    # Print summary
    print(f"  Evaluated: {evaluated} samples")
    print(f"  Match F1={avg_f1:.4f}  Tau={avg_tau:.4f}  LIS={avg_lis:.4f}  Steps={avg_steps:.1f}  Matched={avg_matched:.1f}")
    print(f"  Ordered F1 (a=0.3): Kendall={summary['ordered_f1_kt_a0.30']:.4f}  LIS={summary['ordered_f1_lis_a0.30']:.4f}")

    return {'summary': summary, 'detailed': detailed}


def main():
    parser = argparse.ArgumentParser(description="Ordered Match F1 evaluation for 20 MLLMs")
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5],
                        help='Alpha values for order sensitivity')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to evaluate (default: all 20)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset (with pyarrow fallback for schema issues)
    print("Loading CRYSTAL dataset...")
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(DATASET_PATH)
        print(f"Dataset: {len(dataset)} samples")
    except (TypeError, ValueError) as e:
        print(f"Schema error, loading with pyarrow fallback...")
        arrow_files = sorted(pyglob(f"{DATASET_PATH}/data-*.arrow"))
        tables = [pa.ipc.open_stream(f).read_all() for f in arrow_files]
        full_table = pa.concat_tables(tables)
        # Convert to dict-like access
        dataset = []
        for idx in range(len(full_table)):
            dataset.append({
                'question': full_table['question'][idx].as_py(),
                'answer': full_table['answer'][idx].as_py(),
                'reference_steps': full_table['reference_steps'][idx].as_py() if 'reference_steps' in full_table.column_names else [],
            })
        print(f"Dataset: {len(dataset)} samples (pyarrow)")

    # Initialize evaluator (all-distilroberta-v1, tau=0.35)
    print("Initializing evaluator (all-distilroberta-v1, tau=0.35)...")
    evaluator = MLLMReasoningEvaluator(
        model_name="all-distilroberta-v1",
        similarity_threshold=0.35,
        device=args.device,
    )

    # Select models
    models_to_eval = {}
    if args.models:
        for m in args.models:
            if m in MODELS:
                models_to_eval[m] = MODELS[m]
            else:
                print(f"WARNING: Unknown model '{m}', skipping")
    else:
        models_to_eval = MODELS

    print(f"\nEvaluating {len(models_to_eval)} models with alphas={args.alphas}")

    # Run evaluations
    all_summaries = []
    for model_name, pred_dir in models_to_eval.items():
        result = evaluate_model(model_name, pred_dir, dataset, evaluator, args.alphas)
        if result:
            all_summaries.append(result['summary'])

            # Save detailed per-model results
            model_safe = model_name.replace(' ', '_').replace('.', '_')
            detail_file = Path(args.output_dir) / f"detailed_{model_safe}.csv"
            if result['detailed']:
                keys = result['detailed'][0].keys()
                with open(detail_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(result['detailed'])

    # Save consolidated summary
    if all_summaries:
        summary_file = Path(args.output_dir) / "ordered_f1_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")

        # Save as CSV for easy comparison
        csv_file = Path(args.output_dir) / "ordered_f1_summary.csv"
        keys = all_summaries[0].keys()
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"CSV saved to: {csv_file}")

        # Print comparison table: Kendall Tau vs LIS
        print(f"\n{'='*120}")
        print("COMPARISON: Kendall's Tau vs LIS Ratio (alpha=0.3)")
        print(f"{'='*120}")
        print(f"{'Model':<22} {'Match F1':>9} {'Tau':>7} {'LIS':>7} {'Matched':>8} "
              f"{'OrdF1(KT)':>10} {'OrdF1(LIS)':>11} {'Diff':>7}")
        print('-' * 120)
        for s in sorted(all_summaries, key=lambda x: x.get('match_f1', 0), reverse=True):
            kt = s.get('ordered_f1_kt_a0.30', 0)
            lis = s.get('ordered_f1_lis_a0.30', 0)
            diff = lis - kt
            print(f"{s['model']:<22} {s['match_f1']:>9.4f} {s['kendall_tau']:>7.4f} "
                  f"{s['lis_ratio']:>7.4f} {s['avg_matched']:>8.1f} "
                  f"{kt:>10.4f} {lis:>11.4f} {diff:>+7.4f}")
        print(f"{'='*120}")

        # Print multi-alpha table for both metrics
        for metric_name, key_prefix in [("Kendall's Tau", "ordered_f1_kt_a"),
                                         ("LIS Ratio", "ordered_f1_lis_a")]:
            print(f"\n{'='*100}")
            print(f"Ordered Match F1 using {metric_name}")
            print(f"{'='*100}")
            header = f"{'Model':<22} {'Match F1':>9} "
            header += ' '.join(f"{'a='+str(a):>9}" for a in args.alphas if a > 0)
            print(header)
            print('-' * 100)
            for s in sorted(all_summaries, key=lambda x: x.get(f'{key_prefix}0.30', 0), reverse=True):
                line = f"{s['model']:<22} {s['match_f1']:>9.4f} "
                for a in args.alphas:
                    if a > 0:
                        line += f"{s[f'{key_prefix}{a:.2f}']:>9.4f} "
                print(line)
            print(f"{'='*100}")


if __name__ == "__main__":
    main()
