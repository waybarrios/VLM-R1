#!/usr/bin/env python3
"""
Ablation Study: Test different similarity thresholds and embedding models
This script runs experiments varying:
1. Similarity thresholds (τ)
2. Sentence embedding models

Results are saved to JSON for paper table generation.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from datasets import load_from_disk

# Add mllm_evaluator to path
sys.path.insert(0, '/gpudata3/Wayner/VLM-R1/mllm_evaluator')
from mllm_evaluator import MLLMReasoningEvaluator


def load_dataset_hf(dataset_path: str) -> Dict[int, Dict]:
    """Load dataset using HuggingFace load_from_disk with error handling"""
    print(f"Loading dataset from {dataset_path}...")

    try:
        dataset = load_from_disk(dataset_path)
    except (TypeError, ValueError) as e:
        print(f"Warning: Schema validation error, loading with alternative method...")
        import pyarrow as pa
        from glob import glob

        arrow_files = glob(f"{dataset_path}/data-*.arrow")
        if not arrow_files:
            raise ValueError(f"No Arrow files found in {dataset_path}")

        tables = []
        for arrow_file in sorted(arrow_files):
            stream = pa.ipc.open_stream(arrow_file)
            table = stream.read_all()
            tables.append(table)

        full_table = pa.concat_tables(tables)

        all_data = {}
        for idx in tqdm(range(len(full_table)), desc="Converting dataset"):
            sample = {
                'question': full_table['question'][idx].as_py() if 'question' in full_table.column_names else '',
                'answer': full_table['answer'][idx].as_py() if 'answer' in full_table.column_names else '',
                'reference_steps': full_table['reference_steps'][idx].as_py() if 'reference_steps' in full_table.column_names else []
            }
            all_data[idx] = sample

        print(f"Dataset loaded: {len(all_data)} samples")
        return all_data

    print(f"Dataset info: {len(dataset)} samples")

    all_data = {}
    for idx in tqdm(range(len(dataset)), desc="Converting dataset"):
        sample = dataset[idx]
        all_data[idx] = {
            'question': sample.get('question', ''),
            'answer': sample.get('answer', ''),
            'reference_steps': sample.get('reference_steps', [])
        }

    print(f"Loaded {len(all_data)} samples from dataset")
    return all_data


def load_predictions(predictions_dir: str) -> Dict[int, List[str]]:
    """Load prediction JSON files"""
    predictions_path = Path(predictions_dir)

    if not predictions_path.exists():
        raise ValueError(f"Predictions directory not found: {predictions_dir}")

    # Find all JSON files
    json_files = sorted(predictions_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {predictions_dir}")

    print(f"Loading {len(json_files)} prediction files...")

    all_predictions = {}

    for json_file in tqdm(json_files, desc="Loading predictions"):
        try:
            # Sample idx is the filename (without .json)
            sample_idx = int(json_file.stem)

            with open(json_file, 'r') as f:
                data = json.load(f)

            reasoning_steps = data.get('reasoning_steps', [])
            all_predictions[sample_idx] = reasoning_steps

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue

    print(f"Loaded predictions for {len(all_predictions)} samples")
    return all_predictions


def evaluate_with_config(
    predictions: Dict[int, List[str]],
    dataset: Dict[int, Dict],
    encoder_name: str,
    threshold: float,
    device: str = "cuda:0"
) -> Dict:
    """
    Evaluate with specific encoder and threshold configuration

    Returns:
        Dictionary with aggregated metrics
    """
    print(f"\n{'='*60}")
    print(f"Encoder: {encoder_name}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}")

    # Initialize evaluator
    evaluator = MLLMReasoningEvaluator(
        model_name=encoder_name,
        similarity_threshold=threshold,
        device=device,
        debug_mode=False
    )

    # Collect metrics
    all_f1 = []
    all_precision = []
    all_recall = []
    num_pred_steps_list = []
    num_ref_steps_list = []

    # Evaluate all samples
    for sample_idx in tqdm(predictions.keys(), desc="Evaluating"):
        if sample_idx not in dataset:
            continue

        pred_steps = predictions[sample_idx]
        ref_steps = dataset[sample_idx]['reference_steps']

        # Evaluate
        try:
            metrics = evaluator.evaluate_single(pred_steps, ref_steps, verbose=False)

            all_f1.append(metrics.match_f1)
            all_precision.append(metrics.precision)
            all_recall.append(metrics.recall)
            num_pred_steps_list.append(metrics.num_predicted_steps)
            num_ref_steps_list.append(metrics.num_reference_steps)

        except Exception as e:
            print(f"Error evaluating sample {sample_idx}: {e}")
            continue

    # Compute aggregated statistics
    results = {
        'encoder': encoder_name,
        'threshold': threshold,
        'num_samples': len(all_f1),
        'match_f1': {
            'mean': float(np.mean(all_f1)),
            'std': float(np.std(all_f1)),
            'median': float(np.median(all_f1)),
            'min': float(np.min(all_f1)),
            'max': float(np.max(all_f1))
        },
        'precision': {
            'mean': float(np.mean(all_precision)),
            'std': float(np.std(all_precision))
        },
        'recall': {
            'mean': float(np.mean(all_recall)),
            'std': float(np.std(all_recall))
        },
        'num_predicted_steps': {
            'mean': float(np.mean(num_pred_steps_list))
        },
        'num_reference_steps': {
            'mean': float(np.mean(num_ref_steps_list))
        }
    }

    # Print summary
    print(f"\nResults:")
    print(f"  Match F1:  {results['match_f1']['mean']:.4f} ± {results['match_f1']['std']:.4f}")
    print(f"  Precision: {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}")
    print(f"  Recall:    {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments varying encoder and threshold"
    )
    parser.add_argument(
        "predictions_dir",
        type=str,
        help="Directory containing prediction JSON files"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27",
        help="Path to dataset (Arrow format)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="ablation_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=[
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "all-mpnet-base-v2",
            "all-distilroberta-v1"
        ],
        help="List of encoder models to test"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.35, 0.40, 0.45, 0.50],
        help="List of thresholds to test"
    )

    args = parser.parse_args()

    print("="*60)
    print("ABLATION STUDY: Encoder & Threshold Experiments")
    print("="*60)
    print(f"Predictions: {args.predictions_dir}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_file}")
    print(f"Device: {args.device}")
    print(f"Encoders to test: {len(args.encoders)}")
    print(f"Thresholds to test: {len(args.thresholds)}")
    print(f"Total experiments: {len(args.encoders) * len(args.thresholds)}")
    print("="*60)

    # Load dataset
    dataset = load_dataset_hf(args.dataset_path)

    # Load predictions
    predictions = load_predictions(args.predictions_dir)

    # Run ablation experiments
    all_results = []

    for encoder in args.encoders:
        for threshold in args.thresholds:
            result = evaluate_with_config(
                predictions=predictions,
                dataset=dataset,
                encoder_name=encoder,
                threshold=threshold,
                device=args.device
            )
            all_results.append(result)

    # Save results
    output_data = {
        'metadata': {
            'predictions_dir': args.predictions_dir,
            'dataset_path': args.dataset_path,
            'num_samples': len(predictions),
            'encoders_tested': args.encoders,
            'thresholds_tested': args.thresholds,
            'total_experiments': len(all_results)
        },
        'results': all_results
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_file}")
    print(f"{'='*60}")

    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Encoder':<40} {'Threshold':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("="*100)

    for result in all_results:
        print(f"{result['encoder']:<40} {result['threshold']:>10.2f} "
              f"{result['match_f1']['mean']:>10.4f} "
              f"{result['precision']['mean']:>10.4f} "
              f"{result['recall']['mean']:>10.4f}")

    print("="*100)


if __name__ == "__main__":
    main()
