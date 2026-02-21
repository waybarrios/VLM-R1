#!/usr/bin/env python3
"""
Ablation Study: Simple version without multiprocessing
Processes samples sequentially on one GPU - slower but more reliable
"""


import os, pathlib
HF_CACHE_DIR = "/gpudata3/hf_cache"
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", HF_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", HF_CACHE_DIR)
pathlib.Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

import json
import sys
import os
from pathlib import Path
from typing import Dict, List
import argparse
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add mllm_evaluator to path
sys.path.insert(0, '/gpudata3/Wayner/VLM-R1/mllm_evaluator')
from mllm_evaluator import MLLMReasoningEvaluator


def load_predictions_fast(predictions_dir: str) -> Dict[int, List[str]]:
    """Load prediction JSON files quickly"""
    predictions_path = Path(predictions_dir)

    if not predictions_path.exists():
        raise ValueError(f"Predictions directory not found: {predictions_dir}")

    json_files = sorted(predictions_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {predictions_dir}")

    print(f"Loading {len(json_files)} prediction files...")

    all_predictions = {}

    for json_file in tqdm(json_files, desc="Loading predictions", ncols=80):
        try:
            # Sample idx is the filename (without .json)
            sample_idx = int(json_file.stem)

            with open(json_file, 'r') as f:
                data = json.load(f)

            reasoning_steps = data.get('reasoning_steps', [])
            all_predictions[sample_idx] = reasoning_steps

        except Exception as e:
            continue

    print(f"Loaded predictions for {len(all_predictions)} samples")
    return all_predictions


def load_dataset_fast(dataset_path: str, sample_indices: List[int]) -> Dict[int, Dict]:
    """Load only the needed samples from dataset"""
    import pyarrow as pa
    from glob import glob

    arrow_files = glob(f"{dataset_path}/data-*.arrow")
    if not arrow_files:
        raise ValueError(f"No Arrow files found in {dataset_path}")

    print("Loading dataset...")

    # Read all arrow files
    tables = []
    for arrow_file in sorted(arrow_files):
        stream = pa.ipc.open_stream(arrow_file)
        table = stream.read_all()
        tables.append(table)

    full_table = pa.concat_tables(tables)

    # Only load needed indices
    needed_indices = set(sample_indices)
    all_data = {}

    for idx in tqdm(needed_indices, desc="Loading dataset samples", ncols=80):
        if idx < len(full_table):
            sample = {
                'reference_steps': full_table['reference_steps'][idx].as_py()
                    if 'reference_steps' in full_table.column_names else []
            }
            all_data[idx] = sample

    print(f"Loaded {len(all_data)} dataset samples")
    return all_data


def evaluate_config(
    predictions: Dict[int, List[str]],
    dataset: Dict[int, Dict],
    encoder_name: str,
    threshold: float,
    device: str = "cuda:0"
) -> Dict:
    """
    Evaluate with specific encoder and threshold
    """
    print(f"\nEncoder: {encoder_name}, Threshold: {threshold}")

    # Initialize evaluator once
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

    # Evaluate all samples with progress bar
    sample_indices = list(predictions.keys())

    for sample_idx in tqdm(sample_indices,
                           desc=f"Evaluating {encoder_name[:20]:<20} τ={threshold}",
                           ncols=100):
        if sample_idx not in dataset:
            continue

        pred_steps = predictions[sample_idx]
        ref_steps = dataset[sample_idx]['reference_steps']

        try:
            metrics = evaluator.evaluate_single(pred_steps, ref_steps, verbose=False)

            all_f1.append(metrics.match_f1)
            all_precision.append(metrics.precision)
            all_recall.append(metrics.recall)
            num_pred_steps_list.append(metrics.num_predicted_steps)
            num_ref_steps_list.append(metrics.num_reference_steps)

        except Exception as e:
            # Skip failed samples
            continue

    # Check if we have results
    if len(all_f1) == 0:
        print("⚠️  WARNING: No valid results!")
        return {
            'encoder': encoder_name,
            'threshold': threshold,
            'num_samples': 0,
            'match_f1': {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0},
            'precision': {'mean': 0.0, 'std': 0.0},
            'recall': {'mean': 0.0, 'std': 0.0},
            'num_predicted_steps': {'mean': 0.0},
            'num_reference_steps': {'mean': 0.0}
        }

    # Compute statistics
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

    print(f"Results: F1={results['match_f1']['mean']:.4f} ± {results['match_f1']['std']:.4f}, "
          f"P={results['precision']['mean']:.4f}, R={results['recall']['mean']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments (simple sequential version)"
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
        required=True,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
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

    print("="*80)
    print("ABLATION STUDY - Simple Sequential Version")
    print("="*80)
    print(f"Model: {Path(args.predictions_dir).name}")
    print(f"Output: {args.output_file}")
    print(f"Device: {args.device}")
    print(f"Encoders: {len(args.encoders)}")
    print(f"Thresholds: {len(args.thresholds)}")
    print(f"Total configs: {len(args.encoders) * len(args.thresholds)}")
    print("="*80)
    print()

    # Load predictions
    predictions = load_predictions_fast(args.predictions_dir)

    # Load dataset
    dataset = load_dataset_fast(args.dataset_path, list(predictions.keys()))

    # Run ablation experiments
    all_results = []
    total_configs = len(args.encoders) * len(args.thresholds)
    current_config = 0

    for encoder in args.encoders:
        for threshold in args.thresholds:
            current_config += 1
            print(f"\n[{current_config}/{total_configs}]")

            result = evaluate_config(
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

    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {args.output_file}")
    print(f"{'='*80}")

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
