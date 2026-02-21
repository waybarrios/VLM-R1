#!/usr/bin/env python3
"""
Ablation Study: Sequential models, multi-GPU within each model
Processes one model at a time, but uses 4 GPUs to parallelize sample processing
Shows progress bar with time estimates
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.multiprocessing import Pool, set_start_method
import warnings
warnings.filterwarnings('ignore')

# Add mllm_evaluator to path
sys.path.insert(0, '/gpudata3/Wayner/VLM-R1/mllm_evaluator')
from mllm_evaluator import MLLMReasoningEvaluator

# Try to set multiprocessing start method
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass


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

    print("Loading dataset (fast mode - only needed samples)...")

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


def evaluate_single_sample(args):
    """Evaluate a single sample - for multiprocessing"""
    sample_idx, pred_steps, ref_steps, encoder_name, threshold, gpu_id = args

    device = f"cuda:{gpu_id}"

    try:
        # Create evaluator for this GPU
        evaluator = MLLMReasoningEvaluator(
            model_name=encoder_name,
            similarity_threshold=threshold,
            device=device,
            debug_mode=False
        )

        # Evaluate
        metrics = evaluator.evaluate_single(pred_steps, ref_steps, verbose=False)

        return {
            'sample_idx': sample_idx,
            'f1': metrics.match_f1,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'num_pred': metrics.num_predicted_steps,
            'num_ref': metrics.num_reference_steps
        }

    except Exception as e:
        return {
            'sample_idx': sample_idx,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'num_pred': 0,
            'num_ref': len(ref_steps),
            'error': str(e)
        }


def evaluate_with_multigpu(
    predictions: Dict[int, List[str]],
    dataset: Dict[int, Dict],
    encoder_name: str,
    threshold: float,
    num_gpus: int = 4
) -> Dict:
    """
    Evaluate with multiple GPUs in parallel
    """
    print(f"\nEncoder: {encoder_name}, Threshold: {threshold}")
    print(f"Using {num_gpus} GPUs for parallel processing")

    # Prepare arguments for multiprocessing
    sample_indices = list(predictions.keys())
    args_list = []

    for i, sample_idx in enumerate(sample_indices):
        if sample_idx not in dataset:
            continue

        pred_steps = predictions[sample_idx]
        ref_steps = dataset[sample_idx]['reference_steps']
        gpu_id = i % num_gpus  # Round-robin GPU assignment

        args_list.append((sample_idx, pred_steps, ref_steps, encoder_name, threshold, gpu_id))

    # Process in parallel using multiprocessing
    print(f"Processing {len(args_list)} samples across {num_gpus} GPUs...")

    with Pool(processes=num_gpus) as pool:
        results = list(tqdm(
            pool.imap(evaluate_single_sample, args_list),
            total=len(args_list),
            desc=f"Evaluating {encoder_name[:20]:<20} τ={threshold}",
            ncols=100
        ))

    # Aggregate results
    all_f1 = [r['f1'] for r in results if 'error' not in r]
    all_precision = [r['precision'] for r in results if 'error' not in r]
    all_recall = [r['recall'] for r in results if 'error' not in r]
    num_pred_steps = [r['num_pred'] for r in results if 'error' not in r]
    num_ref_steps = [r['num_ref'] for r in results if 'error' not in r]

    # Check if we have any valid results
    if len(all_f1) == 0:
        print("⚠️  WARNING: No valid results! All samples failed.")
        return {
            'encoder': encoder_name,
            'threshold': threshold,
            'num_samples': 0,
            'match_f1': {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0},
            'precision': {'mean': 0.0, 'std': 0.0},
            'recall': {'mean': 0.0, 'std': 0.0},
            'num_predicted_steps': {'mean': 0.0},
            'num_reference_steps': {'mean': 0.0},
            'error': 'No valid results'
        }

    aggregated = {
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
            'mean': float(np.mean(num_pred_steps))
        },
        'num_reference_steps': {
            'mean': float(np.mean(num_ref_steps))
        }
    }

    print(f"Results: F1={aggregated['match_f1']['mean']:.4f} ± {aggregated['match_f1']['std']:.4f}, "
          f"P={aggregated['precision']['mean']:.4f}, R={aggregated['recall']['mean']:.4f}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments with multi-GPU processing"
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
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use"
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
    print("ABLATION STUDY - Multi-GPU Processing")
    print("="*80)
    print(f"Model: {Path(args.predictions_dir).name}")
    print(f"Output: {args.output_file}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Encoders: {len(args.encoders)}")
    print(f"Thresholds: {len(args.thresholds)}")
    print(f"Total configs: {len(args.encoders) * len(args.thresholds)}")
    print("="*80)
    print()

    # Load predictions
    predictions = load_predictions_fast(args.predictions_dir)

    # Load dataset (only needed samples)
    dataset = load_dataset_fast(args.dataset_path, list(predictions.keys()))

    # Run ablation experiments
    all_results = []
    total_configs = len(args.encoders) * len(args.thresholds)
    current_config = 0

    for encoder in args.encoders:
        for threshold in args.thresholds:
            current_config += 1
            print(f"\n[{current_config}/{total_configs}] ", end="")

            result = evaluate_with_multigpu(
                predictions=predictions,
                dataset=dataset,
                encoder_name=encoder,
                threshold=threshold,
                num_gpus=args.num_gpus
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
            'total_experiments': len(all_results),
            'num_gpus_used': args.num_gpus
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
