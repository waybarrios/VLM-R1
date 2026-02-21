#!/usr/bin/env python3
"""
FINAL metrics computation for CRYSTAL benchmark
Uses DistilRoBERTa-v1 with threshold 0.35 (from ablation study)
Supports multi-GPU processing with progress bars
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from datasets import load_from_disk

# Add mllm_evaluator to path
sys.path.insert(0, '/gpudata3/Wayner/VLM-R1/mllm_evaluator')
from mllm_evaluator import MLLMReasoningEvaluator
from accuracy_calculator import AccuracyCalculator


@dataclass
class MetricResult:
    """Combined metrics for a single prediction"""
    sample_idx: int
    accuracy_correct: bool
    match_f1: float
    precision: float
    recall: float
    num_predicted_steps: int
    num_reference_steps: int
    avg_similarity: float
    answer: str
    match_type: str
    confidence: float


def load_dataset_hf(dataset_path: str) -> Dict[int, Dict]:
    """Load dataset using HuggingFace load_from_disk"""
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

    print(f"Dataset loaded: {len(dataset)} samples")

    all_data = {}
    for idx in tqdm(range(len(dataset)), desc="Converting dataset"):
        sample = dataset[idx]
        all_data[idx] = {
            'question': sample.get('question', ''),
            'answer': sample.get('answer', ''),
            'reference_steps': sample.get('reference_steps', [])
        }

    return all_data


def load_predictions(predictions_dir: str, valid_indices: set) -> Dict[int, Dict]:
    """Load prediction JSON files from directory"""
    predictions = {}
    pred_files = sorted(Path(predictions_dir).glob("*.json"))

    print(f"Loading predictions from {predictions_dir}...")
    print(f"Found {len(pred_files)} JSON files")

    for pred_file in tqdm(pred_files, desc="Loading predictions"):
        try:
            idx = int(pred_file.stem)
            if idx not in valid_indices:
                continue

            with open(pred_file, 'r') as f:
                data = json.load(f)
                predictions[idx] = {
                    'predicted_steps': data.get('reasoning_steps', []),
                    'predicted_answer': data.get('answer', '')
                }
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Warning: Skipping {pred_file}: {e}")
            continue

    print(f"Loaded {len(predictions)} predictions")
    return predictions


def evaluate_batch_gpu(args: Tuple) -> List[Dict]:
    """
    Evaluate a batch of samples on a specific GPU

    Args:
        args: Tuple of (gpu_id, indices, predictions, ground_truth, encoder, threshold)

    Returns:
        List of result dictionaries
    """
    gpu_id, indices, predictions, ground_truth, encoder_name, threshold = args

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Initialize evaluators with specified encoder and threshold
    match_f1_evaluator = MLLMReasoningEvaluator(
        model_name=encoder_name,
        similarity_threshold=threshold,
        device=device,
        debug_mode=False
    )

    accuracy_calculator = AccuracyCalculator(
        use_llm_grader=False,  # Use rule-based matching
        llm_model="llama3.2",
        base_url="http://localhost:11434/v1"
    )

    results = []

    # Create progress bar for this GPU
    pbar = tqdm(
        indices,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
        leave=True
    )

    for idx in pbar:
        try:
            pred = predictions[idx]
            gt = ground_truth[idx]

            # Compute Match F1
            metrics_f1 = match_f1_evaluator.evaluate_single(
                predicted_steps=pred['predicted_steps'],
                reference_steps=gt['reference_steps']
            )

            # Compute Accuracy
            accuracy_result = accuracy_calculator.evaluate_single(
                question=gt['question'],
                predicted_answer=pred['predicted_answer'],
                ground_truth_answer=gt['answer']
            )

            result = {
                'sample_idx': idx,
                'accuracy_correct': accuracy_result.is_correct,
                'match_f1': metrics_f1.match_f1,
                'precision': metrics_f1.precision,
                'recall': metrics_f1.recall,
                'num_predicted_steps': metrics_f1.num_predicted_steps,
                'num_reference_steps': metrics_f1.num_reference_steps,
                'avg_similarity': metrics_f1.avg_similarity,
                'answer': pred['predicted_answer'],
                'match_type': accuracy_result.match_type,
                'confidence': accuracy_result.confidence
            }

            results.append(result)

        except Exception as e:
            print(f"\nError processing sample {idx} on GPU {gpu_id}: {e}")
            continue

    pbar.close()
    return results


def compute_metrics(
    predictions_dir: str,
    dataset_path: str,
    encoder_name: str,
    threshold: float,
    num_gpus: int = 4,
    output_dir: str = "./final_table",
    model_name: str = None
) -> pd.DataFrame:
    """
    Compute final metrics with specified encoder and threshold
    """
    # Load data
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    ground_truth = load_dataset_hf(dataset_path)

    print("\n" + "="*60)
    print("Loading predictions...")
    print("="*60)
    predictions = load_predictions(predictions_dir, set(ground_truth.keys()))

    # Find common indices
    pred_indices = set(predictions.keys())
    gt_indices = set(ground_truth.keys())
    common_indices = sorted(pred_indices.intersection(gt_indices))

    print("\n" + "="*60)
    print("EVALUATION CONFIGURATION")
    print("="*60)
    print(f"Model: {model_name if model_name else Path(predictions_dir).name}")
    print(f"Samples to evaluate: {len(common_indices)}")
    print(f"Encoder: {encoder_name}")
    print(f"Threshold: {threshold}")
    print(f"Number of GPUs: {num_gpus}")
    print("="*60 + "\n")

    # Split work across GPUs
    if torch.cuda.is_available() and num_gpus > 1:
        batch_size = len(common_indices) // num_gpus
        batches = [
            common_indices[i:i + batch_size]
            for i in range(0, len(common_indices), batch_size)
        ]

        # Ensure we don't have more batches than GPUs
        while len(batches) > num_gpus:
            batches[-2].extend(batches[-1])
            batches.pop()

        print(f"Splitting {len(common_indices)} samples across {num_gpus} GPUs:")
        for gpu_id, batch in enumerate(batches):
            print(f"  GPU {gpu_id}: {len(batch)} samples")
        print()

        # Prepare arguments for each GPU
        args_list = [
            (gpu_id, batches[gpu_id], predictions, ground_truth, encoder_name, threshold)
            for gpu_id in range(len(batches))
        ]

        # Run in parallel
        with Pool(processes=num_gpus) as pool:
            batch_results = pool.map(evaluate_batch_gpu, args_list)

        # Flatten results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)

        print(f"\n✓ All GPUs finished! Processed {len(all_results)} samples\n")
    else:
        # Single GPU/CPU processing
        print("Processing on single device...")
        args = (0, common_indices, predictions, ground_truth, encoder_name, threshold)
        all_results = evaluate_batch_gpu(args)
        print(f"\n✓ Processing finished! Processed {len(all_results)} samples\n")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {model_name if model_name else Path(predictions_dir).name}")
    print(f"Total samples evaluated: {len(df)}")
    print(f"Encoder: {encoder_name}")
    print(f"Threshold: {threshold}")

    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print(f"{'='*60}")
    accuracy_mean = df['accuracy_correct'].mean()
    print(f"  Overall Accuracy:     {accuracy_mean:.4f} ({accuracy_mean*100:.2f}%)")
    print(f"  Correct samples:      {df['accuracy_correct'].sum()}/{len(df)}")
    print(f"  Average Confidence:   {df['confidence'].mean():.4f}")

    print(f"\n{'='*60}")
    print("MATCH F1 METRICS")
    print(f"{'='*60}")
    print(f"  Match F1:             {df['match_f1'].mean():.4f} (±{df['match_f1'].std():.4f})")
    print(f"  Precision:            {df['precision'].mean():.4f} (±{df['precision'].std():.4f})")
    print(f"  Recall:               {df['recall'].mean():.4f} (±{df['recall'].std():.4f})")

    print(f"\n{'='*60}")
    print("REASONING STEPS STATISTICS")
    print(f"{'='*60}")
    print(f"  Avg Predicted Steps:  {df['num_predicted_steps'].mean():.2f} (±{df['num_predicted_steps'].std():.2f})")
    print(f"  Avg Reference Steps:  {df['num_reference_steps'].mean():.2f} (±{df['num_reference_steps'].std():.2f})")
    print(f"  Avg Similarity:       {df['avg_similarity'].mean():.4f}")

    # Save results
    model_dir_name = Path(predictions_dir).name
    save_dir = Path(output_dir) / model_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed CSV
    csv_file = save_dir / "metrics_detailed.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Detailed results saved to: {csv_file}")

    # Save summary JSON
    summary = {
        'model_name': model_name if model_name else model_dir_name,
        'encoder': encoder_name,
        'threshold': threshold,
        'total_samples': len(df),
        'accuracy': {
            'mean': float(df['accuracy_correct'].mean()),
            'correct_count': int(df['accuracy_correct'].sum()),
            'confidence_mean': float(df['confidence'].mean())
        },
        'match_f1': {
            'mean': float(df['match_f1'].mean()),
            'std': float(df['match_f1'].std()),
            'min': float(df['match_f1'].min()),
            'max': float(df['match_f1'].max())
        },
        'precision': {
            'mean': float(df['precision'].mean()),
            'std': float(df['precision'].std())
        },
        'recall': {
            'mean': float(df['recall'].mean()),
            'std': float(df['recall'].std())
        },
        'steps': {
            'predicted_mean': float(df['num_predicted_steps'].mean()),
            'predicted_std': float(df['num_predicted_steps'].std()),
            'reference_mean': float(df['num_reference_steps'].mean()),
            'reference_std': float(df['num_reference_steps'].std()),
            'avg_similarity': float(df['avg_similarity'].mean())
        }
    }

    summary_file = save_dir / "metrics_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_file}")

    # Save text summary
    summary_txt_file = save_dir / "metrics_summary.txt"
    with open(summary_txt_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"CRYSTAL BENCHMARK EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model_name if model_name else model_dir_name}\n")
        f.write(f"Encoder: {encoder_name}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Samples: {len(df)}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("ACCURACY METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"  Accuracy:     {summary['accuracy']['mean']:.4f}\n")
        f.write(f"  Correct:      {summary['accuracy']['correct_count']}/{len(df)}\n")
        f.write(f"  Confidence:   {summary['accuracy']['confidence_mean']:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("MATCH F1 METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"  Match F1:     {summary['match_f1']['mean']:.4f} (±{summary['match_f1']['std']:.4f})\n")
        f.write(f"  Precision:    {summary['precision']['mean']:.4f} (±{summary['precision']['std']:.4f})\n")
        f.write(f"  Recall:       {summary['recall']['mean']:.4f} (±{summary['recall']['std']:.4f})\n")
        f.write("\n" + "="*60 + "\n")
        f.write("REASONING STEPS\n")
        f.write("="*60 + "\n")
        f.write(f"  Predicted:    {summary['steps']['predicted_mean']:.2f} (±{summary['steps']['predicted_std']:.2f})\n")
        f.write(f"  Reference:    {summary['steps']['reference_mean']:.2f} (±{summary['steps']['reference_std']:.2f})\n")
        f.write(f"  Similarity:   {summary['steps']['avg_similarity']:.4f}\n")

    print(f"✓ Text summary saved to: {summary_txt_file}")
    print("="*60 + "\n")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute final metrics for CRYSTAL benchmark"
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
        "--encoder",
        type=str,
        default="all-distilroberta-v1",
        help="Sentence encoder to use (default: all-distilroberta-v1)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Similarity threshold (default: 0.35)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./final_table",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Display name for the model"
    )

    args = parser.parse_args()

    # Set multiprocessing start method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Run evaluation
    df = compute_metrics(
        predictions_dir=args.predictions_dir,
        dataset_path=args.dataset_path,
        encoder_name=args.encoder,
        threshold=args.threshold,
        num_gpus=args.num_gpus,
        output_dir=args.output_dir,
        model_name=args.model_name
    )

    print("\n✓ Evaluation complete!\n")


if __name__ == "__main__":
    main()
