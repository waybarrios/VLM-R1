#!/usr/bin/env python3
"""
GRPO Analysis - Compute metrics for GRPO trained model checkpoints
Uses custom MatchF1 settings: threshold=0.35, encoder=all-distilroberta-v1
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from dataclasses import dataclass
import pandas as pd
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
    answer: str
    match_type: str
    confidence: float


def load_dataset_hf(dataset_path: str) -> Dict[int, Dict]:
    """Load dataset using HuggingFace load_from_disk with error handling"""
    print(f"Loading dataset from {dataset_path}...")

    try:
        # Try standard load first
        dataset = load_from_disk(dataset_path)
    except (TypeError, ValueError) as e:
        # If there's a schema error, load without validation
        print(f"Warning: Schema validation error, loading with alternative method...")
        import pyarrow as pa
        import pyarrow.dataset as ds
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

        # Convert to dict format
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
    print(f"Dataset columns: {dataset.column_names}")

    # Convert to dict format indexed by sample index
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


def load_predictions(predictions_dir: str, dataset_indices: set) -> Dict[int, Dict]:
    """
    Load predictions from JSON files
    If a file doesn't exist for a dataset index, create a placeholder
    """
    predictions_dir = Path(predictions_dir)
    predictions = {}

    # First load existing predictions
    json_files = list(predictions_dir.glob("*.json"))
    existing_indices = set()

    for json_file in tqdm(json_files, desc="Loading predictions"):
        try:
            # Filename is the index
            idx = int(json_file.stem)
            existing_indices.add(idx)

            with open(json_file, 'r') as f:
                data = json.load(f)
                predictions[idx] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    # Create placeholders for missing predictions
    missing_indices = dataset_indices - existing_indices
    if missing_indices:
        print(f"Creating placeholders for {len(missing_indices)} missing predictions...")
        for idx in missing_indices:
            predictions[idx] = {
                'reasoning_steps': [],
                'answer': 'insufficient information'
            }

    print(f"Loaded {len(predictions)} predictions ({len(existing_indices)} real, {len(missing_indices)} placeholders)")
    return predictions


def evaluate_batch_gpu(args):
    """Evaluate a batch of predictions on a specific GPU"""
    gpu_id, batch_indices, predictions, ground_truth, model_name, threshold = args

    # Set GPU device
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Initialize evaluators with custom settings
    match_f1_evaluator = MLLMReasoningEvaluator(
        model_name=model_name,  # all-distilroberta-v1
        similarity_threshold=threshold,  # 0.35
        device=device,
        debug_mode=False
    )

    accuracy_calculator = AccuracyCalculator(
        use_llm_grader=False,  # No judge, rule-based only
        llm_model="llama3.2",
        base_url="http://localhost:11434/v1"
    )

    results = []

    # Track running metrics for progress display
    running_metrics = {
        'accuracy': 0.0,
        'match_f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'count': 0
    }

    # Create progress bar with detailed description
    pbar = tqdm(
        batch_indices,
        desc=f"GPU {gpu_id} | Starting...",
        position=gpu_id,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
    )

    for idx in pbar:
        if idx not in ground_truth:
            continue

        try:
            # Get prediction and ground truth
            pred = predictions[idx]
            gt = ground_truth[idx]

            # Extract data
            pred_steps = pred.get("reasoning_steps", [])
            pred_answer = pred.get("answer", "")

            # Ground truth
            ref_steps = gt.get("reference_steps", [])
            gt_answer = gt.get("answer", "")
            question = gt.get("question", "")

            # Check if this is a placeholder (no predictions)
            is_placeholder = len(pred_steps) == 0 and pred_answer == "insufficient information"

            if is_placeholder:
                # For placeholders, matchf1 and accuracy are 0
                pbar.set_description(f"GPU {gpu_id} | Sample {idx} (placeholder)")
                result = {
                    'sample_idx': idx,
                    # Accuracy metrics
                    'accuracy_correct': False,
                    'match_type': 'placeholder',
                    'confidence': 0.0,
                    'predicted_answer': pred_answer,
                    'ground_truth_answer': gt_answer,
                    # Match F1 metrics
                    'match_f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'num_predicted_steps': 0,
                    'num_reference_steps': len(ref_steps),
                    'num_matched_predictions': 0,
                    'num_matched_references': 0,
                    'avg_similarity': 0.0,
                    'max_similarity': 0.0,
                    'threshold_used': threshold
                }
            else:
                # Update progress: Computing Match F1
                pbar.set_description(f"GPU {gpu_id} | Sample {idx} → Match F1")
                metrics_f1 = match_f1_evaluator.evaluate_single(pred_steps, ref_steps, verbose=False)

                # Update progress: Computing Accuracy
                pbar.set_description(f"GPU {gpu_id} | Sample {idx} → Accuracy")
                accuracy_result = accuracy_calculator.evaluate_single(
                    question=question,
                    predicted_answer=pred_answer,
                    ground_truth_answer=gt_answer
                )

                # Combine results with all metrics
                result = {
                    'sample_idx': idx,
                    # Accuracy metrics
                    'accuracy_correct': accuracy_result.is_correct,
                    'match_type': accuracy_result.match_type,
                    'confidence': accuracy_result.confidence,
                    'predicted_answer': pred_answer,
                    'ground_truth_answer': gt_answer,
                    # Match F1 metrics
                    'match_f1': metrics_f1.match_f1,
                    'precision': metrics_f1.precision,
                    'recall': metrics_f1.recall,
                    'num_predicted_steps': metrics_f1.num_predicted_steps,
                    'num_reference_steps': metrics_f1.num_reference_steps,
                    'num_matched_predictions': metrics_f1.num_matched_predictions,
                    'num_matched_references': metrics_f1.num_matched_references,
                    'avg_similarity': metrics_f1.avg_similarity,
                    'max_similarity': metrics_f1.max_similarity,
                    'threshold_used': metrics_f1.threshold_used
                }

            results.append(result)

            # Update running metrics
            running_metrics['accuracy'] += 1 if result['accuracy_correct'] else 0
            running_metrics['match_f1'] += result['match_f1']
            running_metrics['precision'] += result['precision']
            running_metrics['recall'] += result['recall']
            running_metrics['count'] += 1

            # Calculate averages
            if running_metrics['count'] > 0:
                avg_acc = running_metrics['accuracy'] / running_metrics['count']
                avg_f1 = running_metrics['match_f1'] / running_metrics['count']
                avg_prec = running_metrics['precision'] / running_metrics['count']
                avg_rec = running_metrics['recall'] / running_metrics['count']

                # Update progress bar with metrics
                pbar.set_postfix({
                    'Acc': f'{avg_acc:.3f}',
                    'F1': f'{avg_f1:.3f}',
                    'P': f'{avg_prec:.3f}',
                    'R': f'{avg_rec:.3f}'
                })

        except Exception as e:
            pbar.set_description(f"GPU {gpu_id} | Sample {idx} ✗ ERROR")
            print(f"\nError evaluating sample {idx}: {e}")
            continue

    pbar.close()
    return results


def compute_metrics(
    predictions_dir: str,
    dataset_path: str,
    model_name: str = "all-distilroberta-v1",
    threshold: float = 0.35,
    num_gpus: int = 2,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute accuracy and matchf1 metrics for predictions with custom settings

    Args:
        predictions_dir: Directory containing prediction JSON files
        dataset_path: Path to dataset (Arrow format)
        model_name: Sentence transformer model (default: all-distilroberta-v1)
        threshold: Similarity threshold (default: 0.35)
        num_gpus: Number of GPUs to use for parallel processing
        output_dir: Directory to save results (optional)

    Returns:
        DataFrame with evaluation results
    """
    # Load data
    print("Loading dataset...")
    ground_truth = load_dataset_hf(dataset_path)

    print("\nLoading predictions...")
    predictions = load_predictions(predictions_dir, set(ground_truth.keys()))

    # Find common indices
    pred_indices = set(predictions.keys())
    gt_indices = set(ground_truth.keys())
    common_indices = sorted(pred_indices.intersection(gt_indices))

    print(f"\nFound {len(common_indices)} common samples to evaluate")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}")
    print(f"Number of GPUs: {num_gpus}")
    print("\nStarting evaluation...\n")

    # Split work across GPUs
    if torch.cuda.is_available() and num_gpus > 1:
        batch_size = len(common_indices) // num_gpus
        batches = [
            common_indices[i:i + batch_size]
            for i in range(0, len(common_indices), batch_size)
        ]

        # Ensure we don't have more batches than GPUs
        while len(batches) > num_gpus:
            # Merge last batch into second-to-last
            batches[-2].extend(batches[-1])
            batches.pop()

        print(f"Splitting {len(common_indices)} samples across {num_gpus} GPUs:")
        for gpu_id, batch in enumerate(batches):
            print(f"  GPU {gpu_id}: {len(batch)} samples")
        print()

        # Prepare arguments for each GPU
        args_list = [
            (gpu_id, batches[gpu_id], predictions, ground_truth, model_name, threshold)
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
        args = (0, common_indices, predictions, ground_truth, model_name, threshold)
        all_results = evaluate_batch_gpu(args)
        print(f"\n✓ Processing finished! Processed {len(all_results)} samples\n")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples evaluated: {len(df)}")
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}")

    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy:     {df['accuracy_correct'].mean():.4f}")
    print(f"  Correct samples:      {df['accuracy_correct'].sum()}/{len(df)}")
    print(f"  Average Confidence:   {df['confidence'].mean():.4f}")

    print(f"\n{'='*60}")
    print("MATCH F1 METRICS")
    print(f"{'='*60}")
    print(f"  Average Match F1:     {df['match_f1'].mean():.4f} (±{df['match_f1'].std():.4f})")
    print(f"  Average Precision:    {df['precision'].mean():.4f} (±{df['precision'].std():.4f})")
    print(f"  Average Recall:       {df['recall'].mean():.4f} (±{df['recall'].std():.4f})")
    print(f"  Median Match F1:      {df['match_f1'].median():.4f}")

    print(f"\n{'='*60}")
    print("STEP MATCHING DETAILS")
    print(f"{'='*60}")
    print(f"  Avg predicted steps:  {df['num_predicted_steps'].mean():.2f}")
    print(f"  Avg reference steps:  {df['num_reference_steps'].mean():.2f}")
    print(f"  Avg matched (pred):   {df['num_matched_predictions'].mean():.2f}")
    print(f"  Avg matched (ref):    {df['num_matched_references'].mean():.2f}")
    print(f"  Avg similarity:       {df['avg_similarity'].mean():.4f}")
    print(f"  Avg max similarity:   {df['max_similarity'].mean():.4f}")

    # Match type breakdown
    print(f"\n{'='*60}")
    print("MATCH TYPE BREAKDOWN")
    print(f"{'='*60}")
    match_type_counts = df['match_type'].value_counts()
    for match_type, count in match_type_counts.items():
        print(f"  {match_type}: {count} ({count/len(df)*100:.1f}%)")

    print("="*60)

    # Save results if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        predictions_name = Path(predictions_dir).name

        # Create subdirectory for this predictions folder
        predictions_output_dir = output_dir / predictions_name
        predictions_output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed CSV
        output_file = predictions_output_dir / "metrics.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

        # Save summary JSON
        summary_file = predictions_output_dir / "summary.json"
        summary = {
            "experiment_info": {
                "predictions_dir": predictions_dir,
                "dataset_path": dataset_path,
                "model_name": model_name,
                "threshold": threshold,
                "num_gpus": num_gpus,
                "total_samples": len(df)
            },
            "accuracy_metrics": {
                "overall_accuracy": float(df['accuracy_correct'].mean()),
                "correct_samples": int(df['accuracy_correct'].sum()),
                "total_samples": len(df),
                "average_confidence": float(df['confidence'].mean()),
                "median_confidence": float(df['confidence'].median())
            },
            "match_f1_metrics": {
                "average_match_f1": float(df['match_f1'].mean()),
                "std_match_f1": float(df['match_f1'].std()),
                "median_match_f1": float(df['match_f1'].median()),
                "average_precision": float(df['precision'].mean()),
                "std_precision": float(df['precision'].std()),
                "average_recall": float(df['recall'].mean()),
                "std_recall": float(df['recall'].std())
            },
            "step_matching_details": {
                "avg_predicted_steps": float(df['num_predicted_steps'].mean()),
                "avg_reference_steps": float(df['num_reference_steps'].mean()),
                "avg_matched_predictions": float(df['num_matched_predictions'].mean()),
                "avg_matched_references": float(df['num_matched_references'].mean()),
                "avg_similarity": float(df['avg_similarity'].mean()),
                "avg_max_similarity": float(df['max_similarity'].mean())
            },
            "match_type_breakdown": df['match_type'].value_counts().to_dict()
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary JSON saved to: {summary_file}")

        # Save summary TXT (human-readable)
        summary_txt_file = predictions_output_dir / "summary.txt"
        with open(summary_txt_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Predictions dir: {predictions_dir}\n")
            f.write(f"Dataset path: {dataset_path}\n")
            f.write(f"Total samples evaluated: {len(df)}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Threshold: {threshold}\n")

            f.write(f"\n{'='*60}\n")
            f.write("ACCURACY METRICS\n")
            f.write(f"{'='*60}\n")
            f.write(f"  Overall Accuracy:     {df['accuracy_correct'].mean():.4f}\n")
            f.write(f"  Correct samples:      {df['accuracy_correct'].sum()}/{len(df)}\n")
            f.write(f"  Average Confidence:   {df['confidence'].mean():.4f}\n")
            f.write(f"  Median Confidence:    {df['confidence'].median():.4f}\n")

            f.write(f"\n{'='*60}\n")
            f.write("MATCH F1 METRICS\n")
            f.write(f"{'='*60}\n")
            f.write(f"  Average Match F1:     {df['match_f1'].mean():.4f} (±{df['match_f1'].std():.4f})\n")
            f.write(f"  Average Precision:    {df['precision'].mean():.4f} (±{df['precision'].std():.4f})\n")
            f.write(f"  Average Recall:       {df['recall'].mean():.4f} (±{df['recall'].std():.4f})\n")
            f.write(f"  Median Match F1:      {df['match_f1'].median():.4f}\n")

            f.write(f"\n{'='*60}\n")
            f.write("STEP MATCHING DETAILS\n")
            f.write(f"{'='*60}\n")
            f.write(f"  Avg predicted steps:  {df['num_predicted_steps'].mean():.2f}\n")
            f.write(f"  Avg reference steps:  {df['num_reference_steps'].mean():.2f}\n")
            f.write(f"  Avg matched (pred):   {df['num_matched_predictions'].mean():.2f}\n")
            f.write(f"  Avg matched (ref):    {df['num_matched_references'].mean():.2f}\n")
            f.write(f"  Avg similarity:       {df['avg_similarity'].mean():.4f}\n")
            f.write(f"  Avg max similarity:   {df['max_similarity'].mean():.4f}\n")

            f.write(f"\n{'='*60}\n")
            f.write("MATCH TYPE BREAKDOWN\n")
            f.write(f"{'='*60}\n")
            match_type_counts = df['match_type'].value_counts()
            for match_type, count in match_type_counts.items():
                f.write(f"  {match_type}: {count} ({count/len(df)*100:.1f}%)\n")

            f.write("="*60 + "\n")

        print(f"Summary TXT saved to: {summary_txt_file}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute GRPO metrics with custom settings"
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
        "--model-name",
        type=str,
        default="all-distilroberta-v1",
        help="Sentence transformer model (default: all-distilroberta-v1)"
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
        default=2,
        help="Number of GPUs to use for parallel processing (default: 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./GRPO_analysis/results",
        help="Directory to save results (default: ./GRPO_analysis/results)"
    )

    args = parser.parse_args()

    # Set multiprocessing start method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    df = compute_metrics(
        predictions_dir=args.predictions_dir,
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        threshold=args.threshold,
        num_gpus=args.num_gpus,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
