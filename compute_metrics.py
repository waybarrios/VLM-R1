#!/usr/bin/env python3
"""
Fast computation of accuracy and matchf1 metrics for prediction folders
Supports multi-GPU processing for speed
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass
import pandas as pd
import torch
from torch.multiprocessing import Pool, Process, set_start_method
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
    gpu_id, batch_indices, predictions, ground_truth, use_judge, judge_model = args

    # Set GPU device
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # Initialize evaluators
    match_f1_evaluator = MLLMReasoningEvaluator(
        model_name="all-MiniLM-L6-v2",
        device=device,
        debug_mode=False
    )

    accuracy_calculator = AccuracyCalculator(
        use_llm_grader=use_judge,
        llm_model=judge_model if use_judge else "llama3.2",
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
                    'threshold_used': 0.0
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
    use_judge: bool = False,
    judge_model: str = "gpt-oss:120b",
    num_gpus: int = 4,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute accuracy and matchf1 metrics for predictions

    Args:
        predictions_dir: Directory containing prediction JSON files
        dataset_path: Path to dataset (Arrow format)
        use_judge: Whether to use LLM judge for accuracy
        judge_model: Model to use for judging (default: gpt-oss:120b)
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
    print(f"use_judge: {use_judge} ({'USE LLM - slower' if use_judge else 'NO LLM - faster'})")
    if use_judge:
        print(f"Judge model: {judge_model}")
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
            (gpu_id, batches[gpu_id], predictions, ground_truth, use_judge, judge_model)
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
        args = (0, common_indices, predictions, ground_truth, use_judge, judge_model)
        all_results = evaluate_batch_gpu(args)
        print(f"\n✓ Processing finished! Processed {len(all_results)} samples\n")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples evaluated: {len(df)}")
    print(f"use_judge: {use_judge} ({'USE LLM' if use_judge else 'NO LLM - rule-based'})")
    if use_judge:
        print(f"Judge model: {judge_model}")

    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy:     {df['accuracy_correct'].mean():.4f}")
    print(f"  Correct samples:      {df['accuracy_correct'].sum()}/{len(df)}")
    print(f"  Average Confidence:   {df['confidence'].mean():.4f}")

    print(f"\n{'='*60}")
    print("MATCH F1 METRICS (from mllm_evaluator.py)")
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

        # Create filenames (shorter, without predictions_name prefix)
        judge_suffix = "with_judge" if use_judge else "no_judge"

        # Save detailed CSV
        output_file = predictions_output_dir / f"{judge_suffix}_metrics.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

        # Save summary JSON
        summary_file = predictions_output_dir / f"{judge_suffix}_summary.json"
        summary = {
            "experiment_info": {
                "predictions_dir": predictions_dir,
                "dataset_path": dataset_path,
                "use_judge": use_judge,
                "judge_model": judge_model if use_judge else None,
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

        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary JSON saved to: {summary_file}")

        # Save summary TXT (human-readable)
        summary_txt_file = predictions_output_dir / f"{judge_suffix}_summary.txt"
        with open(summary_txt_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Predictions dir: {predictions_dir}\n")
            f.write(f"Dataset path: {dataset_path}\n")
            f.write(f"Total samples evaluated: {len(df)}\n")
            f.write(f"use_judge: {use_judge} ({'USE LLM' if use_judge else 'NO LLM - rule-based'})\n")
            if use_judge:
                f.write(f"Judge model: {judge_model}\n")

            f.write(f"\n{'='*60}\n")
            f.write("ACCURACY METRICS\n")
            f.write(f"{'='*60}\n")
            f.write(f"  Overall Accuracy:     {df['accuracy_correct'].mean():.4f}\n")
            f.write(f"  Correct samples:      {df['accuracy_correct'].sum()}/{len(df)}\n")
            f.write(f"  Average Confidence:   {df['confidence'].mean():.4f}\n")
            f.write(f"  Median Confidence:    {df['confidence'].median():.4f}\n")

            f.write(f"\n{'='*60}\n")
            f.write("MATCH F1 METRICS (from mllm_evaluator.py)\n")
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
        description="Compute accuracy and matchf1 metrics for MLLM predictions"
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
        "--use-judge",
        action="store_true",
        default=False,
        help="If True, use LLM judge (gpt-oss:120b) for accuracy evaluation. If False, use rule-based matching (no LLM). Default: False (no LLM)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-oss:120b",
        help="Model to use for judging (default: gpt-oss:120b)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs to use for parallel processing (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./metrics_results",
        help="Directory to save results (default: ./metrics_results)"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run both tests: first without judge (no LLM), then with judge (with LLM)"
    )

    args = parser.parse_args()

    # Set multiprocessing start method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if args.both:
        # Run without judge (no LLM)
        print("\n" + "="*60)
        print("TEST 1: WITHOUT JUDGE (use_judge=False, NO LLM)")
        print("="*60 + "\n")
        df_no_judge = compute_metrics(
            predictions_dir=args.predictions_dir,
            dataset_path=args.dataset_path,
            use_judge=False,  # False = NO LLM, rule-based accuracy
            judge_model=args.judge_model,
            num_gpus=args.num_gpus,
            output_dir=args.output_dir
        )

        # Run with judge (with LLM)
        print("\n\n" + "="*60)
        print("TEST 2: WITH JUDGE (use_judge=True, USE LLM)")
        print("="*60 + "\n")
        df_with_judge = compute_metrics(
            predictions_dir=args.predictions_dir,
            dataset_path=args.dataset_path,
            use_judge=True,  # True = USE LLM for semantic matching
            judge_model=args.judge_model,
            num_gpus=args.num_gpus,
            output_dir=args.output_dir
        )

        # Compare results
        print("\n\n" + "="*60)
        print("COMPARISON: WITH JUDGE (LLM) vs WITHOUT JUDGE (NO LLM)")
        print("="*60)
        print(f"Accuracy WITH judge (use_judge=True, LLM):      {df_with_judge['accuracy_correct'].mean():.4f}")
        print(f"Accuracy WITHOUT judge (use_judge=False, NO LLM): {df_no_judge['accuracy_correct'].mean():.4f}")
        print(f"Difference: {(df_with_judge['accuracy_correct'].mean() - df_no_judge['accuracy_correct'].mean()):.4f}")
        print(f"\nMatch F1 (same for both): {df_no_judge['match_f1'].mean():.4f}")
        print("="*60)

        # Save comparison summary
        if args.output_dir:
            output_dir = Path(args.output_dir)
            predictions_name = Path(args.predictions_dir).name

            # Create subdirectory for this predictions folder
            predictions_output_dir = output_dir / predictions_name
            predictions_output_dir.mkdir(parents=True, exist_ok=True)

            comparison_file = predictions_output_dir / "comparison.txt"

            with open(comparison_file, 'w') as f:
                f.write("="*60 + "\n")
                f.write("COMPARISON: WITH JUDGE (LLM) vs WITHOUT JUDGE (NO LLM)\n")
                f.write("="*60 + "\n")
                f.write(f"Predictions dir: {args.predictions_dir}\n")
                f.write(f"Dataset path: {args.dataset_path}\n")
                f.write(f"Total samples: {len(df_no_judge)}\n")

                f.write(f"\n{'='*60}\n")
                f.write("ACCURACY COMPARISON\n")
                f.write(f"{'='*60}\n")
                f.write(f"WITHOUT judge (NO LLM):\n")
                f.write(f"  Accuracy: {df_no_judge['accuracy_correct'].mean():.4f}\n")
                f.write(f"  Confidence: {df_no_judge['confidence'].mean():.4f}\n")
                f.write(f"\nWITH judge (USE LLM - {args.judge_model}):\n")
                f.write(f"  Accuracy: {df_with_judge['accuracy_correct'].mean():.4f}\n")
                f.write(f"  Confidence: {df_with_judge['confidence'].mean():.4f}\n")
                f.write(f"\nDifference:\n")
                f.write(f"  Accuracy difference: {(df_with_judge['accuracy_correct'].mean() - df_no_judge['accuracy_correct'].mean()):.4f}\n")
                f.write(f"  Confidence difference: {(df_with_judge['confidence'].mean() - df_no_judge['confidence'].mean()):.4f}\n")

                f.write(f"\n{'='*60}\n")
                f.write("MATCH F1 METRICS (same for both)\n")
                f.write(f"{'='*60}\n")
                f.write(f"  Match F1: {df_no_judge['match_f1'].mean():.4f}\n")
                f.write(f"  Precision: {df_no_judge['precision'].mean():.4f}\n")
                f.write(f"  Recall: {df_no_judge['recall'].mean():.4f}\n")

                f.write("="*60 + "\n")

            print(f"\nComparison summary saved to: {comparison_file}")
    else:
        df = compute_metrics(
            predictions_dir=args.predictions_dir,
            dataset_path=args.dataset_path,
            use_judge=args.use_judge,
            judge_model=args.judge_model,
            num_gpus=args.num_gpus,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
