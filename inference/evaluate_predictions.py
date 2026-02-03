#!/usr/bin/env python3
"""
Evaluation Script for Checkpoint Predictions

Evaluates predictions using two metrics:
1. Accuracy: Answer correctness (with optional LLM judge)
2. Match F1: Reasoning quality (using mllm_evaluator with sentence transformers)

Usage:
    python evaluate_predictions.py \
        --predictions_dir predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-500 \
        --test_dataset_path reasoning_test_with_reference_steps_updated_v27 \
        --use_llm_judge \
        --llm_judge_model gpt-oss:20b
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_from_disk
from tqdm import tqdm

# Add mllm_evaluator to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLLM_EVALUATOR_DIR = PROJECT_ROOT / "mllm_evaluator"
sys.path.insert(0, str(MLLM_EVALUATOR_DIR))

# Import evaluation modules
from accuracy_calculator import AccuracyCalculator
from mllm_evaluator import MLLMReasoningEvaluator


def load_predictions(predictions_dir: str) -> Dict[int, Dict[str, Any]]:
    """Load all prediction files from directory."""
    predictions = {}
    predictions_path = Path(predictions_dir)

    for pred_file in sorted(predictions_path.glob("*.json")):
        if pred_file.name in ["inference_summary.json", "evaluation_results.json"]:
            continue

        try:
            idx = int(pred_file.stem)
            with open(pred_file, "r") as f:
                predictions[idx] = json.load(f)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load {pred_file}: {e}")
            continue

    return predictions


def evaluate_predictions(
    predictions_dir: str,
    test_dataset_path: str,
    use_llm_judge: bool = False,
    llm_judge_model: str = "gpt-oss:20b",
    llm_judge_base_url: str = "http://localhost:11434/v1",
    reasoning_model: str = "all-MiniLM-L6-v2",
    reasoning_threshold: float = None,
    reasoning_device: str = "auto",
    output_file: str = None,
) -> Dict[str, Any]:
    """Evaluate predictions using accuracy and match F1."""

    print(f"Loading predictions from: {predictions_dir}")
    predictions = load_predictions(predictions_dir)
    print(f"Loaded {len(predictions)} predictions")

    print(f"\nLoading test dataset from: {test_dataset_path}")
    test_dataset = load_from_disk(test_dataset_path)
    print(f"Loaded {len(test_dataset)} test samples")

    # Initialize accuracy calculator
    print(f"\nInitializing AccuracyCalculator:")
    print(f"  - LLM judge: {use_llm_judge}")
    if use_llm_judge:
        print(f"  - Model: {llm_judge_model}")
        print(f"  - Base URL: {llm_judge_base_url}")

    accuracy_calculator = AccuracyCalculator(
        use_llm_grader=use_llm_judge,
        llm_model=llm_judge_model,
        base_url=llm_judge_base_url,
    )

    # Initialize reasoning evaluator (mllm_evaluator)
    print(f"\nInitializing MLLMReasoningEvaluator (Match F1):")
    print(f"  - Model: {reasoning_model}")
    print(f"  - Device: {reasoning_device}")

    reasoning_evaluator = MLLMReasoningEvaluator(
        model_name=reasoning_model,
        similarity_threshold=reasoning_threshold if reasoning_threshold is not None else 0.45,
        device=reasoning_device,
        debug_mode=False,
    )
    print(f"  - Similarity threshold: {reasoning_evaluator.similarity_threshold:.3f}")

    # Evaluation results
    results = {
        "total_samples": len(test_dataset),
        "evaluated_samples": 0,
        "missing_predictions": 0,
        # Accuracy metrics
        "accuracy_correct": 0,
        "accuracy_total": 0,
        "accuracy_rate": 0.0,
        # Match F1 metrics
        "match_f1_sum": 0.0,
        "match_f1_avg": 0.0,
        "precision_sum": 0.0,
        "precision_avg": 0.0,
        "recall_sum": 0.0,
        "recall_avg": 0.0,
        # Per-source breakdown
        "by_source": {},
        # Detailed per-sample results
        "detailed_results": [],
    }

    print("\n" + "="*80)
    print("Evaluating predictions...")
    print("="*80)

    for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
        sample = test_dataset[idx]
        question = sample["question"]
        ground_truth = sample["answer"]
        reference_steps = sample.get("reference_steps", [])
        source = sample.get("source", "unknown")

        # Check if prediction exists
        if idx not in predictions:
            results["missing_predictions"] += 1
            continue

        prediction = predictions[idx]
        predicted_answer = prediction.get("answer", "")
        predicted_steps = prediction.get("reasoning_steps", [])

        # Evaluate accuracy
        accuracy_result = accuracy_calculator.evaluate_single(
            question, predicted_answer, ground_truth
        )

        # Evaluate reasoning (Match F1)
        if predicted_steps and reference_steps:
            reasoning_metrics = reasoning_evaluator.evaluate_single(
                predicted_steps, reference_steps, verbose=False
            )
            match_f1 = reasoning_metrics.match_f1
            precision = reasoning_metrics.precision
            recall = reasoning_metrics.recall
        else:
            match_f1 = 0.0
            precision = 0.0
            recall = 0.0

        # Store sample result
        sample_result = {
            "idx": idx,
            "source": source,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "accuracy_correct": accuracy_result.is_correct,
            "accuracy_confidence": accuracy_result.confidence,
            "match_f1": match_f1,
            "precision": precision,
            "recall": recall,
            "num_predicted_steps": len(predicted_steps),
            "num_reference_steps": len(reference_steps),
        }

        results["detailed_results"].append(sample_result)
        results["evaluated_samples"] += 1

        # Update aggregates
        results["accuracy_total"] += 1
        if accuracy_result.is_correct:
            results["accuracy_correct"] += 1

        results["match_f1_sum"] += match_f1
        results["precision_sum"] += precision
        results["recall_sum"] += recall

        # Per-source stats
        if source not in results["by_source"]:
            results["by_source"][source] = {
                "total": 0,
                "accuracy_correct": 0,
                "match_f1_sum": 0.0,
                "precision_sum": 0.0,
                "recall_sum": 0.0,
            }

        source_data = results["by_source"][source]
        source_data["total"] += 1
        if accuracy_result.is_correct:
            source_data["accuracy_correct"] += 1
        source_data["match_f1_sum"] += match_f1
        source_data["precision_sum"] += precision
        source_data["recall_sum"] += recall

    # Calculate averages
    if results["accuracy_total"] > 0:
        results["accuracy_rate"] = results["accuracy_correct"] / results["accuracy_total"]

    if results["evaluated_samples"] > 0:
        n = results["evaluated_samples"]
        results["match_f1_avg"] = results["match_f1_sum"] / n
        results["precision_avg"] = results["precision_sum"] / n
        results["recall_avg"] = results["recall_sum"] / n

        # Per-source averages
        for source, data in results["by_source"].items():
            if data["total"] > 0:
                data["accuracy_rate"] = data["accuracy_correct"] / data["total"]
                data["match_f1_avg"] = data["match_f1_sum"] / data["total"]
                data["precision_avg"] = data["precision_sum"] / data["total"]
                data["recall_avg"] = data["recall_sum"] / data["total"]

    # Save results
    if output_file:
        print(f"\nSaving results to: {output_file}")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def print_summary(results: Dict[str, Any]):
    """Print evaluation summary."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total samples: {results['total_samples']}")
    print(f"Evaluated samples: {results['evaluated_samples']}")
    print(f"Missing predictions: {results['missing_predictions']}")
    print()

    # Accuracy
    print("ACCURACY:")
    acc_rate = results['accuracy_rate'] * 100
    print(f"  Correct: {results['accuracy_correct']}/{results['accuracy_total']} ({acc_rate:.2f}%)")
    print()

    # Match F1
    print("MATCH F1 (Reasoning Quality):")
    print(f"  F1 Score:  {results['match_f1_avg']:.4f}")
    print(f"  Precision: {results['precision_avg']:.4f}")
    print(f"  Recall:    {results['recall_avg']:.4f}")
    print()

    # Per-source breakdown
    if results["by_source"]:
        print("PER-SOURCE BREAKDOWN:")
        print(f"{'Source':<25} {'Total':<8} {'Accuracy':<12} {'Match F1':<12} {'Precision':<12} {'Recall':<10}")
        print("-" * 90)
        for source, data in sorted(results["by_source"].items(), key=lambda x: x[1]["total"], reverse=True):
            acc_pct = data["accuracy_rate"] * 100
            print(f"{source:<25} {data['total']:<8} "
                  f"{acc_pct:>6.2f}%      "
                  f"{data['match_f1_avg']:>7.4f}      "
                  f"{data['precision_avg']:>7.4f}      "
                  f"{data['recall_avg']:>7.4f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint predictions (Accuracy + Match F1)"
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Path to predictions directory",
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=True,
        help="Path to test dataset (HuggingFace format)",
    )
    parser.add_argument(
        "--use_llm_judge",
        action="store_true",
        help="Use LLM judge for accuracy evaluation",
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default="gpt-oss:20b",
        help="LLM judge model name (default: gpt-oss:20b)",
    )
    parser.add_argument(
        "--llm_judge_base_url",
        type=str,
        default="http://localhost:11434/v1",
        help="LLM judge base URL (default: http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for Match F1 (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--reasoning_threshold",
        type=float,
        default=None,
        help="Similarity threshold (default: 0.45 for stricter matching)",
    )
    parser.add_argument(
        "--reasoning_device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for detailed results (JSON)",
    )

    args = parser.parse_args()

    # Set default output file
    if args.output_file is None:
        predictions_path = Path(args.predictions_dir)
        args.output_file = str(predictions_path / "evaluation_results.json")

    # Run evaluation
    results = evaluate_predictions(
        predictions_dir=args.predictions_dir,
        test_dataset_path=args.test_dataset_path,
        use_llm_judge=args.use_llm_judge,
        llm_judge_model=args.llm_judge_model,
        llm_judge_base_url=args.llm_judge_base_url,
        reasoning_model=args.reasoning_model,
        reasoning_threshold=args.reasoning_threshold,
        reasoning_device=args.reasoning_device,
        output_file=args.output_file,
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
