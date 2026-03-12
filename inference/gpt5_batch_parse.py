#!/usr/bin/env python3
"""
GPT-5 Batch API — Step 3: Parse results & evaluate

Reads batch results JSONL, extracts GPT-5 responses, parses JSON using
the same parser as run_simple_vqa.py, saves individual prediction JSONs,
and runs the standard CRYSTAL evaluation pipeline (AccuracyCalculator +
MLLMReasoningEvaluator with all-distilroberta-v1 encoder, threshold 0.35).

Usage:
    python inference/gpt5_batch_parse.py
    python inference/gpt5_batch_parse.py --input inference/gpt5_batch_results.jsonl --output_dir final_table/outputs_testing_gpt5
    python inference/gpt5_batch_parse.py --input inference/gpt5_batch_results.jsonl inference/gpt5_batch_results_part1.jsonl
    python inference/gpt5_batch_parse.py --parse_only   # Skip evaluation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import pyarrow as pa

# Import parse_and_validate_json from run_simple_vqa.py
INFERENCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INFERENCE_DIR))
from run_simple_vqa import parse_and_validate_json

# Import evaluation modules
PROJECT_ROOT = INFERENCE_DIR.parent
MLLM_EVALUATOR_DIR = PROJECT_ROOT / "mllm_evaluator"
sys.path.insert(0, str(MLLM_EVALUATOR_DIR))


def parse_batch_results(input_path: str, predictions_dir: str) -> dict:
    """
    Parse OpenAI Batch API results JSONL into individual prediction JSONs.

    Returns:
        dict with stats: {total, valid, invalid, errors}
    """
    predictions_path = Path(predictions_dir)
    predictions_path.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "valid": 0, "invalid": 0, "api_errors": 0}

    print(f"Parsing batch results from: {input_path}")
    print(f"Saving predictions to: {predictions_dir}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} result lines\n")

    for line in tqdm(lines, desc="Parsing responses"):
        stats["total"] += 1

        result = json.loads(line.strip())
        custom_id = result.get("custom_id", "")

        # Extract sample index from custom_id (format: "sample-{idx}")
        try:
            idx = int(custom_id.split("-", 1)[1])
        except (IndexError, ValueError):
            print(f"Warning: Could not parse custom_id: {custom_id}")
            stats["api_errors"] += 1
            continue

        # Check for API-level errors
        if result.get("error"):
            error = result["error"]
            print(f"Warning: API error for sample {idx}: {error.get('message', error)}")
            stats["api_errors"] += 1
            # Save a fallback prediction
            prediction = {
                "reasoning_steps": [],
                "answer": "insufficient information",
            }
            pred_file = predictions_path / f"{idx}.json"
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(prediction, f, indent=2, ensure_ascii=False)
            continue

        # Extract the response content
        try:
            response_body = result["response"]["body"]
            content = response_body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Malformed response for sample {idx}: {e}")
            stats["api_errors"] += 1
            prediction = {
                "reasoning_steps": [],
                "answer": "insufficient information",
            }
            pred_file = predictions_path / f"{idx}.json"
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(prediction, f, indent=2, ensure_ascii=False)
            continue

        # Parse JSON from GPT-5 output (same parser as local inference)
        prediction, is_valid, validation_error = parse_and_validate_json(content)

        if not is_valid:
            stats["invalid"] += 1
            if stats["invalid"] <= 10:
                snippet = content[:200] + "..." if len(content) > 200 else content
                print(f"Warning: Invalid JSON for sample {idx}: {validation_error}")
                print(f"  Content: {snippet}")
            prediction = {
                "reasoning_steps": [],
                "answer": "insufficient information",
            }
        else:
            stats["valid"] += 1

        # Save individual prediction JSON
        pred_file = predictions_path / f"{idx}.json"
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(prediction, f, indent=2, ensure_ascii=False)

    return stats


def load_dataset_arrow(dataset_path: str) -> pa.Table:
    """Load dataset from Arrow files (avoids datasets library schema issues)."""
    arrow_files = sorted(glob(f"{dataset_path}/data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow files found in {dataset_path}")

    tables = []
    for arrow_file in arrow_files:
        stream = pa.ipc.open_stream(arrow_file)
        tables.append(stream.read_all())

    return pa.concat_tables(tables)


def run_evaluation(predictions_dir: str, dataset_path: str, output_dir: str,
                   encoder: str, threshold: float, model_name: str):
    """
    Run the standard CRYSTAL evaluation pipeline using compute_metrics_final.py logic.

    Uses MLLMReasoningEvaluator (Match F1) + AccuracyCalculator (answer accuracy)
    with the ablation-validated encoder and threshold.
    """
    from mllm_evaluator import MLLMReasoningEvaluator
    from accuracy_calculator import AccuracyCalculator
    import pandas as pd

    # Load dataset via pyarrow (bypasses datasets library schema issues)
    print(f"\nLoading CRYSTAL dataset from: {dataset_path}")
    table = load_dataset_arrow(dataset_path)
    print(f"Loaded {len(table)} samples")

    # Build ground truth dict
    ground_truth = {}
    for idx in tqdm(range(len(table)), desc="Loading ground truth"):
        ground_truth[idx] = {
            "question": table["question"][idx].as_py() or "",
            "answer": table["answer"][idx].as_py() or "",
            "reference_steps": table["reference_steps"][idx].as_py() or [],
        }

    # Load predictions
    predictions = {}
    pred_path = Path(predictions_dir)
    for pred_file in sorted(pred_path.glob("*.json")):
        if pred_file.name in ["metrics_summary.json", "metrics_detailed.csv", "metrics_summary.txt"]:
            continue
        try:
            idx = int(pred_file.stem)
            with open(pred_file, "r") as f:
                data = json.load(f)
                predictions[idx] = {
                    "predicted_steps": data.get("reasoning_steps", []),
                    "predicted_answer": data.get("answer", ""),
                }
        except (ValueError, json.JSONDecodeError):
            continue

    common_indices = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    print(f"Evaluating {len(common_indices)} samples")
    print(f"Encoder: {encoder}, Threshold: {threshold}")

    # Initialize evaluators
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    match_f1_evaluator = MLLMReasoningEvaluator(
        model_name=encoder,
        similarity_threshold=threshold,
        device=device,
        debug_mode=False,
    )

    accuracy_calculator = AccuracyCalculator(
        use_llm_grader=False,
    )

    # Evaluate
    results = []
    for idx in tqdm(common_indices, desc="Evaluating"):
        pred = predictions[idx]
        gt = ground_truth[idx]

        # Accuracy
        acc_result = accuracy_calculator.evaluate_single(
            question=gt["question"],
            predicted_answer=pred["predicted_answer"],
            ground_truth_answer=gt["answer"],
        )

        # Match F1
        if pred["predicted_steps"] and gt["reference_steps"]:
            metrics_f1 = match_f1_evaluator.evaluate_single(
                predicted_steps=pred["predicted_steps"],
                reference_steps=gt["reference_steps"],
            )
            match_f1 = metrics_f1.match_f1
            precision = metrics_f1.precision
            recall = metrics_f1.recall
            num_pred_steps = metrics_f1.num_predicted_steps
            num_ref_steps = metrics_f1.num_reference_steps
            avg_sim = metrics_f1.avg_similarity
        else:
            match_f1 = 0.0
            precision = 0.0
            recall = 0.0
            num_pred_steps = len(pred["predicted_steps"])
            num_ref_steps = len(gt["reference_steps"])
            avg_sim = 0.0

        results.append({
            "sample_idx": idx,
            "accuracy_correct": acc_result.is_correct,
            "match_f1": match_f1,
            "precision": precision,
            "recall": recall,
            "num_predicted_steps": num_pred_steps,
            "num_reference_steps": num_ref_steps,
            "avg_similarity": avg_sim,
            "answer": pred["predicted_answer"],
            "match_type": acc_result.match_type,
            "confidence": acc_result.confidence,
        })

    # Create DataFrame and save results (same format as compute_metrics_final.py)
    df = pd.DataFrame(results)
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed CSV
    csv_file = save_dir / "metrics_detailed.csv"
    df.to_csv(csv_file, index=False)

    # Build and save summary JSON
    summary = {
        "model_name": model_name,
        "encoder": encoder,
        "threshold": threshold,
        "total_samples": len(df),
        "accuracy": {
            "mean": float(df["accuracy_correct"].mean()),
            "correct_count": int(df["accuracy_correct"].sum()),
            "confidence_mean": float(df["confidence"].mean()),
        },
        "match_f1": {
            "mean": float(df["match_f1"].mean()),
            "std": float(df["match_f1"].std()),
            "min": float(df["match_f1"].min()),
            "max": float(df["match_f1"].max()),
        },
        "precision": {
            "mean": float(df["precision"].mean()),
            "std": float(df["precision"].std()),
        },
        "recall": {
            "mean": float(df["recall"].mean()),
            "std": float(df["recall"].std()),
        },
        "steps": {
            "predicted_mean": float(df["num_predicted_steps"].mean()),
            "predicted_std": float(df["num_predicted_steps"].std()),
            "reference_mean": float(df["num_reference_steps"].mean()),
            "reference_std": float(df["num_reference_steps"].std()),
            "avg_similarity": float(df["avg_similarity"].mean()),
        },
    }

    summary_file = save_dir / "metrics_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CRYSTAL BENCHMARK EVALUATION — {model_name}")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(df)}")
    print(f"Encoder: {encoder}, Threshold: {threshold}")
    print(f"\nACCURACY:")
    acc = summary["accuracy"]["mean"]
    print(f"  {acc:.4f} ({acc*100:.2f}%) — {summary['accuracy']['correct_count']}/{len(df)}")
    print(f"\nMATCH F1:")
    print(f"  F1:        {summary['match_f1']['mean']:.4f} (+/-{summary['match_f1']['std']:.4f})")
    print(f"  Precision: {summary['precision']['mean']:.4f} (+/-{summary['precision']['std']:.4f})")
    print(f"  Recall:    {summary['recall']['mean']:.4f} (+/-{summary['recall']['std']:.4f})")
    print(f"\nSTEPS:")
    print(f"  Predicted: {summary['steps']['predicted_mean']:.2f} (+/-{summary['steps']['predicted_std']:.2f})")
    print(f"  Reference: {summary['steps']['reference_mean']:.2f} (+/-{summary['steps']['reference_std']:.2f})")
    print(f"\nResults saved to:")
    print(f"  {csv_file}")
    print(f"  {summary_file}")
    print(f"{'='*60}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Parse GPT-5 Batch API results and run CRYSTAL evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=[str(INFERENCE_DIR / "gpt5_batch_results.jsonl")],
        help="Batch results JSONL file(s) (from gpt5_batch_submit.py). Accepts multiple files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "final_table" / "outputs_testing_gpt5"),
        help="Output directory for predictions and metrics",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27",
        help="Path to CRYSTAL dataset (Arrow format)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="all-distilroberta-v1",
        help="Sentence encoder for Match F1 (default: all-distilroberta-v1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Similarity threshold (default: 0.35)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="GPT-5",
        help="Display name for results (default: GPT-5)",
    )
    parser.add_argument(
        "--parse_only",
        action="store_true",
        help="Only parse results, skip evaluation",
    )
    args = parser.parse_args()

    # Step 1: Parse batch results into individual JSONs
    for input_file in args.input:
        if not Path(input_file).exists():
            print(f"ERROR: Input file not found: {input_file}")
            print("Run gpt5_batch_submit.py first")
            sys.exit(1)

    # Use a predictions subdirectory so metric files don't collide with prediction JSONs
    predictions_dir = str(Path(args.output_dir) / "predictions")

    # Parse all input files (supports multi-part batches)
    combined_stats = {"total": 0, "valid": 0, "invalid": 0, "api_errors": 0}
    for input_file in args.input:
        stats = parse_batch_results(input_file, predictions_dir)
        for k in combined_stats:
            combined_stats[k] += stats[k]

    stats = combined_stats
    print(f"\nParsing complete ({len(args.input)} file(s)):")
    print(f"  Total:      {stats['total']}")
    print(f"  Valid JSON:  {stats['valid']}")
    print(f"  Invalid JSON: {stats['invalid']}")
    print(f"  API errors:  {stats['api_errors']}")
    valid_rate = stats["valid"] / stats["total"] * 100 if stats["total"] > 0 else 0
    print(f"  Valid rate:  {valid_rate:.1f}%")

    if args.parse_only:
        print(f"\nPredictions saved to: {predictions_dir}")
        print("Skipping evaluation (--parse_only)")
        return

    # Step 2: Run evaluation
    print("\n" + "=" * 60)
    print("Running CRYSTAL evaluation pipeline...")
    print("=" * 60)

    run_evaluation(
        predictions_dir=predictions_dir,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        encoder=args.encoder,
        threshold=args.threshold,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
