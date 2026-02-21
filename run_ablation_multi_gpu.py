#!/usr/bin/env python3
"""
Ablation Study: Multi-GPU parallel execution with individual progress bars
Each experiment runs on a dedicated GPU with visible progress
"""

import os
import pathlib
HF_CACHE_DIR = "/gpudata3/hf_cache"
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", HF_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", HF_CACHE_DIR)
pathlib.Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Add mllm_evaluator to path
sys.path.insert(0, '/gpudata3/Wayner/VLM-R1/mllm_evaluator')
from mllm_evaluator import MLLMReasoningEvaluator

# Configuration
DATASET_PATH = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
REASONING_BASE = "/gpudata3/Wayner/reasoning"
OUTPUT_DIR = "ablation_results"

# Models to evaluate
MODELS = {
    "qwen25vl_32b": "outputs_testing_qwen25vl_32b_64k",
    "gemma3_12b": "outputs_testing_gemma3_12b_64k",
    "gemma3_4b": "outputs_testing_gemma3_4b",
    "llava7b": "outputs_testing_llava7b_16",
    "minicpm_v_8b": "outputs_testing_minicpm_v_8b"
}

# Sentence encoders to test
ENCODERS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "all-distilroberta-v1"
]

# Similarity thresholds to test
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]

# GPUs to use
GPUS = [0, 1, 2, 3]

# Thread-safe print
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print"""
    with print_lock:
        print(*args, **kwargs)


def load_predictions_fast(predictions_dir: str) -> Dict[int, List[str]]:
    """Load prediction JSON files quickly"""
    predictions_path = Path(predictions_dir)

    if not predictions_path.exists():
        raise ValueError(f"Predictions directory not found: {predictions_dir}")

    json_files = sorted(predictions_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {predictions_dir}")

    all_predictions = {}

    for json_file in json_files:
        try:
            sample_idx = int(json_file.stem)
            with open(json_file, 'r') as f:
                data = json.load(f)
            reasoning_steps = data.get('reasoning_steps', [])
            all_predictions[sample_idx] = reasoning_steps
        except Exception:
            continue

    return all_predictions


def load_dataset_fast(dataset_path: str, sample_indices: List[int]) -> Dict[int, Dict]:
    """Load only the needed samples from dataset"""
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
    needed_indices = set(sample_indices)
    all_data = {}

    for idx in needed_indices:
        if idx < len(full_table):
            sample = {
                'reference_steps': full_table['reference_steps'][idx].as_py()
                    if 'reference_steps' in full_table.column_names else []
            }
            all_data[idx] = sample

    return all_data


def evaluate_single_config(
    model_name: str,
    predictions: Dict[int, List[str]],
    dataset: Dict[int, Dict],
    encoder_name: str,
    threshold: float,
    gpu_id: int,
    exp_id: int,
    total_exps: int
) -> Dict:
    """
    Evaluate a single configuration (1 model, 1 encoder, 1 threshold) on 1 GPU
    """
    device = f"cuda:{gpu_id}"

    # Check if already completed
    output_file = f"{OUTPUT_DIR}/partial_{model_name}_{encoder_name.replace('-', '_')}_tau{threshold}.json"
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            safe_print(f"[Exp {exp_id}/{total_exps}] GPU{gpu_id}: {model_name} | {encoder_name[:20]:<20} | τ={threshold:.2f} - CACHED ✓")
            return {
                'status': 'cached',
                'model': model_name,
                'encoder': encoder_name,
                'threshold': threshold,
                'gpu': gpu_id,
                'exp_id': exp_id,
                'results': data['results']
            }
        except Exception:
            pass  # If loading fails, re-run

    safe_print(f"[Exp {exp_id}/{total_exps}] GPU{gpu_id}: Starting {model_name} | {encoder_name[:20]:<20} | τ={threshold:.2f}")

    start_time = time.time()

    try:
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

        # Evaluate all samples with progress bar
        sample_indices = list(predictions.keys())

        # Create progress bar with GPU-specific description
        desc = f"GPU{gpu_id} {model_name[:12]:<12} {encoder_name[:15]:<15} τ={threshold:.2f}"
        pbar = tqdm(
            sample_indices,
            desc=desc,
            position=gpu_id,  # Each GPU gets its own line
            leave=True,
            ncols=120,
            colour=['blue', 'green', 'yellow', 'red'][gpu_id % 4]
        )

        for sample_idx in pbar:
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
            except Exception:
                continue

            # Update progress bar with current average
            if len(all_f1) > 0:
                pbar.set_postfix({
                    'F1': f"{np.mean(all_f1):.3f}",
                    'P': f"{np.mean(all_precision):.3f}",
                    'R': f"{np.mean(all_recall):.3f}"
                })

        pbar.close()

        # Check if we have results
        if len(all_f1) == 0:
            safe_print(f"[Exp {exp_id}/{total_exps}] GPU{gpu_id}: {model_name} - NO VALID RESULTS ⚠️")
            return {
                'status': 'no_results',
                'model': model_name,
                'encoder': encoder_name,
                'threshold': threshold,
                'gpu': gpu_id,
                'exp_id': exp_id
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

        elapsed = time.time() - start_time

        # Save partial result
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model': model_name,
                    'encoder': encoder_name,
                    'threshold': threshold,
                    'gpu': gpu_id
                },
                'results': results
            }, f, indent=2)

        f1 = results['match_f1']['mean']
        p = results['precision']['mean']
        r = results['recall']['mean']

        safe_print(f"[Exp {exp_id}/{total_exps}] GPU{gpu_id}: {model_name} | {encoder_name[:20]:<20} | τ={threshold:.2f} - F1={f1:.4f} P={p:.4f} R={r:.4f} ({elapsed:.1f}s) ✓")

        return {
            'status': 'success',
            'model': model_name,
            'encoder': encoder_name,
            'threshold': threshold,
            'gpu': gpu_id,
            'exp_id': exp_id,
            'elapsed': elapsed,
            'results': results
        }

    except Exception as e:
        elapsed = time.time() - start_time
        safe_print(f"[Exp {exp_id}/{total_exps}] GPU{gpu_id}: {model_name} | {encoder_name[:20]:<20} | τ={threshold:.2f} - ERROR: {str(e)} ❌")
        return {
            'status': 'error',
            'model': model_name,
            'encoder': encoder_name,
            'threshold': threshold,
            'gpu': gpu_id,
            'exp_id': exp_id,
            'error': str(e),
            'elapsed': elapsed
        }


def main():
    print("=" * 100)
    print("ABLATION STUDY - MULTI-GPU PARALLEL EXECUTION WITH PROGRESS BARS")
    print("=" * 100)
    print(f"Models: {len(MODELS)}")
    print(f"Encoders: {len(ENCODERS)}")
    print(f"Thresholds: {len(THRESHOLDS)}")
    print(f"GPUs: {len(GPUS)} ({GPUS})")

    total_experiments = len(MODELS) * len(ENCODERS) * len(THRESHOLDS)
    print(f"Total experiments: {total_experiments}")
    print(f"Experiments per GPU: ~{total_experiments // len(GPUS)}")
    print("=" * 100)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all model predictions and dataset once
    print("Loading predictions and dataset for all models...")
    print()

    all_model_data = {}
    dataset_cache = None

    for model_name, model_dir in MODELS.items():
        predictions_dir = f"{REASONING_BASE}/{model_dir}"
        print(f"Loading {model_name}...")

        try:
            predictions = load_predictions_fast(predictions_dir)

            # Load dataset only once (all models use same dataset)
            if dataset_cache is None:
                print(f"Loading dataset...")
                dataset_cache = load_dataset_fast(DATASET_PATH, list(predictions.keys()))

            all_model_data[model_name] = {
                'predictions': predictions,
                'dataset': dataset_cache
            }
            print(f"  ✓ {model_name}: {len(predictions)} samples loaded\n")
        except Exception as e:
            print(f"  ❌ {model_name}: ERROR - {str(e)}\n")
            continue

    print(f"\n{'=' * 100}")
    print(f"Loaded {len(all_model_data)} models successfully")
    print(f"{'=' * 100}\n")

    # Generate all experiment configurations
    experiments = []
    exp_id = 0

    for model_name in MODELS.keys():
        if model_name not in all_model_data:
            continue
        for encoder in ENCODERS:
            for threshold in THRESHOLDS:
                exp_id += 1
                experiments.append({
                    'id': exp_id,
                    'model_name': model_name,
                    'encoder': encoder,
                    'threshold': threshold
                })

    total_exps = len(experiments)
    print(f"Generated {total_exps} experiment configurations\n")

    # Run experiments in parallel using ThreadPoolExecutor
    print(f"{'=' * 100}")
    print("RUNNING EXPERIMENTS IN PARALLEL (ThreadPoolExecutor)")
    print(f"{'=' * 100}\n")

    all_results = []
    completed = 0
    failed = 0

    # Use threads (better for CUDA than processes)
    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        future_to_exp = {}

        # Submit all experiments
        for i, exp in enumerate(experiments):
            gpu_id = GPUS[i % len(GPUS)]
            model_data = all_model_data.get(exp['model_name'])

            if model_data is None:
                continue

            future = executor.submit(
                evaluate_single_config,
                exp['model_name'],
                model_data['predictions'],
                model_data['dataset'],
                exp['encoder'],
                exp['threshold'],
                gpu_id,
                exp['id'],
                total_exps
            )
            future_to_exp[future] = exp

        # Collect results as they complete
        for future in as_completed(future_to_exp):
            try:
                result = future.result()
                all_results.append(result)

                if result['status'] in ['success', 'cached']:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                safe_print(f"Future failed with exception: {str(e)}")

    # Summary
    print(f"\n{'=' * 100}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'=' * 100}")
    print(f"Success/Cached: {completed}/{total_exps}")
    print(f"Failed: {failed}/{total_exps}")
    print(f"{'=' * 100}\n")

    # Save all experiments to JSON
    print("Saving results...")

    all_experiments_file = f"{OUTPUT_DIR}/all_experiments_summary.json"
    with open(all_experiments_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_experiments': total_exps,
                'completed': completed,
                'failed': failed,
                'models': list(MODELS.keys()),
                'encoders': ENCODERS,
                'thresholds': THRESHOLDS,
                'gpus': GPUS
            },
            'experiments': all_results
        }, f, indent=2)
    print(f"  ✓ All experiments summary: {all_experiments_file}")

    # Aggregate by model
    print("\nAggregating results by model...")
    model_results = {}

    for result in all_results:
        if result['status'] not in ['success', 'cached'] or 'results' not in result:
            continue

        model_name = result['model']
        if model_name not in model_results:
            model_results[model_name] = []

        model_results[model_name].append(result['results'])

    for model_name, results in model_results.items():
        model_file = f"{OUTPUT_DIR}/{model_name}_complete_ablation.json"
        with open(model_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model': model_name,
                    'num_configurations': len(results),
                    'encoders': ENCODERS,
                    'thresholds': THRESHOLDS
                },
                'results': results
            }, f, indent=2)
        print(f"  ✓ {model_name}: {len(results)} configs → {model_file}")

    # Create summary tables
    print(f"\n{'=' * 100}")
    print("RESULTS SUMMARY BY MODEL")
    print(f"{'=' * 100}\n")

    for model_name in sorted(model_results.keys()):
        print(f"\n{model_name.upper()}")
        print("-" * 100)
        print(f"{'Encoder':<40} {'Threshold':>10} {'F1':>12} {'Precision':>12} {'Recall':>12}")
        print("-" * 100)

        model_configs = [r for r in all_results
                        if r.get('model') == model_name and r['status'] in ['success', 'cached'] and 'results' in r]

        for result in sorted(model_configs, key=lambda x: (x['encoder'], x['threshold'])):
            res = result['results']
            print(f"{res['encoder']:<40} {res['threshold']:>10.2f} "
                  f"{res['match_f1']['mean']:>12.4f} "
                  f"{res['precision']['mean']:>12.4f} "
                  f"{res['recall']['mean']:>12.4f}")

        print("-" * 100)

    print(f"\n{'=' * 100}")
    print("✓ ABLATION STUDY COMPLETE")
    print(f"{'=' * 100}")
    print("Output files:")
    print(f"  - Summary: {all_experiments_file}")
    print(f"  - Per-model: {OUTPUT_DIR}/<model>_complete_ablation.json")
    print(f"  - Partial results: {OUTPUT_DIR}/partial_*.json")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
