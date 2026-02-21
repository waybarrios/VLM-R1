#!/usr/bin/env python3
"""
Ablation Study: Efficient parallel execution across 4 GPUs
Runs experiments in parallel batches, saves partial results
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

# Configuration
DATASET_PATH = "/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27"
REASONING_BASE = "/gpudata3/Wayner/reasoning"
OUTPUT_DIR = "ablation_results"

# Models
MODELS = {
    "qwen25vl_32b": "outputs_testing_qwen25vl_32b_64k",
    "gemma3_12b": "outputs_testing_gemma3_12b_64k",
    "gemma3_4b": "outputs_testing_gemma3_4b",
    "llava7b": "outputs_testing_llava7b_16",
    "minicpm_v_8b": "outputs_testing_minicpm_v_8b"
}

# Encoders
ENCODERS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "all-distilroberta-v1"
]

# Thresholds
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]

# GPUs
GPUS = [0, 1, 2, 3]
NUM_GPUS = len(GPUS)


def run_single_experiment(model_name: str, model_dir: str, encoder: str,
                          threshold: float, gpu_id: int, exp_id: int, total_exps: int) -> Dict:
    """
    Run a single ablation experiment (1 model, 1 encoder, 1 threshold)
    """
    predictions_dir = f"{REASONING_BASE}/{model_dir}"
    output_file = f"{OUTPUT_DIR}/partial_{model_name}_{encoder.replace('-', '_')}_{threshold}.json"
    log_file = f"{OUTPUT_DIR}/logs/exp_{exp_id}_gpu{gpu_id}.log"

    # Create logs directory
    os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)

    print(f"\n[Exp {exp_id}/{total_exps}] GPU {gpu_id}: {model_name} | {encoder[:20]} | τ={threshold}")
    print(f"  Log: {log_file}")

    # Check if already completed
    if os.path.exists(output_file):
        print(f"  ✓ Already completed, skipping...")
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            return {
                'status': 'cached',
                'model': model_name,
                'encoder': encoder,
                'threshold': threshold,
                'gpu': gpu_id,
                'exp_id': exp_id,
                'results': data['results'][0] if data.get('results') else None
            }
        except:
            pass  # If loading fails, re-run

    # Run experiment with output to log file
    cmd = [
        "python3", "run_ablation_simple.py",
        predictions_dir,
        "--dataset-path", DATASET_PATH,
        "--output-file", output_file,
        "--device", f"cuda:{gpu_id}",
        "--encoders", encoder,
        "--thresholds", str(threshold)
    ]

    try:
        start_time = time.time()

        # Run with output to log file (shows progress bars)
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600  # 10 minute timeout per experiment
            )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Load results
            with open(output_file, 'r') as f:
                data = json.load(f)

            exp_result = data['results'][0] if data.get('results') else None

            if exp_result:
                f1 = exp_result['match_f1']['mean']
                p = exp_result['precision']['mean']
                r = exp_result['recall']['mean']
                print(f"  ✓ GPU {gpu_id}: F1={f1:.4f}, P={p:.4f}, R={r:.4f} ({elapsed:.1f}s)")

            return {
                'status': 'success',
                'model': model_name,
                'encoder': encoder,
                'threshold': threshold,
                'gpu': gpu_id,
                'exp_id': exp_id,
                'elapsed': elapsed,
                'log_file': log_file,
                'results': exp_result
            }
        else:
            print(f"  ❌ GPU {gpu_id}: FAILED (see {log_file})")
            # Show last few lines of log
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    print(f"  Last error: {lines[-1] if lines else 'No output'}")
            except:
                pass

            return {
                'status': 'failed',
                'model': model_name,
                'encoder': encoder,
                'threshold': threshold,
                'gpu': gpu_id,
                'exp_id': exp_id,
                'log_file': log_file
            }

    except subprocess.TimeoutExpired:
        print(f"  ⏱️  GPU {gpu_id}: TIMEOUT (>10min) - see {log_file}")
        return {
            'status': 'timeout',
            'model': model_name,
            'encoder': encoder,
            'threshold': threshold,
            'gpu': gpu_id,
            'exp_id': exp_id,
            'log_file': log_file
        }
    except Exception as e:
        print(f"  ❌ GPU {gpu_id}: ERROR - {str(e)}")
        return {
            'status': 'error',
            'model': model_name,
            'encoder': encoder,
            'threshold': threshold,
            'gpu': gpu_id,
            'exp_id': exp_id,
            'error': str(e)
        }


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*80)
    print("ABLATION STUDY - PARALLEL EFFICIENT EXECUTION")
    print("="*80)
    print(f"Models: {len(MODELS)}")
    print(f"Encoders: {len(ENCODERS)}")
    print(f"Thresholds: {len(THRESHOLDS)}")
    print(f"GPUs: {NUM_GPUS} ({GPUS})")
    print(f"Total experiments: {len(MODELS) * len(ENCODERS) * len(THRESHOLDS)}")
    print(f"Batches (4 GPUs): {(len(MODELS) * len(ENCODERS) * len(THRESHOLDS) + NUM_GPUS - 1) // NUM_GPUS}")
    print("="*80)
    print()

    # Generate all experiments
    all_experiments = []
    exp_id = 0
    for model_name, model_dir in MODELS.items():
        for encoder in ENCODERS:
            for threshold in THRESHOLDS:
                exp_id += 1
                all_experiments.append({
                    'id': exp_id,
                    'model_name': model_name,
                    'model_dir': model_dir,
                    'encoder': encoder,
                    'threshold': threshold
                })

    total_exps = len(all_experiments)
    print(f"Generated {total_exps} experiments\n")

    # Run experiments in parallel batches
    all_results = []
    completed = 0
    failed = 0

    # Use ProcessPoolExecutor to run in parallel
    with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
        # Submit all experiments
        future_to_exp = {}

        for i, exp in enumerate(all_experiments):
            gpu_id = GPUS[i % NUM_GPUS]

            future = executor.submit(
                run_single_experiment,
                exp['model_name'],
                exp['model_dir'],
                exp['encoder'],
                exp['threshold'],
                gpu_id,
                exp['id'],
                total_exps
            )
            future_to_exp[future] = exp

        # Collect results as they complete
        print(f"\n{'='*80}")
        print("RUNNING EXPERIMENTS IN PARALLEL")
        print(f"{'='*80}\n")

        with tqdm(total=total_exps, desc="Total Progress", ncols=100) as pbar:
            for future in as_completed(future_to_exp):
                result = future.result()
                all_results.append(result)

                if result['status'] == 'success' or result['status'] == 'cached':
                    completed += 1
                else:
                    failed += 1

                pbar.update(1)
                pbar.set_postfix({
                    'completed': completed,
                    'failed': failed
                })

    # Save all results
    results_file = f"{OUTPUT_DIR}/all_experiments.json"
    with open(results_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_experiments': total_exps,
                'completed': completed,
                'failed': failed,
                'models': list(MODELS.keys()),
                'encoders': ENCODERS,
                'thresholds': THRESHOLDS
            },
            'experiments': all_results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Success: {completed}/{total_exps}")
    print(f"Failed: {failed}/{total_exps}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*80}\n")

    # Aggregate results by model
    print("Aggregating results by model...")
    model_results = {}

    for result in all_results:
        if result['status'] not in ['success', 'cached'] or not result.get('results'):
            continue

        model_name = result['model']

        if model_name not in model_results:
            model_results[model_name] = []

        model_results[model_name].append(result['results'])

    # Save per-model results
    for model_name, results in model_results.items():
        model_file = f"{OUTPUT_DIR}/{model_name}_ablation.json"
        with open(model_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model': model_name,
                    'num_configs': len(results),
                    'encoders': ENCODERS,
                    'thresholds': THRESHOLDS
                },
                'results': results
            }, f, indent=2)
        print(f"  ✓ {model_name}: {len(results)} configs → {model_file}")

    # Aggregate across all models
    print("\nAggregating across all models...")

    import numpy as np

    aggregated = {}
    configs = set()

    for result in all_results:
        if result['status'] not in ['success', 'cached'] or not result.get('results'):
            continue

        config = (result['encoder'], result['threshold'])
        configs.add(config)

    for encoder, threshold in sorted(configs):
        key = f"{encoder}___{threshold}"
        f1_means = []
        precision_means = []
        recall_means = []

        for result in all_results:
            if (result['status'] in ['success', 'cached'] and
                result.get('results') and
                result['encoder'] == encoder and
                result['threshold'] == threshold):

                f1_means.append(result['results']['match_f1']['mean'])
                precision_means.append(result['results']['precision']['mean'])
                recall_means.append(result['results']['recall']['mean'])

        if f1_means:
            aggregated[key] = {
                'encoder': encoder,
                'threshold': threshold,
                'num_models': len(f1_means),
                'match_f1_across_models': {
                    'mean': float(np.mean(f1_means)),
                    'std': float(np.std(f1_means)),
                    'min': float(np.min(f1_means)),
                    'max': float(np.max(f1_means))
                },
                'precision_across_models': {
                    'mean': float(np.mean(precision_means)),
                    'std': float(np.std(precision_means))
                },
                'recall_across_models': {
                    'mean': float(np.mean(recall_means)),
                    'std': float(np.std(recall_means))
                }
            }

    # Save aggregated results
    aggregate_file = f"{OUTPUT_DIR}/ablation_aggregate.json"
    with open(aggregate_file, 'w') as f:
        json.dump({
            'metadata': {
                'num_models': len(model_results),
                'models': list(model_results.keys()),
                'num_configurations': len(aggregated),
                'encoders': ENCODERS,
                'thresholds': THRESHOLDS
            },
            'aggregated_results': list(aggregated.values())
        }, f, indent=2)

    print(f"✓ Aggregated results: {aggregate_file}\n")

    # Print summary table
    print("="*110)
    print("AGGREGATED SUMMARY (MEAN ACROSS ALL MODELS)")
    print("="*110)
    print(f"{'Encoder':<40} {'Threshold':>10} {'F1':>12} {'Precision':>12} {'Recall':>12}")
    print("="*110)

    for config in sorted(aggregated.values(), key=lambda x: (x['encoder'], x['threshold'])):
        print(f"{config['encoder']:<40} {config['threshold']:>10.2f} "
              f"{config['match_f1_across_models']['mean']:>12.4f} "
              f"{config['precision_across_models']['mean']:>12.4f} "
              f"{config['recall_across_models']['mean']:>12.4f}")

    print("="*110)

    print(f"\n{'='*80}")
    print("✓ COMPLETE")
    print(f"{'='*80}")
    print("Results:")
    print(f"  - All experiments: {OUTPUT_DIR}/all_experiments.json")
    print(f"  - Per model: {OUTPUT_DIR}/<model>_ablation.json")
    print(f"  - Aggregated: {OUTPUT_DIR}/ablation_aggregate.json")
    print(f"  - Partial results: {OUTPUT_DIR}/partial_*.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
