#!/usr/bin/env python3
"""
Simplified parallel inference - much more robust and easier to debug
Each GPU processes a different subset of data in parallel
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import time
from pathlib import Path
from datasets import load_from_disk


def split_dataset(dataset_path: str, num_splits: int, output_base_dir: str):
    """Split dataset into N parts for parallel processing."""
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    total_samples = len(dataset)

    print(f"Total samples: {total_samples}")
    print(f"Splitting into {num_splits} parts...\n")

    # Clean up old splits if they exist
    split_base = Path(output_base_dir)
    if split_base.exists():
        print(f"Cleaning up old splits in {output_base_dir}")
        shutil.rmtree(split_base)

    split_base.mkdir(parents=True, exist_ok=True)

    splits = []
    samples_per_split = (total_samples + num_splits - 1) // num_splits

    for i in range(num_splits):
        start_idx = i * samples_per_split
        end_idx = min(start_idx + samples_per_split, total_samples)

        if start_idx >= total_samples:
            break

        split_dataset = dataset.select(range(start_idx, end_idx))
        split_dir = f"{output_base_dir}/split_{i}"

        print(f"  Split {i}: samples {start_idx}-{end_idx} ({len(split_dataset)} samples)")
        split_dataset.save_to_disk(split_dir)

        splits.append({
            "split_id": i,
            "gpu_id": None,  # Will be assigned later
            "dataset_path": split_dir,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_samples": len(split_dataset)
        })

    return splits


def run_inference_on_gpu(
    gpu_id: int,
    checkpoint_dir: str,
    dataset_path: str,
    predictions_dir: str,
    start_idx: int,
    batch_size: int,
    timeout: int,
):
    """Run inference on a specific GPU with timeout."""

    script_path = os.path.join(
        os.path.dirname(__file__),
        "run_deepspeed_checkpoint_inference.py"
    )

    cmd = [
        "python",
        script_path,
        "--checkpoint_dir", checkpoint_dir,
        "--test_dataset_path", dataset_path,
        "--predictions_dir", predictions_dir,
        "--device_ids", str(gpu_id),
        "--start_idx_offset", str(start_idx),
        "--batch_size", str(batch_size),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting inference...")
    start_time = time.time()

    try:
        # Simple subprocess.run with timeout - much more robust than Popen
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"[GPU {gpu_id}] ✓ Completed in {elapsed:.1f}s")
            return {"success": True, "gpu_id": gpu_id, "time": elapsed}
        else:
            print(f"[GPU {gpu_id}] ✗ Failed with exit code {result.returncode}")
            print(f"[GPU {gpu_id}] Error: {result.stderr[-500:]}")  # Last 500 chars
            return {"success": False, "gpu_id": gpu_id, "error": "non-zero exit", "code": result.returncode}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_id}] ✗ TIMEOUT after {elapsed:.1f}s")
        return {"success": False, "gpu_id": gpu_id, "error": "timeout", "time": elapsed}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[GPU {gpu_id}] ✗ Exception: {str(e)}")
        return {"success": False, "gpu_id": gpu_id, "error": str(e), "time": elapsed}


def merge_results(predictions_dir: str, gpu_results: list, num_gpus: int, gpu_ids: list):
    """Merge results from all GPUs into final prediction directory."""

    print("\n" + "="*80)
    print("Merging results from all GPUs...")
    print("="*80)

    os.makedirs(predictions_dir, exist_ok=True)

    all_results = []
    total_copied = 0

    for result in gpu_results:
        if not result["success"]:
            continue

        gpu_id = result["gpu_id"]
        pred_dir = f"{predictions_dir}_gpu{gpu_id}_temp"
        pred_path = Path(pred_dir)

        if not pred_path.exists():
            print(f"⚠ GPU {gpu_id} temp dir not found: {pred_dir}")
            continue

        # Copy individual prediction files
        files_copied = 0
        for json_file in pred_path.glob("*.json"):
            if json_file.name not in ["inference_summary.json", "inference_stats.json"]:
                dest_file = Path(predictions_dir) / json_file.name
                shutil.copy(json_file, dest_file)
                files_copied += 1

        # Load summary
        summary_file = pred_path / "inference_summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                results = json.load(f)
                all_results.extend(results)

        print(f"✓ GPU {gpu_id}: Copied {files_copied} files")
        total_copied += files_copied

        # Clean up temp directory
        try:
            shutil.rmtree(pred_path)
            print(f"✓ GPU {gpu_id}: Cleaned up temp directory")
        except Exception as e:
            print(f"⚠ GPU {gpu_id}: Failed to cleanup temp dir: {e}")

    # Save merged summary
    summary_file = os.path.join(predictions_dir, "inference_summary.json")
    with open(summary_file, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Calculate merged stats
    total_processed = len(all_results)
    valid_count = sum(1 for r in all_results if r.get("is_valid", False))

    stats = {
        "checkpoint": predictions_dir,
        "total_samples": total_processed,
        "valid_predictions": valid_count,
        "invalid_predictions": total_processed - valid_count,
        "validation_rate": valid_count / total_processed if total_processed > 0 else 0,
        "num_gpus": num_gpus,
        "gpu_ids": gpu_ids,
    }

    stats_file = os.path.join(predictions_dir, "inference_stats.json")
    with open(stats_file, "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✓ Merge completed!")
    print("="*80)
    print(f"Total files: {total_copied}")
    print(f"Total samples: {total_processed}")
    print(f"Valid: {valid_count} ({stats['validation_rate']*100:.1f}%)")
    print(f"Invalid: {total_processed - valid_count}")
    print("="*80)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Run parallel inference across multiple GPUs")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_dataset_path", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Comma-separated GPU IDs")
    parser.add_argument("--temp_dir", type=str, default="temp_splits", help="Temp dir for dataset splits")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds per GPU (default: 1800s = 30min)")

    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    num_gpus = len(gpu_ids)

    print("="*80)
    print("Parallel Inference - Simplified & Robust")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Dataset: {args.test_dataset_path}")
    print(f"Output: {args.predictions_dir}")
    print(f"GPUs: {gpu_ids}")
    print(f"Batch size: {args.batch_size}")
    print(f"Timeout: {args.timeout}s per GPU")
    print("="*80)

    # Split dataset
    splits = split_dataset(args.test_dataset_path, num_gpus, args.temp_dir)

    print("\n" + "="*80)
    print("Launching parallel inference on all GPUs...")
    print("="*80 + "\n")

    # Launch all GPU processes using multiprocessing for true parallelism
    from concurrent.futures import ProcessPoolExecutor, as_completed

    futures = []
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        for gpu_id, split_info in zip(gpu_ids, splits):
            split_predictions_dir = f"{args.predictions_dir}_gpu{gpu_id}_temp"

            future = executor.submit(
                run_inference_on_gpu,
                gpu_id=gpu_id,
                checkpoint_dir=args.checkpoint_dir,
                dataset_path=split_info["dataset_path"],
                predictions_dir=split_predictions_dir,
                start_idx=split_info["start_idx"],
                batch_size=args.batch_size,
                timeout=args.timeout,
            )
            futures.append(future)

        # Wait for all to complete and collect results
        gpu_results = []
        for future in as_completed(futures):
            result = future.result()
            gpu_results.append(result)

    # Check results
    print("\n" + "="*80)
    print("GPU Completion Status:")
    print("="*80)

    successful = sum(1 for r in gpu_results if r["success"])
    failed = len(gpu_results) - successful

    for result in sorted(gpu_results, key=lambda x: x["gpu_id"]):
        gpu_id = result["gpu_id"]
        if result["success"]:
            print(f"[GPU {gpu_id}] ✓ Success ({result['time']:.1f}s)")
        else:
            print(f"[GPU {gpu_id}] ✗ Failed: {result.get('error', 'unknown')}")

    print(f"\nTotal: {successful} success, {failed} failed")

    if successful == 0:
        print("\n✗ All GPUs failed - cannot merge results")
        # Cleanup
        if Path(args.temp_dir).exists():
            shutil.rmtree(args.temp_dir)
        return 1

    # Merge results from successful GPUs
    stats = merge_results(args.predictions_dir, gpu_results, num_gpus, gpu_ids)

    # Clean up dataset splits
    if Path(args.temp_dir).exists():
        try:
            shutil.rmtree(args.temp_dir)
            print(f"\n✓ Cleaned up dataset splits: {args.temp_dir}")
        except Exception as e:
            print(f"\n⚠ Failed to cleanup {args.temp_dir}: {e}")

    print("\n" + "="*80)
    print("✓ Parallel inference completed!")
    print(f"Results saved to: {args.predictions_dir}")
    print("="*80 + "\n")

    return 0 if successful == num_gpus else 1


if __name__ == "__main__":
    sys.exit(main())
