#!/usr/bin/env python3
"""
Simple parallel checkpoint inference
Each GPU processes a DIFFERENT checkpoint (not dataset split)
Much simpler - no dataset splitting, no merging needed
"""

import os
import sys
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠ tqdm not available - install with: pip install tqdm")
    print("  Running without progress bars...\n")


def run_checkpoint_on_gpu(gpu_id, checkpoint_dir, test_dataset, predictions_dir, batch_size, timeout, pbar=None):
    """Run inference for one checkpoint on one GPU."""

    checkpoint_name = Path(checkpoint_dir).name

    if pbar:
        pbar.set_description(f"GPU {gpu_id}: {checkpoint_name}")
    else:
        print(f"[GPU {gpu_id}] Starting {checkpoint_name}...", flush=True)

    cmd = [
        "python",
        "inference/run_deepspeed_checkpoint_inference.py",
        "--checkpoint_dir", checkpoint_dir,
        "--test_dataset_path", test_dataset,
        "--predictions_dir", predictions_dir,
        "--device_ids", str(gpu_id),
        "--batch_size", str(batch_size),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start_time = time.time()

    # Create log file for this GPU/checkpoint
    log_file = f"gpu{gpu_id}_{checkpoint_name}.log"

    try:
        # Run with output redirected to log file so we can debug issues
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            msg = f"\n[GPU {gpu_id}] ✓ {checkpoint_name} completed in {elapsed:.1f}s"
            if pbar:
                pbar.write(msg)
                pbar.update(1)
            else:
                print(msg, flush=True)

            # Delete log file on success
            try:
                os.remove(log_file)
            except:
                pass

            return {
                "gpu_id": gpu_id,
                "checkpoint": checkpoint_name,
                "success": True,
                "time": elapsed
            }
        else:
            msg = f"\n[GPU {gpu_id}] ✗ {checkpoint_name} failed (exit code {result.returncode})"
            if pbar:
                pbar.write(msg)
                pbar.write(f"   See log: {log_file}")
            else:
                print(msg, flush=True)
                print(f"   See log: {log_file}")

            # Show last 20 lines of error
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    print(f"\n   Last 20 lines of {log_file}:")
                    print("   " + "   ".join(lines[-20:]))
            except:
                pass

            return {
                "gpu_id": gpu_id,
                "checkpoint": checkpoint_name,
                "success": False,
                "error": f"exit code {result.returncode}",
                "time": elapsed
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        msg = f"\n[GPU {gpu_id}] ✗ {checkpoint_name} TIMEOUT after {elapsed:.1f}s"
        if pbar:
            pbar.write(msg)
            pbar.write(f"   See log: {log_file}")
        else:
            print(msg, flush=True)
            print(f"   See log: {log_file}")
        return {
            "gpu_id": gpu_id,
            "checkpoint": checkpoint_name,
            "success": False,
            "error": "timeout",
            "time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        msg = f"\n[GPU {gpu_id}] ✗ {checkpoint_name} exception: {e}"
        if pbar:
            pbar.write(msg)
        else:
            print(msg, flush=True)

        # Try to save log info
        try:
            with open(log_file, 'a') as f:
                f.write(f"\n\nException: {e}\n")
        except:
            pass

        return {
            "gpu_id": gpu_id,
            "checkpoint": checkpoint_name,
            "success": False,
            "error": str(e),
            "time": elapsed
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run multiple checkpoints in parallel, one per GPU")
    parser.add_argument("--output_dirs", nargs='+', required=True, help="Output directories containing checkpoints")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Comma-separated GPU IDs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per checkpoint (seconds)")
    parser.add_argument("--checkpoint_range", type=str, default="400-1500-100",
                       help="Checkpoint range: start-end-step (e.g., 400-1500-100)")

    args = parser.parse_args()

    # Parse checkpoint range
    start, end, step = map(int, args.checkpoint_range.split('-'))
    checkpoint_nums = list(range(start, end + 1, step))

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    print("="*80)
    print("Parallel Checkpoint Inference - Simple & Efficient")
    print("="*80)
    print(f"GPU IDs: {gpu_ids}")
    print(f"Batch size: {args.batch_size}")
    print(f"Timeout: {args.timeout}s per checkpoint")
    print(f"Checkpoints: {checkpoint_nums}")
    print(f"Experiments: {len(args.output_dirs)}")
    print("="*80)
    print()

    # Collect all checkpoint tasks
    tasks = []
    skipped_count = 0

    for output_dir in args.output_dirs:
        exp_name = Path(output_dir).name
        for ckpt_num in checkpoint_nums:
            checkpoint_dir = f"{output_dir}/checkpoint-{ckpt_num}"

            # Check if checkpoint exists
            if not Path(checkpoint_dir).exists():
                print(f"⚠ Skipping {exp_name}/checkpoint-{ckpt_num} (not found)")
                continue

            pred_dir = f"predictions/{exp_name}/checkpoint-{ckpt_num}"

            # Check if already completed (stats file exists and has correct number of samples)
            stats_file = Path(pred_dir) / "inference_stats.json"
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    # Check if it has the expected number of samples (6372 for full dataset)
                    if stats.get('total_samples', 0) >= 6300:  # Allow some margin
                        print(f"✓ Skipping {exp_name}/checkpoint-{ckpt_num} (already completed: {stats['total_samples']} samples)")
                        skipped_count += 1
                        continue
                    else:
                        print(f"⚠ {exp_name}/checkpoint-{ckpt_num} incomplete ({stats.get('total_samples', 0)} samples) - will resume")
                except Exception as e:
                    print(f"⚠ Could not read stats for {exp_name}/checkpoint-{ckpt_num}: {e} - will reprocess")

            tasks.append({
                "checkpoint_dir": checkpoint_dir,
                "predictions_dir": pred_dir,
                "checkpoint_name": f"{exp_name}/checkpoint-{ckpt_num}"
            })

    if skipped_count > 0:
        print(f"\nAlready completed: {skipped_count}")

    if not tasks:
        print("\n✓ All checkpoints already completed!")
        return 0

    print(f"Remaining tasks: {len(tasks)}")
    print()

    # Process checkpoints in batches (one batch per GPU set)
    num_gpus = len(gpu_ids)
    all_results = []
    num_batches = (len(tasks) + num_gpus - 1) // num_gpus

    for batch_idx, batch_start in enumerate(range(0, len(tasks), num_gpus), 1):
        batch_tasks = tasks[batch_start:batch_start + num_gpus]

        print("\n" + "="*80)
        print(f"Batch {batch_idx}/{num_batches}")
        print("="*80)
        for i, task in enumerate(batch_tasks):
            gpu_id = gpu_ids[i]
            print(f"  GPU {gpu_id}: {task['checkpoint_name']}")
        print("="*80 + "\n")

        # Run this batch in parallel using threads (simpler than processes)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []

            for i, task in enumerate(batch_tasks):
                gpu_id = gpu_ids[i]

                future = executor.submit(
                    run_checkpoint_on_gpu,
                    gpu_id=gpu_id,
                    checkpoint_dir=task["checkpoint_dir"],
                    test_dataset=args.test_dataset,
                    predictions_dir=task["predictions_dir"],
                    batch_size=args.batch_size,
                    timeout=args.timeout,
                    pbar=None,  # No overall pbar - each GPU has its own
                )
                futures.append((future, gpu_id, task["checkpoint_name"]))

            # Wait for all GPUs in this batch to complete
            batch_results = []
            for future, gpu_id, ckpt_name in futures:
                try:
                    result = future.result()
                    batch_results.append(result)
                    all_results.append(result)
                except Exception as e:
                    msg = f"\n[GPU {gpu_id}] ✗ Exception in {ckpt_name}: {e}"
                    print(msg, flush=True)
                    batch_results.append({
                        "gpu_id": gpu_id,
                        "checkpoint": ckpt_name,
                        "success": False,
                        "error": str(e),
                        "time": 0
                    })
                    all_results.append(batch_results[-1])

        # Print batch summary
        print("\n" + "-"*80)
        print(f"Batch {batch_idx} Summary:")
        print("-"*80)
        for result in sorted(batch_results, key=lambda x: x["gpu_id"]):
            status = "✓ OK" if result["success"] else "✗ FAIL"
            time_str = f"{result['time']:.1f}s" if result['time'] > 0 else ""
            error_str = f"({result.get('error', '')})" if not result["success"] else ""
            print(f"  [GPU {result['gpu_id']}] {status} {result['checkpoint']} {time_str} {error_str}")
        print("-"*80)

    # Final summary
    print()
    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)

    total = len(all_results)
    success = sum(1 for r in all_results if r["success"])
    failed = total - success

    print(f"Total executed: {total}")
    print(f"  Completed: {success}")
    print(f"  Failed: {failed}")
    if skipped_count > 0:
        print(f"  Previously completed: {skipped_count}")
    print()

    if failed > 0:
        print("Failed checkpoints:")
        for result in all_results:
            if not result["success"]:
                print(f"  - {result['checkpoint']}: {result.get('error', 'unknown')}")
        print()

    print("="*80)
    print("✓ All batches completed!")
    print("="*80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
