#!/usr/bin/env python3
"""
Monitor checkpoint progress and detect stuck processes
"""

import os
import sys
import time
import argparse
from pathlib import Path


def monitor_checkpoint(predictions_dir, timeout_no_new_files=300):
    """
    Monitor a checkpoint directory for new files.
    Returns True if making progress, False if stuck.
    """

    if not os.path.exists(predictions_dir):
        print(f"Directory {predictions_dir} doesn't exist yet - process might be loading...")
        return True  # Still loading

    # Count JSON files (excluding summary/stats)
    json_files = []
    for f in os.listdir(predictions_dir):
        if f.endswith('.json') and f not in ['inference_summary.json', 'inference_stats.json']:
            json_files.append(f)

    num_files = len(json_files)

    if num_files == 0:
        # Check how long the directory has existed
        dir_creation_time = os.path.getctime(predictions_dir)
        time_since_creation = time.time() - dir_creation_time

        if time_since_creation > timeout_no_new_files:
            return False  # Stuck - directory created but no files after timeout
        else:
            return True  # Still loading model

    # Check timestamp of most recent file
    most_recent_time = max(os.path.getmtime(os.path.join(predictions_dir, f)) for f in json_files)
    time_since_last_file = time.time() - most_recent_time

    if time_since_last_file > timeout_no_new_files:
        return False  # Stuck - no new files in timeout period

    return True  # Making progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dirs", nargs='+', required=True)
    parser.add_argument("--timeout", type=int, default=300, help="Seconds without new files = stuck")
    parser.add_argument("--check_interval", type=int, default=30, help="Check every N seconds")

    args = parser.parse_args()

    print(f"Monitoring {len(args.predictions_dirs)} checkpoints...")
    print(f"Timeout: {args.timeout}s without new files")
    print(f"Check interval: {args.check_interval}s")
    print()

    while True:
        all_ok = True

        for pred_dir in args.predictions_dirs:
            is_progressing = monitor_checkpoint(pred_dir, args.timeout)

            # Count files
            num_files = 0
            if os.path.exists(pred_dir):
                num_files = len([f for f in os.listdir(pred_dir)
                               if f.endswith('.json') and
                               f not in ['inference_summary.json', 'inference_stats.json']])

            status = "✓ OK" if is_progressing else "✗ STUCK"
            print(f"{status} {pred_dir}: {num_files} files")

            if not is_progressing:
                all_ok = False
                print(f"  ⚠ WARNING: No new files in {args.timeout}s - process may be stuck!")

        print()

        if not all_ok:
            print("Some processes appear stuck!")
            break

        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
