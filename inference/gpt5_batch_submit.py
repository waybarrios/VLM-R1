#!/usr/bin/env python3
"""
GPT-5 Batch API — Step 2: Submit batch job & monitor

Uploads JSONL to OpenAI, creates a batch job, polls for completion,
and downloads results.

Requires: OPENAI_API_KEY environment variable

Usage:
    python inference/gpt5_batch_submit.py
    python inference/gpt5_batch_submit.py --input inference/gpt5_batch_requests.jsonl
    python inference/gpt5_batch_submit.py --check <batch_id>   # Resume monitoring
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI


INFERENCE_DIR = Path(__file__).resolve().parent


def upload_file(client: OpenAI, input_path: str) -> str:
    """Upload JSONL file to OpenAI for batch processing."""
    print(f"Uploading {input_path} ...")
    with open(input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"Uploaded: file_id={file_obj.id} ({file_obj.bytes / 1024 / 1024:.1f} MB)")
    return file_obj.id


def create_batch(client: OpenAI, file_id: str) -> str:
    """Create a batch job."""
    print(f"\nCreating batch job with file_id={file_id} ...")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "GPT-5 CRYSTAL benchmark evaluation"},
    )
    print(f"Batch created: id={batch.id}, status={batch.status}")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str, poll_interval: int = 60) -> dict:
    """Poll batch job until completion."""
    print(f"\nMonitoring batch {batch_id} ...")
    print(f"Polling every {poll_interval}s (Ctrl+C to stop — you can resume with --check {batch_id})\n")

    terminal_states = {"completed", "failed", "expired", "cancelled"}
    prev_status = None

    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts

        if batch.status != prev_status:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {batch.status}")
            if counts:
                print(f"  Completed: {counts.completed}/{counts.total}  "
                      f"Failed: {counts.failed}")
            prev_status = batch.status

        if batch.status in terminal_states:
            return batch

        time.sleep(poll_interval)


def download_results(client: OpenAI, batch: dict, output_path: str):
    """Download batch results to file."""
    if batch.status != "completed":
        print(f"\nBatch ended with status: {batch.status}")
        if batch.errors and batch.errors.data:
            print("Errors:")
            for err in batch.errors.data[:10]:
                print(f"  - {err.code}: {err.message}")
        return False

    output_file_id = batch.output_file_id
    if not output_file_id:
        print("ERROR: Batch completed but no output_file_id found")
        return False

    print(f"\nDownloading results from output_file_id={output_file_id} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(output_file_id)
    content.write_to_file(output_path)

    # Count lines
    with open(output_path, "r") as f:
        n_lines = sum(1 for _ in f)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved {n_lines} results to {output_path} ({size_mb:.1f} MB)")

    # Also download error file if present
    if batch.error_file_id:
        error_path = output_path.replace(".jsonl", "_errors.jsonl")
        error_content = client.files.content(batch.error_file_id)
        error_content.write_to_file(error_path)
        print(f"Error file saved to: {error_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Submit and monitor OpenAI Batch API job for GPT-5 evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(INFERENCE_DIR / "gpt5_batch_requests.jsonl"),
        help="Input JSONL file (from gpt5_batch_prepare.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(INFERENCE_DIR / "gpt5_batch_results.jsonl"),
        help="Output JSONL file for batch results",
    )
    parser.add_argument(
        "--check",
        type=str,
        default=None,
        help="Resume monitoring an existing batch by ID (skip upload + create)",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=60,
        help="Seconds between status checks (default: 60)",
    )
    args = parser.parse_args()

    # Verify API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    client = OpenAI()

    if args.check:
        # Resume monitoring an existing batch
        batch_id = args.check
        print(f"Resuming monitoring of batch {batch_id}")
    else:
        # Full flow: upload → create → poll
        if not Path(args.input).exists():
            print(f"ERROR: Input file not found: {args.input}")
            print("Run gpt5_batch_prepare.py first")
            sys.exit(1)

        file_id = upload_file(client, args.input)
        batch_id = create_batch(client, file_id)

    # Poll until done
    batch = poll_batch(client, batch_id, poll_interval=args.poll_interval)

    # Download results
    success = download_results(client, batch, args.output)

    if success:
        print(f"\nNext step: python inference/gpt5_batch_parse.py --input {args.output}")
    else:
        print(f"\nBatch {batch_id} did not complete successfully.")
        print(f"Check status: python inference/gpt5_batch_submit.py --check {batch_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
