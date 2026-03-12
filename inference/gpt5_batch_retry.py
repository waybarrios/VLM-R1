#!/usr/bin/env python3
"""
GPT-5 Batch API — Retry empty responses with higher max_completion_tokens.

Reads retry indices JSON, generates JSONL only for those samples, submits,
polls, downloads, and merges results back.

Usage:
    python inference/gpt5_batch_retry.py                              # 431 empties, 12288 tokens
    python inference/gpt5_batch_retry.py --max_tokens 16384           # even more tokens
    python inference/gpt5_batch_retry.py --retry_indices inference/gpt5_retry_indices_12k.json
"""

import sys
import json
import base64
import argparse
from io import BytesIO
from pathlib import Path
from glob import glob

import pyarrow as pa
from PIL import Image
from tqdm import tqdm

INFERENCE_DIR = Path(__file__).resolve().parent

# ── Evaluation prompt (loaded from external file for single source of truth) ──
PROMPT_TEMPLATE_FILE = INFERENCE_DIR / "prompt_template.txt"
PROMPT_TEMPLATE = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")


def load_dataset_arrow(dataset_path):
    arrow_files = sorted(glob(f"{dataset_path}/data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow files found in {dataset_path}")
    tables = []
    for f in arrow_files:
        stream = pa.ipc.open_stream(f)
        tables.append(stream.read_all())
    return pa.concat_tables(tables)


def encode_image_base64(image_bytes, max_dim=1024, jpeg_quality=85):
    pil_img = Image.open(BytesIO(image_bytes))
    if max(pil_img.size) > max_dim:
        pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def format_user_question(question, options, choices):
    if choices:
        formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(choices)])
    elif options:
        formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(options)])
    else:
        formatted_options = ""
    return f"{question}\n\n{formatted_options}".strip()


def main():
    parser = argparse.ArgumentParser(description="Retry empty GPT-5 responses with more tokens")
    parser.add_argument("--max_tokens", type=int, default=12288)
    parser.add_argument("--dataset_path", type=str,
                        default="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27")
    parser.add_argument("--retry_indices", type=str,
                        default=str(INFERENCE_DIR / "gpt5_retry_indices_12k.json"))
    parser.add_argument("--model", type=str, default="gpt-5")
    args = parser.parse_args()

    # Load retry indices
    with open(args.retry_indices) as f:
        indices = json.load(f)
    print(f"Retrying {len(indices)} samples with max_completion_tokens={args.max_tokens}")

    # Load dataset
    table = load_dataset_arrow(args.dataset_path)
    print(f"Dataset: {len(table)} samples")

    # Generate JSONL (separate file to preserve previous retry)
    output_jsonl = INFERENCE_DIR / "gpt5_batch_retry_12k.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for idx in tqdm(indices, desc="Encoding retry samples"):
            image_dict = table["image"][idx].as_py()
            question = table["question"][idx].as_py()
            options = table["options"][idx].as_py() if "options" in table.column_names else None
            choices = table["choices"][idx].as_py() if "choices" in table.column_names else None

            image_data_url = encode_image_base64(image_dict["bytes"])
            user_question = format_user_question(question, options, choices)
            prompt_text = PROMPT_TEMPLATE.replace("{USER_INSTRUCTION}", user_question)

            request = {
                "custom_id": f"sample-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                                {"type": "text", "text": prompt_text},
                            ],
                        },
                    ],
                    "max_completion_tokens": args.max_tokens,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")

    size_mb = output_jsonl.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(indices)} requests to {output_jsonl} ({size_mb:.1f} MB)")

    # Submit
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY")
        sys.exit(1)

    from openai import OpenAI
    import time

    client = OpenAI()

    print(f"\nUploading {output_jsonl} ...")
    with open(output_jsonl, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"Uploaded: file_id={file_obj.id} ({file_obj.bytes / 1024 / 1024:.1f} MB)")

    print("Creating batch ...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"GPT-5 CRYSTAL retry ({args.max_tokens} tokens, updated prompt)"},
    )
    print(f"Batch: id={batch.id}, status={batch.status}")

    # Poll
    print(f"Polling every 60s ...")
    while True:
        batch = client.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"[{time.strftime('%H:%M:%S')}] {batch.status} — "
              f"{counts.completed}/{counts.total} done, {counts.failed} failed")
        if batch.status in {"completed", "failed", "expired", "cancelled"}:
            break
        time.sleep(60)

    if batch.status != "completed" or not batch.output_file_id:
        print(f"Batch failed: {batch.status}")
        sys.exit(1)

    # Download
    retry_results = INFERENCE_DIR / "gpt5_batch_retry_12k_results.jsonl"
    content = client.files.content(batch.output_file_id)
    content.write_to_file(str(retry_results))
    print(f"Downloaded results to {retry_results}")

    # Merge into existing predictions
    predictions_dir = Path("/gpudata3/Wayner/VLM-R1/final_table/outputs_testing_gpt5/predictions")
    sys.path.insert(0, str(INFERENCE_DIR))
    from run_simple_vqa import parse_and_validate_json

    updated = 0
    still_empty = 0
    with open(retry_results) as f:
        for line in f:
            result = json.loads(line)
            idx = int(result["custom_id"].split("-")[1])
            content_text = result["response"]["body"]["choices"][0]["message"]["content"]

            if not content_text.strip():
                still_empty += 1
                continue

            prediction, is_valid, _ = parse_and_validate_json(content_text)
            if not is_valid:
                prediction = {"reasoning_steps": [], "answer": "insufficient information"}

            pred_file = predictions_dir / f"{idx}.json"
            with open(pred_file, "w") as pf:
                json.dump(prediction, pf, indent=2, ensure_ascii=False)
            updated += 1

    print(f"\nMerged: {updated} predictions updated, {still_empty} still empty")

    # Recompute summary metrics over all 6372 predictions
    print("\n" + "=" * 60)
    print("Recomputing summary metrics over all predictions...")
    print("=" * 60)

    total = 0
    empty_count = 0
    correct_count = 0
    f1_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    step_sum = 0

    # Quick accuracy check (just answer match, no semantic eval)
    for pred_file in sorted(predictions_dir.glob("*.json")):
        try:
            idx = int(pred_file.stem)
        except ValueError:
            continue
        with open(pred_file) as pf:
            d = json.load(pf)
        total += 1
        steps = d.get("reasoning_steps", [])
        step_sum += len(steps)
        if not steps:
            empty_count += 1

    summary = {
        "samples": total,
        "still_empty": empty_count,
        "pred_steps": step_sum / total if total else 0,
        "retry_updated": updated,
        "retry_still_empty": still_empty,
    }

    summary_file = predictions_dir.parent / "metrics_summary_after_retry_12k.json"
    with open(summary_file, "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"Samples: {total}, Still empty: {empty_count}, Avg steps: {summary['pred_steps']:.1f}")
    print(f"Saved quick summary to {summary_file}")
    print(f"\nTo run full evaluation with Match F1:")
    print(f"  python inference/gpt5_batch_parse.py --parse_only")
    print(f"  # Then: python compute_metrics.py (or re-run gpt5_batch_parse.py without --parse_only)")


if __name__ == "__main__":
    main()
