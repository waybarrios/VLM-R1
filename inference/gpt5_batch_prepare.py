#!/usr/bin/env python3
"""
GPT-5 Batch API — Step 1: Prepare batch JSONL requests

Loads CRYSTAL dataset (via pyarrow for schema compatibility), encodes
images as base64, formats questions with the exact system prompt from
run_simple_vqa.py, and writes OpenAI Batch API JSONL format.

Usage:
    python inference/gpt5_batch_prepare.py
    python inference/gpt5_batch_prepare.py --dataset_path /path/to/dataset --output inference/gpt5_batch_requests.jsonl
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


def encode_image_base64(image_bytes: bytes, max_dim: int = 1024,
                        jpeg_quality: int = 85) -> str:
    """Encode raw image bytes to base64 data URL for OpenAI vision API.

    Resizes to max_dim and encodes as JPEG to reduce JSONL size (~77% smaller
    than PNG).  OpenAI bills by tiles, not bytes, so this does not affect
    token cost but drastically speeds up upload.
    """
    pil_img = Image.open(BytesIO(image_bytes))
    # Resize if larger than max_dim on any side
    if max(pil_img.size) > max_dim:
        pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    # RGBA → RGB (JPEG doesn't support alpha)
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def format_user_question(question: str, options, choices) -> str:
    """
    Format question + options exactly as run_simple_vqa.py does.

    In CRYSTAL, options are typically embedded in the question text already
    (options/choices columns are None). This handles both cases.
    """
    if choices:
        formatted_options = "\n".join(
            [f"{chr(65+i)}) {c}" for i, c in enumerate(choices)]
        )
    elif options:
        formatted_options = "\n".join(
            [f"{chr(65+i)}) {c}" for i, c in enumerate(options)]
        )
    else:
        formatted_options = ""

    return f"{question}\n\n{formatted_options}".strip()


def build_batch_request(custom_id: str, prompt_text: str,
                        image_data_url: str, model: str, max_tokens: int,
                        reasoning_effort: str | None = None) -> dict:
    """Build a single OpenAI Batch API request line.

    prompt_text is the full PROMPT_TEMPLATE with {USER_INSTRUCTION} already
    filled in.
    """
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            },
        ],
        "max_completion_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare OpenAI Batch API JSONL for GPT-5 evaluation on CRYSTAL"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27",
        help="Path to CRYSTAL dataset (Arrow format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(INFERENCE_DIR / "gpt5_batch_requests.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI model name (default: gpt-5)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens per response (default: 4096)",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort level (default: None = model default). Use 'none' to disable reasoning tokens.",
    )
    args = parser.parse_args()

    # OpenAI Batch API limit: 200 MB per file
    MAX_FILE_BYTES = 190 * 1024 * 1024  # 190 MB (with safety margin)

    # Load dataset via pyarrow (bypasses datasets library schema issues)
    print(f"Loading CRYSTAL dataset from: {args.dataset_path}")
    table = load_dataset_arrow(args.dataset_path)
    total = len(table)
    print(f"Loaded {total} samples")
    print(f"Columns: {table.column_names}")

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write batch JSONL, splitting into multiple files if > 190 MB
    print(f"\nWriting batch requests (splitting at {MAX_FILE_BYTES // (1024*1024)} MB per file)")
    print(f"Model: {args.model}, max_tokens: {args.max_tokens}, reasoning_effort: {args.reasoning_effort}")

    part = 0
    current_bytes = 0
    current_lines = 0
    output_files = []

    def get_output_path(part_num):
        if part_num == 0:
            return output_path
        stem = output_path.stem
        return output_path.with_name(f"{stem}_part{part_num}{output_path.suffix}")

    current_file_path = get_output_path(0)
    output_files.append(current_file_path)
    f = open(current_file_path, "w", encoding="utf-8")

    try:
        for idx in tqdm(range(total), desc="Encoding samples"):
            # Extract fields from Arrow table
            image_dict = table["image"][idx].as_py()
            question = table["question"][idx].as_py()
            options = table["options"][idx].as_py() if "options" in table.column_names else None
            choices = table["choices"][idx].as_py() if "choices" in table.column_names else None

            # Encode image as base64 (image stored as {"bytes": ..., "path": ...})
            image_data_url = encode_image_base64(image_dict["bytes"])

            # Format question with options and fill into prompt template
            user_question = format_user_question(question, options, choices)
            prompt_text = PROMPT_TEMPLATE.replace("{USER_INSTRUCTION}", user_question)

            # Build batch request
            request = build_batch_request(
                custom_id=f"sample-{idx}",
                prompt_text=prompt_text,
                image_data_url=image_data_url,
                model=args.model,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
            )

            line = json.dumps(request, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))

            # Check if we need to start a new file
            if current_bytes + line_bytes > MAX_FILE_BYTES and current_lines > 0:
                f.close()
                size_mb = current_file_path.stat().st_size / (1024 * 1024)
                print(f"\n  Part {part}: {current_lines} requests ({size_mb:.1f} MB)")
                part += 1
                current_file_path = get_output_path(part)
                output_files.append(current_file_path)
                f = open(current_file_path, "w", encoding="utf-8")
                current_bytes = 0
                current_lines = 0

            f.write(line)
            current_bytes += line_bytes
            current_lines += 1
    finally:
        f.close()

    # Summary
    print(f"\nDone! Wrote {total} batch requests across {len(output_files)} file(s):")
    for fp in output_files:
        size_mb = fp.stat().st_size / (1024 * 1024)
        # Count lines
        with open(fp) as fh:
            n = sum(1 for _ in fh)
        print(f"  {fp.name}: {n} requests ({size_mb:.1f} MB)")

    if len(output_files) == 1:
        print(f"\nNext step: python inference/gpt5_batch_submit.py --input {output_files[0]}")
    else:
        print(f"\nNext step: Submit each file as a separate batch job:")
        for fp in output_files:
            print(f"  python inference/gpt5_batch_submit.py --input {fp}")


if __name__ == "__main__":
    main()
