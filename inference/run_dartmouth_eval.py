#!/usr/bin/env python3
"""
Run CRYSTAL evaluation via Dartmouth Chat API.

Supports any vision model available on chat.dartmouth.edu:
  - meta.llama-3.2-11b-vision-instruct
  - qwen.qwen3-vl-32b-instruct-fp8
  - openai_responses.gpt-5.2-chat-latest
  - vertex_ai.gemini-2.5-pro
  - anthropic.claude-sonnet-4-6
  - etc.

Usage:
    python inference/run_dartmouth_eval.py --model meta.llama-3.2-11b-vision-instruct
    python inference/run_dartmouth_eval.py --model qwen.qwen3-vl-32b-instruct-fp8 --workers 4
    python inference/run_dartmouth_eval.py --model meta.llama-3.2-11b-vision-instruct --resume
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
from io import BytesIO
from pathlib import Path
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
from PIL import Image
from tqdm import tqdm

INFERENCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INFERENCE_DIR.parent

# Load prompt template
PROMPT_TEMPLATE = (INFERENCE_DIR / "prompt_template.txt").read_text(encoding="utf-8")

# Dartmouth Chat API
DARTMOUTH_API_URL = "https://chat.dartmouth.edu/api/chat/completions"


class BudgetExhaustedError(Exception):
    """Raised when Dartmouth API reports budget exceeded."""
    pass


def check_budget(api_key: str, model: str = "vertex_ai.gemini-2.5-flash") -> dict:
    """Check remaining budget by making a minimal API call.
    Returns {'ok': True/False, 'spend': float, 'budget': float, 'remaining': float}."""
    resp = requests.post(
        DARTMOUTH_API_URL,
        headers={"Authorization": f"bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 1, "stream": False},
        timeout=30,
    )
    if resp.status_code == 200:
        return {"ok": True, "message": "Credits available"}
    if resp.status_code == 400:
        try:
            err = resp.json().get("error", {})
            msg = err.get("message", "")
            if "budget" in msg.lower() or err.get("type") == "budget_exceeded":
                import re
                spend_m = re.search(r'Spend=([\d.]+)', msg)
                budget_m = re.search(r'Budget=([\d.]+)', msg)
                spend = float(spend_m.group(1)) if spend_m else 0
                budget = float(budget_m.group(1)) if budget_m else 0
                return {"ok": False, "spend": spend, "budget": budget,
                        "remaining": budget - spend, "message": msg}
        except Exception:
            pass
    return {"ok": True, "message": f"HTTP {resp.status_code} (assuming OK)"}


def load_api_key(key_file: str) -> str:
    path = Path(key_file)
    if not path.exists():
        print(f"ERROR: API key file not found: {key_file}")
        sys.exit(1)
    return path.read_text().strip()


def load_dataset_arrow(dataset_path: str) -> pa.Table:
    arrow_files = sorted(glob(f"{dataset_path}/data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow files found in {dataset_path}")
    tables = []
    for f in arrow_files:
        stream = pa.ipc.open_stream(f)
        tables.append(stream.read_all())
    return pa.concat_tables(tables)


def encode_image_base64(image_bytes: bytes, max_dim: int = 1024,
                        jpeg_quality: int = 85) -> str:
    pil_img = Image.open(BytesIO(image_bytes))
    if max(pil_img.size) > max_dim:
        pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def format_user_question(question: str, options, choices) -> str:
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


def parse_and_validate_json(content: str):
    """Parse JSON from model output. Returns (prediction, is_valid, error)."""
    if not content or not content.strip():
        return {"reasoning_steps": [], "answer": "insufficient information"}, False, "empty"

    text = content.strip()

    # Strip <details> tags (Dartmouth wraps reasoning in HTML tags for some models)
    import re
    text = re.sub(r'<details[^>]*>.*?</details>', '', text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        d = json.loads(text)
        if isinstance(d, dict) and "reasoning_steps" in d and "answer" in d:
            steps = d["reasoning_steps"]
            answer = d["answer"]
            if isinstance(steps, list) and isinstance(answer, str) and answer.strip():
                d["reasoning_steps"] = [s.strip() for s in steps if isinstance(s, str) and s.strip()]
                d["answer"] = answer.strip()
                return d, True, None
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from text
    try:
        start = text.index("{")
        depth = 0
        end = start
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = text[start:end]
        # Fix common issues
        candidate = candidate.replace("'", '"')
        candidate = re.sub(r',\s*}', '}', candidate)
        candidate = re.sub(r',\s*]', ']', candidate)
        d = json.loads(candidate)
        if isinstance(d, dict) and "reasoning_steps" in d and "answer" in d:
            steps = d["reasoning_steps"]
            answer = d["answer"]
            if isinstance(steps, list) and isinstance(answer, str):
                d["reasoning_steps"] = [s.strip() for s in steps if isinstance(s, str) and s.strip()]
                d["answer"] = answer.strip() if answer.strip() else "insufficient information"
                return d, True, None
    except (ValueError, json.JSONDecodeError):
        pass

    # Fallback: if model returned just the answer without JSON (common with smaller models)
    clean = text.strip().rstrip(".")
    if clean and len(clean) < 200 and "{" not in clean:
        return {"reasoning_steps": [], "answer": clean}, True, None

    return {"reasoning_steps": [], "answer": "insufficient information"}, False, f"parse error"


# Few-shot example to help smaller models produce structured JSON with reasoning
FEW_SHOT_USER = (
    "Question: What is the smallest object?\n"
    "A) The left object  B) The middle object  C) The right object\n\n"
    "Respond as JSON with reasoning_steps (at least 5 visual observations about the image) and answer."
)
FEW_SHOT_ASSISTANT = json.dumps({
    "reasoning_steps": [
        "The image shows three objects placed on a flat surface.",
        "The leftmost object is tall and dark colored.",
        "The middle object is shorter and lighter in color.",
        "The rightmost object is medium-sized and dark.",
        "Comparing all three, the middle object is clearly the shortest and narrowest.",
    ],
    "answer": "B"
})


def call_dartmouth(api_key: str, model: str, image_data_url: str,
                   prompt_text: str, max_tokens: int,
                   use_few_shot: bool = True,
                   disable_thinking: bool = False,
                   max_retries: int = 3, retry_delay: float = 5.0) -> str:
    """Call Dartmouth Chat API with retries and optional few-shot."""
    headers = {
        "Authorization": f"bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []

    # Few-shot example (helps smaller models follow JSON format)
    if use_few_shot:
        messages.append({"role": "user", "content": FEW_SHOT_USER})
        messages.append({"role": "assistant", "content": FEW_SHOT_ASSISTANT})

    # Actual request with image
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": prompt_text},
        ],
    })

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if disable_thinking:
        payload["thinking"] = {"type": "disabled", "budget_tokens": 0}

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                DARTMOUTH_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 400:
                err_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                err_msg = err_data.get("error", {}).get("message", "")
                err_type = err_data.get("error", {}).get("type", "")
                if "budget" in err_msg.lower() or err_type == "budget_exceeded":
                    raise BudgetExhaustedError(f"BUDGET EXHAUSTED: {err_msg}")
                print(f"  HTTP 400: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return ""
            elif resp.status_code == 429:
                wait = retry_delay * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return ""
        except BudgetExhaustedError:
            raise
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return ""
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return ""

    return ""


def process_sample(args_tuple):
    """Process a single sample. Used by ThreadPoolExecutor."""
    idx, api_key, model, image_bytes, question, options, choices, max_tokens, pred_file, raw_log_file, use_few_shot, disable_thinking = args_tuple

    # Encode image
    image_data_url = encode_image_base64(image_bytes)

    # Format prompt
    user_question = format_user_question(question, options, choices)
    prompt_text = PROMPT_TEMPLATE.replace("{USER_INSTRUCTION}", user_question)

    # Call API
    content = call_dartmouth(api_key, model, image_data_url, prompt_text, max_tokens, use_few_shot=use_few_shot, disable_thinking=disable_thinking)

    # Log raw response
    if raw_log_file:
        raw_entry = {"idx": idx, "raw_response": content, "question": question[:100]}
        with open(raw_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(raw_entry, ensure_ascii=False) + "\n")

    # Parse response
    prediction, is_valid, error = parse_and_validate_json(content)

    # Save
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2, ensure_ascii=False)

    return idx, is_valid, len(prediction.get("reasoning_steps", []))


def main():
    parser = argparse.ArgumentParser(
        description="Run CRYSTAL evaluation via Dartmouth Chat API"
    )
    parser.add_argument(
        "--model", type=str, default="meta.llama-3.2-11b-vision-instruct",
        help="Dartmouth model ID",
    )
    parser.add_argument(
        "--key_file", type=str, default=str(PROJECT_ROOT / "chart_dt.txt"),
        help="Path to API key file",
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: final_table/outputs_testing_<model_short>)",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent workers (be gentle with Dartmouth servers)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples that already have predictions")
    parser.add_argument("--no_few_shot", action="store_true",
                        help="Disable few-shot example (for large models that follow instructions)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests in seconds")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of samples (0 = all, useful for testing)")
    parser.add_argument("--disable_thinking", action="store_true",
                        help="Disable thinking/reasoning tokens (for Gemini 2.5 models). Saves cost and speeds up inference.")
    parser.add_argument("--index_file", type=str, default=None,
                        help="JSON file with list of indices to process (for sharding across keys)")
    args = parser.parse_args()

    # Derive short model name for output directory
    model_short = args.model.split(".")[-1].replace("-instruct", "").replace("-fp8", "")
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "final_table" / f"outputs_testing_{model_short}")

    predictions_dir = Path(args.output_dir) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    api_key = load_api_key(args.key_file)

    print("=" * 60)
    print(f"CRYSTAL Evaluation — Dartmouth Chat API")
    print(f"=" * 60)
    use_few_shot = not args.no_few_shot
    print(f"Model:      {args.model}")
    print(f"Output:     {args.output_dir}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Workers:    {args.workers}")
    print(f"Resume:     {args.resume}")
    print(f"Few-shot:   {use_few_shot}")
    print(f"Thinking:   {'disabled' if args.disable_thinking else 'enabled'}")
    print()

    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    table = load_dataset_arrow(args.dataset_path)
    total = len(table)
    print(f"Loaded {total} samples")

    # Determine which samples to process
    if args.index_file:
        with open(args.index_file) as f:
            indices = json.load(f)
        print(f"Loaded {len(indices)} indices from {args.index_file}")
    else:
        indices = list(range(total))
    if args.limit > 0:
        indices = indices[:args.limit]
        print(f"Limited to first {args.limit} samples")

    if args.resume:
        existing = set()
        for f in predictions_dir.glob("*.json"):
            try:
                existing.add(int(f.stem))
            except ValueError:
                pass
        before = len(indices)
        indices = [i for i in indices if i not in existing]
        print(f"Resume: skipping {before - len(indices)} existing, {len(indices)} remaining")

    if not indices:
        print("All samples already processed!")
    else:
        print(f"\nProcessing {len(indices)} samples...")

        # Stats
        valid_count = 0
        invalid_count = 0
        total_steps = 0

        # Raw response log for debugging
        raw_log_file = str(Path(args.output_dir) / "raw_responses.jsonl")

        # Pre-flight budget check
        budget_info = check_budget(api_key, args.model)
        if not budget_info["ok"]:
            print(f"\n*** BUDGET EXHAUSTED before starting! ***")
            print(f"    {budget_info['message']}")
            print(f"    Switch API key (--key_file) and use --resume to continue.")
            sys.exit(1)
        print(f"Budget check: OK")

        # Sequential with optional delay (safer for shared infrastructure)
        budget_abort = False
        if args.workers <= 1:
            for idx in tqdm(indices, desc="Evaluating"):
                image_dict = table["image"][idx].as_py()
                question = table["question"][idx].as_py()
                options = table["options"][idx].as_py() if "options" in table.column_names else None
                choices = table["choices"][idx].as_py() if "choices" in table.column_names else None
                pred_file = str(predictions_dir / f"{idx}.json")

                try:
                    _, is_valid, n_steps = process_sample(
                        (idx, api_key, args.model, image_dict["bytes"],
                         question, options, choices, args.max_tokens, pred_file, raw_log_file, use_few_shot, args.disable_thinking)
                    )
                except BudgetExhaustedError as e:
                    print(f"\n*** {e} ***")
                    budget_abort = True
                    break

                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                total_steps += n_steps

                if args.delay > 0:
                    time.sleep(args.delay)
        else:
            # Parallel with ThreadPoolExecutor
            tasks = []
            for idx in indices:
                image_dict = table["image"][idx].as_py()
                question = table["question"][idx].as_py()
                options = table["options"][idx].as_py() if "options" in table.column_names else None
                choices = table["choices"][idx].as_py() if "choices" in table.column_names else None
                pred_file = str(predictions_dir / f"{idx}.json")
                tasks.append(
                    (idx, api_key, args.model, image_dict["bytes"],
                     question, options, choices, args.max_tokens, pred_file, raw_log_file, use_few_shot, args.disable_thinking)
                )

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_sample, t): t[0] for t in tasks}
                with tqdm(total=len(futures), desc="Evaluating") as pbar:
                    for future in as_completed(futures):
                        try:
                            _, is_valid, n_steps = future.result()
                            if is_valid:
                                valid_count += 1
                            else:
                                invalid_count += 1
                            total_steps += n_steps
                        except BudgetExhaustedError as e:
                            print(f"\n*** {e} ***")
                            budget_abort = True
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        except Exception as e:
                            invalid_count += 1
                            print(f"  Worker error: {e}")
                        pbar.update(1)

        if budget_abort:
            processed = valid_count + invalid_count
            print(f"\nAborted due to budget exhaustion.")
            print(f"  Completed: {processed}/{len(indices)} before abort")
            print(f"  Use --resume with a new --key_file to continue.")
            sys.exit(1)

        processed = valid_count + invalid_count
        print(f"\nProcessing complete:")
        print(f"  Valid:   {valid_count}/{processed} ({100*valid_count/processed:.1f}%)")
        print(f"  Invalid: {invalid_count}/{processed}")
        print(f"  Avg steps: {total_steps/processed:.1f}")

    # Quick summary over all predictions
    print(f"\n{'=' * 60}")
    print("Computing summary over all predictions...")
    total_preds = 0
    empty = 0
    step_sum = 0
    for f in sorted(predictions_dir.glob("*.json")):
        try:
            int(f.stem)
        except ValueError:
            continue
        with open(f) as fh:
            d = json.load(fh)
        total_preds += 1
        steps = d.get("reasoning_steps", [])
        step_sum += len(steps)
        if not steps:
            empty += 1

    summary = {
        "model": args.model,
        "samples": total_preds,
        "empty_steps": empty,
        "avg_steps": step_sum / total_preds if total_preds else 0,
    }
    summary_file = Path(args.output_dir) / "quick_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Predictions: {total_preds}, Empty: {empty}, Avg steps: {summary['avg_steps']:.1f}")
    print(f"Saved to: {summary_file}")
    print(f"\nTo run full Match F1 evaluation:")
    print(f"  python inference/gpt5_batch_parse.py --input /dev/null --parse_only")
    print(f"  # Or use compute_metrics.py pointed at {args.output_dir}/predictions")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
