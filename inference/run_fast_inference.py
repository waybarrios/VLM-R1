#!/usr/bin/env python3
"""
Fast inference script for VQA - Follows the pattern from test_rec_r1.py
Uses HuggingFace directly (NO DeepSpeed) with qwen_vl_utils
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, Tuple
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src/open-r1-multimodal/src")
sys.path.insert(0, SRC_DIR)

# Apply Qwen2.5-VL monkey patches
from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn
monkey_patch_qwen2_5vl_flash_attn()
print("✓ Monkey patches applied")


def set_deterministic_mode(seed: int = 42):
    """Configure for deterministic generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Deterministic mode enabled (seed={seed})")


def get_system_prompt() -> str:
    """Returns the exact system prompt used during training."""
    return """You are a vision-language model. First, analyze the provided image(s) and any user text silently. Do NOT reveal your internal reasoning.

Return ONLY a single, valid JSON object with this exact schema:
{"reasoning_steps": [], "answer": ""}

Rules for "reasoning_steps":
- Decide the number of steps based on task complexity; include enough to make the answer evident without filler.
- Include some inference from visual information, always anchored to visible cues.
- Write single-clause sentences, each adding a new, directly checkable fact or cue-based inference.
- You may include cautious, visually grounded commonsense using words such as "appears", "suggests", or "likely", but always anchor it to visible cues (lighting/shadows; perspective/vanishing lines/horizon/tilt; scale/relative size; focus/DOF; parallax; occlusion/contact shadows; reflections/transparency; material/texture; symmetry/patterns/alignment; position/orientation/foreground–background; density/motion cues; human pose/gaze/gesture; interactions/affordances; object state; physics plausibility; signage/text/logos/typography; numbers/units; plots/charts: type, axes/ticks/units, scale (lin/log), legend↔series, gridlines/baseline, error bars/CI, trendlines, outliers/binning, colorbar; maps: scale bar, north arrow; math/geometry: labels/givens, unit checks, angle rules, Pythagorean, distance/slope, transformations, area/volume, circle theorems, trig (incl. sine/cosine laws), vectors, systems/quadratics, combinatorics, logs/exponents, probability/statistics, exact forms, conversions, plots, graphs, math equations, diagrams).
- Keep each step ≤14 words. No multi-sentence items. No chains like "because/therefore". No internal monologue.

Rules for "answer":
- Provide the final answer grounded strictly in visible content and given text.
- If information is missing or ambiguous, set "answer" to "insufficient information" and include steps noting what is missing (e.g., "Noted the license plate is unreadable due to blur.").
- Multiple-choice: if options have letters, return only the single best LETTER (e.g., "B"); if unlabeled, return the exact option text verbatim.
- Numeric: include required units; obey requested rounding; otherwise give exact/simplest form.

What to notice in steps (express as sentences, not labels):
- Objects & attributes (classes, colors, materials, states), logos/brands if clearly visible.
- Positions & spatial relations (left/right/above/below/front/behind, proximity, alignment, orientation, foreground/background).
- Depth cues (relative size, position in frame vs. horizon, sharpness/detail, shadow contact, occlusion order).
- Scene & lighting/time cues (indoor/outdoor, daylight vs. night, weather indications, activity/no-activity).
- Occlusion effects and how they affect certainty.
- Text/OCR with exact casing/punctuation ("Read text: 'SPEED LIMIT 25'. ").
- Counts & quantities for distinct instances; approximate only if visually justified.
- Graphics/plots/diagrams: axes, ticks, units, legends; read exact values rather than guessing.

Output formatting:
- Output only the JSON object. No extra keys, comments, code fences, or prose.
- Use double quotes for all strings; no trailing commas; any valid JSON whitespace is acceptable."""


def parse_and_validate_json(content: str) -> Tuple[Dict[str, Any], bool, str]:
    """Parse and validate JSON from model output."""
    # Remove markdown code blocks
    content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

    json_str = None
    parsed = None

    # Method 1: Try to parse entire content as JSON
    try:
        parsed = json.loads(content_cleaned)
        json_str = content_cleaned
    except (json.JSONDecodeError, Exception):
        pass

    # Method 2: Extract outermost { } pair
    if parsed is None:
        first_brace = content_cleaned.find('{')
        if first_brace != -1:
            brace_count = 0
            for idx in range(first_brace, len(content_cleaned)):
                if content_cleaned[idx] == '{':
                    brace_count += 1
                elif content_cleaned[idx] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = content_cleaned[first_brace:idx+1]
                        break

    if json_str:
        # Fix common JSON issues
        json_str = json_str.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
        json_str = re.sub(r'''(?<=[:,\[])\s*'([^']*)'(?=\s*[,\]\}])''', r' "\1"', json_str)
        json_str = re.sub(r'"\s*\n\s*"', '",\n    "', json_str)
        json_str = re.sub(r'"\s+(?=")', '", ', json_str)
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

        # Try parsing
        if parsed is None:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as e:
                return None, False, f"JSON parse error: {str(e)}"

    # Validate structure
    if parsed is None:
        return None, False, "No valid JSON found"

    if not isinstance(parsed, dict):
        return None, False, "JSON is not an object"

    # Check required keys
    if "reasoning_steps" not in parsed:
        return None, False, "Missing 'reasoning_steps' key"

    if "answer" not in parsed:
        return None, False, "Missing 'answer' key"

    # Validate types
    if not isinstance(parsed["reasoning_steps"], list):
        return None, False, "'reasoning_steps' must be a list"

    if not isinstance(parsed["answer"], str):
        return None, False, "'answer' must be a string"

    # Validate that all items in reasoning_steps are strings
    for idx, step in enumerate(parsed["reasoning_steps"]):
        if not isinstance(step, str):
            return None, False, f"'reasoning_steps[{idx}]' must be a string, not {type(step).__name__}"

    # Clean reasoning steps
    reasoning_steps = [s.strip() for s in parsed["reasoning_steps"] if s and s.strip()]

    # Require at least 1 valid reasoning step
    if not reasoning_steps:
        return None, False, "'reasoning_steps' is empty or contains no valid strings"

    # Return validated data
    return {
        "reasoning_steps": reasoning_steps,
        "answer": parsed["answer"].strip()
    }, True, None


def run_batch_inference(
    checkpoint_path: str,
    test_dataset_path: str,
    predictions_dir: str,
    device: str = "cuda",
    batch_size: int = 2,
    start_idx: int = 0,
    end_idx: int = None,
):
    """Run inference on test dataset - following test_rec_r1.py pattern."""

    print(f"\n{'='*80}")
    print(f"Running fast inference on checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    # Enable deterministic mode
    set_deterministic_mode(seed=42)

    start_time = time.time()

    # Create predictions directory
    os.makedirs(predictions_dir, exist_ok=True)

    # Load model - use torch.device("cuda") which uses the GPU set by CUDA_VISIBLE_DEVICES
    print(f"Loading model from: {checkpoint_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to("cuda")  # Will use the GPU from CUDA_VISIBLE_DEVICES
    model.eval()

    # Load processor - NO manual min_pixels/max_pixels (uses checkpoint defaults)
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    print(f"✓ Model loaded on: {torch.cuda.current_device()} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    print("✓ Processor loaded (using checkpoint defaults)")
    print("✓ Generation: greedy decoding (deterministic)")

    # Load dataset
    print(f"\nLoading test dataset from: {test_dataset_path}")
    hf_dataset = load_from_disk(test_dataset_path)
    print(f"Loaded {len(hf_dataset)} test samples\n")

    # Determine range
    if end_idx is None:
        end_idx = len(hf_dataset)
    end_idx = min(end_idx, len(hf_dataset))

    # Check for existing predictions
    print(f"Checking for existing predictions...")
    existing_predictions = set()
    if os.path.exists(predictions_dir):
        for f in os.listdir(predictions_dir):
            if f.endswith('.json') and f not in ['inference_summary.json', 'inference_stats.json']:
                try:
                    idx = int(f.replace('.json', ''))
                    existing_predictions.add(idx)
                except ValueError:
                    pass

    if existing_predictions:
        print(f"Found {len(existing_predictions)} existing predictions - will skip these")
    else:
        print(f"No existing predictions found")

    # Prepare system prompt
    system_prompt = get_system_prompt()

    # Run inference
    print(f"\nRunning inference on samples {start_idx} to {end_idx} (batch_size={batch_size})...")
    all_results = []
    valid_count = 0
    error_count = 0
    skipped_count = 0

    # Process in batches
    for batch_start in tqdm(range(start_idx, end_idx, batch_size), desc="Inference", ncols=100):
        batch_end = min(batch_start + batch_size, end_idx)
        batch_indices = []
        batch_messages = []

        # Prepare batch messages
        for idx in range(batch_start, batch_end):
            # Check if prediction already exists
            if idx in existing_predictions:
                skipped_count += 1
                prediction_file = os.path.join(predictions_dir, f"{idx}.json")
                try:
                    with open(prediction_file, "r", encoding='utf-8') as f:
                        existing_pred = json.load(f)
                    all_results.append({
                        "idx": idx,
                        "prediction": existing_pred,
                        "raw_output": "[skipped - already exists]",
                        "is_valid": True,
                        "error": None,
                    })
                    valid_count += 1
                except Exception as e:
                    print(f"Warning: Could not load existing file {prediction_file}: {e}")
                continue

            # Get sample
            item = hf_dataset[idx]

            # Format options/choices if present - EXACTLY like training (grpo_rec.py lines 269-277)
            if item.get("choices"):
                formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item["choices"])])
            elif item.get("options"):
                formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item["options"])])
            else:
                formatted_options = ""

            # Format user question with options/choices if they exist
            user_question = f"{item['question']}\n\n{formatted_options}".strip()

            # Create messages - EXACTLY like training (grpo_rec.py lines 317-331)
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_question},
                    ],
                },
            ]

            batch_indices.append(idx)
            batch_messages.append(messages)
            # Store image separately for process_vision_info
            batch_messages[-1][1]["content"][0]["image"] = item["image"]

        # Skip if all samples were already processed
        if not batch_messages:
            continue

        try:
            # Preparation for inference - EXACTLY like test_rec_r1.py
            text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in batch_messages]

            # Use process_vision_info - KEY STEP from test_rec_r1.py
            image_inputs, video_inputs = process_vision_info(batch_messages)

            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            # Inference: Generation of the output - EXACTLY like test_rec_r1.py
            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=8192,
                do_sample=False  # Greedy decoding
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Process outputs
            for idx, raw_output in zip(batch_indices, batch_output_text):
                try:
                    # Parse and validate JSON
                    prediction, is_valid, validation_error = parse_and_validate_json(raw_output)

                    if not is_valid:
                        prediction = {
                            "reasoning_steps": [],
                            "answer": "insufficient information"
                        }
                        error_msg = f"Validation failed: {validation_error}"
                        error_count += 1
                    else:
                        error_msg = None
                        valid_count += 1

                    result = {
                        "idx": idx,
                        "prediction": prediction,
                        "raw_output": raw_output,
                        "is_valid": is_valid,
                        "error": error_msg,
                    }

                    # Save immediately
                    prediction_file = os.path.join(predictions_dir, f"{idx}.json")
                    with open(prediction_file, "w", encoding='utf-8') as f:
                        json.dump(prediction, f, indent=2, ensure_ascii=False)

                    all_results.append(result)

                except Exception as e:
                    error_msg = f"Processing error: {str(e)}"
                    result = {
                        "idx": idx,
                        "prediction": {
                            "reasoning_steps": [],
                            "answer": "insufficient information"
                        },
                        "raw_output": raw_output,
                        "is_valid": False,
                        "error": error_msg,
                    }
                    all_results.append(result)
                    error_count += 1

                    # Save error prediction
                    prediction_file = os.path.join(predictions_dir, f"{idx}.json")
                    with open(prediction_file, "w", encoding='utf-8') as f:
                        json.dump(result["prediction"], f, indent=2, ensure_ascii=False)

        except Exception as e:
            # Batch processing error
            print(f"\nBatch error at {batch_start}-{batch_end}: {str(e)}")
            for idx in batch_indices:
                error_msg = f"Batch generation error: {str(e)}"
                result = {
                    "idx": idx,
                    "prediction": {
                        "reasoning_steps": [],
                        "answer": "insufficient information"
                    },
                    "raw_output": "",
                    "is_valid": False,
                    "error": error_msg,
                }
                all_results.append(result)
                error_count += 1

                # Save error prediction
                prediction_file = os.path.join(predictions_dir, f"{idx}.json")
                with open(prediction_file, "w", encoding='utf-8') as f:
                    json.dump(result["prediction"], f, indent=2, ensure_ascii=False)

    # Save summary
    summary_file = os.path.join(predictions_dir, "inference_summary.json")
    with open(summary_file, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Statistics
    elapsed_time = time.time() - start_time
    total_processed = len(all_results)
    valid_rate = valid_count / total_processed if total_processed > 0 else 0

    stats = {
        "checkpoint": checkpoint_path,
        "total_samples": end_idx - start_idx,
        "processed_samples": total_processed,
        "skipped_samples": skipped_count,
        "newly_generated": total_processed - skipped_count,
        "valid_predictions": valid_count,
        "invalid_predictions": error_count,
        "validation_rate": valid_rate,
        "elapsed_time_seconds": elapsed_time,
        "avg_time_per_sample": elapsed_time / (total_processed - skipped_count) if (total_processed - skipped_count) > 0 else 0,
    }

    stats_file = os.path.join(predictions_dir, "inference_stats.json")
    with open(stats_file, "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"✓ Inference completed!")
    print(f"{'='*80}")
    print(f"\n📁 Processed: {total_processed} samples")
    print(f"⏭️  Skipped: {skipped_count} (already existed)")
    print(f"✅ Valid: {valid_count} ({valid_rate*100:.1f}%)")
    print(f"❌ Invalid: {error_count} ({(1-valid_rate)*100:.1f}%)")
    print(f"⏱️  Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    if total_processed - skipped_count > 0:
        print(f"⚡ Speed: {elapsed_time/(total_processed - skipped_count):.2f}s per sample")
    print(f"💾 Saved to: {predictions_dir}")
    print(f"{'='*80}\n")

    # Clean up
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Fast VQA inference using HuggingFace (pattern from test_rec_r1.py)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--chunk_total", type=int, default=None, help="Total number of chunks (for parallel processing)")
    parser.add_argument("--chunk_index", type=int, default=None, help="Index of this chunk (0-based)")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (default: all)")

    args = parser.parse_args()

    # Calculate start and end indices for chunking
    start_idx = args.start_idx
    end_idx = args.end_idx

    if args.chunk_total is not None and args.chunk_index is not None:
        # Load dataset to get total size
        from datasets import load_from_disk
        dataset = load_from_disk(args.test_dataset_path)
        total_samples = len(dataset)

        # Calculate chunk size
        chunk_size = (total_samples + args.chunk_total - 1) // args.chunk_total
        start_idx = args.chunk_index * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)

        print(f"Chunk {args.chunk_index + 1}/{args.chunk_total}: processing samples {start_idx} to {end_idx-1} (total: {total_samples})")

    run_batch_inference(
        checkpoint_path=args.checkpoint_dir,
        test_dataset_path=args.test_dataset_path,
        predictions_dir=args.predictions_dir,
        device=args.device,
        batch_size=args.batch_size,
        start_idx=start_idx,
        end_idx=end_idx,
    )


if __name__ == "__main__":
    main()
