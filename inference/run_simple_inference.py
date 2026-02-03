#!/usr/bin/env python3
"""
Fast inference script using HuggingFace directly - NO DeepSpeed.
Uses the same pixel resolution as training for consistency.
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
from transformers import (
    Qwen2_5VLForConditionalGeneration,
    AutoProcessor,
)
from datasets import load_from_disk

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src/open-r1-multimodal/src")
sys.path.insert(0, SRC_DIR)

# Apply Qwen2.5-VL monkey patches
from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn
monkey_patch_qwen2_5vl_flash_attn()
print("✓ Monkey patches applied")


def set_deterministic_mode(seed: int = 42):
    """Configure PyTorch for deterministic generation."""
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


def load_model_and_processor(checkpoint_path: str, device: str = "cuda"):
    """Load model and processor with same pixel configuration as training."""
    print(f"Loading model from: {checkpoint_path}")
    print(f"Target device: {device}")

    # Load model directly with HuggingFace (NO DeepSpeed)
    model = Qwen2_5VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically distribute across available GPUs
        attn_implementation="flash_attention_2",
        trust_remote_code=False,
    )

    # Configure generation for deterministic output
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()

    # Load processor with SAME pixel configuration as training
    # From train_vqa_multi_deepspeed.sh: MAX_PIXELS=602112, MIN_PIXELS=3136
    min_pixels = 3136       # 56*56 (same as training)
    max_pixels = 602112     # Same as training

    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    print(f"✓ Model loaded on: {device}")
    print(f"✓ Pixel configuration: min={min_pixels}, max={max_pixels} (same as training)")
    print("✓ Generation: greedy decoding (deterministic)")

    return model, processor


def run_inference(
    model,
    processor,
    image,
    question: str,
    choices: list = None,
    max_new_tokens: int = 8192,
):
    """Run inference on a single sample."""
    system_prompt = get_system_prompt()

    # Format options/choices if present
    if choices:
        formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(choices)])
        user_question = f"{question}\n\n{formatted_options}".strip()
    else:
        user_question = question

    # Create messages
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_question},
            ],
        },
    ]

    # Apply chat template
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode output
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    raw_output = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Parse and validate JSON
    prediction, is_valid, validation_error = parse_and_validate_json(raw_output)

    if not is_valid:
        prediction = {
            "reasoning_steps": [],
            "answer": "insufficient information"
        }
        error_msg = f"Validation failed: {validation_error}"
    else:
        error_msg = None

    return {
        "prediction": prediction,
        "raw_output": raw_output,
        "is_valid": is_valid,
        "error": error_msg,
    }


def run_batch_inference(
    checkpoint_path: str,
    test_dataset_path: str,
    predictions_dir: str,
    device: str = "cuda",
    start_idx: int = 0,
    end_idx: int = None,
):
    """Run inference on test dataset."""

    print(f"\n{'='*80}")
    print(f"Running inference on checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    # Enable deterministic mode
    set_deterministic_mode(seed=42)

    start_time = time.time()

    # Create predictions directory
    os.makedirs(predictions_dir, exist_ok=True)

    # Load model and processor
    model, processor = load_model_and_processor(checkpoint_path, device)

    # Load dataset
    print(f"Loading test dataset from: {test_dataset_path}")
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

    # Run inference
    print(f"Running inference on samples {start_idx} to {end_idx}...")
    all_results = []
    valid_count = 0
    error_count = 0
    skipped_count = 0

    for idx in tqdm(range(start_idx, end_idx), desc="Inference", ncols=100):
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

        # Extract data
        image = item["image"]
        question = item["question"]
        answer = item["answer"]
        choices = item.get("choices") or item.get("options")

        try:
            # Run inference
            result = run_inference(
                model,
                processor,
                image,
                question,
                choices,
            )

            result["idx"] = idx

            if result["is_valid"]:
                valid_count += 1
            else:
                error_count += 1

            # Save immediately
            prediction_file = os.path.join(predictions_dir, f"{idx}.json")
            with open(prediction_file, "w", encoding='utf-8') as f:
                json.dump(result["prediction"], f, indent=2, ensure_ascii=False)

            all_results.append(result)

        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
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
        "avg_time_per_sample": elapsed_time / total_processed if total_processed > 0 else 0,
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
    parser = argparse.ArgumentParser(description="Fast inference using HuggingFace (NO DeepSpeed)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (default: all)")

    args = parser.parse_args()

    run_batch_inference(
        checkpoint_path=args.checkpoint_dir,
        test_dataset_path=args.test_dataset_path,
        predictions_dir=args.predictions_dir,
        device=args.device,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )


if __name__ == "__main__":
    main()
