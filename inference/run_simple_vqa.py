#!/usr/bin/env python3
"""
Simple VQA inference - Based on test_rec_r1.py
No DeepSpeed, no distributed training, just pure HuggingFace inference
"""

import os
import sys
import json
import re
from pathlib import Path
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


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


def parse_and_validate_json(content: str):
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

    if "reasoning_steps" not in parsed or "answer" not in parsed:
        return None, False, "Missing required keys"

    if not isinstance(parsed["reasoning_steps"], list):
        return None, False, "'reasoning_steps' must be a list"

    if not isinstance(parsed["answer"], str):
        return None, False, "'answer' must be a string"

    for idx, step in enumerate(parsed["reasoning_steps"]):
        if not isinstance(step, str):
            return None, False, f"'reasoning_steps[{idx}]' must be a string"

    reasoning_steps = [s.strip() for s in parsed["reasoning_steps"] if s and s.strip()]

    if not reasoning_steps:
        return None, False, "'reasoning_steps' is empty"

    return {
        "reasoning_steps": reasoning_steps,
        "answer": parsed["answer"].strip()
    }, True, None


def main():
    parser = argparse.ArgumentParser(description="Simple VQA inference (no DeepSpeed, no distributed)")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_dataset_path", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use (default: 0)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    args = parser.parse_args()

    # Set GPU via environment variable (device_map="auto" will use it)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU: {args.gpu}")

    # Enable deterministic mode
    set_deterministic_mode(seed=42)

    start_time = time.time()

    # Create predictions directory
    os.makedirs(args.predictions_dir, exist_ok=True)

    # Load model - Following official Qwen2.5-VL pattern
    print(f"\nLoading model from: {args.checkpoint_dir}")

    # We recommend enabling flash_attention_2 for better acceleration and memory saving
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # Default processor (uses checkpoint's min_pixels/max_pixels settings)
    processor = AutoProcessor.from_pretrained(args.checkpoint_dir)

    print(f"✓ Model and processor loaded")

    # Load dataset
    print(f"\nLoading test dataset from: {args.test_dataset_path}")
    dataset = load_from_disk(args.test_dataset_path)
    print(f"Loaded {len(dataset)} test samples\n")

    # Get system prompt
    system_prompt = get_system_prompt()

    # Check for existing predictions
    existing_predictions = set()
    if os.path.exists(args.predictions_dir):
        for f in os.listdir(args.predictions_dir):
            if f.endswith('.json') and f not in ['inference_summary.json', 'inference_stats.json']:
                try:
                    idx = int(f.replace('.json', ''))
                    existing_predictions.add(idx)
                except ValueError:
                    pass

    if existing_predictions:
        print(f"Found {len(existing_predictions)} existing predictions - will skip these\n")

    total_samples = len(dataset)
    print(f"Total samples: {total_samples}, Already done: {len(existing_predictions)}")
    print(f"Will process: {total_samples - len(existing_predictions)} samples\n")

    # Run inference - process on-the-fly instead of pre-loading all messages
    valid_count = 0
    error_count = 0

    for idx in tqdm(range(total_samples), desc="Inference"):
        # Skip if already exists
        if idx in existing_predictions:
            continue

        # Get item
        item = dataset[idx]

        # Format options/choices
        if item.get("choices"):
            formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item["choices"])])
        elif item.get("options"):
            formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item["options"])])
        else:
            formatted_options = ""

        user_question = f"{item['question']}\n\n{formatted_options}".strip()

        # Create message
        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": user_question},
                ],
            },
        ]

        batch_messages = [message]
        batch_indices = [idx]

        # Preparation for inference - EXACTLY like test_rec_r1.py
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Debug: Print before generation
        if idx % 100 == 0:
            print(f"\n[DEBUG] Sample {idx}: Starting generation...")
            sys.stdout.flush()

        # Inference: Generation of the output (EXACTLY like test_rec_r1.py)
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)

        # Debug: Print after generation
        if idx % 100 == 0:
            print(f"[DEBUG] Sample {idx}: Generation completed")
            sys.stdout.flush()

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Process outputs
        for idx, raw_output in zip(batch_indices, batch_output_text):
            prediction, is_valid, validation_error = parse_and_validate_json(raw_output)

            if not is_valid:
                prediction = {
                    "reasoning_steps": [],
                    "answer": "insufficient information"
                }
                error_count += 1
            else:
                valid_count += 1

            # Save prediction
            prediction_file = os.path.join(args.predictions_dir, f"{idx}.json")
            with open(prediction_file, "w", encoding='utf-8') as f:
                json.dump(prediction, f, indent=2, ensure_ascii=False)

    # Statistics
    elapsed_time = time.time() - start_time
    total_processed = valid_count + error_count
    valid_rate = valid_count / total_processed if total_processed > 0 else 0

    print(f"\n{'='*80}")
    print(f"✓ Inference completed!")
    print(f"{'='*80}")
    print(f"\n📁 Processed: {total_processed} samples")
    print(f"✅ Valid: {valid_count} ({valid_rate*100:.1f}%)")
    print(f"❌ Invalid: {error_count} ({(1-valid_rate)*100:.1f}%)")
    print(f"⏱️  Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    if total_processed > 0:
        print(f"⚡ Speed: {elapsed_time/total_processed:.2f}s per sample")
    print(f"💾 Saved to: {args.predictions_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
