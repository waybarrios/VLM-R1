#!/usr/bin/env python3
"""
Inference script for DeepSpeed checkpoints - loads from ZeRO-3 format directly
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import time
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)
from datasets import load_from_disk
from tqdm import tqdm
import traceback

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src/open-r1-multimodal/src")
sys.path.insert(0, SRC_DIR)

# Apply Qwen2.5-VL monkey patches
from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn
monkey_patch_qwen2_5vl_flash_attn()
print("✓ Monkey patches applied")


def set_deterministic_mode(seed: int = 42):
    """Configure PyTorch and CUDA for maximum determinism."""
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure PyTorch for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for additional determinism
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Enable deterministic algorithms (may impact performance)
    try:
        torch.use_deterministic_algorithms(True)
        print("✓ Deterministic algorithms enabled")
    except Exception as e:
        print(f"⚠ Could not enable all deterministic algorithms: {e}")
        print("  (This is normal for some operations)")

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
    """Parse and validate JSON from model output (same as training validation)."""
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

    # Validate that all items in reasoning_steps are strings (not objects/dicts)
    for idx, step in enumerate(parsed["reasoning_steps"]):
        if not isinstance(step, str):
            return None, False, f"'reasoning_steps[{idx}]' must be a string, not {type(step).__name__}"

    # Clean and validate reasoning steps
    reasoning_steps = [s.strip() for s in parsed["reasoning_steps"] if s and s.strip()]

    # Require at least 1 valid reasoning step (same as training validation)
    if not reasoning_steps:
        return None, False, "'reasoning_steps' is empty or contains no valid strings"

    # Return validated data
    return {
        "reasoning_steps": reasoning_steps,
        "answer": parsed["answer"].strip()
    }, True, None


class VQATestDataset(Dataset):
    """Dataset wrapper for VQA test data."""

    def __init__(self, hf_dataset, processor, system_prompt: str):
        self.dataset = hf_dataset
        self.processor = processor
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Format options/choices if present
        if item.get("choices"):
            formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item["choices"])])
        elif item.get("options"):
            formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item["options"])])
        else:
            formatted_options = ""

        # Format user question
        user_question = f"{item['question']}\n\n{formatted_options}".strip()

        # Create messages with system prompt
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_question},
                ],
            },
        ]

        return {
            "idx": idx,
            "image": item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "messages": messages,
            "source": item.get("source", "unknown"),
        }


def load_deepspeed_checkpoint(checkpoint_path: str, device_ids: List[int]):
    """Load DeepSpeed checkpoint - safetensors are already available."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Target GPUs: {device_ids}")

    # Set memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set CUDA_VISIBLE_DEVICES to only use specified GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

    # Load directly from safetensors (already converted by training script)
    # Force model to load on single GPU (cuda:0 after CUDA_VISIBLE_DEVICES is set)
    # Using "sdpa" instead of "flash_attention_2" to avoid compatibility issues
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Force all on cuda:0 (which is the visible GPU)
        attn_implementation="sdpa",  # Scaled Dot Product Attention (PyTorch native, more stable)
        trust_remote_code=False,  # Qwen2.5-VL is in transformers now
    )

    # Configure generation config for deterministic generation
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    print("✓ Loaded checkpoint from safetensors")
    print(f"✓ Model loaded on GPUs: {device_ids}")
    print("✓ Generation config set to greedy decoding (deterministic)")
    return model


def run_inference_batch(model, processor, batch, max_new_tokens=8192):
    """Run inference on a batch of samples - processes all samples in parallel.

    Uses greedy decoding (do_sample=False) for fully deterministic generation.
    Combined with seed setting and CUDA deterministic mode, this ensures
    reproducible results across runs.
    """
    results = []

    if len(batch) == 0:
        return results

    # Get model device
    model_device = next(model.parameters()).device

    # Prepare all inputs
    batch_indices = []
    batch_text_prompts = []
    batch_images = []

    for item in batch:
        batch_indices.append(item["idx"])

        # Apply chat template
        text_prompt = processor.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True
        )
        batch_text_prompts.append(text_prompt)
        batch_images.append(item["image"])

    try:
        # Process all images and texts in a single batch
        inputs = processor(
            text=batch_text_prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate for all samples in batch (greedy decoding for determinism)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding (fully deterministic)
                num_beams=1,  # No beam search
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode each output
        input_len = inputs["input_ids"].shape[1]

        for i, (idx, output) in enumerate(zip(batch_indices, output_ids)):
            try:
                generated_ids = output[input_len:]
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

                results.append({
                    "idx": idx,
                    "prediction": prediction,
                    "raw_output": raw_output,
                    "is_valid": is_valid,
                    "error": error_msg,
                })

            except Exception as e:
                error_msg = f"Decode error: {str(e)}"
                results.append({
                    "idx": idx,
                    "prediction": {
                        "reasoning_steps": [],
                        "answer": "insufficient information"
                    },
                    "raw_output": "",
                    "is_valid": False,
                    "error": error_msg,
                })

    except Exception as e:
        # If batch processing fails, fall back to processing individually
        error_msg = f"Batch generation error: {str(e)}"
        print(f"  Batch error, falling back to individual processing: {error_msg}")

        for item in batch:
            idx = item["idx"]
            try:
                messages = item["messages"]
                image = item["image"]

                text_prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = processor(
                    text=[text_prompt],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )

                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                # Fallback to individual generation (greedy decoding for determinism)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # Greedy decoding (fully deterministic)
                        num_beams=1,  # No beam search
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                    )

                input_len = inputs["input_ids"].shape[1]
                generated_ids = output_ids[0][input_len:]
                raw_output = processor.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

                prediction, is_valid, validation_error = parse_and_validate_json(raw_output)

                if not is_valid:
                    prediction = {
                        "reasoning_steps": [],
                        "answer": "insufficient information"
                    }
                    error_msg = f"Validation failed: {validation_error}"
                else:
                    error_msg = None

                results.append({
                    "idx": idx,
                    "prediction": prediction,
                    "raw_output": raw_output,
                    "is_valid": is_valid,
                    "error": error_msg,
                })

            except Exception as e:
                error_msg = f"Generation error: {str(e)}"
                results.append({
                    "idx": idx,
                    "prediction": {
                        "reasoning_steps": [],
                        "answer": "insufficient information"
                    },
                    "raw_output": "",
                    "is_valid": False,
                    "error": error_msg,
                })

    return results


def run_inference_on_checkpoint(
    checkpoint_path: str,
    test_dataset_path: str,
    predictions_dir: str,
    device_ids: List[int],
    batch_size: int = 1,
    start_idx_offset: int = 0,
):
    """Run inference on a single checkpoint."""

    print(f"\n{'='*80}")
    print(f"Processing checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    # Enable deterministic mode
    set_deterministic_mode(seed=42)

    start_time = time.time()

    # Create predictions directory
    os.makedirs(predictions_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_deepspeed_checkpoint(checkpoint_path, device_ids)
    model.eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    # Load dataset
    print(f"Loading test dataset from: {test_dataset_path}")
    hf_dataset = load_from_disk(test_dataset_path)
    print(f"Loaded {len(hf_dataset)} test samples\n")

    # Create dataset
    system_prompt = get_system_prompt()
    dataset = VQATestDataset(hf_dataset, processor, system_prompt)

    # Check how many predictions already exist
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
        print(f"No existing predictions found - processing all {len(dataset)} samples")

    # Run inference
    print(f"Running inference with batch_size={batch_size}...")
    all_results = []
    valid_count = 0
    error_count = 0
    skipped_count = 0

    # Process in batches
    num_samples = len(dataset)
    checkpoint_name = Path(checkpoint_path).name
    pbar = tqdm(
        range(0, num_samples, batch_size),
        desc=f"[GPU {device_ids[0]}] {checkpoint_name}",
        unit="batch",
        ncols=100,
        position=device_ids[0] + 1,  # Each GPU gets its own line
        leave=True
    )
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, num_samples)
        batch_items = []

        # Collect items for this batch, checking for existing predictions
        for local_idx in range(batch_start, batch_end):
            item = dataset[local_idx]
            global_idx = local_idx + start_idx_offset
            item["idx"] = global_idx

            # Check if prediction already exists (for resuming interrupted runs)
            if global_idx in existing_predictions:
                skipped_count += 1
                # Load existing prediction for summary
                prediction_file = os.path.join(predictions_dir, f"{global_idx}.json")
                try:
                    with open(prediction_file, "r", encoding='utf-8') as f:
                        existing_pred = json.load(f)
                    all_results.append({
                        "idx": global_idx,
                        "prediction": existing_pred,
                        "raw_output": "[skipped - already exists]",
                        "is_valid": True,
                        "error": None,
                    })
                    valid_count += 1
                except Exception as e:
                    pbar.write(f"Warning: Could not load existing file {prediction_file}: {e}")
                continue

            batch_items.append(item)

        # Process batch if there are items to process
        if batch_items:
            results = run_inference_batch(
                model,
                processor,
                batch_items,
            )

            for result in results:
                if result["is_valid"]:
                    valid_count += 1
                else:
                    error_count += 1

                # Save immediately after generation
                idx = result["idx"]
                prediction_file = os.path.join(predictions_dir, f"{idx}.json")
                with open(prediction_file, "w", encoding='utf-8') as f:
                    json.dump(result["prediction"], f, indent=2, ensure_ascii=False)

            all_results.extend(results)

        # Update progress bar with stats (even if batch was empty due to all skipped)
        pbar.set_postfix({
            'valid': valid_count,
            'errors': error_count,
            'skipped': skipped_count
        }, refresh=True)

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
        "total_samples": len(dataset),
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
    print(f"\n📁 Coverage: {len(dataset)} samples")
    print(f"⏭️  Skipped: {skipped_count} (already existed)")
    print(f"✅ Valid: {valid_count} ({valid_rate*100:.1f}%)")
    print(f"❌ Invalid: {error_count} ({(1-valid_rate)*100:.1f}%)")
    print(f"⏱️  Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    print(f"💾 Saved to: {predictions_dir}")
    print(f"{'='*80}\n")

    # Clean up
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_dataset_path", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--device_ids", type=str, default="5")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--start_idx_offset", type=int, default=0, help="Offset for preserving original indices")

    args = parser.parse_args()

    device_ids = [int(x.strip()) for x in args.device_ids.split(",")]

    run_inference_on_checkpoint(
        checkpoint_path=args.checkpoint_dir,
        test_dataset_path=args.test_dataset_path,
        predictions_dir=args.predictions_dir,
        device_ids=device_ids,
        batch_size=args.batch_size,
        start_idx_offset=args.start_idx_offset,
    )


if __name__ == "__main__":
    main()
