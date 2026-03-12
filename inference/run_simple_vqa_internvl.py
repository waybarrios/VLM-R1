#!/usr/bin/env python3
"""
Simple VQA inference for InternVL3.5 models.
Adapted from run_simple_vqa.py (Qwen) for InternVL architecture.
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
from transformers import AutoModel, AutoProcessor, AutoConfig, AutoTokenizer
from datasets import load_from_disk
import argparse
import warnings
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# InternVL constants
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = best_ratio[0] * image_size
    target_height = best_ratio[1] * image_size
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def set_deterministic_mode(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Deterministic mode enabled (seed={seed})")


def get_system_prompt() -> str:
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
    content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()
    json_str = None
    parsed = None

    try:
        parsed = json.loads(content_cleaned)
        json_str = content_cleaned
    except (json.JSONDecodeError, Exception):
        pass

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
        json_str = json_str.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
        json_str = re.sub(r'''(?<=[:,\[])\s*'([^']*)'(?=\s*[,\]\}])''', r' "\1"', json_str)
        json_str = re.sub(r'"\s*\n\s*"', '",\n    "', json_str)
        json_str = re.sub(r'"\s+(?=")', '", ', json_str)
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

        if parsed is None:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as e:
                return None, False, f"JSON parse error: {str(e)}"

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
    parser = argparse.ArgumentParser(description="Simple VQA inference for InternVL3.5")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--test_dataset_path", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use (default: 0)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_anyres_num", type=int, default=12,
                        help="Max number of tiles for dynamic resolution (default: 12)")
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU: {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")

    set_deterministic_mode(seed=42)
    start_time = time.time()
    os.makedirs(args.predictions_dir, exist_ok=True)

    # Load InternVL model
    print(f"\nLoading InternVL model from: {args.checkpoint_dir}")
    config = AutoConfig.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    image_size = config.vision_config.image_size

    model = AutoModel.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map={"": 0},
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=True)
    print(f"Model loaded. Image size: {image_size}, max_anyres: {args.max_anyres_num}")

    # Load dataset
    print(f"\nLoading test dataset from: {args.test_dataset_path}")
    dataset = load_from_disk(args.test_dataset_path)
    print(f"Loaded {len(dataset)} test samples\n")

    system_prompt = get_system_prompt()

    # Check existing predictions
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

    # Generation config
    generation_config = dict(max_new_tokens=512, do_sample=False)

    valid_count = 0
    error_count = 0

    for idx in tqdm(range(total_samples), desc="Inference"):
        if idx in existing_predictions:
            continue

        item = dataset[idx]

        # Format options
        if item.get("choices"):
            formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item["choices"])])
        elif item.get("options"):
            formatted_options = "\n".join([f"{chr(65+i)}) {c}" for i, c in enumerate(item["options"])])
        else:
            formatted_options = ""

        user_question = f"{item['question']}\n\n{formatted_options}".strip()

        # Process image
        pil_image = item["image"]
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        pixel_values = load_image(pil_image, input_size=image_size, max_num=args.max_anyres_num)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # model.chat() auto-inserts <image> token and replaces with image embeddings
        # So we only pass the text question — no image tokens
        query = f"{system_prompt}\n\n{user_question}"

        if idx % 100 == 0:
            print(f"\n[DEBUG] Sample {idx}: {pixel_values.shape[0]} patches, starting generation...")
            sys.stdout.flush()

        # Use model.chat() for generation
        try:
            response = model.chat(
                tokenizer,
                pixel_values=pixel_values,
                question=query,
                generation_config=generation_config,
                verbose=False,
            )
        except Exception as e:
            print(f"\n[ERROR] Sample {idx}: {e}")
            response = ""

        if idx % 100 == 0:
            print(f"[DEBUG] Sample {idx}: Generation completed, output length: {len(response)}")
            sys.stdout.flush()

        # Parse and save
        prediction, is_valid, validation_error = parse_and_validate_json(response)

        if not is_valid:
            prediction = {
                "reasoning_steps": [],
                "answer": "insufficient information"
            }
            error_count += 1
        else:
            valid_count += 1

        prediction_file = os.path.join(args.predictions_dir, f"{idx}.json")
        with open(prediction_file, "w", encoding='utf-8') as f:
            json.dump(prediction, f, indent=2, ensure_ascii=False)

    # Stats
    elapsed_time = time.time() - start_time
    total_processed = valid_count + error_count
    valid_rate = valid_count / total_processed if total_processed > 0 else 0

    print(f"\n{'='*80}")
    print(f"Inference completed!")
    print(f"{'='*80}")
    print(f"\nProcessed: {total_processed} samples")
    print(f"Valid: {valid_count} ({valid_rate*100:.1f}%)")
    print(f"Invalid: {error_count} ({(1-valid_rate)*100:.1f}%)")
    print(f"Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    if total_processed > 0:
        print(f"Speed: {elapsed_time/total_processed:.2f}s per sample")
    print(f"Saved to: {args.predictions_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
