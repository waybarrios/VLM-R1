#!/usr/bin/env python3
"""
Test monkey patch with a real cat image.
"""

import os
import sys
import torch
from PIL import Image
import requests
from io import BytesIO

# Add source directory to path
sys.path.insert(0, '/gpudata3/Wayner/VLM-R1/src/open-r1-multimodal/src')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Import and apply monkey patches
from open_r1.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_qwen2_5vl_forward,
    monkey_patch_torch_load
)

print("=" * 80)
print("Testing Monkey Patch with Real Cat Image")
print("=" * 80)

# Apply monkey patches
print("\n[1/5] Applying monkey patches...")
monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_qwen2_5vl_forward()
monkey_patch_torch_load()
print("✓ Monkey patches applied")

# Load model and processor
print("\n[2/5] Loading model and processor...")
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(model_name)
print(f"✓ Model loaded on device: {model.device}")

# Load the cat image
print("\n[3/5] Loading cat image...")
cat_url = "https://img.freepik.com/free-photo/portrait-beautiful-purebred-pussycat-with-shorthair-orange-collar-neck-sitting-floor-reacting-camera-flash-scared-looking-light-indoor_8353-12551.jpg"
print(f"Image URL: {cat_url}")

try:
    response = requests.get(cat_url, timeout=10)
    cat_image = Image.open(BytesIO(response.content))
    print(f"✓ Cat image loaded: {cat_image.size} pixels, mode: {cat_image.mode}")
except Exception as e:
    print(f"✗ Error loading image: {e}")
    sys.exit(1)

# Test with the comprehensive system prompt
print("\n[4/5] Generating response with VQA-style JSON format...")
print("-" * 80)

system_prompt = """You are a vision-language model. First, analyze the provided image(s) and any user text silently. Do NOT reveal your internal reasoning.

Return ONLY a single, valid JSON object with this exact schema:
{"reasoning_steps": [], "answer": ""}

Rules for "reasoning_steps":
- Decide the number of steps based on task complexity; include enough to make the answer evident without filler.
- Include some inference from visual information, always anchored to visible cues.
- Write single-clause sentences, each adding a new, directly checkable fact or cue-based inference.
- Keep each step ≤14 words. No multi-sentence items. No chains like "because/therefore". No internal monologue.

Rules for "answer":
- Provide the final answer grounded strictly in visible content and given text.
- If information is missing or ambiguous, set "answer" to "insufficient information" and include steps noting what is missing.

Output formatting:
- Output only the JSON object. No extra keys, comments, code fences, or prose.
- Use double quotes for all strings; no trailing commas; any valid JSON whitespace is acceptable."""

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
            {"type": "text", "text": "What type of animal is in this image?"},
        ],
    },
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(f"Question: What type of animal is in this image?")
print(f"Prompt length: {len(prompt)} characters")

inputs = processor(
    text=[prompt],
    images=[cat_image],
    return_tensors="pt",
    padding=True,
).to(model.device)

print(f"Input IDs shape: {inputs.input_ids.shape}")
print(f"Pixel values shape: {inputs.pixel_values.shape}")
print(f"Image grid thw: {inputs.image_grid_thw}")

# Generate response
print("\nGenerating response...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=1.0,
        pad_token_id=processor.tokenizer.pad_token_id
    )

output_text = processor.batch_decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print("\n" + "=" * 80)
print("[5/5] RESULTS")
print("=" * 80)
print("\nFull Output:")
print("-" * 80)
print(output_text)
print("-" * 80)

# Try to extract and validate JSON
import json
import re

print("\n" + "=" * 80)
print("JSON Validation")
print("=" * 80)

try:
    # Extract JSON from output after "assistant"
    if "assistant" in output_text:
        assistant_part = output_text.split("assistant", 1)[1].strip()
    else:
        assistant_part = output_text

    # Find JSON by looking for opening brace and matching closing brace
    json_str = None
    start_idx = assistant_part.find('{')

    if start_idx != -1:
        # Count braces to find the matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(assistant_part)):
            char = assistant_part[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the matching closing brace
                        json_str = assistant_part[start_idx:i+1]
                        break

    if json_str:
        print("\nExtracted JSON:")
        print("-" * 80)
        print(json_str)
        print("-" * 80)

        parsed = json.loads(json_str)

        print("\n✓ Valid JSON!")
        print(f"\nReasoning Steps ({len(parsed.get('reasoning_steps', []))}):")
        for i, step in enumerate(parsed.get('reasoning_steps', []), 1):
            print(f"  {i}. {step}")

        print(f"\nAnswer: {parsed.get('answer', 'N/A')}")

        # Check if answer is correct
        answer_lower = parsed.get('answer', '').lower()
        if 'cat' in answer_lower or 'feline' in answer_lower or 'kitten' in answer_lower:
            print("\n✓✓✓ CORRECT! The model identified it as a cat!")
        else:
            print(f"\n⚠ Answer doesn't mention cat: '{parsed.get('answer')}'")

    else:
        print("\n✗ No valid JSON format detected in output")
        print("Output does not contain the expected JSON structure.")

except json.JSONDecodeError as e:
    print(f"\n✗ JSON parsing failed: {e}")
except Exception as e:
    print(f"\n✗ Error during validation: {e}")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
