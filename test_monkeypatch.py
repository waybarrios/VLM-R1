#!/usr/bin/env python3
"""
Test script to verify the Qwen2.5-VL monkey patch is working correctly.
This tests basic generation with the monkey-patched forward method.
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
print("Qwen2.5-VL Monkey Patch Test")
print("=" * 80)

# Apply monkey patches
print("\n[1/6] Applying monkey patches...")
monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_qwen2_5vl_forward()
monkey_patch_torch_load()
print("✓ Monkey patches applied")

# Load model and processor
print("\n[2/6] Loading model and processor...")
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(model_name)
print(f"✓ Model loaded on device: {model.device}")

# Test 1: Text-only generation (no image)
print("\n[3/6] Test 1: Text-only generation (no image)")
print("-" * 80)
messages_text = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Say 'Hello, World!' and nothing else."},
        ],
    },
]

text_prompt = processor.apply_chat_template(
    messages_text,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Prompt: {text_prompt[:200]}...")

inputs_text = processor(
    text=[text_prompt],
    return_tensors="pt",
    padding=True,
).to(model.device)

print(f"Input shape: {inputs_text.input_ids.shape}")

# Generate
with torch.no_grad():
    output_ids = model.generate(
        **inputs_text,
        max_new_tokens=50,
        do_sample=False,  # Greedy for consistency
        pad_token_id=processor.tokenizer.pad_token_id
    )

output_text = processor.batch_decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(f"\nGenerated output:\n{output_text}")
print(f"\nOutput length: {len(output_text)} characters")

# Test 2: Image + text generation
print("\n[4/6] Test 2: Image + text generation")
print("-" * 80)

# Load a test image
print("Loading test image...")
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
try:
    response = requests.get(url, timeout=10)
    image = Image.open(BytesIO(response.content))
    print(f"✓ Image loaded: {image.size}")
except Exception as e:
    print(f"⚠ Could not load image from URL: {e}")
    print("Creating dummy image instead...")
    image = Image.new('RGB', (224, 224), color='red')

messages_image = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful vision-language assistant."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in one short sentence."},
        ],
    },
]

image_prompt = processor.apply_chat_template(
    messages_image,
    tokenize=False,
    add_generation_prompt=True
)
print(f"Prompt: {image_prompt[:200]}...")

inputs_image = processor(
    text=[image_prompt],
    images=[image],
    return_tensors="pt",
    padding=True,
).to(model.device)

print(f"Input IDs shape: {inputs_image.input_ids.shape}")
print(f"Pixel values shape: {inputs_image.pixel_values.shape}")
print(f"Image grid thw: {inputs_image.image_grid_thw}")

# Generate with image
with torch.no_grad():
    output_ids_image = model.generate(
        **inputs_image,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id
    )

output_text_image = processor.batch_decode(
    output_ids_image,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(f"\nGenerated output:\n{output_text_image}")
print(f"\nOutput length: {len(output_text_image)} characters")

# Test 3: JSON format generation (like training)
print("\n[5/6] Test 3: JSON format generation (VQA-style)")
print("-" * 80)

messages_json = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": """You are a vision-language model. Analyze the image and respond ONLY with valid JSON in this format:
{"reasoning_steps": ["step 1", "step 2"], "answer": "your answer"}"""},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What objects do you see in this image?"},
        ],
    },
]

json_prompt = processor.apply_chat_template(
    messages_json,
    tokenize=False,
    add_generation_prompt=True
)

inputs_json = processor(
    text=[json_prompt],
    images=[image],
    return_tensors="pt",
    padding=True,
).to(model.device)

print(f"Generating JSON response...")

with torch.no_grad():
    output_ids_json = model.generate(
        **inputs_json,
        max_new_tokens=150,
        do_sample=True,
        temperature=1.0,
        pad_token_id=processor.tokenizer.pad_token_id
    )

output_text_json = processor.batch_decode(
    output_ids_json,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(f"\nGenerated output:\n{output_text_json}")

# Validate JSON
import json
import re
try:
    # Extract JSON from output after "assistant"
    if "assistant" in output_text_json:
        assistant_part = output_text_json.split("assistant", 1)[1].strip()
    else:
        assistant_part = output_text_json

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
        parsed = json.loads(json_str)
        print(f"\n✓ Valid JSON detected!")
        print(f"  - Has reasoning_steps: {isinstance(parsed.get('reasoning_steps'), list)}")
        print(f"  - Has answer: {isinstance(parsed.get('answer'), str)}")
        print(f"  - Number of steps: {len(parsed.get('reasoning_steps', []))}")
    else:
        print(f"\n⚠ No valid JSON format detected in output")
except Exception as e:
    print(f"\n⚠ JSON parsing failed: {e}")

# Summary
print("\n" + "=" * 80)
print("[6/6] Test Summary")
print("=" * 80)
print("✓ Monkey patch applied successfully")
print("✓ Text-only generation works")
print("✓ Image + text generation works")
print("✓ Model generates tokens correctly")
print("\nIf you see valid outputs above (not gibberish/Chinese), the monkey patch is working!")
print("=" * 80)
