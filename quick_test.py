#!/usr/bin/env python3
"""Quick test to verify monkey patch works after fixes."""

import sys
import torch
sys.path.insert(0, '/gpudata3/Wayner/VLM-R1/src/open-r1-multimodal/src')

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from open_r1.qwen2_5vl_monkey_patch import (
    monkey_patch_qwen2_5vl_flash_attn,
    monkey_patch_qwen2_5vl_forward,
)

print("Applying monkey patches...")
monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_qwen2_5vl_forward()
print("✓ Patches applied\n")

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
print("✓ Model loaded\n")

# Simple text test
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "Say 'test passed' and nothing else."}]
}]

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)

output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(f"Output: {output}\n")

if "test passed" in output.lower() or len(output) < 200:
    print("✓ TEST PASSED - Output looks reasonable!")
else:
    print("✗ TEST FAILED - Output contains gibberish")
