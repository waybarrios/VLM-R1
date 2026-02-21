#!/usr/bin/env python3
"""
Test the improved MMBench system prompt with a few sample questions.
"""
import sys
sys.path.insert(0, '/gpudata3/Wayner/original/lmms-eval')

from lmms_eval.tasks.mmbench.en_utils_reasoning import get_reasoning_system_prompt, extract_answer_from_json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from datasets import load_dataset
import json

# Load the model
print("Loading model...")
model_path = "/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1400"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Get the system prompt
system_prompt = get_reasoning_system_prompt()

# Load a few samples from MMBench
print("Loading MMBench dataset...")
dataset = load_dataset("lmms-lab/MMBench", "en", split="dev", token=True)

# Test on 5 samples
print("\n" + "="*80)
print("Testing improved prompt on 5 samples from MMBench EN Dev")
print("="*80 + "\n")

for i in range(5):
    sample = dataset[i]

    # Build question with options
    question = sample['question']
    hint = sample.get('hint', '')

    options = []
    for opt in ['A', 'B', 'C', 'D', 'E']:
        if sample.get(opt) and sample[opt] != 'nan':
            options.append(f"{opt}. {sample[opt]}")

    options_text = "\n".join(options)

    if hint and hint != 'nan':
        full_question = f"{hint}\n\n{question}\n\nOptions:\n{options_text}"
    else:
        full_question = f"{question}\n\nOptions:\n{options_text}"

    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample['image']},
                {"type": "text", "text": full_question}
            ]
        }
    ]

    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False
        )

    # Decode
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Extract answer
    extracted = extract_answer_from_json(response)
    target = sample['answer']

    # Print results
    print(f"Sample {i+1}:")
    print(f"  Question: {question[:80]}...")
    print(f"  Target: {target}")
    print(f"  Response: {response[:150]}...")
    print(f"  Extracted: '{extracted}'")
    print(f"  Correct: {'✅' if extracted == target else '❌'}")

    # Check if response has extra text
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "answer" in parsed:
            raw_answer = parsed["answer"]
            if len(raw_answer) > 1:
                print(f"  ⚠️  WARNING: Answer has extra text: '{raw_answer}'")
            else:
                print(f"  ✅ Good! Answer is clean: '{raw_answer}'")
    except:
        print(f"  ⚠️  WARNING: Invalid JSON response")

    print()

print("="*80)
print("Test complete!")
print("="*80)
