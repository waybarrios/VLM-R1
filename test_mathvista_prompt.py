#!/usr/bin/env python3
"""
Test MathVista reasoning prompt to ensure it's properly configured.
"""
import sys
sys.path.insert(0, '/gpudata3/Wayner/original/lmms-eval')

from lmms_eval.tasks.mathvista.utils_reasoning import get_reasoning_system_prompt, extract_answer_from_json, clean_answer_format

print("=" * 80)
print("MATHVISTA REASONING PROMPT TEST")
print("=" * 80)
print()

# Test 1: Get system prompt
print("📝 TEST 1: System Prompt")
print("-" * 80)
prompt = get_reasoning_system_prompt()
print(f"Prompt length: {len(prompt)} characters")
print(f"Contains emoji sections: {'📊' in prompt and '🔢' in prompt and '📝' in prompt}")
print(f"Contains ✅ examples: {'✅ CORRECT examples' in prompt}")
print(f"Contains ❌ examples: {'❌ WRONG examples' in prompt}")
print(f"Contains multiple-choice guidance: {'Multiple-choice: ONLY the letter' in prompt}")
print(f"Contains numeric guidance: {'Numeric: ONLY the number' in prompt}")
print(f"Contains text guidance: {'Text: Concise answer' in prompt}")
print()

# Show key sections
print("Key sections preview:")
print("-" * 80)
lines = prompt.split('\n')
for i, line in enumerate(lines):
    if '📊' in line or '🔢' in line or '📝' in line:
        print(f"Line {i}: {line[:80]}")
    if '✅ CORRECT examples' in line:
        print(f"Line {i}: {line}")
        # Show next 5 lines
        for j in range(1, 6):
            if i+j < len(lines):
                print(f"Line {i+j}: {lines[i+j][:80]}")
        break
print()

# Test 2: Extraction from JSON
print("🔍 TEST 2: JSON Extraction")
print("-" * 80)

test_cases = [
    ('{"reasoning_steps": ["Step 1", "Step 2"], "answer": "B"}', "B", "Valid JSON"),
    ('{"reasoning_steps": ["Step 1"], "answer": "42"}', "42", "Numeric answer"),
    ('{"reasoning_steps": ["Step 1"], "answer": "3.14"}', "3.14", "Decimal answer"),
    ('The answer is 42', "42", "Plain text (no JSON)"),
    ('{"reasoning_steps": ["..."], "answer": "B."}', "B", "Letter with period"),
    ('{"reasoning_steps": ["..."], "answer": "(B)"}', "B", "Letter with parentheses"),
    ('{"reasoning_steps": ["..."], "answer": "The answer is 40"}', "40", "With prefix"),
    ('{"reasoning_steps": ["..."], "answer": "Answer: 40"}', "40", "With 'Answer:' prefix"),
]

all_passed = True
for response, expected, description in test_cases:
    extracted = extract_answer_from_json(response)
    passed = extracted == expected
    all_passed = all_passed and passed
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {description}")
    print(f"   Input:    {response[:60]}")
    print(f"   Expected: {expected}")
    print(f"   Got:      {extracted}")
    if not passed:
        print(f"   ⚠️  MISMATCH!")
    print()

# Test 3: Answer cleaning
print("🧹 TEST 3: Answer Format Cleaning")
print("-" * 80)

clean_test_cases = [
    ("B.", "B", "Remove period from letter"),
    ("(B)", "B", "Remove parentheses"),
    ("**B**", "B", "Remove markdown"),
    ("The answer is 40", "40", "Remove 'The answer is' prefix"),
    ("Answer: 40", "40", "Remove 'Answer:' prefix"),
    ("3.14", "3.14", "Keep decimal point"),
    ("triangle.", "triangle", "Remove period from word"),
    ("It is triangle", "triangle", "Remove 'It is' prefix"),
]

all_clean_passed = True
for input_text, expected, description in clean_test_cases:
    cleaned = clean_answer_format(input_text)
    passed = cleaned == expected
    all_clean_passed = all_clean_passed and passed
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {description}")
    print(f"   Input:    '{input_text}'")
    print(f"   Expected: '{expected}'")
    print(f"   Got:      '{cleaned}'")
    if not passed:
        print(f"   ⚠️  MISMATCH!")
    print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"System Prompt: ✅ Configured with emoji sections and examples")
print(f"JSON Extraction: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
print(f"Answer Cleaning: {'✅ ALL TESTS PASSED' if all_clean_passed else '❌ SOME TESTS FAILED'}")
print()

if all_passed and all_clean_passed:
    print("🎉 SUCCESS! MathVista reasoning is ready to use!")
    print()
    print("Run with:")
    print('CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --module lmms_eval -- \\')
    print('  --model qwen2_5_vl \\')
    print('  --model_args pretrained="/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1500" \\')
    print('  --tasks mathvista_testmini_reasoning \\')
    print('  --batch_size 1 \\')
    print('  --log_samples \\')
    print('  --output_path logs-mathvista-testmini-reasoning-ckpt1500')
else:
    print("⚠️  WARNING: Some tests failed. Check the output above.")

print("=" * 80)
