#!/usr/bin/env python3
"""
Simple test of the improved MMBench prompt using existing samples.
"""
import json
import sys
sys.path.insert(0, '/gpudata3/Wayner/original/lmms-eval')

from lmms_eval.tasks.mmbench.en_utils_reasoning import get_reasoning_system_prompt

# Get the prompt
system_prompt = get_reasoning_system_prompt()

print("="*80)
print("IMPROVED MMBENCH SYSTEM PROMPT")
print("="*80)
print()
print(system_prompt)
print()
print("="*80)
print()

# Load existing samples to check format
samples_path = "/gpudata3/Wayner/original/logs-mmbech_reasoning-final-grpo-1500/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133__checkpoint-1500/20251113_033518_samples_mmbench_en_dev_reasoning.jsonl"

print("Checking what OLD prompt produced (from existing samples):")
print()

with open(samples_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break

        sample = json.loads(line)
        response = sample['filtered_resps'][0]
        target = sample['target']

        # Try to parse JSON
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                answer = parsed.get("answer", "")

                # Check format
                has_extra = len(answer) > 1
                has_period = "." in answer
                has_colon = ":" in answer

                status = "❌ NEEDS FIXING" if has_extra else "✅ GOOD"

                print(f"Sample {i+1}: Target={target}, Answer='{answer}' {status}")

                if has_extra:
                    issues = []
                    if has_period:
                        issues.append("has period")
                    if has_colon:
                        issues.append("has colon")
                    if len(answer) > 10:
                        issues.append("has full text")
                    print(f"          Issues: {', '.join(issues)}")
        except:
            print(f"Sample {i+1}: Invalid JSON")

print()
print("="*80)
print("The NEW prompt explicitly shows these as WRONG examples:")
print("  ❌ \"B.\"")
print("  ❌ \"B: The car is red\"")
print("  ❌ \"B. Maryland\"")
print("  ❌ \"(C)\"")
print("="*80)
