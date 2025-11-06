#!/usr/bin/env python3
"""Test format validation with parse-first approach."""

import json
import re

def format_reward_vqa_new(content):
    """New format validation using parse-first approach."""
    reward = 0.0
    parsed = None

    try:
        # Remove markdown code fences if present
        content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

        # Method 1: Try to parse the entire content as JSON
        try:
            parsed = json.loads(content_cleaned)
        except json.JSONDecodeError:
            # Method 2: Find outermost { } pair and extract that
            first_brace = content_cleaned.find('{')
            if first_brace != -1:
                # Find matching closing brace using brace counting
                brace_count = 0
                for idx in range(first_brace, len(content_cleaned)):
                    if content_cleaned[idx] == '{':
                        brace_count += 1
                    elif content_cleaned[idx] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = content_cleaned[first_brace:idx+1]

                            # Fix trailing commas before parsing
                            json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

                            parsed = json.loads(json_str)
                            break

        # Validate structure if we successfully parsed JSON
        if parsed is not None:
            # Check if it has the required keys with correct types
            if "reasoning_steps" in parsed and "answer" in parsed:
                if isinstance(parsed["reasoning_steps"], list) and isinstance(parsed["answer"], str):
                    reward = 1.0

    except (json.JSONDecodeError, Exception):
        pass

    return reward, parsed


def test_format_validation():
    """Test various format cases."""

    test_cases = [
        # (description, content, expected_reward)
        (
            "Valid JSON - User's example",
            '''{"reasoning_steps": [
 "Observed objects labeled 'dish soap', 'marbles', and 'wet ice cube'.",
 "Noted the 'dish soap' appears to be slippery.",
 "Identified the 'marbles' being smooth and round.",
 "Noted the 'wet ice cube' being transparent.",
 "Recognized they all have a property of being translucent.",
 "Inference: translucent is the common property among the three objects.",
 "Concluded 'translucent' is the correct choice."
], "answer": "B) translucent"}''',
            1.0
        ),
        (
            "Valid JSON with trailing comma",
            '''{"reasoning_steps": ["step1", "step2",], "answer": "A"}''',
            1.0
        ),
        (
            "Valid JSON - Clean",
            '{"reasoning_steps": ["step1", "step2"], "answer": "A"}',
            1.0
        ),
        (
            "Valid JSON in markdown",
            '''```json
{"reasoning_steps": ["step1"], "answer": "B"}
```''',
            1.0
        ),
        (
            "Valid JSON with extra text",
            'Here is my answer: {"reasoning_steps": ["step1"], "answer": "C"}',
            1.0
        ),
        (
            "Invalid - Missing reasoning_steps",
            '{"answer": "A"}',
            0.0
        ),
        (
            "Invalid - Missing answer",
            '{"reasoning_steps": ["step1"]}',
            0.0
        ),
        (
            "Invalid - reasoning_steps is not a list",
            '{"reasoning_steps": "not a list", "answer": "A"}',
            0.0
        ),
        (
            "Invalid - answer is not a string",
            '{"reasoning_steps": ["step1"], "answer": 123}',
            0.0
        ),
        (
            "Invalid - Not JSON at all",
            "This is just plain text without JSON",
            0.0
        ),
        (
            "Invalid - Malformed JSON",
            '{"reasoning_steps": ["step1", "answer": "A"}',
            0.0
        ),
    ]

    print("=" * 80)
    print("Format Validation Test - Parse-First Approach")
    print("=" * 80)
    print()

    all_passed = True
    for i, (description, content, expected) in enumerate(test_cases, 1):
        reward, parsed = format_reward_vqa_new(content)

        status = "✅" if reward == expected else "❌"
        print(f"{status} Test {i}: {description}")
        print(f"   Expected: {expected}, Got: {reward}")

        if reward == 1.0 and parsed:
            print(f"   Keys: {list(parsed.keys())}")
            print(f"   reasoning_steps is list: {isinstance(parsed.get('reasoning_steps'), list)}")
            print(f"   answer is string: {isinstance(parsed.get('answer'), str)}")
        elif reward == 0.0:
            if parsed:
                print(f"   Parsed but invalid structure")
                print(f"   Keys: {list(parsed.keys())}")
            else:
                print(f"   Failed to parse as JSON")

        if reward != expected:
            all_passed = False
            print(f"   ⚠️ MISMATCH!")

        print()

    print("=" * 80)
    if all_passed:
        print("✅✅✅ ALL TESTS PASSED!")
        print()
        print("Format validation now:")
        print("  1. Parses JSON first (two methods: direct + brace matching)")
        print("  2. Validates structure (has required keys with correct types)")
        print("  3. Returns reward 1.0 for valid, 0.0 for invalid")
    else:
        print("❌ Some tests failed")

    return all_passed


if __name__ == "__main__":
    print("\n")
    success = test_format_validation()
    exit(0 if success else 1)
