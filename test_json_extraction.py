#!/usr/bin/env python3
"""Test JSON extraction with the reported failing example."""

import json
import re

def extract_json_new_method(content):
    """New robust JSON extraction method."""
    content_cleaned = re.sub(r'```json\s*|\s*```', '', content).strip()

    json_str = None
    parsed = None

    # Method 1: Try to parse the entire content as JSON
    try:
        parsed = json.loads(content_cleaned)
        if "reasoning_steps" in parsed and "answer" in parsed:
            json_str = content_cleaned
            return parsed, "Method 1: Direct JSON parsing"
    except (json.JSONDecodeError, Exception):
        pass

    # Method 2: Find outermost { } pair and extract that
    if parsed is None:
        first_brace = content_cleaned.find('{')
        if first_brace != -1:
            # Find matching closing brace
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
        # 1. Replace smart quotes with regular quotes
        json_str = json_str.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")

        # 2. Replace single quotes with double quotes for string values
        json_str = re.sub(r'''(?<=[:,\[])\s*'([^']*)'(?=\s*[,\]\}])''', r' "\1"', json_str)

        # 3. Add missing commas between array items
        json_str = re.sub(r'"\s*\n\s*"', '",\n    "', json_str)
        json_str = re.sub(r'"\s+(?=")', '", ', json_str)

        # 4. Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)

        # Try parsing (if not already parsed in Method 1)
        if parsed is None:
            try:
                parsed = json.loads(json_str)
                return parsed, "Method 2: Brace matching with fixes"
            except json.JSONDecodeError:
                # If still fails, try more aggressive fixing
                json_str_fixed = json_str.replace("'", '"')
                json_str_fixed = re.sub(r'"\s*\n\s*"', '",\n    "', json_str_fixed)
                json_str_fixed = re.sub(r'"\s+(?=")', '", ', json_str_fixed)
                json_str_fixed = re.sub(r',(\s*[\]}])', r'\1', json_str_fixed)
                parsed = json.loads(json_str_fixed)
                return parsed, "Method 2: Brace matching with aggressive fixes"

    return None, "Failed to extract JSON"

def test_reported_example():
    """Test the exact example that was failing."""

    # This is the content that was reported as failing
    content = '''{"reasoning_steps": [
 "Observed objects labeled 'dish soap', 'marbles', and 'wet ice cube'.",
 "Noted the 'dish soap' appears to be slippery.",
 "Identified the 'marbles' being smooth and round.",
 "Noted the 'wet ice cube' being transparent.",
 "Recognized they all have a property of being translucent.",
 "Inference: translucent is the common property among the three objects.",
 "Concluded 'translucent' is the correct choice."
], "answer": "B) translucent"}'''

    print("=" * 80)
    print("Testing Reported Failing Example")
    print("=" * 80)
    print()
    print("Content:")
    print(content)
    print()
    print("=" * 80)

    try:
        parsed, method = extract_json_new_method(content)

        if parsed:
            print(f"✅ SUCCESS - {method}")
            print()
            print("Extracted JSON:")
            print(json.dumps(parsed, indent=2))
            print()
            print(f"Number of reasoning steps: {len(parsed.get('reasoning_steps', []))}")
            print(f"Answer field: '{parsed.get('answer', '')}'")
            print()

            # Test extraction
            predicted_answer = parsed.get("answer", "")
            if predicted_answer:
                print(f"✅ Predicted Answer: '{predicted_answer}'")
                print(f"✅ Answer is NOT empty!")
            else:
                print(f"❌ Predicted Answer is empty")

        else:
            print(f"❌ FAILED - {method}")

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 80)

    return parsed is not None and parsed.get("answer", "") != ""

def test_various_formats():
    """Test various JSON formats."""

    test_cases = [
        # (description, content, should_work)
        (
            "Clean JSON",
            '{"reasoning_steps": ["step1", "step2"], "answer": "A"}',
            True
        ),
        (
            "JSON with trailing comma",
            '{"reasoning_steps": ["step1", "step2",], "answer": "A"}',
            True
        ),
        (
            "JSON with newlines",
            '''{"reasoning_steps": [
                "step1",
                "step2"
            ], "answer": "A"}''',
            True
        ),
        (
            "JSON with markdown code fence",
            '''```json
            {"reasoning_steps": ["step1"], "answer": "B"}
            ```''',
            True
        ),
        (
            "JSON with extra text before",
            'Here is my answer: {"reasoning_steps": ["step1"], "answer": "C"}',
            True
        ),
        (
            "JSON with extra text after",
            '{"reasoning_steps": ["step1"], "answer": "D"} - This is my final answer.',
            True
        ),
    ]

    print()
    print("=" * 80)
    print("Testing Various Formats")
    print("=" * 80)
    print()

    all_passed = True
    for i, (description, content, should_work) in enumerate(test_cases, 1):
        print(f"Test {i}: {description}")
        try:
            parsed, method = extract_json_new_method(content)
            if should_work:
                if parsed and parsed.get("answer"):
                    print(f"  ✅ Extracted answer: '{parsed.get('answer')}'")
                else:
                    print(f"  ❌ Failed to extract (expected to work)")
                    all_passed = False
            else:
                if parsed:
                    print(f"  ❌ Extracted (expected to fail)")
                    all_passed = False
                else:
                    print(f"  ✅ Correctly failed")
        except Exception as e:
            if should_work:
                print(f"  ❌ Exception: {e}")
                all_passed = False
            else:
                print(f"  ✅ Correctly raised exception")
        print()

    return all_passed

if __name__ == "__main__":
    print("\n")

    # Test reported example
    test1 = test_reported_example()

    # Test various formats
    test2 = test_various_formats()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if test1 and test2:
        print("✅✅✅ ALL TESTS PASSED!")
        print()
        print("The new JSON extraction method successfully:")
        print("  1. Parses the reported failing example")
        print("  2. Extracts 'B) translucent' from the answer field")
        print("  3. Handles various JSON formats robustly")
        exit(0)
    else:
        print("❌ Some tests failed")
        exit(1)
