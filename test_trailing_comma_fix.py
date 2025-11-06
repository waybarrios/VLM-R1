#!/usr/bin/env python3
"""Test script to verify trailing comma handling in JSON parsing."""

import json
import re

def test_trailing_comma_fix():
    """Test that the trailing comma fix works correctly."""

    # Example from the user's report - has trailing comma after last array item
    test_json = '''{
    "reasoning_steps": [
        "Noted a tray with tomatoes beside oranges from Aidan's lunch.",
        "Assumed an almond appears on Bonnie's lunch tray beside the vegetable mix.",
        "Noted a lettuce in Aidan's lunch appears to be part of a sandwich alongside tomatoes.",
        "Noted a tomato mix appears in Aidan's lunch beside oranges.",
        "Concluded tomatoes were more commonly associated with the salad, likely eaten raw.",
    ],
    "answer": "C) Bonnie can trade her broccoli for Aidan's oranges."
}'''

    print("Original JSON (with trailing comma):")
    print(test_json)
    print("\n" + "="*60 + "\n")

    # Try parsing without fix - should fail
    try:
        parsed = json.loads(test_json)
        print("❌ UNEXPECTED: Original JSON parsed without error")
    except json.JSONDecodeError as e:
        print(f"✓ Expected: Original JSON fails to parse")
        print(f"  Error: {e}")

    print("\n" + "="*60 + "\n")

    # Apply the trailing comma fix
    json_str_fixed = re.sub(r',(\s*[\]}])', r'\1', test_json)

    print("Fixed JSON (trailing commas removed):")
    print(json_str_fixed)
    print("\n" + "="*60 + "\n")

    # Try parsing with fix - should succeed
    try:
        parsed = json.loads(json_str_fixed)
        print("✓ Fixed JSON parses successfully!")
        print(f"  Keys: {list(parsed.keys())}")
        print(f"  reasoning_steps is list: {isinstance(parsed['reasoning_steps'], list)}")
        print(f"  answer is string: {isinstance(parsed['answer'], str)}")
        print(f"  Number of reasoning steps: {len(parsed['reasoning_steps'])}")

        # Verify structure matches expected format
        if "reasoning_steps" in parsed and "answer" in parsed:
            if isinstance(parsed["reasoning_steps"], list) and isinstance(parsed["answer"], str):
                print("\n✓✓✓ FORMAT VALIDATION PASSES (reward = 1.0)")
                return True
    except json.JSONDecodeError as e:
        print(f"❌ Fixed JSON still fails to parse: {e}")
        return False

    return False

if __name__ == "__main__":
    success = test_trailing_comma_fix()
    exit(0 if success else 1)
