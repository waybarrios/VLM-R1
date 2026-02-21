#!/usr/bin/env python3
"""
Recompute MMBench EN Dev reasoning metrics with improved letter extraction.
This script processes existing samples and applies the updated extraction logic.
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

def extract_letter_only(answer_text):
    """
    Extract only the letter (A-E) from the answer text.
    Handles formats like:
    - "B" → "B"
    - "B." → "B"
    - "B. the air pressure" → "B"
    - "B: wavelength" → "B"
    - "(B)" → "B"
    """
    if not answer_text:
        return answer_text

    answer_text = answer_text.strip()

    # Pattern 1: Letter followed by period/colon/parenthesis and optional text
    match = re.match(r'^([A-E])[\s]*[.:)]', answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 2: Just the letter (e.g., "A", "B")
    if len(answer_text) == 1 and answer_text.upper() in "ABCDE":
        return answer_text.upper()

    # Pattern 3: Parenthesis around letter (e.g., "(B)")
    match = re.match(r'^\(?([A-E])\)?$', answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 4: "Option B" or "Answer B" or "The answer is B"
    match = re.search(r'\b([A-E])\b', answer_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: return original if no pattern matches
    return answer_text


def extract_answer_from_json(response):
    """Extract the answer field from JSON response, then extract only the letter."""
    # Remove markdown code blocks
    response_cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()

    answer_text = None

    # Try to parse as JSON
    try:
        parsed = json.loads(response_cleaned)
        if isinstance(parsed, dict) and "answer" in parsed:
            answer_text = parsed["answer"].strip()
    except (json.JSONDecodeError, Exception):
        pass

    # Method 2: Extract outermost { } and parse
    if answer_text is None:
        first_brace = response_cleaned.find('{')
        if first_brace != -1:
            brace_count = 0
            for idx in range(first_brace, len(response_cleaned)):
                if response_cleaned[idx] == '{':
                    brace_count += 1
                elif response_cleaned[idx] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response_cleaned[first_brace:idx+1]
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and "answer" in parsed:
                                answer_text = parsed["answer"].strip()
                        except:
                            pass
                        break

    # Method 3: Regex to extract "answer" field
    if answer_text is None:
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response_cleaned)
        if answer_match:
            answer_text = answer_match.group(1).strip()

    # Fallback: return the original response
    if answer_text is None:
        answer_text = response.strip()

    # Extract only the letter from the answer
    return extract_letter_only(answer_text)


def main():
    samples_path = Path("/gpudata3/Wayner/original/logs-mmbech_reasoning-final-grpo-1500/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133__checkpoint-1500/20251113_033518_samples_mmbench_en_dev_reasoning.jsonl")

    if not samples_path.exists():
        print(f"Error: Samples file not found at {samples_path}")
        sys.exit(1)

    print("="*80)
    print("MMBench EN Dev Reasoning - Recomputed Metrics")
    print("="*80)
    print()

    # Track overall accuracy
    total = 0
    correct = 0

    # Track by category
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    # Track by L2 category
    l2_category_correct = defaultdict(int)
    l2_category_total = defaultdict(int)

    # Track extraction improvements
    had_extra_text = 0
    extraction_fixed = 0

    # Track errors
    errors = []

    with open(samples_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            total += 1

            response = sample['filtered_resps'][0]
            target = sample['target']
            category = sample['gpt_eval_score']['category']
            l2_category = sample['gpt_eval_score']['L2-category']

            # Get original prediction (before our improved extraction)
            original_pred = sample['gpt_eval_score']['prediction']

            # Apply improved extraction
            extracted_letter = extract_answer_from_json(response)

            # Track if we improved extraction
            if len(original_pred) > 2:
                had_extra_text += 1
                if original_pred != extracted_letter and extracted_letter == target:
                    extraction_fixed += 1

            # Check correctness
            is_correct = (extracted_letter == target)

            if is_correct:
                correct += 1
                category_correct[category] += 1
                l2_category_correct[l2_category] += 1
            else:
                # Track first 10 errors for analysis
                if len(errors) < 10:
                    errors.append({
                        'index': sample['gpt_eval_score']['index'],
                        'question': sample['gpt_eval_score']['question'][:100],
                        'target': target,
                        'original_pred': original_pred,
                        'extracted': extracted_letter,
                        'category': category
                    })

            category_total[category] += 1
            l2_category_total[l2_category] += 1

    # Compute overall accuracy
    overall_acc = 100 * correct / total

    print(f"📊 Overall Results:")
    print(f"   Total samples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {overall_acc:.2f}%")
    print()
    print(f"   Baseline target: 78.52%")
    print(f"   Difference: {overall_acc - 78.52:+.2f}pp")
    print()

    # Show extraction improvements
    print(f"🔧 Extraction Improvements:")
    print(f"   Samples with extra text: {had_extra_text} ({100*had_extra_text/total:.1f}%)")
    print(f"   Fixed by improved extraction: {extraction_fixed}")
    print()

    # Show category breakdown
    print(f"📋 Accuracy by Category:")
    for category in sorted(category_total.keys()):
        cat_acc = 100 * category_correct[category] / category_total[category]
        print(f"   {category:30s}: {cat_acc:6.2f}% ({category_correct[category]:4d}/{category_total[category]:4d})")
    print()

    # Show L2 category breakdown
    print(f"📋 Accuracy by L2 Category:")
    for l2_cat in sorted(l2_category_total.keys()):
        l2_acc = 100 * l2_category_correct[l2_cat] / l2_category_total[l2_cat]
        print(f"   {l2_cat:30s}: {l2_acc:6.2f}% ({l2_category_correct[l2_cat]:4d}/{l2_category_total[l2_cat]:4d})")
    print()

    # Show sample errors
    if errors:
        print(f"❌ Sample Errors (first 10):")
        for i, err in enumerate(errors, 1):
            print(f"\n   Error {i}:")
            print(f"      Index: {err['index']}")
            print(f"      Question: {err['question']}...")
            print(f"      Category: {err['category']}")
            print(f"      Target: {err['target']}")
            print(f"      Original prediction: '{err['original_pred']}'")
            print(f"      Extracted letter: '{err['extracted']}'")
        print()

    print("="*80)

    # Final verdict
    if overall_acc >= 78.52:
        print(f"✅ SUCCESS! Achieved {overall_acc:.2f}% (target: 78.52%)")
    else:
        print(f"❌ Below target. Achieved {overall_acc:.2f}% vs target 78.52%")
        print(f"   Gap: {78.52 - overall_acc:.2f}pp")
        print(f"   This suggests the model predictions are genuinely worse, not just extraction issues.")

    print("="*80)


if __name__ == "__main__":
    main()
