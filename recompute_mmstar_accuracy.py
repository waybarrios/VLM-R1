#!/usr/bin/env python3
"""
Recompute MMStar accuracy from samples with improved answer extraction.
"""
import json
import sys
import re
from pathlib import Path
from collections import defaultdict


def extract_answer_from_json(response):
    """Extract the answer field from JSON response."""
    # Remove markdown code blocks
    response_cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()

    # Try to parse as JSON
    try:
        # Method 1: Direct JSON parse
        parsed = json.loads(response_cleaned)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"].strip()
    except (json.JSONDecodeError, Exception):
        pass

    # Method 2: Extract outermost { } and parse
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
                            return parsed["answer"].strip()
                    except:
                        pass
                    break

    # Method 3: Regex to extract "answer" field
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response_cleaned)
    if answer_match:
        return answer_match.group(1).strip()

    # Fallback: return the original response
    return response.strip()


def extract_answer_letter(answer_text):
    """Extract the answer letter from the answer field - IMPROVED VERSION."""
    if not answer_text:
        return None

    # Clean the answer
    answer_text = answer_text.strip()
    answer_text_upper = answer_text.upper()

    # Pattern 1: Starts with letter followed by colon (e.g., "B: The golf course")
    match = re.match(r'^([A-D])[\s]*:', answer_text_upper)
    if match:
        return match.group(1)

    # Pattern 2: Just the letter (e.g., "A", "B")
    if len(answer_text) == 1 and answer_text_upper in "ABCD":
        return answer_text_upper

    # Pattern 3: Letter with parentheses (e.g., "(A)", "(B)")
    match = re.search(r'\(([A-D])\)', answer_text_upper)
    if match:
        return match.group(1)

    # Pattern 4: Letter with dot or bracket (e.g., "A.", "A)")
    match = re.match(r'^([A-D])[\s]*[\.\)\]]', answer_text_upper)
    if match:
        return match.group(1)

    # Pattern 5: "Answer: A" or "Option A" format
    match = re.search(r'(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])', answer_text_upper)
    if match:
        return match.group(1)

    # Pattern 6: First letter if it's A-D
    if answer_text_upper and answer_text_upper[0] in "ABCD":
        return answer_text_upper[0]

    # Pattern 7: Find any single A-D letter in the text
    letters = re.findall(r'\b([A-D])\b', answer_text_upper)
    if len(letters) == 1:
        return letters[0]

    # Pattern 8: If all else fails, find first occurrence of A-D
    match = re.search(r'([A-D])', answer_text_upper)
    if match:
        return match.group(1)

    return None


def exact_match(pred, gt):
    """Exact match from MMStar utils."""
    answer = gt.lower().replace("\n", " ").strip()
    predict = pred.lower().replace("\n", " ").strip() if pred else ""

    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception as e:
        return 0.0
    return 0.0


def recompute_accuracy(samples_file):
    """Recompute accuracy from samples JSONL file with improved extraction."""

    # Read all samples
    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"\n{'='*70}")
    print(f"Recomputing MMStar Accuracy with Improved Answer Extraction")
    print(f"{'='*70}\n")
    print(f"Total samples: {len(samples)}\n")

    # Recompute scores
    l2_category_scores = defaultdict(list)
    category_scores = defaultdict(list)
    total_correct = 0
    total_wrong = 0

    extraction_failures = []

    for sample in samples:
        # Get raw response
        raw_response = sample['filtered_resps'][0]

        # Extract answer from JSON
        answer_text = extract_answer_from_json(raw_response)

        # Extract letter
        pred_letter = extract_answer_letter(answer_text)

        # Get ground truth
        gt_letter = sample['target']

        # Get categories
        if 'average' in sample:
            l2_category = sample['average']['l2_category']
        else:
            l2_category = "unknown"

        # Find L1 category
        category = None
        for cat_name in sample.keys():
            if cat_name not in ['doc_id', 'target', 'filtered_resps', 'doc_hash', 'average', 'input']:
                category = cat_name
                break

        if category is None:
            category = "unknown"

        # Calculate score
        if pred_letter:
            score = exact_match(pred_letter, gt_letter)
        else:
            score = 0.0
            extraction_failures.append({
                'question_id': sample['doc_id'],
                'l2_category': l2_category,
                'answer_text': answer_text,
                'raw_response': raw_response[:200]
            })

        # Store scores
        l2_category_scores[l2_category].append(score)
        category_scores[category].append(score)

        if score == 1.0:
            total_correct += 1
        else:
            total_wrong += 1
            if pred_letter:  # Only log if we extracted something
                print(f"❌ Q{sample['doc_id']:4d} ({l2_category:40s}): pred={pred_letter}, gt={gt_letter} | {answer_text[:60]}")

    # Print L2 category results
    print(f"\n{'='*70}")
    print(f"L2 Category Results (18 subcategories)")
    print(f"{'='*70}")

    l2_category_avg_score = {}
    for l2_category, scores in sorted(l2_category_scores.items()):
        avg_score = sum(scores) / len(scores) * 100.0
        l2_category_avg_score[l2_category] = avg_score
        print(f"{l2_category:50s}: {avg_score:6.2f}% (n={len(scores):4d})")

    # Print L1 category results
    print(f"\n{'='*70}")
    print(f"L1 Category Results (6 main categories)")
    print(f"{'='*70}")

    for category, scores in sorted(category_scores.items()):
        if category != "unknown":
            avg_score = sum(scores) / len(scores) * 100.0
            print(f"{category:50s}: {avg_score:6.2f}% (n={len(scores):4d})")

    # Calculate overall average (average of L2 categories)
    overall_avg = sum(l2_category_avg_score.values()) / len(l2_category_avg_score) if l2_category_avg_score else 0

    print(f"\n{'='*70}")
    print(f"Overall Results")
    print(f"{'='*70}")
    print(f"{'Average (mean of L2 categories)':50s}: {overall_avg:6.2f}%")
    print(f"{'Simple accuracy':50s}: {total_correct / (total_correct + total_wrong) * 100:6.2f}% ({total_correct}/{total_correct + total_wrong})")
    print(f"{'='*70}\n")

    if extraction_failures:
        print(f"\n⚠️  Failed to extract answer from {len(extraction_failures)} samples:")
        for failure in extraction_failures[:10]:  # Show first 10
            print(f"  Q{failure['question_id']} ({failure['l2_category']}): {failure['answer_text'][:80]}")
        if len(extraction_failures) > 10:
            print(f"  ... and {len(extraction_failures) - 10} more")

    return overall_avg


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python recompute_mmstar_accuracy.py <samples.jsonl>")
        sys.exit(1)

    samples_file = sys.argv[1]

    if not Path(samples_file).exists():
        print(f"Error: File not found: {samples_file}")
        sys.exit(1)

    accuracy = recompute_accuracy(samples_file)
    print(f"\n✓ Final Average Accuracy: {accuracy:.2f}%\n")
