#!/usr/bin/env python3
"""Test option matching for VQA multiple choice questions."""

import sys
sys.path.insert(0, '/gpudata3/Wayner/VLM-R1/mllm_evaluator')

from accuracy_calculator import AccuracyCalculator

def test_option_matching():
    """Test the exact case from the user's report."""

    # User's example case
    problem = """What can Aiden and Bonnie trade to each get what they want?

A) Bonnie can trade her almonds for Aiden's tomatoes.
B) Aiden can trade his tomatoes for Bonnie's broccoli.
C) Bonnie can trade her broccoli for Aiden's oranges.
D) Aiden can trade his tomatoes for Bonnie's sandwich."""

    predicted_answer = "D"  # Model predicted D
    ground_truth = "Aiden can trade his tomatoes for Bonnie's broccoli."  # This is option B

    print("=" * 80)
    print("Testing Option Matching Fix")
    print("=" * 80)
    print()
    print("Problem:")
    print(problem)
    print()
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Ground Truth: {ground_truth}")
    print()

    # Initialize calculator (no LLM grader needed for this test)
    calculator = AccuracyCalculator(use_llm_grader=False)

    # Evaluate
    result = calculator.evaluate_single(problem, predicted_answer, ground_truth)

    print("=" * 80)
    print("RESULT:")
    print("=" * 80)
    print(f"Normalized Prediction: {result.normalized_prediction}")
    print(f"Normalized Ground Truth: {result.normalized_ground_truth}")
    print(f"Match Type: {result.match_type}")
    print(f"Is Correct: {result.is_correct}")
    print(f"Confidence: {result.confidence}")
    print()

    # Verify expectations
    expected_pred = "D"
    expected_gt = "B"
    expected_correct = False  # D != B, so should be incorrect

    if result.normalized_prediction == expected_pred:
        print(f"✅ Correctly extracted predicted choice: {expected_pred}")
    else:
        print(f"❌ Failed to extract predicted choice. Expected: {expected_pred}, Got: {result.normalized_prediction}")
        return False

    if result.normalized_ground_truth == expected_gt:
        print(f"✅ Correctly matched ground truth to option: {expected_gt}")
    else:
        print(f"❌ Failed to match ground truth. Expected: {expected_gt}, Got: {result.normalized_ground_truth}")
        return False

    if result.is_correct == expected_correct:
        print(f"✅ Correctly determined answer is incorrect (D != B)")
    else:
        print(f"❌ Incorrect evaluation. Expected: {expected_correct}, Got: {result.is_correct}")
        return False

    print()
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("The fix correctly:")
    print("1. Extracts 'D' from predicted answer")
    print("2. Matches ground truth text 'Aiden can trade his tomatoes for Bonnie's broccoli.' to option B")
    print("3. Compares D vs B and correctly marks as incorrect")
    print()
    return True

def test_correct_case():
    """Test when the model predicts the correct answer."""

    problem = """What can Aiden and Bonnie trade to each get what they want?

A) Bonnie can trade her almonds for Aiden's tomatoes.
B) Aiden can trade his tomatoes for Bonnie's broccoli.
C) Bonnie can trade her broccoli for Aiden's oranges.
D) Aiden can trade his tomatoes for Bonnie's sandwich."""

    predicted_answer = "B"  # Model predicts B
    ground_truth = "Aiden can trade his tomatoes for Bonnie's broccoli."  # This is option B

    print("=" * 80)
    print("Testing Correct Answer Case")
    print("=" * 80)
    print()
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Ground Truth: {ground_truth}")
    print()

    calculator = AccuracyCalculator(use_llm_grader=False)
    result = calculator.evaluate_single(problem, predicted_answer, ground_truth)

    print(f"Normalized Prediction: {result.normalized_prediction}")
    print(f"Normalized Ground Truth: {result.normalized_ground_truth}")
    print(f"Is Correct: {result.is_correct}")
    print()

    if result.is_correct and result.normalized_prediction == "B" and result.normalized_ground_truth == "B":
        print("✅ Correctly marked B == B as correct!")
        return True
    else:
        print("❌ Failed to recognize correct answer")
        return False

def test_alternative_formats():
    """Test various format combinations - all should work symmetrically."""

    problem = """What is the capital of France?

A) London
B) Paris
C) Berlin
D) Madrid"""

    test_cases = [
        # All valid representations of option B (Paris) - should ALL be correct
        ("B", "B", True, "Letter vs Letter"),
        ("B", "Paris", True, "Letter vs Full text"),
        ("B", "B) Paris", True, "Letter vs Formatted option"),
        ("Paris", "B", True, "Full text vs Letter"),
        ("Paris", "Paris", True, "Full text vs Full text"),
        ("Paris", "B) Paris", True, "Full text vs Formatted option"),
        ("B) Paris", "B", True, "Formatted option vs Letter"),
        ("B) Paris", "Paris", True, "Formatted option vs Full text"),
        ("B) Paris", "B) Paris", True, "Formatted option vs Formatted option"),

        # Wrong answers - should be incorrect
        ("A", "Paris", False, "Wrong letter vs Full text"),
        ("A", "B", False, "Wrong letter vs Correct letter"),
        ("London", "Paris", False, "Wrong city vs Correct city"),
        ("D", "B", False, "Wrong letter vs Correct letter"),
    ]

    print("=" * 80)
    print("Testing Alternative Formats - Symmetric Matching")
    print("=" * 80)
    print()

    calculator = AccuracyCalculator(use_llm_grader=False)
    all_passed = True

    for i, (pred, gt, expected, description) in enumerate(test_cases, 1):
        result = calculator.evaluate_single(problem, pred, gt)
        status = "✅" if result.is_correct == expected else "❌"
        print(f"{status} Test {i}: {description}")
        print(f"   Pred: '{pred}' → '{result.normalized_prediction}'")
        print(f"   GT: '{gt}' → '{result.normalized_ground_truth}'")
        print(f"   Expected: {expected}, Got: {result.is_correct}")
        print()

        if result.is_correct != expected:
            all_passed = False

    return all_passed

def test_user_requirement():
    """
    Test the exact user requirement:
    "For the example that pastes the answer to B or 'Aiden can trade...' or 'B) Aiden can trade...'
    both must be correct."
    """
    problem = """What can Aiden and Bonnie trade to each get what they want?

A) Bonnie can trade her almonds for Aiden's tomatoes.
B) Aiden can trade his tomatoes for Bonnie's broccoli.
C) Bonnie can trade her broccoli for Aiden's oranges.
D) Aiden can trade his tomatoes for Bonnie's sandwich."""

    ground_truth = "Aiden can trade his tomatoes for Bonnie's broccoli."  # Option B

    # All these predictions should be marked as CORRECT (they all refer to option B)
    correct_predictions = [
        "B",
        "Aiden can trade his tomatoes for Bonnie's broccoli.",
        "B) Aiden can trade his tomatoes for Bonnie's broccoli.",
    ]

    print("=" * 80)
    print("User Requirement Test: All formats for option B must be correct")
    print("=" * 80)
    print(f"\nGround Truth: {ground_truth}")
    print()

    calculator = AccuracyCalculator(use_llm_grader=False)
    all_passed = True

    for i, prediction in enumerate(correct_predictions, 1):
        result = calculator.evaluate_single(problem, prediction, ground_truth)
        status = "✅" if result.is_correct else "❌"

        print(f"{status} Test {i}: Prediction = '{prediction}'")
        print(f"   Normalized Pred: {result.normalized_prediction}")
        print(f"   Normalized GT: {result.normalized_ground_truth}")
        print(f"   Is Correct: {result.is_correct}")

        if not result.is_correct:
            print(f"   ❌ FAILED: Should be correct!")
            all_passed = False
        else:
            print(f"   ✅ PASSED: Correctly marked as correct")
        print()

    if all_passed:
        print("=" * 80)
        print("✅✅✅ ALL USER REQUIREMENTS MET!")
        print("=" * 80)
        print("\nAll three formats are correctly recognized as the same answer:")
        print("  • 'B' (letter only)")
        print("  • 'Aiden can trade his tomatoes for Bonnie's broccoli.' (full text)")
        print("  • 'B) Aiden can trade his tomatoes for Bonnie's broccoli.' (formatted)")
        print()

    return all_passed

if __name__ == "__main__":
    print("\n")

    # Run tests
    test1 = test_option_matching()
    print("\n")

    test2 = test_correct_case()
    print("\n")

    test3 = test_user_requirement()
    print("\n")

    test4 = test_alternative_formats()
    print("\n")

    if test1 and test2 and test3:
        print("🎉 All critical tests passed! The fix works correctly.")
        exit(0)
    else:
        print("❌ Some tests failed.")
        exit(1)
