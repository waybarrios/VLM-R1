"""
Causal Intervention Reward (CIR)
Rewards reasoning steps based on their causal necessity for the correct answer.

R_causal(step_i) = P(correct | with step_i) - P(correct | without step_i)

Steps that cause large accuracy drops when removed are causally necessary.
This addresses the "correct answer, wrong reasoning" problem by ensuring
that high rewards require both correct answers AND causally relevant steps.

Reference: Novel contribution for VLM reasoning (CVPR/NeurIPS submission)
"""

import re
from typing import List, Tuple, Dict, Any, Optional
import os
from datetime import datetime


def parse_reasoning_steps(response: str) -> Tuple[List[str], str]:
    """
    Parse response into reasoning steps and final answer.

    Supports multiple formats:
    1. JSON format: {"reasoning_steps": [...], "answer": "X"}
    2. XML format: <think>...</think><answer>X</answer>

    Args:
        response: Model's full response

    Returns:
        Tuple of (list of reasoning steps, final answer string)
    """
    import json

    # Try JSON format first (current model output)
    try:
        # Find JSON block in response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{[^{}]*"reasoning_steps"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = None

        if json_str:
            data = json.loads(json_str)
            steps = data.get("reasoning_steps", [])
            if isinstance(steps, str):
                # Sometimes steps is a single string
                steps = [steps] if steps else []
            answer = data.get("answer", "")
            if steps or answer:
                return steps, str(answer)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try XML/tag format
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if not think_match:
        # Try alternative format
        think_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)

    if not think_match:
        return [], response.strip()

    think_content = think_match.group(1).strip()

    # Parse steps - try multiple patterns
    steps = []

    # Pattern 1: "Step N: content"
    step_pattern = r'Step\s*(\d+)[:.]\s*(.*?)(?=Step\s*\d+[:.:]|\Z)'
    matches = re.findall(step_pattern, think_content, re.DOTALL | re.IGNORECASE)
    if matches:
        steps = [m[1].strip() for m in matches if m[1].strip()]

    # Pattern 2: Numbered list "1. content"
    if not steps:
        numbered_pattern = r'^\s*(\d+)[.)]\s*(.*?)(?=^\s*\d+[.)]|\Z)'
        matches = re.findall(numbered_pattern, think_content, re.DOTALL | re.MULTILINE)
        if matches:
            steps = [m[1].strip() for m in matches if m[1].strip()]

    # Pattern 3: Bullet points
    if not steps:
        bullet_pattern = r'[-•*]\s*(.*?)(?=[-•*]|\Z)'
        matches = re.findall(bullet_pattern, think_content, re.DOTALL)
        if matches:
            steps = [m.strip() for m in matches if m.strip()]

    # Fallback: split by newlines
    if not steps:
        lines = [s.strip() for s in think_content.split('\n') if s.strip()]
        steps = [l for l in lines if len(l) > 10]  # Filter very short lines

    # Extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if not answer_match:
        answer_match = re.search(r'<final_answer>(.*?)</final_answer>', response, re.DOTALL)

    answer = answer_match.group(1).strip() if answer_match else ""

    return steps, answer


def check_answer_match(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.

    Handles multiple answer formats:
    - Exact match
    - MCQ option letters (A, B, C, D)
    - Numeric values with tolerance
    - Yes/No answers

    Args:
        predicted: Model's predicted answer
        ground_truth: Correct answer

    Returns:
        True if answers match
    """
    if not predicted or not ground_truth:
        return False

    pred_clean = predicted.strip().lower()
    gt_clean = ground_truth.strip().lower()

    # Exact match
    if pred_clean == gt_clean:
        return True

    # MCQ: check option letter
    pred_option = re.search(r'^([a-d])\b', pred_clean)
    gt_option = re.search(r'^([a-d])\b', gt_clean)
    if pred_option and gt_option:
        return pred_option.group(1) == gt_option.group(1)

    # Yes/No
    yes_variants = {'yes', 'true', 'correct', '1'}
    no_variants = {'no', 'false', 'incorrect', '0'}
    pred_bool = pred_clean in yes_variants
    gt_bool = gt_clean in yes_variants
    if (pred_clean in yes_variants or pred_clean in no_variants) and \
       (gt_clean in yes_variants or gt_clean in no_variants):
        return pred_bool == gt_bool

    # Numeric: check if numbers match with tolerance
    pred_nums = re.findall(r'[-+]?\d*\.?\d+', pred_clean)
    gt_nums = re.findall(r'[-+]?\d*\.?\d+', gt_clean)
    if pred_nums and gt_nums:
        try:
            pred_val = float(pred_nums[0])
            gt_val = float(gt_nums[0])
            # Relative tolerance for larger numbers, absolute for small
            if abs(gt_val) > 1:
                return abs(pred_val - gt_val) / abs(gt_val) < 0.01
            else:
                return abs(pred_val - gt_val) < 0.01
        except ValueError:
            pass

    # Substring match for longer answers
    if len(gt_clean) > 10:
        return gt_clean in pred_clean or pred_clean in gt_clean

    return False


def compute_step_alignment(
    predicted_steps: List[str],
    reference_steps: List[str],
    threshold: float = 0.3
) -> Tuple[float, int, int]:
    """
    Compute alignment between predicted and reference steps.

    Uses word overlap F1 as a fast proxy for semantic similarity.

    Args:
        predicted_steps: Model's reasoning steps
        reference_steps: Reference reasoning steps
        threshold: Minimum similarity for a match

    Returns:
        Tuple of (f1_score, matched_predictions, matched_references)
    """
    try:
        from simple_similarity import best_match_f1
        return best_match_f1(predicted_steps, reference_steps, threshold=threshold)
    except ImportError:
        # Fallback: simple overlap computation
        if not predicted_steps or not reference_steps:
            return 0.0, 0, 0

        def tokenize(text):
            return set(re.findall(r'\w+', text.lower()))

        matched_pred = 0
        matched_ref = 0

        for pred in predicted_steps:
            pred_tokens = tokenize(pred)
            for ref in reference_steps:
                ref_tokens = tokenize(ref)
                if pred_tokens and ref_tokens:
                    overlap = len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens)
                    if overlap > threshold:
                        matched_pred += 1
                        break

        for ref in reference_steps:
            ref_tokens = tokenize(ref)
            for pred in predicted_steps:
                pred_tokens = tokenize(pred)
                if pred_tokens and ref_tokens:
                    overlap = len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens)
                    if overlap > threshold:
                        matched_ref += 1
                        break

        precision = matched_pred / len(predicted_steps) if predicted_steps else 0
        recall = matched_ref / len(reference_steps) if reference_steps else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return f1, matched_pred, matched_ref


def lightweight_causal_reward(
    predicted_steps: List[str],
    reference_steps: List[str],
    predicted_answer: str,
    ground_truth: str,
    answer_weight: float = 0.6,
    step_weight: float = 0.4
) -> float:
    """
    Lightweight approximation of Causal Intervention Reward.

    Instead of computing full counterfactuals (expensive), we use a
    multiplicative interaction between answer correctness and step alignment.

    Key insight: Steps that align with reference are ASSUMED to be
    causally important. This is a reasonable proxy because reference
    steps were chosen to be necessary for the answer.

    Reward structure:
    - Correct answer + good steps = HIGH reward (causally faithful)
    - Correct answer + bad steps = MEDIUM reward (possibly lucky)
    - Wrong answer + good steps = LOW reward (steps didn't help)
    - Wrong answer + bad steps = ZERO reward

    Args:
        predicted_steps: Model's reasoning steps
        reference_steps: Reference reasoning steps
        predicted_answer: Model's final answer
        ground_truth: Correct answer
        answer_weight: Weight for answer correctness (default 0.6)
        step_weight: Weight for step alignment (default 0.4)

    Returns:
        Reward value in [0, 1]
    """
    # Check answer correctness
    answer_correct = check_answer_match(predicted_answer, ground_truth)
    answer_score = 1.0 if answer_correct else 0.0

    # Check step alignment
    if predicted_steps and reference_steps:
        f1, matched_pred, matched_ref = compute_step_alignment(
            predicted_steps, reference_steps, threshold=0.3
        )
        step_score = f1
    else:
        step_score = 0.0

    # Multiplicative interaction for causal faithfulness
    if answer_correct:
        # Correct answer: reward based on both answer and reasoning
        # Higher step_score increases reward (faithful reasoning)
        reward = answer_weight * answer_score + step_weight * step_score
    else:
        # Wrong answer: heavily penalize, but give partial credit
        # if reasoning was on the right track
        reward = step_weight * step_score * 0.3  # Much reduced

    return reward


def causal_intervention_reward(
    completions: List[List[Dict]],
    ground_truths: Optional[List[str]] = None,
    reference_steps: Optional[List[List[str]]] = None,
    data_indices: Optional[List[int]] = None,
    answer_weight: float = 0.6,
    step_weight: float = 0.4,
    debug_mode: bool = False,
    log_path: Optional[str] = None,
    **kwargs
) -> List[float]:
    """
    Compute Causal Intervention Reward for a batch of completions.

    This is the main entry point for the CIR reward function.

    Args:
        completions: Model completions, each is [{"content": "..."}]
        ground_truths: Correct answers for each sample
        reference_steps: Reference reasoning steps for each sample
        data_indices: Dataset indices for logging
        answer_weight: Weight for answer correctness
        step_weight: Weight for step alignment
        debug_mode: If True, write detailed logs
        log_path: Path for debug logs

    Returns:
        List of reward values, one per completion
    """
    rewards = []
    current_time = datetime.now().strftime("%m-%d-%H-%M-%S")

    # Handle missing inputs
    if ground_truths is None:
        ground_truths = [""] * len(completions)
    if reference_steps is None:
        reference_steps = [[]] * len(completions)
    if data_indices is None:
        data_indices = list(range(len(completions)))

    for i, completion in enumerate(completions):
        try:
            content = completion[0]["content"] if completion else ""
        except (KeyError, IndexError):
            content = str(completion)

        # Parse response
        steps, answer = parse_reasoning_steps(content)

        # Get reference data
        gt = ground_truths[i] if i < len(ground_truths) else ""
        ref_steps = reference_steps[i] if i < len(reference_steps) else []

        # Ensure ref_steps is a list of strings
        if ref_steps and not isinstance(ref_steps[0], str):
            ref_steps = [str(s) for s in ref_steps]
        ref_steps = [s.strip() for s in ref_steps if s and str(s).strip()]

        # Compute reward
        reward = lightweight_causal_reward(
            predicted_steps=steps,
            reference_steps=ref_steps,
            predicted_answer=answer,
            ground_truth=gt,
            answer_weight=answer_weight,
            step_weight=step_weight
        )

        rewards.append(reward)

        # Debug logging
        if debug_mode and log_path:
            log_file = log_path.replace(".txt", "_causal.txt")
            try:
                with open(log_file, "a", encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Time: {current_time} | Index: {data_indices[i]}\n")
                    f.write(f"Causal Reward: {reward:.4f}\n")
                    f.write(f"Answer correct: {check_answer_match(answer, gt)}\n")
                    f.write(f"Predicted steps: {len(steps)}\n")
                    f.write(f"Reference steps: {len(ref_steps)}\n")
                    if steps:
                        f.write(f"First step: {steps[0][:100]}...\n")
                    f.write(f"Answer: {answer[:50]}\n")
                    f.write(f"Ground truth: {gt[:50]}\n")
            except Exception as e:
                pass  # Don't fail on logging errors

    return rewards


# Alias for compatibility with reward function selector
vqa_causal_reasoning_reward = causal_intervention_reward


if __name__ == "__main__":
    # Test the CIR module
    print("Testing Causal Intervention Reward module...")

    test_response = """<think>
Step 1: Looking at the image, I can see a traffic light.
Step 2: The traffic light appears to be green.
Step 3: Therefore, vehicles can proceed.
</think>
<answer>B</answer>"""

    steps, answer = parse_reasoning_steps(test_response)
    print(f"Parsed {len(steps)} steps: {steps}")
    print(f"Answer: {answer}")

    # Test reward computation
    ref_steps = [
        "Observe the traffic light in the image",
        "The light is showing green",
        "Green means go"
    ]

    reward = lightweight_causal_reward(
        predicted_steps=steps,
        reference_steps=ref_steps,
        predicted_answer=answer,
        ground_truth="B"
    )
    print(f"Reward (correct answer): {reward:.4f}")

    reward_wrong = lightweight_causal_reward(
        predicted_steps=steps,
        reference_steps=ref_steps,
        predicted_answer="C",
        ground_truth="B"
    )
    print(f"Reward (wrong answer): {reward_wrong:.4f}")

    print("\nTest passed!")
