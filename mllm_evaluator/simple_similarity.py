"""
Simple text similarity metrics that work reliably in multi-GPU environments.
Includes both word overlap (pure Python) and semantic similarity (SentenceTransformer on CPU).
"""

from typing import List, Tuple, Optional, Any
from collections import Counter
import re
import numpy as np


def tokenize(text: str) -> List[str]:
    """Simple word tokenization."""
    return re.findall(r'\w+', text.lower())


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Jaccard similarity: |intersection| / |union|
    Robust and deterministic.
    """
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def word_overlap_similarity(text1: str, text2: str) -> float:
    """
    Word overlap: count of common words / geometric mean of lengths
    More lenient than Jaccard.
    """
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)

    # Count common words (with multiplicity)
    common = sum((counter1 & counter2).values())

    # Geometric mean of lengths
    denom = (len(tokens1) * len(tokens2)) ** 0.5

    return common / denom if denom > 0 else 0.0


def semantic_match_f1(
    predicted_steps: List[str],
    reference_steps: List[str],
    model: Any,  # SentenceTransformer instance (CPU-only for DeepSpeed)
    threshold: float = 0.70
) -> Tuple[float, int, int]:
    """
    Calculate F1 score using SEMANTIC similarity (SentenceTransformer + cosine).

    Reuses the same algorithm as MLLMReasoningEvaluator but takes a pre-initialized
    model as parameter for DeepSpeed compatibility (CPU-only, process-local).

    This captures semantic equivalence that word overlap misses:
    - "The light is green" ≈ "Traffic signal shows green" (HIGH score)
    - Word overlap would give LOW score due to different words

    Args:
        predicted_steps: List of predicted reasoning steps
        reference_steps: List of reference reasoning steps
        model: SentenceTransformer model (must be CPU-only for DeepSpeed)
        threshold: Cosine similarity threshold (0.70 recommended)

    Returns:
        (f1_score, matched_predictions, matched_references)
    """
    if not predicted_steps or not reference_steps:
        return 0.0, 0, 0

    # Compute embeddings (same as MLLMReasoningEvaluator._compute_embeddings)
    pred_embeddings = model.encode(predicted_steps, convert_to_tensor=False, show_progress_bar=False)
    ref_embeddings = model.encode(reference_steps, convert_to_tensor=False, show_progress_bar=False)

    # Compute cosine similarity matrix (same as MLLMReasoningEvaluator._compute_similarity_matrix)
    pred_norm = pred_embeddings / (np.linalg.norm(pred_embeddings, axis=1, keepdims=True) + 1e-8)
    ref_norm = ref_embeddings / (np.linalg.norm(ref_embeddings, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = np.dot(pred_norm, ref_norm.T)

    # Greedy matching (same as MLLMReasoningEvaluator._find_matches)
    similarities = []
    for i in range(len(predicted_steps)):
        for j in range(len(reference_steps)):
            if similarity_matrix[i, j] > threshold:
                similarities.append((similarity_matrix[i, j], i, j))

    similarities.sort(reverse=True)

    matched_preds = set()
    matched_refs = set()

    for sim, pred_idx, ref_idx in similarities:
        if pred_idx not in matched_preds and ref_idx not in matched_refs:
            matched_preds.add(pred_idx)
            matched_refs.add(ref_idx)

    # Calculate F1 (same as MLLMReasoningEvaluator.evaluate_single)
    n_pred = len(predicted_steps)
    n_ref = len(reference_steps)
    n_matched_pred = len(matched_preds)
    n_matched_ref = len(matched_refs)

    precision = n_matched_pred / n_pred if n_pred > 0 else 0.0
    recall = n_matched_ref / n_ref if n_ref > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, n_matched_pred, n_matched_ref


def best_match_f1(predicted_steps: List[str],
                  reference_steps: List[str],
                  threshold: float = 0.3) -> Tuple[float, int, int]:
    """
    Calculate F1 score for step matching using simple word overlap.

    Args:
        predicted_steps: List of predicted reasoning steps
        reference_steps: List of reference reasoning steps
        threshold: Similarity threshold for matching (0.3 is more lenient)

    Returns:
        (f1_score, matched_predictions, matched_references)
    """
    if not predicted_steps:
        return 0.0, 0, 0

    if not reference_steps:
        return 0.0, 0, 0

    # Calculate similarity matrix
    similarities = []
    for i, pred in enumerate(predicted_steps):
        for j, ref in enumerate(reference_steps):
            sim = word_overlap_similarity(pred, ref)
            if sim > threshold:
                similarities.append((sim, i, j))

    # Sort by descending similarity
    similarities.sort(reverse=True)

    # Greedy matching (no double assignments)
    matched_preds = set()
    matched_refs = set()

    for sim, pred_idx, ref_idx in similarities:
        if pred_idx not in matched_preds and ref_idx not in matched_refs:
            matched_preds.add(pred_idx)
            matched_refs.add(ref_idx)

    n_pred = len(predicted_steps)
    n_ref = len(reference_steps)
    n_matched_pred = len(matched_preds)
    n_matched_ref = len(matched_refs)

    precision = n_matched_pred / n_pred if n_pred > 0 else 0.0
    recall = n_matched_ref / n_ref if n_ref > 0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, n_matched_pred, n_matched_ref


if __name__ == "__main__":
    # Test the similarity functions
    pred = [
        "Noted blue cap on meter",
        "Read meter reads 'ALL'",
        "Detected black number '17' underneath"
    ]

    ref = [
        "The image shows a scene with visible text.",
        "Identify any text displayed in yellow color.",
        "Locate the number '17' displayed in yellow.",
        "Verify the presence of additional text.",
        "Ensure that the number '17' is prominently marked.",
        "Confirm that the text '17' is not obstructed."
    ]

    f1, matched_pred, matched_ref = best_match_f1(pred, ref, threshold=0.3)

    print(f"F1 Score: {f1:.3f}")
    print(f"Matched predictions: {matched_pred}/{len(pred)}")
    print(f"Matched references: {matched_ref}/{len(ref)}")
    print(f"Precision: {matched_pred/len(pred):.3f}")
    print(f"Recall: {matched_ref/len(ref):.3f}")
