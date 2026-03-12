#!/usr/bin/env python3
"""
MLLM Reasoning Evaluator
Simplified evaluator focusing on Match F1 metric with semantic analysis
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    match_f1: float
    precision: float
    recall: float
    num_predicted_steps: int
    num_reference_steps: int
    num_matched_predictions: int
    num_matched_references: int
    avg_similarity: float
    max_similarity: float
    threshold_used: float
    # Ordered Match F1 fields (populated when alpha > 0)
    kendall_tau: float = 1.0
    tau_normalized: float = 1.0
    lis_ratio: float = 1.0
    order_score: float = 1.0  # normalized score from chosen order metric
    ordered_match_f1: float = 0.0
    alpha_used: float = 0.0
    order_metric_used: str = "none"


class MLLMReasoningEvaluator:
    """
    Simplified evaluator for MLLM reasoning processes using Match F1 metric
    
    Match F1 measures the quality of step matching between predicted and reference reasoning:
    - Precision: fraction of predicted steps that match a reference step
    - Recall: fraction of reference steps that are matched by a prediction
    - F1: harmonic mean of precision and recall
    
    References:
    - Rajpurkar et al. (2016): SQuAD uses F1 for token-level answer matching
    - Wang et al. (2023): Self-Consistency improves reasoning through answer aggregation
    - Reimers & Gurevych (2019): Sentence-BERT for semantic similarity
    """
    
    def __init__(
        self,
        model_name: str = "all-distilroberta-v1",
        similarity_threshold: Optional[float] = None,
        device: Optional[str] = None,
        debug_mode: bool = False
    ):
        """
        Initialize the evaluator
        
        Args:
            model_name: Sentence transformer model to use
            similarity_threshold: Custom threshold (None for model-optimized)
            device: Device to use ('auto', 'cuda', 'cpu')
            debug_mode: Enable detailed debugging output
        """
        
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if debug_mode:
            print(f"Initializing MLLM Evaluator on device: {self.device}")

        # Load model with explicit device to avoid multi-process tensor shape issues
        # Set cache folder to avoid conflicts in multi-process environments
        import os
        cache_folder = os.path.expanduser("~/.cache/sentence_transformers")

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_folder
        )

        # Ensure model is in eval mode and on correct device
        self.model.eval()
        if hasattr(self.model, '_first_module'):
            self.model._first_module().to(self.device)
        self.model_name = model_name
        self.debug_mode = debug_mode
        
        # Model-specific optimized thresholds (empirically determined)
        # Lowered thresholds to account for different reasoning granularities
        self.model_thresholds = {
            "all-MiniLM-L6-v2": 0.35,  # Lowered from 0.45 for VQA reasoning
            "all-MiniLM-L12-v2": 0.37,  # Lowered from 0.47
            "all-mpnet-base-v2": 0.38,  # Lowered from 0.48
            "all-distilroberta-v1": 0.35,  # Ablation-validated (τ=0.35, Paper Section 4.3)
            "paraphrase-multilingual-MiniLM-L12-v2": 0.33,  # Lowered from 0.43
            "paraphrase-multilingual-mpnet-base-v2": 0.35  # Lowered from 0.45
        }
        
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        else:
            self.similarity_threshold = self.model_thresholds.get(model_name, 0.45)
        
        if debug_mode:
            print(f"Model: {model_name}")
            print(f"Similarity threshold: {self.similarity_threshold}")
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute sentence embeddings for list of texts"""
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    
    def _compute_similarity_matrix(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity matrix between two sets of embeddings
        
        Cosine similarity: sim(A, B) = (A · B) / (||A|| * ||B||)
        """
        if embeddings1.size == 0 or embeddings2.size == 0:
            return np.array([[]])
        
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        return np.dot(embeddings1_norm, embeddings2_norm.T)
    
    def _find_matches(self,
                     similarity_matrix: np.ndarray,
                     threshold: float) -> Tuple[set, set, List[Tuple[int, int, float]]]:
        """
        Find optimal 1:1 matching between predicted and reference steps

        Uses greedy algorithm:
        1. Find all pairs with similarity > threshold
        2. Sort by descending similarity
        3. Assign matches greedily (no double assignments)

        Returns:
            matched_preds: set of matched prediction indices
            matched_refs: set of matched reference indices
            match_pairs: list of (pred_idx, ref_idx, similarity) tuples
        """
        matched_refs = set()
        matched_preds = set()
        match_pairs = []

        similarities = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                if similarity_matrix[i, j] > threshold:
                    similarities.append((similarity_matrix[i, j], i, j))

        similarities.sort(reverse=True)

        for sim, pred_idx, ref_idx in similarities:
            if pred_idx not in matched_preds and ref_idx not in matched_refs:
                matched_preds.add(pred_idx)
                matched_refs.add(ref_idx)
                match_pairs.append((pred_idx, ref_idx, sim))

                if self.debug_mode:
                    print(f"Match: P{pred_idx} <-> R{ref_idx} (sim: {sim:.3f})")

        return matched_preds, matched_refs, match_pairs

    @staticmethod
    def _compute_kendall_tau(match_pairs: List[Tuple[int, int, float]]) -> float:
        """
        Compute Kendall's Tau from matched pairs to measure order preservation.

        Sort matched pairs by reference index, then check if predicted indices
        are monotonically increasing (concordant) or not (discordant).

        Returns:
            tau in [-1, 1]. +1 = perfect order, 0 = random, -1 = reversed.
            Returns 1.0 if fewer than 2 matches (order undefined).
        """
        if len(match_pairs) < 2:
            return 1.0

        # Sort by reference index, extract predicted indices
        sorted_by_ref = sorted(match_pairs, key=lambda x: x[1])
        pred_indices = [p[0] for p in sorted_by_ref]

        # Count concordant and discordant pairs
        k = len(pred_indices)
        concordant = 0
        discordant = 0
        for i in range(k):
            for j in range(i + 1, k):
                if pred_indices[i] < pred_indices[j]:
                    concordant += 1
                elif pred_indices[i] > pred_indices[j]:
                    discordant += 1
                # ties are ignored

        total_pairs = k * (k - 1) / 2
        if total_pairs == 0:
            return 1.0

        return (concordant - discordant) / total_pairs

    @staticmethod
    def _compute_lis_ratio(match_pairs: List[Tuple[int, int, float]]) -> float:
        """
        Compute LIS (Longest Increasing Subsequence) ratio from matched pairs.

        Sort matched pairs by reference index, then find the longest increasing
        subsequence of predicted indices. The ratio LIS/k measures what fraction
        of matched steps are in the correct relative order.

        Returns:
            ratio in [0, 1]. 1.0 = all matched steps in correct order.
            Returns 1.0 if fewer than 2 matches (order undefined).
        """
        if len(match_pairs) < 2:
            return 1.0

        # Sort by reference index, extract predicted indices
        sorted_by_ref = sorted(match_pairs, key=lambda x: x[1])
        pred_indices = [p[0] for p in sorted_by_ref]

        # O(k log k) LIS using patience sorting
        from bisect import bisect_left
        tails = []
        for val in pred_indices:
            pos = bisect_left(tails, val)
            if pos == len(tails):
                tails.append(val)
            else:
                tails[pos] = val

        return len(tails) / len(pred_indices)

    def evaluate_single(self,
                       predicted_steps: List[str],
                       reference_steps: List[str],
                       verbose: bool = None,
                       alpha: float = 0.0,
                       order_metric: str = "kendall_tau") -> EvaluationMetrics:
        """
        Evaluate a single sample using Match F1 metric

        Match F1 = 2 * (Precision * Recall) / (Precision + Recall)
        where:
        - Precision = |matched_predictions| / |total_predictions|
        - Recall = |matched_references| / |total_references|

        When alpha > 0, computes Ordered Match F1:
        Ordered_F1 = F1 * ((1 - alpha) + alpha * order_score)

        Args:
            predicted_steps: List of predicted reasoning steps
            reference_steps: List of reference/ground truth reasoning steps
            verbose: Whether to show debug information
            alpha: Order sensitivity in [0, 1]. 0 = ignore order, 0.3 = recommended.
            order_metric: "kendall_tau" or "lis" (Longest Increasing Subsequence ratio).

        Returns:
            EvaluationMetrics object with Match F1 and related metrics
        """
        if verbose is None:
            verbose = self.debug_mode
            
        if not reference_steps:
            raise ValueError("Reference steps cannot be empty")
        
        if not predicted_steps:
            return EvaluationMetrics(
                match_f1=0.0,
                precision=0.0,
                recall=0.0,
                num_predicted_steps=0,
                num_reference_steps=len(reference_steps),
                num_matched_predictions=0,
                num_matched_references=0,
                avg_similarity=0.0,
                max_similarity=0.0,
                threshold_used=self.similarity_threshold,
                alpha_used=alpha,
                order_metric_used=order_metric
            )

        pred_embeddings = self._compute_embeddings(predicted_steps)
        ref_embeddings = self._compute_embeddings(reference_steps)

        similarity_matrix = self._compute_similarity_matrix(pred_embeddings, ref_embeddings)

        matched_preds, matched_refs, match_pairs = self._find_matches(
            similarity_matrix, self.similarity_threshold)

        n_predicted = len(predicted_steps)
        n_reference = len(reference_steps)
        n_matched_preds = len(matched_preds)
        n_matched_refs = len(matched_refs)

        precision = n_matched_preds / n_predicted if n_predicted > 0 else 0.0
        recall = n_matched_refs / n_reference if n_reference > 0 else 0.0
        match_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_similarity = np.mean(similarity_matrix) if similarity_matrix.size > 0 else 0.0
        max_similarity = np.max(similarity_matrix) if similarity_matrix.size > 0 else 0.0

        # Compute both order metrics (always, for comparison)
        tau = self._compute_kendall_tau(match_pairs)
        tau_norm = (tau + 1.0) / 2.0
        lis = self._compute_lis_ratio(match_pairs)

        # Select which metric drives the ordered F1
        if order_metric == "lis":
            order_score = lis
        else:  # kendall_tau
            order_score = tau_norm

        ordered_f1 = match_f1 * ((1.0 - alpha) + alpha * order_score) if alpha > 0 else match_f1

        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Precision: {precision:.3f} ({n_matched_preds}/{n_predicted})")
            print(f"  Recall: {recall:.3f} ({n_matched_refs}/{n_reference})")
            print(f"  Match F1: {match_f1:.3f}")
            if alpha > 0:
                print(f"  Kendall's Tau: {tau:.3f} (normalized: {tau_norm:.3f})")
                print(f"  LIS ratio: {lis:.3f}")
                print(f"  Order metric: {order_metric} (score: {order_score:.3f})")
                print(f"  Ordered Match F1 (alpha={alpha}): {ordered_f1:.3f}")

        return EvaluationMetrics(
            match_f1=match_f1,
            precision=precision,
            recall=recall,
            num_predicted_steps=n_predicted,
            num_reference_steps=n_reference,
            num_matched_predictions=n_matched_preds,
            num_matched_references=n_matched_refs,
            avg_similarity=avg_similarity,
            max_similarity=max_similarity,
            threshold_used=self.similarity_threshold,
            kendall_tau=tau,
            tau_normalized=tau_norm,
            lis_ratio=lis,
            order_score=order_score,
            ordered_match_f1=ordered_f1,
            alpha_used=alpha,
            order_metric_used=order_metric
        )
    
    def evaluate_dataset(self,
                        predictions: Dict[int, Dict],
                        ground_truth: Dict[int, Dict],
                        verbose: bool = False) -> pd.DataFrame:
        """
        Evaluate entire dataset with Match F1 metric
        
        Args:
            predictions: Dict mapping sample_id to prediction data
                        Format: {id: {"reasoning_steps": [...], "answer": "..."}}
            ground_truth: Dict mapping sample_id to ground truth data
                         Format: {id: {"reference_steps": [...]}}
            verbose: Whether to show progress information
            
        Returns:
            DataFrame with evaluation results for each sample
        """
        pred_indices = set(predictions.keys())
        gt_indices = set(ground_truth.keys())
        
        if pred_indices != gt_indices:
            missing_pred = gt_indices - pred_indices
            missing_gt = pred_indices - gt_indices
            if missing_pred:
                print(f"Warning: Missing predictions for indices: {missing_pred}")
            if missing_gt:
                print(f"Warning: Missing ground truth for indices: {missing_gt}")
        
        common_indices = pred_indices.intersection(gt_indices)
        print(f"Evaluating {len(common_indices)} samples...")
        
        results = []
        
        for idx in tqdm(common_indices, disable=not verbose):
            try:
                pred_steps = predictions[idx]["reasoning_steps"]
                ref_steps = ground_truth[idx]["reference_steps"]
                answer = predictions[idx].get("answer", "")
                
                metrics = self.evaluate_single(pred_steps, ref_steps, verbose=False)
                
                result = {
                    'sample_idx': idx,
                    'answer': answer,
                    'match_f1': metrics.match_f1,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'num_predicted_steps': metrics.num_predicted_steps,
                    'num_reference_steps': metrics.num_reference_steps,
                    'num_matched_predictions': metrics.num_matched_predictions,
                    'num_matched_references': metrics.num_matched_references,
                    'avg_similarity': metrics.avg_similarity,
                    'max_similarity': metrics.max_similarity,
                    'threshold_used': metrics.threshold_used
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating sample {idx}: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            print(f"\n=== Evaluation Summary ===")
            print(f"Samples evaluated: {len(df)}")
            print(f"Model: {self.model_name} (threshold: {self.similarity_threshold:.3f})")
            print(f"Average Match F1: {df['match_f1'].mean():.3f} (±{df['match_f1'].std():.3f})")
            print(f"Average Precision: {df['precision'].mean():.3f}")
            print(f"Average Recall: {df['recall'].mean():.3f}")
        
        return df
    
    def generate_summary(self, results_df: pd.DataFrame) -> Dict:
        """Generate statistical summary of evaluation results"""
        if len(results_df) == 0:
            return {}
        
        summary = {
            'total_samples': len(results_df),
            'model_name': self.model_name,
            'similarity_threshold': self.similarity_threshold,
            'avg_match_f1': results_df['match_f1'].mean(),
            'std_match_f1': results_df['match_f1'].std(),
            'median_match_f1': results_df['match_f1'].median(),
            'avg_precision': results_df['precision'].mean(),
            'avg_recall': results_df['recall'].mean(),
            'top_10_percent_threshold': results_df['match_f1'].quantile(0.9),
            'bottom_10_percent_threshold': results_df['match_f1'].quantile(0.1)
        }
        
        return summary


def load_json_data(file_path: str) -> Dict:
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(results_df: pd.DataFrame, output_path: str):
    """Save evaluation results to CSV file"""
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def demo():
    """Demonstration of the MLLM Reasoning Evaluator"""
    print("🚀 MLLM Reasoning Evaluator Demo (Match F1 Focus)")
    print("=" * 50)
    
    predictions = {
        0: {
            "reasoning_steps": [
                "I observe three objects on a surface",
                "The middle object is white and smaller",
                "Therefore the answer is C"
            ],
            "answer": "C"
        }
    }
    
    ground_truth = {
        0: {
            "reference_steps": [
                "Three devices are on a desk",
                "Center device is white and compact",
                "Size comparison shows center is smallest",
                "Select option C"
            ]
        }
    }
    
    evaluator = MLLMReasoningEvaluator(model_name="all-distilroberta-v1")
    
    results_df = evaluator.evaluate_dataset(predictions, ground_truth, verbose=True)
    
    if len(results_df) > 0:
        print("\nResults:")
        print(results_df[['match_f1', 'precision', 'recall']].round(3))
        
        summary = evaluator.generate_summary(results_df)
        print(f"\nSummary:")
        print(f"Match F1: {summary['avg_match_f1']:.3f}")
        print(f"Precision: {summary['avg_precision']:.3f}")
        print(f"Recall: {summary['avg_recall']:.3f}")
    
    return evaluator, results_df


if __name__ == "__main__":
    demo()