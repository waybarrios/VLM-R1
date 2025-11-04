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
        model_name: str = "all-MiniLM-L6-v2",
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
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name
        self.debug_mode = debug_mode
        
        # Model-specific optimized thresholds (empirically determined)
        self.model_thresholds = {
            "all-MiniLM-L6-v2": 0.45,
            "all-MiniLM-L12-v2": 0.47,
            "all-mpnet-base-v2": 0.48,
            "all-distilroberta-v1": 0.50,
            "paraphrase-multilingual-MiniLM-L12-v2": 0.43,
            "paraphrase-multilingual-mpnet-base-v2": 0.45
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
                     threshold: float) -> Tuple[set, set]:
        """
        Find optimal 1:1 matching between predicted and reference steps
        
        Uses greedy algorithm:
        1. Find all pairs with similarity > threshold
        2. Sort by descending similarity
        3. Assign matches greedily (no double assignments)
        """
        matched_refs = set()
        matched_preds = set()
        
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
                
                if self.debug_mode:
                    print(f"Match: P{pred_idx} <-> R{ref_idx} (sim: {sim:.3f})")
        
        return matched_preds, matched_refs
    
    def evaluate_single(self, 
                       predicted_steps: List[str], 
                       reference_steps: List[str],
                       verbose: bool = None) -> EvaluationMetrics:
        """
        Evaluate a single sample using Match F1 metric
        
        Match F1 = 2 * (Precision * Recall) / (Precision + Recall)
        where:
        - Precision = |matched_predictions| / |total_predictions|
        - Recall = |matched_references| / |total_references|
        
        Args:
            predicted_steps: List of predicted reasoning steps
            reference_steps: List of reference/ground truth reasoning steps
            verbose: Whether to show debug information
            
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
                threshold_used=self.similarity_threshold
            )
        
        pred_embeddings = self._compute_embeddings(predicted_steps)
        ref_embeddings = self._compute_embeddings(reference_steps)
        
        similarity_matrix = self._compute_similarity_matrix(pred_embeddings, ref_embeddings)
        
        matched_preds, matched_refs = self._find_matches(
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
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Precision: {precision:.3f} ({n_matched_preds}/{n_predicted})")
            print(f"  Recall: {recall:.3f} ({n_matched_refs}/{n_reference})")
            print(f"  Match F1: {match_f1:.3f}")
        
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
            threshold_used=self.similarity_threshold
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
    
    evaluator = MLLMReasoningEvaluator(model_name="all-MiniLM-L6-v2")
    
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