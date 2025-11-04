#!/usr/bin/env python3
"""
Accuracy Calculator for MLLM Evaluation
Handles various answer formats: yes/no, numbers, multiple choice, text
Uses OpenAI API compatible interface for Ollama
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class AccuracyResult:
    """Container for accuracy evaluation results"""
    is_correct: bool
    predicted_answer: str
    ground_truth_answer: str
    normalized_prediction: str
    normalized_ground_truth: str
    match_type: str  # 'exact', 'numeric', 'numeric_rounded', 'choice', 'llm_verified'
    confidence: float = 1.0


class AnswerNormalizer:
    """Normalize answers to standard format"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Basic text normalization: lowercase, remove extra spaces, punctuation"""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip().lower()
        # Remove periods that are NOT part of decimal numbers (e.g., sentence-ending periods)
        text = re.sub(r'(?<!\d)\.(?!\d)', '', text)  # Remove periods not between digits
        # Remove other common punctuation
        text = re.sub(r'[,()\[\]{}]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """Extract numeric value from text"""
        # Don't use normalize_text here as it might affect decimal points
        if not isinstance(text, str):
            text = str(text)
        text = text.strip().lower()
        
        # Remove common prefix patterns
        patterns = [
            r'the answer is:?\s*',
            r'the result is:?\s*',
            r'answer:?\s*',
            r'result:?\s*',
            r'approximately:?\s*',
            r'about:?\s*',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Try to find numbers (including decimals and negatives)
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        return None
    
    @staticmethod
    def compare_numbers(predicted: float, ground_truth: float, 
                       relative_tolerance: float = 0.1,
                       absolute_tolerance: float = 0.05,
                       delta: float = 1e-10) -> Tuple[bool, str, float]:
        """
        Compare two numbers with flexible tolerance for rounding
        
        Implements the tolerance criterion from Equation (2):
        |pred - gt| <= eps_abs  OR  |pred - gt| / max(|gt|, delta) <= eps_rel
        
        Strategy:
        1. Exact match -> confidence 1.0
        2. Very close (abs diff < 0.01) -> confidence 1.0
        3. Within absolute tolerance -> confidence 0.95
        4. Within relative tolerance (10% by default) -> confidence 0.9
        5. Same order of magnitude and close -> confidence 0.8
        6. Otherwise -> not correct
        
        Args:
            predicted: Predicted numeric value
            ground_truth: Ground truth numeric value
            relative_tolerance: Relative error tolerance (default 10%)
            absolute_tolerance: Absolute difference tolerance (default 0.05)
            delta: Small value to prevent division by zero (default 1e-10)
            
        Returns:
            (is_correct, match_type, confidence)
        """
        # Handle exact match
        if predicted == ground_truth:
            return True, "numeric_exact", 1.0
        
        # Calculate differences
        abs_diff = abs(predicted - ground_truth)
        
        # Apply tolerance criterion from Equation (2)
        # Check absolute tolerance first
        if abs_diff <= absolute_tolerance:
            return True, "numeric_rounded", 0.95
        
        # Check relative tolerance with delta to prevent division by zero
        denominator = max(abs(ground_truth), delta)
        rel_diff = abs_diff / denominator
        
        if rel_diff <= relative_tolerance:
            return True, "numeric_rounded", 0.9
        
        # Very close match (e.g., 0.67 vs 0.6723...)
        if abs_diff < 0.01:
            return True, "numeric_rounded", 1.0
        
        # Check if same order of magnitude and reasonably close
        # E.g., 0.6 vs 0.67234... (both between 0.1 and 1.0)
        if ground_truth != 0:
            # Determine number of decimal places in prediction
            pred_str = str(predicted).rstrip('0').rstrip('.')
            if '.' in pred_str:
                pred_decimals = len(pred_str.split('.')[1])
            else:
                pred_decimals = 0
            
            # If prediction has fewer decimals, it's likely rounded
            # Allow more tolerance
            if pred_decimals <= 2:
                # For predictions with 1-2 decimal places
                if rel_diff < 0.15:  # 15% tolerance for rounded answers
                    return True, "numeric_rounded", 0.85
                
                # Check if prediction rounds to ground truth at that precision
                rounded_gt = round(ground_truth, pred_decimals)
                if abs(predicted - rounded_gt) < 1e-6:
                    return True, "numeric_rounded", 0.95
        
        return False, "numeric_mismatch", 0.0
    
    @staticmethod
    def extract_choice(text: str) -> Optional[str]:
        """
        Extract multiple choice answer (A, B, C, D, etc.)
        
        Handles two cases:
        1. Short answer (single letter or letter with minimal formatting)
        2. Long answer (e.g., "a) option-text") - extract only the letter
        """
        original_text = text
        text = AnswerNormalizer.normalize_text(text)
        
        # Case 1: If text is very short (1-5 chars), likely just a letter answer
        if len(text) <= 5:
            # Match patterns like: a, (a), a), [a], {a}
            match = re.search(r'([a-z])', text)
            if match:
                return match.group(1).upper()
        
        # Case 2: Longer text - extract letter from patterns like "a) option-text"
        # Try multiple patterns in order of specificity
        patterns = [
            r'^([a-z])\)',  # Matches "a) text"
            r'^\(([a-z])\)',  # Matches "(a) text"
            r'^([a-z])\.',  # Matches "a. text"
            r'option\s+([a-z])',  # Matches "option a"
            r'choice\s+([a-z])',  # Matches "choice a"
            r'answer\s+is\s+([a-z])',  # Matches "answer is a"
            r'\(([a-z])\)',  # Matches "(a)" anywhere
            r'\b([a-z])\)',  # Matches "a)" as a word
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).upper()
        
        return None
    
    @staticmethod
    def is_yes_no_question(text: str) -> bool:
        """Check if answer is yes/no type"""
        text = AnswerNormalizer.normalize_text(text)
        return text in ['yes', 'no', 'true', 'false']
    
    @staticmethod
    def normalize_yes_no(text: str) -> str:
        """Normalize yes/no answers"""
        text = AnswerNormalizer.normalize_text(text)
        
        # Handle common variations
        yes_patterns = ['yes', 'true', 'correct', 'affirmative']
        no_patterns = ['no', 'false', 'incorrect', 'negative']
        
        for pattern in yes_patterns:
            if pattern in text:
                return 'yes'
        
        for pattern in no_patterns:
            if pattern in text:
                return 'no'
        
        return text


class LLMGrader:
    """Use local LLM via Ollama (OpenAI API compatible) to verify text answers"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434/v1"):
        """
        Initialize LLM grader with OpenAI compatible API
        
        Args:
            model: Ollama model name (e.g., "llama3.2", "mistral", "phi3")
            base_url: Ollama API base URL
        """
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key='ollama',  # required, but unused
        )
    
    def verify_answer(self, question: str, predicted: str, ground_truth: str) -> Tuple[bool, float]:
        """
        Use LLM to verify if predicted answer matches ground truth semantically
        
        Returns:
            (is_correct, confidence)
        """
        system_prompt = """You are an expert evaluator for question answering systems.
Your task is to determine if two answers are semantically equivalent.
Consider synonyms, paraphrases, and different phrasings of the same content.
Respond ONLY with a JSON object containing two fields:
- "correct": boolean (true if answers are equivalent, false otherwise)
- "confidence": float between 0.0 and 1.0 (your confidence in the judgment)

Example responses:
{"correct": true, "confidence": 0.95}
{"correct": false, "confidence": 0.85}"""

        user_prompt = f"""Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted}

Are these answers semantically equivalent?"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                # Remove markdown code blocks if present
                response_text = re.sub(r'```json\s*|\s*```', '', response_text)
                parsed = json.loads(response_text)
                
                is_correct = parsed.get('correct', False)
                confidence = float(parsed.get('confidence', 0.5))
                
                # Ensure confidence is in valid range
                confidence = max(0.0, min(1.0, confidence))
                
                return is_correct, confidence
                
            except json.JSONDecodeError:
                # Fallback: look for boolean keywords in response
                response_lower = response_text.lower()
                
                if 'true' in response_lower or '"correct": true' in response_lower:
                    return True, 0.7
                elif 'false' in response_lower or '"correct": false' in response_lower:
                    return False, 0.7
                
                # If we can't parse, fall back to string matching
                pred_norm = AnswerNormalizer.normalize_text(predicted)
                gt_norm = AnswerNormalizer.normalize_text(ground_truth)
                
                if pred_norm == gt_norm:
                    return True, 1.0
                elif pred_norm in gt_norm or gt_norm in pred_norm:
                    return True, 0.8
                return False, 0.3
            
        except Exception as e:
            print(f"LLM grading error: {e}")
            
            # Fallback to simple string matching
            pred_norm = AnswerNormalizer.normalize_text(predicted)
            gt_norm = AnswerNormalizer.normalize_text(ground_truth)
            
            if pred_norm == gt_norm:
                return True, 1.0
            elif pred_norm in gt_norm or gt_norm in pred_norm:
                return True, 0.8
            return False, 0.3


class AccuracyCalculator:
    """Calculate accuracy for MLLM predictions"""
    
    def __init__(self, 
                 use_llm_grader: bool = True, 
                 llm_model: str = "llama3.2", 
                 base_url: str = "http://localhost:11434/v1",
                 numeric_relative_tolerance: float = 0.1,
                 numeric_absolute_tolerance: float = 0.05,
                 delta: float = 1e-10):
        """
        Initialize accuracy calculator
        
        Args:
            use_llm_grader: Whether to use LLM for text answer verification
            llm_model: Ollama model name
            base_url: Ollama API base URL
            numeric_relative_tolerance: Relative error tolerance for numbers (default 10%)
            numeric_absolute_tolerance: Absolute difference tolerance (default 0.05)
            delta: Small value to prevent division by zero (default 1e-10)
        """
        self.normalizer = AnswerNormalizer()
        self.use_llm_grader = use_llm_grader
        self.llm_grader = LLMGrader(model=llm_model, base_url=base_url) if use_llm_grader else None
        self.numeric_relative_tolerance = numeric_relative_tolerance
        self.numeric_absolute_tolerance = numeric_absolute_tolerance
        self.delta = delta
    
    def evaluate_single(self, 
                       question: str,
                       predicted_answer: str, 
                       ground_truth_answer: str) -> AccuracyResult:
        """
        Evaluate a single prediction against ground truth
        
        Implements Equation (1) where C(pred, gt) returns 1 if correct, 0 otherwise.
        
        Handles multiple answer types:
        - Numeric answers (with tolerance as per Equation 2)
        - Multiple choice (A, B, C, D) - with smart extraction
        - Yes/No questions
        - Free-form text (using LLM grader for semantic equivalence)
        - Exact text matching with normalization
        """
        pred_norm = self.normalizer.normalize_text(predicted_answer)
        gt_norm = self.normalizer.normalize_text(ground_truth_answer)
        
        # 1. Try numeric matching with tolerance (Equation 2)
        pred_num = self.normalizer.extract_number(predicted_answer)
        gt_num = self.normalizer.extract_number(ground_truth_answer)
        
        if pred_num is not None and gt_num is not None:
            is_correct, match_type, confidence = self.normalizer.compare_numbers(
                pred_num, gt_num,
                relative_tolerance=self.numeric_relative_tolerance,
                absolute_tolerance=self.numeric_absolute_tolerance,
                delta=self.delta
            )
            return AccuracyResult(
                is_correct=is_correct,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=str(pred_num),
                normalized_ground_truth=str(gt_num),
                match_type=match_type,
                confidence=confidence
            )
        
        # 2. Try yes/no matching (must come before choice to avoid "Yes" -> "Y")
        if self.normalizer.is_yes_no_question(gt_norm):
            pred_yn = self.normalizer.normalize_yes_no(predicted_answer)
            gt_yn = self.normalizer.normalize_yes_no(ground_truth_answer)
            
            is_correct = pred_yn == gt_yn
            return AccuracyResult(
                is_correct=is_correct,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=pred_yn,
                normalized_ground_truth=gt_yn,
                match_type='yes_no',
                confidence=1.0
            )
        
        # 3. Try multiple choice matching
        pred_choice = self.normalizer.extract_choice(predicted_answer)
        gt_choice = self.normalizer.extract_choice(ground_truth_answer)
        
        if pred_choice is not None and gt_choice is not None:
            is_correct = pred_choice == gt_choice
            return AccuracyResult(
                is_correct=is_correct,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=pred_choice,
                normalized_ground_truth=gt_choice,
                match_type='choice',
                confidence=1.0
            )
        
        # 4. Try exact text matching (after normalization)
        if pred_norm == gt_norm:
            return AccuracyResult(
                is_correct=True,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=pred_norm,
                normalized_ground_truth=gt_norm,
                match_type='exact',
                confidence=1.0
            )
        
        # 5. For longer text answers, use LLM grader for semantic matching
        # Use LLM judge for sentences/paragraphs, but not for single-word mismatches
        gt_words = gt_norm.split()
        pred_words = pred_norm.split()
        
        # If both are single words and don't match, it's incorrect (no need for LLM)
        if len(gt_words) == 1 and len(pred_words) == 1:
            return AccuracyResult(
                is_correct=False,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=pred_norm,
                normalized_ground_truth=gt_norm,
                match_type='exact',
                confidence=1.0
            )
        
        # For longer answers, use LLM grader
        if self.use_llm_grader and self.llm_grader:
            is_correct, confidence = self.llm_grader.verify_answer(
                question, predicted_answer, ground_truth_answer
            )
            return AccuracyResult(
                is_correct=is_correct,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                normalized_prediction=pred_norm,
                normalized_ground_truth=gt_norm,
                match_type='llm_verified',
                confidence=confidence
            )
        
        # 6. Fallback: substring matching
        is_correct = (pred_norm in gt_norm or gt_norm in pred_norm)
        return AccuracyResult(
            is_correct=is_correct,
            predicted_answer=predicted_answer,
            ground_truth_answer=ground_truth_answer,
            normalized_prediction=pred_norm,
            normalized_ground_truth=gt_norm,
            match_type='exact',
            confidence=0.7 if is_correct else 0.3
        )
    
    def evaluate_dataset(self, 
                        predictions: Dict[int, Dict],
                        ground_truth: Dict[int, Dict]) -> Dict:
        """
        Evaluate entire dataset
        
        Implements Equation (1): Accuracy = (1/N) * sum(C(pred_i, gt_i))
        
        Args:
            predictions: {id: {"answer": "...", "question": "..."}}
            ground_truth: {id: {"answer": "..."}}
        
        Returns:
            Dictionary with accuracy metrics and detailed results
        """
        results = []
        
        for idx in tqdm(predictions.keys(), desc="Evaluating accuracy", unit="sample"):

            if idx not in ground_truth:
                continue
            
            question = predictions[idx].get("question", "")
            pred_answer = predictions[idx].get("answer", "")
            gt_answer = ground_truth[idx].get("answer", "")
            
            result = self.evaluate_single(question, pred_answer, gt_answer)
            
            results.append({
                'sample_idx': idx,
                'is_correct': result.is_correct,
                'predicted': result.predicted_answer,
                'ground_truth': result.ground_truth_answer,
                'normalized_pred': result.normalized_prediction,
                'normalized_gt': result.normalized_ground_truth,
                'match_type': result.match_type,
                'confidence': result.confidence
            })
        
        # Calculate metrics - Equation (1)
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = correct / total if total > 0 else 0.0
        
        # Weighted accuracy by confidence
        weighted_correct = sum(r['confidence'] for r in results if r['is_correct'])
        weighted_accuracy = weighted_correct / total if total > 0 else 0.0
        
        # Per-type accuracy
        type_stats = {}
        all_match_types = set(r['match_type'] for r in results)
        
        for match_type in all_match_types:
            type_results = [r for r in results if r['match_type'] == match_type]
            if type_results:
                type_correct = sum(1 for r in type_results if r['is_correct'])
                avg_confidence = sum(r['confidence'] for r in type_results) / len(type_results)
                type_stats[match_type] = {
                    'count': len(type_results),
                    'correct': type_correct,
                    'accuracy': type_correct / len(type_results),
                    'avg_confidence': avg_confidence
                }
        
        return {
            'overall_accuracy': accuracy,
            'weighted_accuracy': weighted_accuracy,
            'total_samples': total,
            'correct_samples': correct,
            'type_statistics': type_stats,
            'detailed_results': results
        }


def demo():
    """Demo of accuracy calculator with numeric rounding examples"""
    print("🎯 Accuracy Calculator Demo (OpenAI API + Numeric Rounding)")
    print("=" * 60)
    
    calculator = AccuracyCalculator(
        use_llm_grader=True, 
        llm_model="llama3.2",
        numeric_relative_tolerance=0.1,  # 10% tolerance
        numeric_absolute_tolerance=0.05,  # 0.05 absolute tolerance
        delta=1e-10  # Small value to prevent division by zero
    )
    
    # Test cases focusing on numeric rounding
    test_cases = [
        # Numeric rounding examples
        {
            'question': 'What is the probability?',
            'predicted': '0.67',
            'ground_truth': '0.67234572345763',
            'expected': '✅ (rounded to 2 decimals)'
        },
        {
            'question': 'What is the value?',
            'predicted': '0.6',
            'ground_truth': '0.67234572345763',
            'expected': '✅ (rounded to 1 decimal, ~10% diff)'
        },
        {
            'question': 'What is the exact value?',
            'predicted': '0.67234572345763',
            'ground_truth': '0.67234572345763',
            'expected': '✅ (exact match)'
        },
        {
            'question': 'What is the percentage?',
            'predicted': 'The answer is 0.5',
            'ground_truth': '0.45',
            'expected': '✅ (within tolerance)'
        },
        {
            'question': 'Count the objects',
            'predicted': '5',
            'ground_truth': '5.0',
            'expected': '✅ (integer vs float)'
        },
        {
            'question': 'What is 0.9999 rounded?',
            'predicted': '1',
            'ground_truth': '0.9999',
            'expected': '✅ (very close to 1)'
        },
        # Yes/No
        {
            'question': 'Is the traffic light green?',
            'predicted': 'Yes',
            'ground_truth': 'No',
            'expected': '❌'
        },
        # Multiple choice - short answer
        {
            'question': 'What color is the sky?',
            'predicted': 'A',
            'ground_truth': 'A',
            'expected': '✅'
        },
        # Multiple choice - with formatting
        {
            'question': 'What color is the sky?',
            'predicted': '(A)',
            'ground_truth': 'A',
            'expected': '✅'
        },
        # Multiple choice - long answer with option text
        {
            'question': 'What is the capital of France?',
            'predicted': 'a) Paris',
            'ground_truth': 'A',
            'expected': '✅ (extracted letter from option)'
        },
        # Text with LLM - long answer
        {
            'question': 'What is the capital of France?',
            'predicted': 'The capital of France is Paris',
            'ground_truth': 'Paris',
            'expected': '✅ (LLM verified)'
        },
        # Single word mismatch - no LLM needed
        {
            'question': 'What is the capital of France?',
            'predicted': 'London',
            'ground_truth': 'Paris',
            'expected': '❌ (single word mismatch)'
        },
    ]
    
    print("\nTest Cases:\n")
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test.get('expected', '')}")
        print(f"Question: {test['question']}")
        print(f"Predicted: {test['predicted']}")
        print(f"Ground Truth: {test['ground_truth']}")
        
        result = calculator.evaluate_single(
            test['question'],
            test['predicted'],
            test['ground_truth']
        )
        
        status = '✅ Correct' if result.is_correct else '❌ Incorrect'
        print(f"Result: {status}")
        print(f"Match Type: {result.match_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Normalized - Pred: {result.normalized_prediction}, GT: {result.normalized_ground_truth}")
        print("-" * 60)
    
    # Demonstrate tolerance settings
    print("\n\n📊 Tolerance Settings Demo")
    print("=" * 60)
    
    test_numeric = {
        'question': 'What is the value?',
        'predicted': '0.6',
        'ground_truth': '0.67234572345763'
    }
    
    tolerances = [
        (0.05, 0.01),  # Strict
        (0.10, 0.05),  # Default
        (0.20, 0.10),  # Lenient
    ]
    
    for rel_tol, abs_tol in tolerances:
        calc = AccuracyCalculator(
            use_llm_grader=False,
            numeric_relative_tolerance=rel_tol,
            numeric_absolute_tolerance=abs_tol
        )
        
        result = calc.evaluate_single(
            test_numeric['question'],
            test_numeric['predicted'],
            test_numeric['ground_truth']
        )
        
        print(f"\nTolerance: rel={rel_tol:.0%}, abs={abs_tol:.2f}")
        print(f"  Result: {'✅ Correct' if result.is_correct else '❌ Incorrect'}")
        print(f"  Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    demo()