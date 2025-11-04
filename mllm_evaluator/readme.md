
# MLLM Reasoning Evaluator

A production-ready evaluator for analyzing the quality of reasoning processes in Multimodal Large Language Models (MLLMs) using Match F1 metric and comprehensive answer accuracy verification.

## 🎯 Overview

The MLLM Reasoning Evaluator provides semantic evaluation of reasoning steps and answer correctness for multimodal language models. It uses Match F1 to measure reasoning quality and supports multiple answer formats for accuracy calculation.

### Key Features

- **Match F1 Metric**: Precision and recall of reasoning step matching using semantic similarity
- **Multi-Format Accuracy**: Handles yes/no, numeric, multiple choice, and free-form text answers
- **LLM-Based Grading**: Uses local LLM via Ollama (OpenAI API compatible) for semantic answer verification
- **GPU Acceleration**: CUDA support for fast large-scale evaluation
- **Flexible Normalization**: Robust answer extraction and normalization

## 📊 Metrics

### Match F1 (Reasoning Quality)

Measures how well predicted reasoning steps align with reference steps:

```
Precision = |matched_predictions| / |total_predictions|
Recall = |matched_references| / |total_references|
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**References:**
- Rajpurkar et al. (2016): SQuAD - F1 for answer matching
- Wang et al. (2023): Self-Consistency in reasoning
- Reimers & Gurevych (2019): Sentence-BERT for semantic similarity

### Accuracy (Answer Correctness)
a
Measures final answer correctness across multiple formats:
- **Yes/No**: Normalized boolean matching
- **Numeric**: Extracted number comparison (with tolerance)
- **Multiple Choice**: Letter extraction and matching (A, B, C, D)
- **Free-form Text**: LLM-based semantic verification

## 🚀 Installation

### Requirements

```bash
pip install sentence-transformers pandas numpy torch openai tqdm
```

### Ollama Setup (for LLM grader)

The accuracy calculator uses Ollama through the OpenAI-compatible API.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2

# Verify installation
ollama list
```

### Supported Ollama Models

- **llama3.2** (default, recommended)
- **llama3.1**
- **mistral**
- **phi3**
- **gemma2**

## 💡 Quick Start

### Basic Usage - Reasoning Evaluation

```python
from mllm_evaluator import MLLMReasoningEvaluator

evaluator = MLLMReasoningEvaluator(model_name="all-MiniLM-L6-v2")

predictions = {
    0: {
        "reasoning_steps": [
            "I observe three objects",
            "The middle object is smaller",
            "Therefore the answer is C"
        ],
        "answer": "C"
    }
}

ground_truth = {
    0: {
        "reference_steps": [
            "Three devices are visible",
            "Center device is compact",
            "Select option C"
        ],
        "answer": "C"
    }
}

results_df = evaluator.evaluate_dataset(predictions, ground_truth)
print(results_df[['match_f1', 'precision', 'recall']])
```

### Basic Usage - Accuracy Evaluation

```python
from accuracy_calculator import AccuracyCalculator

# Initialize with LLM grader (requires Ollama)
calculator = AccuracyCalculator(
    use_llm_grader=True,
    llm_model="llama3.2",
    base_url="http://localhost:11434/v1"
)

predictions = {
    0: {"question": "Is the light green?", "answer": "Yes"},
    1: {"question": "How many objects?", "answer": "The answer is 5"},
    2: {"question": "What color?", "answer": "The answer is (A)"}
}

ground_truth = {
    0: {"answer": "No"},
    1: {"answer": "5"},
    2: {"answer": "A"}
}

results = calculator.evaluate_dataset(predictions, ground_truth)
print(f"Accuracy: {results['overall_accuracy']:.3f}")
```

### Without LLM Grader (Fallback Mode)

```python
from accuracy_calculator import AccuracyCalculator

# Use only rule-based matching (no Ollama required)
calculator = AccuracyCalculator(use_llm_grader=False)

results = calculator.evaluate_dataset(predictions, ground_truth)
print(f"Accuracy: {results['overall_accuracy']:.3f}")
```

### Combined Evaluation

```python
from mllm_evaluator import MLLMReasoningEvaluator
from accuracy_calculator import AccuracyCalculator

# Evaluate reasoning
reasoning_eval = MLLMReasoningEvaluator()
reasoning_results = reasoning_eval.evaluate_dataset(predictions, ground_truth)

# Evaluate accuracy
accuracy_calc = AccuracyCalculator(use_llm_grader=True)
accuracy_results = accuracy_calc.evaluate_dataset(predictions, ground_truth)

print(f"Match F1: {reasoning_results['match_f1'].mean():.3f}")
print(f"Accuracy: {accuracy_results['overall_accuracy']:.3f}")
```

## 🤖 Available Models

### Embedding Models (Reasoning Evaluation)

#### Fast Models (Development)
```python
evaluator = MLLMReasoningEvaluator(model_name="all-MiniLM-L6-v2")
# Threshold: 0.45, Use case: Development, testing
```

#### Production Models (High Quality)
```python
evaluator = MLLMReasoningEvaluator(model_name="all-mpnet-base-v2")
# Threshold: 0.48, Use case: Production, research
```

### LLM Models (Accuracy Evaluation)

#### Recommended Models
```python
# Llama 3.2 (default, balanced)
calculator = AccuracyCalculator(llm_model="llama3.2")

# Llama 3.1 (larger, more accurate)
calculator = AccuracyCalculator(llm_model="llama3.1")

# Mistral (fast, efficient)
calculator = AccuracyCalculator(llm_model="mistral")

# Phi-3 (lightweight)
calculator = AccuracyCalculator(llm_model="phi3")
```

## 📝 Data Formats

### Predictions Format

```json
{
  "sample_id": {
    "question": "Question text",
    "reasoning_steps": ["Step 1", "Step 2", "Step 3"],
    "answer": "Final answer"
  }
}
```

### Ground Truth Format

```json
{
  "sample_id": {
    "reference_steps": ["Reference 1", "Reference 2"],
    "answer": "Ground truth answer"
  }
}
```

## 🎯 Answer Format Handling

### Yes/No Answers
```python
# Input variations
"Yes", "yes", "YES", "The answer is yes"
# Normalized to: "yes"

# Ground truth
"No", "no", "NO", "false"
# Normalized to: "no"
```

### Numeric Answers
```python
# Input variations
"5", "The answer is 5", "There are 5 objects", "Result: 5.0"
# Normalized to: 5.0

# Floating point
"3.14", "The value is 3.14", "approximately 3.14"
# Normalized to: 3.14
```

### Multiple Choice
```python
# Input variations
"A", "(A)", "A)", "a", "(a)", "a)", "The answer is A", "Option A"
# Normalized to: "A"

# All letters supported: A, B, C, D, E, F, etc.
```

### Free-form Text (LLM Verification)
```python
# Example 1
Predicted: "The capital of France is Paris"
Ground Truth: "Paris"
Result: ✅ Correct (LLM verifies semantic equivalence)

# Example 2
Predicted: "The sky appears blue"
Ground Truth: "blue"
Result: ✅ Correct (LLM extracts meaning)

# Example 3
Predicted: "It's raining"
Ground Truth: "sunny weather"
Result: ❌ Incorrect (LLM detects contradiction)
```

## 📊 Output Format

### Reasoning Results (DataFrame)

| Column | Description | Range |
|--------|-------------|-------|
| `match_f1` | F1 score of step matching | 0.0-1.0 |
| `precision` | Fraction of predictions matched | 0.0-1.0 |
| `recall` | Fraction of references matched | 0.0-1.0 |
| `num_matched_predictions` | Count of matched predictions | 0+ |
| `num_matched_references` | Count of matched references | 0+ |
| `avg_similarity` | Average similarity score | 0.0-1.0 |

### Accuracy Results (Dictionary)

```python
{
  'overall_accuracy': 0.85,
  'total_samples': 100,
  'correct_samples': 85,
  'type_statistics': {
    'exact': {'count': 20, 'correct': 19, 'accuracy': 0.950},
    'numeric': {'count': 30, 'correct': 28, 'accuracy': 0.933},
    'choice': {'count': 40, 'correct': 35, 'accuracy': 0.875},
    'llm_verified': {'count': 10, 'correct': 3, 'accuracy': 0.300}
  },
  'detailed_results': [
    {
      'sample_idx': 0,
      'is_correct': True,
      'predicted': 'Yes',
      'ground_truth': 'No',
      'normalized_pred': 'yes',
      'normalized_gt': 'no',
      'match_type': 'exact',
      'confidence': 1.0
    },
    # ... more results
  ]
}
```

## 🔧 Configuration

### Custom Threshold for Reasoning

```python
evaluator = MLLMReasoningEvaluator(
    model_name="all-MiniLM-L6-v2",
    similarity_threshold=0.5  # Custom threshold (default: 0.45)
)
```

### LLM Grader Configuration

```python
calculator = AccuracyCalculator(
    use_llm_grader=True,
    llm_model="llama3.2",  # Model name
    base_url="http://localhost:11434/v1"  # Ollama API endpoint
)
```

### Custom Ollama Endpoint

If Ollama is running on a different host or port:

```python
calculator = AccuracyCalculator(
    use_llm_grader=True,
    llm_model="llama3.2",
    base_url="http://192.168.1.100:11434/v1"  # Custom endpoint
)
```

### Disable LLM Grader

For faster evaluation without semantic verification:

```python
calculator = AccuracyCalculator(use_llm_grader=False)
# Falls back to exact/substring matching
# No Ollama required
```

## 📈 Interpreting Results

### Match F1 Scores
- **>0.7**: Excellent reasoning alignment
- **0.5-0.7**: Good alignment with room for improvement
- **<0.5**: Poor alignment, review reasoning quality

### Precision vs Recall
- **High Precision, Low Recall**: Correct but incomplete reasoning
- **Low Precision, High Recall**: Verbose but contains errors
- **Balanced**: Well-aligned reasoning

### Accuracy Scores
- **>0.9**: Excellent answer correctness
- **0.7-0.9**: Good performance
- **<0.7**: Needs improvement

### Match Types Distribution

Understanding which matching method was used:

```python
results = calculator.evaluate_dataset(predictions, ground_truth)

for match_type, stats in results['type_statistics'].items():
    print(f"{match_type}: {stats['accuracy']:.3f} ({stats['count']} samples)")

# Example output:
# exact: 0.950 (20 samples)
# numeric: 0.933 (30 samples)
# choice: 0.875 (40 samples)
# llm_verified: 0.300 (10 samples)
```

## 🧪 Running Tests

### Run Complete Demo

```bash
python demo.py
```

**Expected output:**
- Reasoning evaluation with Match F1
- Accuracy evaluation with multi-format support
- Combined metrics
- Model comparison

### Run Individual Components

```bash
# Test reasoning evaluator only
python -c "from mllm_evaluator import demo; demo()"

# Test accuracy calculator only
python -c "from accuracy_calculator import demo; demo()"
```

## 🔍 Troubleshooting

### Low Match F1
- Try lower similarity thresholds
- Check if reference steps are too specific
- Verify embedding model suits your domain

### Accuracy Issues

#### Pattern matching not working
- Ensure answers are properly formatted
- Check normalization rules match your data
- Add custom patterns if needed

#### LLM grader errors

**Error: Connection refused**
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve
```

**Error: Model not found**
```bash
# Pull the model
ollama pull llama3.2

# List available models
ollama list
```

**Error: Timeout**
```python
# The LLM grader has a 30-second timeout
# If responses are slow, check system resources
# Or use a smaller/faster model

calculator = AccuracyCalculator(llm_model="phi3")  # Faster model
```

### LLM Grader Not Working

1. **Check Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

2. **Test model directly:**
```bash
ollama run llama3.2 "Hello"
```

3. **Check Python OpenAI library:**
```bash
pip install --upgrade openai
```

4. **Fallback to rule-based matching:**
```python
calculator = AccuracyCalculator(use_llm_grader=False)
```

### Performance Issues

#### Slow LLM grading
- Use a smaller model: `phi3` or `mistral`
- Disable LLM grader for bulk processing
- Process in smaller batches

#### Memory issues
- Use CPU mode for embeddings
- Process dataset in chunks
- Use lighter embedding models

## 🎓 Advanced Usage

### Batch Processing with Progress

```python
from tqdm import tqdm
from accuracy_calculator import AccuracyCalculator

calculator = AccuracyCalculator(use_llm_grader=True)

# Process large dataset
results_list = []
for batch_start in tqdm(range(0, len(predictions), 100)):
    batch_end = batch_start + 100
    batch_pred = {k: predictions[k] for k in list(predictions.keys())[batch_start:batch_end]}
    batch_gt = {k: ground_truth[k] for k in list(ground_truth.keys())[batch_start:batch_end]}
    
    batch_results = calculator.evaluate_dataset(batch_pred, batch_gt)
    results_list.append(batch_results)
```

### Custom Answer Normalization

```python
from accuracy_calculator import AnswerNormalizer

class CustomNormalizer(AnswerNormalizer):
    @staticmethod
    def normalize_text(text: str) -> str:
        # Add custom normalization rules
        text = super().normalize_text(text)
        # Remove custom prefixes
        text = text.replace("my answer is:", "")
        return text

# Use custom normalizer
calculator = AccuracyCalculator(use_llm_grader=True)
calculator.normalizer = CustomNormalizer()
```

### Export Results

```python
import pandas as pd

# Reasoning results
reasoning_df = evaluator.evaluate_dataset(predictions, ground_truth)
reasoning_df.to_csv("reasoning_results.csv", index=False)

# Accuracy results
accuracy_results = calculator.evaluate_dataset(predictions, ground_truth)
accuracy_df = pd.DataFrame(accuracy_results['detailed_results'])
accuracy_df.to_csv("accuracy_results.csv", index=False)
```

## 📚 API Reference

### MLLMReasoningEvaluator

```python
class MLLMReasoningEvaluator:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: Optional[float] = None,
        device: Optional[str] = None,
        debug_mode: bool = False
    )
    
    def evaluate_single(
        self,
        predicted_steps: List[str],
        reference_steps: List[str],
        verbose: bool = None
    ) -> EvaluationMetrics
    
    def evaluate_dataset(
        self,
        predictions: Dict[int, Dict],
        ground_truth: Dict[int, Dict],
        verbose: bool = False
    ) -> pd.DataFrame
```

### AccuracyCalculator

```python
class AccuracyCalculator:
    def __init__(
        self,
        use_llm_grader: bool = True,
        llm_model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1"
    )
    
    def evaluate_single(
        self,
        question: str,
        predicted_answer: str,
        ground_truth_answer: str
    ) -> AccuracyResult
    
    def evaluate_dataset(
        self,
        predictions: Dict[int, Dict],
        ground_truth: Dict[int, Dict]
    ) -> Dict
```

## 🔗 Integration Examples

### Hugging Face Datasets

```python
from datasets import load_dataset
from mllm_evaluator import MLLMReasoningEvaluator
from accuracy_calculator import AccuracyCalculator

# Load dataset
dataset = load_dataset("your-dataset")

# Convert to evaluator format
predictions = {}
ground_truth = {}

for i, sample in enumerate(dataset["test"]):
    predictions[i] = {
        "question": sample["question"],
        "reasoning_steps": sample["model_reasoning"],
        "answer": sample["predicted_answer"]
    }
    ground_truth[i] = {
        "reference_steps": sample["gold_reasoning"],
        "answer": sample["gold_answer"]
    }

# Evaluate
reasoning_eval = MLLMReasoningEvaluator()
accuracy_calc = AccuracyCalculator()

reasoning_results = reasoning_eval.evaluate_dataset(predictions, ground_truth)
accuracy_results = accuracy_calc.evaluate_dataset(predictions, ground_truth)
```

## 📚 References

- Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- Ollama Documentation: https://ollama.com/
- OpenAI Python Library: https://github.com/openai/openai-python

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests.

## 🙏 Acknowledgments

- Sentence-Transformers for embedding models
- Ollama for local LLM inference
- OpenAI for the Python client library

---

**Version**: 2.0.0  
**Supported Python**: 3.8+  
**GPU Support**: CUDA 11.0+  
**LLM Backend**: Ollama (OpenAI API compatible)
