# Compute Metrics Script

Fast computation of **accuracy** and **matchf1** metrics for prediction folders with multi-GPU support.

## Key Features

- ✅ Loads dataset using HuggingFace `load_from_disk`
- ✅ Creates placeholders for missing predictions (matchf1=0, accuracy=0)
- ✅ Multi-GPU support (default: 4 GPUs)
- ✅ Two evaluation modes: **with judge (LLM)** and **without judge (no LLM)**

## `use_judge` Parameter

### `use_judge=False` (Default - NO LLM)
- **Faster** evaluation
- Uses **rule-based matching** for accuracy:
  - Exact text matching (normalized)
  - Numeric matching with tolerance
  - Multiple choice extraction
  - Yes/No matching
- **No LLM calls** - cheaper and faster

### `use_judge=True` (USE LLM)
- **Slower** but more accurate
- Uses **LLM judge** (gpt-oss:120b) for semantic similarity
- Better at handling paraphrases and semantic equivalence
- **Makes LLM API calls** - more expensive

## Usage Examples

### 1. Run BOTH tests (recommended)
```bash
# Runs WITHOUT judge first, then WITH judge
python compute_metrics.py \
    /gpudata3/Wayner/reasoning/outputs_testing_llava7b_16 \
    --dataset-path /gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27 \
    --num-gpus 4 \
    --output-dir ./metrics_results \
    --both \
    --judge-model "gpt-oss:120b"
```

Or simply:
```bash
bash run_compute_metrics.sh
```

### 2. Run WITHOUT judge only (use_judge=False, NO LLM)
```bash
python compute_metrics.py \
    /gpudata3/Wayner/reasoning/outputs_testing_llava7b_16 \
    --dataset-path /gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27 \
    --num-gpus 4 \
    --output-dir ./metrics_results
```

Or:
```bash
bash run_compute_metrics_no_judge.sh
```

### 3. Run WITH judge only (use_judge=True, USE LLM)
```bash
python compute_metrics.py \
    /gpudata3/Wayner/reasoning/outputs_testing_llava7b_16 \
    --dataset-path /gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27 \
    --num-gpus 4 \
    --output-dir ./metrics_results \
    --use-judge \
    --judge-model "gpt-oss:120b"
```

Or:
```bash
bash run_compute_metrics_with_judge.sh
```

## Arguments

```
positional arguments:
  predictions_dir       Directory containing prediction JSON files

options:
  --dataset-path        Path to HuggingFace dataset folder
  --use-judge          If True: USE LLM judge for accuracy
                        If False: NO LLM, rule-based accuracy (default: False)
  --judge-model        LLM model to use (default: gpt-oss:120b)
  --num-gpus           Number of GPUs for parallel processing (default: 4)
  --output-dir         Directory to save results (default: ./metrics_results)
  --both               Run both tests: WITHOUT judge then WITH judge
```

## Output Files

When `--both` is used, creates two CSV files:
- `outputs_testing_llava7b_16_no_judge_metrics.csv` (use_judge=False, NO LLM)
- `outputs_testing_llava7b_16_with_judge_metrics.csv` (use_judge=True, USE LLM)

Each CSV contains per-sample metrics:
- `sample_idx`: Index of the sample
- `accuracy_correct`: Boolean, whether answer is correct
- `match_f1`: Match F1 score for reasoning steps
- `precision`: Precision of step matching
- `recall`: Recall of step matching
- `answer`: Predicted answer
- `match_type`: Type of matching used
- `confidence`: Confidence score
- `num_predicted_steps`: Number of predicted reasoning steps
- `num_reference_steps`: Number of reference reasoning steps

## Missing Predictions

If a prediction file doesn't exist for a dataset index, the script automatically creates a placeholder:
```json
{
  "reasoning_steps": [],
  "answer": "insufficient information"
}
```

For these samples:
- `matchf1 = 0`
- `accuracy = 0`

## GPU Configuration

The script uses GPUs 0, 1, 2, 3 by default. Set which GPUs to use:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Functions Used

- **Match F1**: Uses `mllm_evaluator.py` → `MLLMReasoningEvaluator.evaluate_single()`
- **Accuracy**: Uses `accuracy_calculator.py` → `AccuracyCalculator.evaluate_single()`
