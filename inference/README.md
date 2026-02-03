# VLM Checkpoint Inference

This directory contains scripts for running inference on trained VLM checkpoints.

## Overview

The inference pipeline:
1. Automatically finds all checkpoints in a training output directory
2. Loads each checkpoint model
3. Runs inference on a test dataset
4. Generates predictions in JSON format with `reasoning_steps` and `answer` keys
5. Saves individual predictions as `0.json`, `1.json`, etc. (indexed by dataset position)

## Quick Start

### Basic Usage

```bash
cd /gpudata3/Wayner/VLM-R1/inference
./run_inference.sh
```

This will use default paths:
- **Output directory**: `/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040`
- **Test dataset**: `reasoning_test_with_reference_steps_updated_v27`
- **Predictions directory**: `predictions/`

### Custom Paths

```bash
./run_inference.sh \
    /path/to/output/directory \
    /path/to/test/dataset \
    /path/to/predictions/directory
```

### Process Specific Checkpoint Only

```bash
CHECKPOINT_FILTER="checkpoint-500" ./run_inference.sh
```

### Use Specific GPUs

```bash
GPU_IDS="0,1" ./run_inference.sh
```

### Adjust Inference Parameters

```bash
BATCH_SIZE=2 \
MAX_NEW_TOKENS=1024 \
TEMPERATURE=0.5 \
TOP_P=0.95 \
./run_inference.sh
```

## Direct Python Usage

For more control, you can use the Python script directly:

```bash
python run_checkpoint_inference.py \
    --output_dir /gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040 \
    --test_dataset_path reasoning_test_with_reference_steps_updated_v27 \
    --predictions_base_dir predictions \
    --batch_size 1 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --device_ids 0,1,2,3 \
    --checkpoint_filter "checkpoint-1000"
```

### Arguments

- `--output_dir`: Path to training output directory containing checkpoints (required)
- `--test_dataset_path`: Path to test dataset in HuggingFace format (required)
- `--predictions_base_dir`: Base directory for saving predictions (default: `predictions`)
- `--batch_size`: Batch size for inference (default: `1`)
- `--max_new_tokens`: Maximum number of tokens to generate (default: `512`)
- `--temperature`: Temperature for sampling (default: `0.7`)
- `--top_p`: Top-p for nucleus sampling (default: `0.9`)
- `--device_ids`: Comma-separated list of GPU device IDs (default: `0,1,2,3`)
- `--checkpoint_filter`: Optional regex pattern to filter checkpoints

## Output Structure

Predictions are saved in the following structure:

```
predictions/
└── qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/
    ├── checkpoint-500/
    │   ├── 0.json
    │   ├── 1.json
    │   ├── 2.json
    │   ├── ...
    │   └── inference_summary.json
    └── checkpoint-1000/
        ├── 0.json
        ├── 1.json
        ├── 2.json
        ├── ...
        └── inference_summary.json
```

### Prediction File Format

Each `{idx}.json` file contains:

```json
{
  "reasoning_steps": [
    "Step 1: Visual observation about the image",
    "Step 2: Another observation or inference",
    "Step 3: Final reasoning leading to answer"
  ],
  "answer": "C"
}
```

### Summary File

The `inference_summary.json` contains all predictions plus raw model outputs and any errors:

```json
[
  {
    "idx": 0,
    "prediction": {
      "reasoning_steps": [...],
      "answer": "C"
    },
    "raw_output": "Full text output from model...",
    "error": null
  },
  ...
]
```

## Test Dataset Format

The test dataset should be a HuggingFace dataset with the following features:

```python
Dataset({
    features: [
        'image',           # PIL Image
        'question',        # str
        'answer',          # str (ground truth)
        'source',          # str (optional)
        'options',         # list (optional)
        'choices',         # list (optional)
        'reference_steps'  # list (optional)
    ],
    num_rows: N
})
```

Load with:

```python
from datasets import load_from_disk
test_dataset = load_from_disk("reasoning_test_with_reference_steps_updated_v27")
```

## Multi-GPU Inference

The script automatically uses all specified GPUs via `torch.nn.DataParallel`:

```bash
# Use GPUs 0, 1, 2, 3
GPU_IDS="0,1,2,3" ./run_inference.sh

# Use only GPUs 0 and 1
GPU_IDS="0,1" ./run_inference.sh

# Use only GPU 0
GPU_IDS="0" ./run_inference.sh
```

## System Prompt

The inference uses the exact same system prompt as training:

```
You are a vision-language model. First, analyze the provided image(s) and any user text silently. Do NOT reveal your internal reasoning.

Return ONLY a single, valid JSON object with this exact schema:
{"reasoning_steps": [], "answer": ""}

Rules for "reasoning_steps":
- Decide the number of steps based on task complexity; include enough to make the answer evident without filler.
- Include some inference from visual information, always anchored to visible cues.
- Write single-clause sentences, each adding a new, directly checkable fact or cue-based inference.
...
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers>=4.52.4 (for Flash Attention support)
- datasets>=3.0.0
- Pillow
- tqdm

## Troubleshooting

### Out of Memory Error

Try reducing the batch size:

```bash
BATCH_SIZE=1 ./run_inference.sh
```

Or use fewer GPUs:

```bash
GPU_IDS="0" ./run_inference.sh
```

### Model Loading Issues

Ensure the checkpoint directory contains:
- `config.json`
- `model-*.safetensors` or `pytorch_model.bin`
- `tokenizer.json` and related tokenizer files
- `preprocessor_config.json`

### JSON Parsing Errors

The script includes robust JSON extraction that handles:
- Direct JSON output
- JSON in code blocks (```json ... ```)
- JSON embedded in text
- Malformed output (falls back to empty format)

Check `inference_summary.json` for raw outputs and error details.

## Performance Tips

1. **Batch Size**: Start with `BATCH_SIZE=1` for memory-constrained GPUs
2. **Temperature**: Lower temperature (e.g., `0.3`) for more deterministic outputs
3. **Multi-GPU**: Use all available GPUs for faster processing
4. **Checkpoint Filter**: Process one checkpoint at a time if running low on disk space

## Evaluating Predictions

Evaluate predictions using two key metrics:

1. **Accuracy**: Answer correctness (with optional LLM judge)
2. **Match F1**: Reasoning quality using `mllm_evaluator.py` with sentence transformers

### Basic Usage (Without LLM Judge)

```bash
python evaluate_predictions.py \
    --predictions_dir predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-500 \
    --test_dataset_path reasoning_test_with_reference_steps_updated_v27
```

### With LLM Judge (More Accurate)

```bash
# Start Ollama first: ollama serve
python evaluate_predictions.py \
    --predictions_dir predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-500 \
    --test_dataset_path reasoning_test_with_reference_steps_updated_v27 \
    --use_llm_judge \
    --llm_judge_model gpt-oss:20b
```

### Advanced Options

```bash
python evaluate_predictions.py \
    --predictions_dir predictions/.../checkpoint-500 \
    --test_dataset_path reasoning_test_with_reference_steps_updated_v27 \
    --use_llm_judge \
    --llm_judge_model gpt-oss:20b \
    --reasoning_model all-MiniLM-L6-v2 \
    --reasoning_threshold 0.35 \
    --reasoning_device cuda
```

**Parameters**:
- `--use_llm_judge`: Enable LLM-based semantic matching for accuracy
- `--llm_judge_model`: Ollama model name (default: gpt-oss:20b)
- `--reasoning_model`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `--reasoning_threshold`: Similarity threshold (default: 0.45 for stricter matching)
- `--reasoning_device`: Device for reasoning evaluator (auto/cuda/cpu)

### Evaluation Output

Example output:
```
================================================================================
EVALUATION RESULTS
================================================================================
Total samples: 6372
Evaluated samples: 6372
Missing predictions: 0

ACCURACY:
  Correct: 5432/6372 (85.23%)

MATCH F1 (Reasoning Quality):
  F1 Score:  0.7234
  Precision: 0.7512
  Recall:    0.6987

PER-SOURCE BREAKDOWN:
Source                    Total    Accuracy     Match F1     Precision    Recall
------------------------------------------------------------------------------------------
realwordqa                1234      87.45%      0.7123       0.7401       0.6876
mmmu                      2345      83.21%      0.6987       0.7234       0.6754
...
================================================================================
```

The JSON file (`evaluation_results.json`) contains:
- Overall metrics (accuracy, match F1, precision, recall)
- Per-source breakdown
- Detailed per-sample results with predicted/reference steps

## Example Workflow

1. Train a model:
   ```bash
   ./train_vqa_multi_deepspeed.sh
   ```

2. Run inference on all checkpoints:
   ```bash
   cd inference
   ./run_inference.sh
   ```

3. Evaluate predictions:
   ```bash
   python evaluate_predictions.py \
       --predictions_dir predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/checkpoint-500 \
       --test_dataset_path reasoning_test_with_reference_steps_updated_v27 \
       --use_llm_judge \
       --llm_judge_model gpt-oss:20b
   ```

4. Analyze specific predictions:
   ```python
   import json

   # Load a prediction
   with open("predictions/.../checkpoint-500/0.json") as f:
       pred = json.load(f)
   print("Reasoning steps:", pred["reasoning_steps"])
   print("Answer:", pred["answer"])

   # Load evaluation results
   with open("predictions/.../checkpoint-500/evaluation_results.json") as f:
       eval_results = json.load(f)

   # Print overall metrics
   print(f"Accuracy: {eval_results['accuracy_rate']*100:.2f}%")
   print(f"Match F1: {eval_results['match_f1_avg']:.4f}")
   print(f"Precision: {eval_results['precision_avg']:.4f}")
   print(f"Recall: {eval_results['recall_avg']:.4f}")

   # Find incorrect predictions
   incorrect = [r for r in eval_results["detailed_results"] if not r["accuracy_correct"]]
   print(f"Found {len(incorrect)} incorrect predictions")

   # Find low F1 reasoning
   low_f1 = [r for r in eval_results["detailed_results"] if r["match_f1"] < 0.5]
   print(f"Found {len(low_f1)} samples with low Match F1")

   # Analyze first incorrect prediction
   if incorrect:
       sample = incorrect[0]
       print(f"Question: {sample['question']}")
       print(f"Ground truth: {sample['ground_truth']}")
       print(f"Prediction: {sample['prediction']}")
       print(f"Reasoning: {sample['reasoning_steps']}")
   ```

5. Compare checkpoints:
   ```bash
   # Evaluate multiple checkpoints
   for checkpoint in predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/*/; do
       echo "Evaluating $(basename $checkpoint)..."
       python evaluate_predictions.py \
           --predictions_dir "$checkpoint" \
           --test_dataset_path reasoning_test_with_reference_steps_updated_v27 \
           --use_llm_judge
   done

   # Compare metrics across checkpoints
   echo -e "\nCheckpoint Comparison:"
   echo "Checkpoint        Accuracy    Match F1    Precision   Recall"
   echo "----------------------------------------------------------------"
   for checkpoint in predictions/qwen2.5-vl-3b-vqa-deepspeed-20251104_172040/*/; do
       if [ -f "$checkpoint/evaluation_results.json" ]; then
           name=$(basename "$checkpoint")
           python -c "
import json
r = json.load(open('$checkpoint/evaluation_results.json'))
print(f'${name:<16}  {r[\"accuracy_rate\"]*100:>6.2f}%    {r[\"match_f1_avg\"]:>6.4f}    {r[\"precision_avg\"]:>6.4f}    {r[\"recall_avg\"]:>6.4f}')
"
       fi
   done
   ```
