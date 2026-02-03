# GRPO Analysis

This folder contains scripts to analyze GRPO-trained model checkpoints using custom MatchF1 settings.

## Configuration

- **Encoder**: `all-distilroberta-v1`
- **Threshold**: `0.35`
- **Dataset**: `reasoning_test_with_reference_steps_updated_v27`
- **Available GPUs**: 0, 1

## Structure

```
GRPO_analysis/
├── README.md                    # This file
├── compute_grpo_metrics.py      # Main evaluation script
├── run_grpo_analysis.sh         # Batch evaluation script
├── consolidate_results.py       # Results consolidation script
├── consolidate_results.sh       # Consolidation wrapper
└── results/                     # Output directory (created after running)
    ├── baseline_qwen25vl_3b/   # Baseline results
    ├── checkpoint-100/          # Checkpoint results
    ├── checkpoint-200/
    └── ...
```

## Usage

### Step 1: Run Evaluation

Evaluate baseline + all checkpoints:

```bash
cd /gpudata3/Wayner/VLM-R1
bash GRPO_analysis/run_grpo_analysis.sh
```

This will:
- Evaluate baseline model (`outputs_testing_qwen25vl_3b`)
- Evaluate all checkpoints in `predictions_final/`
- Use GPUs 0 and 1 in parallel
- Save results to `GRPO_analysis/results/`

### Step 2: Consolidate Results

Generate comparison tables:

```bash
bash GRPO_analysis/consolidate_results.sh
```

Or directly:

```bash
python GRPO_analysis/consolidate_results.py
```

This creates:
- `consolidated_results.csv` - Full data table
- `summary_table.csv` - Key metrics only
- `results_table.md` - Markdown formatted table
- `CONSOLIDATED_RESULTS.txt` - Detailed human-readable report
- `consolidated_summary.json` - JSON format results

### Step 3: View Results

Quick view:
```bash
cat GRPO_analysis/CONSOLIDATED_RESULTS.txt
```

Or open CSV files in your preferred tool.

## Individual Model Evaluation

To evaluate a single checkpoint:

```bash
python GRPO_analysis/compute_grpo_metrics.py \
    /gpudata3/Wayner/VLM-R1/predictions_final/checkpoint-500 \
    --dataset-path /gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27 \
    --model-name all-distilroberta-v1 \
    --threshold 0.35 \
    --num-gpus 2 \
    --output-dir ./GRPO_analysis/results
```

## Data Sources

### Baseline
- **Location**: `/gpudata3/Wayner/reasoning/outputs_testing_qwen25vl_3b`
- **Model**: Qwen2.5-VL-3B (pre-GRPO)

### Checkpoints
- **Location**: `/gpudata3/Wayner/VLM-R1/predictions_final/`
- **Format**: `checkpoint-{step}/` (e.g., checkpoint-100, checkpoint-200, ...)
- **Model**: Qwen2.5-VL-3B trained with GRPO

### Test Dataset
- **Location**: `/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27`
- **Format**: HuggingFace Arrow dataset
- **Fields**: `question`, `answer`, `reference_steps`

## Prediction Format

Each model has predictions stored as individual JSON files:
```json
{
  "reasoning_steps": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "answer": "A"
}
```

Filename corresponds to sample index (e.g., `0.json`, `1.json`, ...).

## Metrics Computed

### Accuracy Metrics
- Overall accuracy (correct/total)
- Confidence scores
- Match type breakdown (exact, semantic, numeric, etc.)

### Match F1 Metrics
- **Match F1**: Harmonic mean of precision and recall
- **Precision**: Fraction of predicted steps matching reference
- **Recall**: Fraction of reference steps matched by predictions
- Median F1 scores

### Step Matching Details
- Average predicted steps per sample
- Average reference steps per sample
- Average matched predictions/references
- Average cosine similarity
- Maximum similarity scores

## Output Format

Each evaluated model produces:

1. **metrics.csv** - Detailed per-sample metrics
   - Columns: sample_idx, accuracy_correct, match_f1, precision, recall, etc.

2. **summary.json** - Aggregated statistics in JSON
   - Experiment info, accuracy metrics, Match F1 metrics, etc.

3. **summary.txt** - Human-readable summary
   - Quick overview of all metrics

## Customization

To modify encoder or threshold, edit the variables in `run_grpo_analysis.sh`:

```bash
MODEL_NAME="all-distilroberta-v1"  # Change encoder model
THRESHOLD=0.35                      # Change similarity threshold
```

Available encoder models:
- `all-MiniLM-L6-v2` (default in original)
- `all-MiniLM-L12-v2`
- `all-mpnet-base-v2`
- `all-distilroberta-v1` (current)
- `paraphrase-multilingual-MiniLM-L12-v2`

## Notes

- The scripts use `spawn` multiprocessing to avoid tensor shape issues
- GPU memory is managed automatically
- Missing predictions are handled with placeholders (zero metrics)
- All scripts are designed to be resumable (can re-run without issues)

## Troubleshooting

### Out of GPU memory
Reduce `NUM_GPUS` or evaluate checkpoints individually.

### Missing predictions
The script creates placeholders automatically (metrics = 0).

### Import errors
Ensure the mllm_evaluator module is available:
```bash
ls /gpudata3/Wayner/VLM-R1/mllm_evaluator/
```

## References

This analysis uses the methodology from:
- `compute_metrics.py` - Base evaluation framework
- `mllm_evaluator.py` - Match F1 computation with semantic embeddings
- `accuracy_calculator.py` - Answer accuracy evaluation
