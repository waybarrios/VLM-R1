# Output Structure

When you run `compute_metrics.py`, all results are organized in subdirectories by prediction folder name.

## 📁 Directory Structure

```
metrics_results/
└── outputs_testing_llava7b_16/          # Subdirectory per predictions folder
    ├── no_judge_metrics.csv             # Detailed metrics WITHOUT judge
    ├── no_judge_summary.json            # Summary JSON WITHOUT judge
    ├── no_judge_summary.txt             # Summary TXT WITHOUT judge
    ├── with_judge_metrics.csv           # Detailed metrics WITH judge
    ├── with_judge_summary.json          # Summary JSON WITH judge
    ├── with_judge_summary.txt           # Summary TXT WITH judge
    └── comparison.txt                   # Comparison between both
```

## 📊 File Descriptions

### 1. `no_judge_metrics.csv` / `with_judge_metrics.csv`
Detailed per-sample results with all metrics:
- `sample_idx`: Sample index
- `accuracy_correct`: Boolean, whether answer is correct
- `match_f1`: Match F1 score
- `precision`: Precision of step matching
- `recall`: Recall of step matching
- `predicted_answer`: The predicted answer
- `ground_truth_answer`: The correct answer
- `num_predicted_steps`: Number of predicted reasoning steps
- `num_reference_steps`: Number of reference reasoning steps
- `avg_similarity`: Average similarity score
- And more...

### 2. `no_judge_summary.json` / `with_judge_summary.json`
Aggregate metrics in JSON format:
```json
{
  "experiment_info": {...},
  "accuracy_metrics": {
    "overall_accuracy": 0.8234,
    "correct_samples": 5248,
    "average_confidence": 0.5945
  },
  "match_f1_metrics": {
    "average_match_f1": 0.6541,
    "average_precision": 0.7012,
    "average_recall": 0.6892
  },
  ...
}
```

### 3. `no_judge_summary.txt` / `with_judge_summary.txt`
Human-readable summary with all aggregate metrics formatted nicely.

### 4. `comparison.txt` (only when using `--both`)
Side-by-side comparison of results with and without judge.

## 🔄 Multiple Prediction Folders

If you evaluate multiple prediction folders, each gets its own subdirectory:

```
metrics_results/
├── outputs_testing_llava7b_16/
│   ├── no_judge_metrics.csv
│   ├── no_judge_summary.json
│   ├── no_judge_summary.txt
│   └── ...
├── outputs_testing_llava13b_32/
│   ├── no_judge_metrics.csv
│   ├── no_judge_summary.json
│   ├── no_judge_summary.txt
│   └── ...
└── outputs_testing_qwen2_8/
    ├── no_judge_metrics.csv
    ├── no_judge_summary.json
    ├── no_judge_summary.txt
    └── ...
```

## 📖 Quick Access

```bash
# View summary for a specific model
cat metrics_results/outputs_testing_llava7b_16/no_judge_summary.txt

# View comparison
cat metrics_results/outputs_testing_llava7b_16/comparison.txt

# Load JSON in Python
import json
with open('metrics_results/outputs_testing_llava7b_16/no_judge_summary.json') as f:
    data = json.load(f)
```

## 🎯 Benefits

✅ **Organized**: Each prediction folder has its own directory
✅ **Clean**: No long filenames with prefixes
✅ **Scalable**: Easy to evaluate multiple models
✅ **Comparable**: All files for one model are together
