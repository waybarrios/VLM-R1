# All Models Evaluation Summary (No Judge Mode)

**Date:** 2025-11-08
**Dataset:** reasoning_test_with_reference_steps_updated_v27
**Total Samples:** 6,372
**Evaluation Mode:** No Judge (Rule-based, no LLM)

---

## Quick Comparison Table

| Model | Accuracy | Match F1 | Precision | Recall | Std Dev |
|-------|----------|----------|-----------|--------|---------|
| LLaVA-7B-16 | TBD | TBD | TBD | TBD | TBD |
| Gemma3-4B | TBD | TBD | TBD | TBD | TBD |
| MiniCPM-V-8B | 16.90% | 0.193 ± 0.169 | 0.635 ± 0.463 | 0.120 ± 0.123 | High Precision |
| Gemma3-12B-64K | TBD | TBD | TBD | TBD | TBD |
| Qwen2.5-VL-32B-64K | TBD | TBD | TBD | TBD | TBD |

---

## Detailed Results

### 1. LLaVA-7B-16
```
Overall Accuracy:     TBD
Average Match F1:     TBD
Average Precision:    TBD
Average Recall:       TBD
```

**Analysis:**
- Loading results...

---

### 2. Gemma3-4B
```
Overall Accuracy:     TBD
Average Match F1:     TBD
Average Precision:    TBD
Average Recall:       TBD
```

**Analysis:**
- Loading results...

---

### 3. MiniCPM-V-8B
```
Overall Accuracy:     0.1690 (16.90%)
Correct samples:      1077/6372
Average Confidence:   0.4140
Median Confidence:    0.3000

Average Match F1:     0.1929 (±0.1691)
Average Precision:    0.6353 (±0.4633)
Average Recall:       0.1199 (±0.1225)
Median Match F1:      0.2000

Step Matching:
  Avg predicted steps:  1.31
  Avg reference steps:  11.56
  Avg matched (pred):   1.18
  Avg matched (ref):    1.18
```

**Analysis:**
- **Very High Precision (63.5%)** but **Very Low Recall (12.0%)**
  - Model generates few steps, but most are relevant
  - Missing majority of reference reasoning steps
  - Pattern: Conservative reasoning (quality over quantity)

- **High Standard Deviation** (F1: ±0.169, Precision: ±0.463)
  - Inconsistent performance across questions
  - Likely struggles with complex/long-reasoning questions

- **Low Accuracy (16.9%)**
  - Final answers often incorrect
  - 28.8% placeholders (no predictions)

- **Match Type Breakdown:**
  - Exact matches: 48.4%
  - Placeholders: 28.8% (needs improvement)
  - Choice questions: 9.0%

**Recommendations:**
- Encourage more comprehensive reasoning (increase recall)
- Reduce placeholder rate
- Focus on complex questions (analyze by difficulty tier)

---

### 4. Gemma3-12B-64K
```
Overall Accuracy:     TBD
Average Match F1:     TBD
Average Precision:    TBD
Average Recall:       TBD
```

**Analysis:**
- Loading results...

---

### 5. Qwen2.5-VL-32B-64K
```
Overall Accuracy:     TBD
Average Match F1:     TBD
Average Precision:    TBD
Average Recall:       TBD
```

**Analysis:**
- Loading results...

---

## Key Observations

### Precision vs Recall Patterns

**High Precision, Low Recall** (e.g., MiniCPM-V-8B):
- Few but accurate reasoning steps
- Conservative/safe predictions
- Missing important reasoning
- **Action:** Encourage more detailed reasoning

**High Recall, Low Precision:**
- Many steps, some irrelevant
- Verbose reasoning
- **Action:** Encourage concise reasoning

**Balanced (P ≈ R):**
- Ideal scenario
- Appropriate number of relevant steps

---

## Dataset Complexity Analysis

**Distribution:**
- Easy (< 0.3): 3,897 samples (61.2%)
- Medium (0.3-0.5): 2,376 samples (37.3%)
- Hard (0.5-0.7): 98 samples (1.5%)
- Very Hard (≥ 0.7): 1 sample (0.0%)

**Key Correlations:**
- Reference steps: r = +0.910 (very strong)
- Question length: r = +0.665 (strong)
- Question words: r = +0.650 (strong)

---

## Performance by Difficulty Tier

*(To be filled after running analyze script)*

| Difficulty | N | Accuracy | Match F1 | Precision | Recall |
|------------|---|----------|----------|-----------|--------|
| Easy | 3897 | TBD | TBD | TBD | TBD |
| Medium | 2376 | TBD | TBD | TBD | TBD |
| Hard | 98 | TBD | TBD | TBD | TBD |
| Very Hard | 1 | TBD | TBD | TBD | TBD |

---

## Files Generated

Each model has the following outputs in `metrics_results/[model_name]/`:
- `no_judge_metrics.csv` - Detailed per-sample metrics
- `no_judge_summary.json` - Machine-readable summary
- `no_judge_summary.txt` - Human-readable summary

Additional files:
- `dataset_complexity_scores.json` - Unbiased complexity analysis
- `METRICS_GUIDE.md` - Complete metrics documentation

---

## How to Use These Results in Papers

### Minimal Reporting:
```latex
Our model achieves X% accuracy and Y Match F1 on the VQA reasoning benchmark.
```

### Better Reporting:
```latex
Our model achieves X% accuracy and Y ± Z Match F1 (median: M),
with balanced precision (P) and recall (R).
```

### Best Reporting:
```latex
Our model achieves X% accuracy and Y ± Z Match F1 (median: M, IQR: [Q1, Q3]).
Performance varies by question difficulty: easy questions (F1 = A ± B) vs.
very hard questions (F1 = C ± D). The high standard deviation reflects the
diverse complexity of our dataset, consistent with prior visual reasoning
benchmarks [citations].
```

---

## Next Steps

1. **Analyze by Difficulty Tier:**
   ```bash
   python analyze_by_difficulty.py \
       dataset_complexity_scores.json \
       metrics_results/[model_name]/no_judge_metrics.csv
   ```

2. **Generate Publication Figures:**
   ```bash
   python visualize_for_paper.py \
       metrics_results/[model_name]/no_judge_metrics.csv
   ```

3. **Compare Models:**
   ```bash
   python compare_models.py \
       metrics_results/*/no_judge_summary.json
   ```

---

*Generated: 2025-11-08*
*Documentation: See METRICS_GUIDE.md for detailed explanations*
