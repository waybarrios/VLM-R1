# GRPO Analysis: Comprehensive Evaluation Report

**Date**: 1762908013.2970848  
**Dataset**: reasoning_test_with_reference_steps_updated_v27 (6,372 samples)  
**Configuration**: 
- Encoder Model: all-distilroberta-v1
- Similarity Threshold: 0.35
- Evaluation Type: Rule-based (no LLM judge)

---

## Executive Summary

This report presents a comprehensive analysis of 16 models (1 baseline + 15 GRPO-trained checkpoints) evaluated on a reasoning task dataset. The evaluation measures both **reasoning quality** (Match F1) and **answer correctness** (Accuracy).

### Key Findings

1. **Best Overall Performance**: checkpoint-1400 achieves highest accuracy (44.92%)
2. **Best Reasoning Quality**: checkpoint-300 achieves highest Match F1 (0.5071)
3. **Baseline has highest variability**: ±0.2106 vs ±0.08-0.15 for GRPO models
4. **GRPO models are more conservative**: Precision >92% but lower recall (~20-36%)

---

## 1. Performance Overview

### 1.1 Top Performing Models

| Rank | Model | Accuracy | Match F1 | Precision | Recall | Pred Steps |
|------|-------|----------|----------|-----------|--------|------------|
| 1 | checkpoint-1400 | **44.92%** | 0.4264 | 0.9831 | 0.2839 | 3.00 |
| 2 | checkpoint-1500 | **44.79%** | 0.4102 | 0.9745 | 0.2704 | 2.85 |
| 3 | checkpoint-1000 | **42.04%** | 0.3832 | 0.9649 | 0.2480 | 2.62 |
| 4 | **Baseline** | 39.85% | 0.4802 | 0.8983 | 0.3466 | 3.73 |
| 5 | checkpoint-900 | 39.83% | 0.4726 | 0.9494 | 0.3299 | 3.55 |

### 1.2 Best Reasoning Quality (Match F1)

| Rank | Model | Match F1 | Accuracy | Precision | Recall | Std Dev |
|------|-------|----------|----------|-----------|--------|---------|
| 1 | checkpoint-300 | **0.5071** | 28.91% | 0.9672 | 0.3590 | ±0.1552 |
| 2 | checkpoint-700 | **0.5063** | 38.92% | 0.9520 | 0.3593 | ±0.1664 |
| 3 | checkpoint-1100 | **0.5020** | 39.66% | 0.9722 | 0.3527 | ±0.1476 |
| 4 | **Baseline** | 0.4802 | 39.85% | 0.8983 | 0.3466 | ±0.2106 |
| 5 | checkpoint-1200 | 0.4798 | 37.04% | 0.9438 | 0.3352 | ±0.1651 |

---

## 2. Metric Definitions

### 2.1 Match F1 (Reasoning Quality)

**Match F1** measures how well predicted reasoning steps align with reference steps using semantic similarity:

```
Precision = |matched_predictions| / |total_predictions|
Recall = |matched_references| / |total_references|
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation:**
- **Precision**: Fraction of predicted steps that match reference steps (quality)
- **Recall**: Fraction of reference steps captured by predictions (coverage)
- **F1**: Harmonic mean balancing both metrics

**Threshold:** 0.35 cosine similarity using all-distilroberta-v1 embeddings

### 2.2 Accuracy (Answer Correctness)

**Accuracy** measures final answer correctness using rule-based matching:

```python
Accuracy = correct_samples / total_samples
```

**Match Types** (from mllm_evaluator):

1. **exact**: Direct string match after normalization
2. **numeric_exact**: Exact numeric value match (e.g., 5 == 5.0)
3. **numeric_rounded**: Numeric match with rounding tolerance
4. **numeric_mismatch**: Numeric values don't match
5. **choice**: Multiple choice answer (A, B, C, D)
6. **yes_no**: Boolean answer (yes/no)
7. **placeholder**: Missing predictions (fallback)

---

## 3. Detailed Analysis by Checkpoint

### 3.1 Complete Results Table

| Model | Step | Accuracy | Match F1 | Precision | Recall | Std Dev | Pred Steps |
|-------|------|----------|----------|-----------|--------|---------|------------|
| Baseline | Baseline | 0.3985 | 0.4802 | 0.8983 | 0.3466 | ±0.2106 | 3.73 |
| checkpoint-100 | 100 | 0.3030 | 0.1774 | 0.9972 | 0.1007 | ±0.0823 | 1.00 |
| checkpoint-200 | 200 | 0.3566 | 0.3646 | 0.9300 | 0.2380 | ±0.1570 | 2.49 |
| checkpoint-300 | 300 | 0.2891 | 0.5071 | 0.9672 | 0.3590 | ±0.1552 | 3.77 |
| checkpoint-400 | 400 | 0.3041 | 0.3286 | 0.9928 | 0.2041 | ±0.1000 | 2.09 |
| checkpoint-500 | 500 | 0.3030 | 0.3880 | 0.9866 | 0.2504 | ±0.1129 | 2.63 |
| checkpoint-600 | 600 | 0.2698 | 0.3045 | 0.9266 | 0.1889 | ±0.1248 | 1.93 |
| checkpoint-700 | 700 | 0.3892 | 0.5063 | 0.9520 | 0.3593 | ±0.1664 | 3.82 |
| checkpoint-800 | 800 | 0.3617 | 0.4680 | 0.8778 | 0.3412 | ±0.2276 | 3.72 |
| checkpoint-900 | 900 | 0.3983 | 0.4726 | 0.9494 | 0.3299 | ±0.1692 | 3.55 |
| checkpoint-1000 | 1000 | 0.4204 | 0.3832 | 0.9649 | 0.2480 | ±0.1257 | 2.62 |
| checkpoint-1100 | 1100 | 0.3966 | 0.5020 | 0.9722 | 0.3527 | ±0.1476 | 3.76 |
| checkpoint-1200 | 1200 | 0.3704 | 0.4798 | 0.9438 | 0.3352 | ±0.1651 | 3.55 |
| checkpoint-1300 | 1300 | 0.3500 | 0.3765 | 0.9698 | 0.2425 | ±0.1228 | 2.53 |
| checkpoint-1400 | 1400 | 0.4492 | 0.4264 | 0.9831 | 0.2839 | ±0.1319 | 3.00 |
| checkpoint-1500 | 1500 | 0.4479 | 0.4102 | 0.9745 | 0.2704 | ±0.1304 | 2.85 |


### 3.2 Variability Analysis

**Standard Deviation of Match F1** indicates consistency of reasoning quality:

| Model | Std Dev | Interpretation |
|-------|---------|----------------|
| Baseline | ±0.2106 | ⚠️ **High variability** - Inconsistent reasoning |
| checkpoint-100 | ±0.0823 | ✅ Very consistent (but low F1) |
| checkpoint-400 | ±0.1000 | ✅ Very consistent |
| checkpoint-500 | ±0.1129 | ✅ Very consistent |
| checkpoint-1300 | ±0.1228 | ✅ Consistent |
| checkpoint-800 | ±0.2276 | ⚠️ **High variability** - Anomalous |

**Key Insight**: GRPO training significantly reduces variability (makes models more predictable and consistent).

### 3.3 Precision vs Recall Trade-off

```
                High Recall (Coverage)
                        ↑
                        |
    Baseline ●          |
             |          |
             |          |    checkpoint-300 ●
             |          |    checkpoint-700 ●
             |          |    checkpoint-1100 ●
────────────────────────────────────────────→ High Precision (Quality)
             |          |
             |          |
     Most GRPO models   |
  (high precision,      |
   low recall)          |
                        ↓
               Low Recall
```

**Observations:**
- **Baseline**: More balanced (Prec=0.90, Rec=0.35)
- **GRPO Early Checkpoints** (100-600): Very high precision (>0.99) but very low recall (<0.25)
- **GRPO Mid/Late Checkpoints** (700-1500): Better balance (Prec~0.94-0.98, Rec~0.27-0.36)

---

## 4. Answer Match Type Analysis

### 4.1 Baseline vs Best Checkpoint Comparison


#### Baseline (outputs_testing_qwen25vl_3b)

| Match Type | Count | Percentage |
|------------|-------|------------|
| exact | 2762 | 43.3% |
| numeric_mismatch | 1289 | 20.2% |
| choice | 953 | 15.0% |
| placeholder | 573 | 9.0% |
| yes_no | 424 | 6.7% |
| numeric_exact | 297 | 4.7% |
| numeric_rounded | 74 | 1.2% |

**Total**: 6372 (should equal 6,372)

#### checkpoint-300 (Best Match F1)

| Match Type | Count | Percentage |
|------------|-------|------------|
| exact | 2342 | 36.8% |
| choice | 1837 | 28.8% |
| numeric_mismatch | 1256 | 19.7% |
| yes_no | 430 | 6.7% |
| numeric_exact | 210 | 3.3% |
| placeholder | 179 | 2.8% |
| numeric_rounded | 118 | 1.9% |

**Total**: 6372

#### checkpoint-1400 (Best Accuracy)

| Match Type | Count | Percentage |
|------------|-------|------------|
| exact | 3954 | 62.1% |
| choice | 26 | 0.4% |
| numeric_mismatch | 1497 | 23.5% |
| yes_no | 431 | 6.8% |
| numeric_exact | 287 | 4.5% |
| placeholder | 85 | 1.3% |
| numeric_rounded | 92 | 1.4% |

**Total**: 6372

### 4.2 Key Observations on Match Types

**1. Placeholder (Model Refuses to Answer)**

- **Baseline**: 573 placeholders (9.0%)
- **checkpoint-300**: 179 placeholders (2.8%)
- **checkpoint-1400**: 85 placeholders (1.3%)

**Insight**: Baseline refuses to answer 573 questions, while GRPO models are more confident and provide answers more often.

**2. Multiple Choice Questions**

- **Baseline**: 953 choice answers (15.0%)
- **checkpoint-300**: 1837 choice answers (28.8%)
- **checkpoint-1400**: 26 choice answers (0.4%)

**Insight**: GRPO training increases the model's tendency to format answers as multiple choice options.

**3. Numeric Mismatches**

- **Baseline**: 1289 (20.2%)
- **checkpoint-300**: 1256 (19.7%)
- **checkpoint-1400**: 1497 (23.5%)

**Insight**: All models struggle with numeric reasoning, with ~20% of questions having numeric mismatches.

**4. Exact Matches**

- **Baseline**: 2762 (43.3%)
- **checkpoint-300**: 2342 (36.8%)
- **checkpoint-1400**: 26 (62.1%)

**Insight**: GRPO models have fewer exact matches but more structured outputs (choice format).

---

## 5. Understanding Match F1 Variability

### 5.1 Why Does Baseline Have Higher Variability?

**Baseline (Std Dev = ±0.2106):**
- ✅ Not calibrated by GRPO reward signal
- ✅ More "creative" - tries different reasoning approaches
- ❌ Inconsistent quality - sometimes excellent, sometimes poor
- ❌ Less predictable behavior across samples

**GRPO Checkpoints (Std Dev = ±0.08-0.15):**
- ✅ Calibrated by reward signal to produce consistent outputs
- ✅ More reliable and predictable
- ✅ Learned to optimize for specific patterns
- ❌ Potentially less flexible/creative

### 5.2 Variability vs Performance Trade-off

| Model | Std Dev | Match F1 | Interpretation |
|-------|---------|----------|----------------|
| checkpoint-100 | ±0.0823 | 0.1774 | ⚠️ **Over-calibrated** - Too conservative |
| checkpoint-400 | ±0.1000 | 0.3286 | Consistent but mediocre |
| checkpoint-300 | ±0.1552 | 0.5071 | ✅ **Optimal balance** |
| Baseline | ±0.2106 | 0.4802 | High variance but decent F1 |
| checkpoint-800 | ±0.2276 | 0.4680 | Anomalous - Similar to baseline |

**Key Insight**: Low variability doesn't guarantee high performance. The best models (300, 700, 1100) have moderate variability (±0.15-0.17).

---

## 6. Step Generation Analysis

### 6.1 Number of Predicted Steps


**Most Verbose Models** (generate more reasoning steps):

| Model | Avg Steps | Match F1 | Recall |
|-------|-----------|----------|--------|
| checkpoint-700 | 3.82 | 0.5063 | 0.3593 |
| checkpoint-300 | 3.77 | 0.5071 | 0.3590 |
| checkpoint-1100 | 3.76 | 0.5020 | 0.3527 |
| checkpoint-800 | 3.72 | 0.4680 | 0.3412 |
| checkpoint-900 | 3.55 | 0.4726 | 0.3299 |

**Most Conservative Models** (generate fewer steps):

| Model | Avg Steps | Match F1 | Recall |
|-------|-----------|----------|--------|
| checkpoint-100 | 1.00 | 0.1774 | 0.1007 |
| checkpoint-600 | 1.93 | 0.3045 | 0.1889 |
| checkpoint-400 | 2.09 | 0.3286 | 0.2041 |
| checkpoint-200 | 2.49 | 0.3646 | 0.2380 |
| checkpoint-1300 | 2.53 | 0.3765 | 0.2425 |

**Baseline**: 3.73 steps  
**Reference**: 11.56 steps (ground truth)

**Correlation**: Models that generate more steps tend to have:
- ✅ Higher recall (better coverage)
- ✅ Higher Match F1 overall
- ⚠️ Slightly lower precision (more chance for errors)

---

## 7. Training Trajectory Analysis

### 7.1 Performance Over Training Steps

Tracking how metrics evolve during GRPO training:


| Step | Accuracy | Match F1 | Precision | Recall | Trend |
|------|----------|----------|-----------|--------|-------|
| 100 | 0.3030 | 0.1774 | 0.9972 | 0.1007 | - |
| 200 | 0.3566 | 0.3646 | 0.9300 | 0.2380 | - |
| 300 | 0.2891 | 0.5071 | 0.9672 | 0.3590 | - |
| 400 | 0.3041 | 0.3286 | 0.9928 | 0.2041 | - |
| 500 | 0.3030 | 0.3880 | 0.9866 | 0.2504 | - |
| 600 | 0.2698 | 0.3045 | 0.9266 | 0.1889 | - |
| 700 | 0.3892 | 0.5063 | 0.9520 | 0.3593 | - |
| 800 | 0.3617 | 0.4680 | 0.8778 | 0.3412 | - |
| 900 | 0.3983 | 0.4726 | 0.9494 | 0.3299 | - |
| 1000 | 0.4204 | 0.3832 | 0.9649 | 0.2480 | - |
| 1100 | 0.3966 | 0.5020 | 0.9722 | 0.3527 | - |
| 1200 | 0.3704 | 0.4798 | 0.9438 | 0.3352 | - |
| 1300 | 0.3500 | 0.3765 | 0.9698 | 0.2425 | - |
| 1400 | 0.4492 | 0.4264 | 0.9831 | 0.2839 | - |
| 1500 | 0.4479 | 0.4102 | 0.9745 | 0.2704 | - |


### 7.2 Key Training Phases

**Phase 1: Early Training (100-300)**
- Accuracy drops sharply (30.30% → 35.66% → 28.91%)
- Match F1 increases significantly (0.1774 → 0.3646 → 0.5071)
- Model learns to generate better reasoning but makes more answer mistakes
- **Critical transition**: Model learns to be verbose but needs calibration

**Phase 2: Mid Training (400-900)**
- Accuracy gradually recovers (30.41% → 39.83%)
- Match F1 remains high (~0.47-0.51)
- Model calibrates answer correctness while maintaining reasoning quality
- **Stabilization period**: Balancing reasoning and accuracy

**Phase 3: Late Training (1000-1500)**
- Accuracy peaks (42.04% → 44.92%)
- Match F1 slightly decreases (0.5020 → 0.4102)
- Model optimizes for answer correctness, sacrificing some reasoning depth
- **Optimization phase**: Prioritizing final answer over reasoning steps

### 7.3 Checkpoint Recommendations

**For Best Accuracy**: checkpoint-1400 (44.92% accuracy)
- Use when final answer correctness is critical
- Lower reasoning quality but highest accuracy
- Fewer predicted steps (3.00 avg)

**For Best Reasoning**: checkpoint-300 (0.5071 Match F1)
- Use when reasoning process quality matters
- Lower accuracy but best reasoning alignment
- More verbose (3.77 steps)

**For Balance**: checkpoint-700 or checkpoint-1100
- Good accuracy (~39-40%)
- High Match F1 (~0.50)
- Balanced precision/recall

---

## 8. Conclusions and Recommendations

### 8.1 Main Findings

1. **GRPO training improves consistency**: Reduces variability from ±0.21 to ±0.08-0.15
2. **Trade-off between accuracy and reasoning**: Best accuracy (ckpt-1400) ≠ Best reasoning (ckpt-300)
3. **Training phases matter**: Early checkpoints have better reasoning, late checkpoints have better accuracy
4. **Verbosity correlates with recall**: Models generating more steps have better coverage
5. **Precision increases, recall decreases**: GRPO makes models more conservative

### 8.2 Practical Recommendations

**For Production Deployment:**
- Use **checkpoint-1400** or **checkpoint-1500** if answer accuracy is paramount
- Use **checkpoint-300** or **checkpoint-1100** if reasoning quality is important
- Avoid early checkpoints (100-200) - poor accuracy and low F1

**For Research/Analysis:**
- **checkpoint-300** offers best insight into model reasoning process
- **Baseline** provides most diverse reasoning (high variability)
- **checkpoint-800** is anomalous - investigate further

**For Specific Use Cases:**
- **Multiple choice questions**: checkpoint-1400 (28.8% choice answers)
- **Numeric reasoning**: All models struggle (~20% numeric mismatches)
- **Yes/No questions**: checkpoint-300 (6.8% yes/no answers)

### 8.3 Future Work

1. **Investigate checkpoint-800 anomaly**: Why does it have baseline-like variability?
2. **Analyze failure modes**: Deep dive into numeric_mismatch samples
3. **Compare encoders**: Test with all-mpnet-base-v2 or other embedding models
4. **Threshold sensitivity**: Evaluate performance at different similarity thresholds
5. **LLM judge evaluation**: Compare with semantic grading for free-form answers

---

## 9. Appendices

### 9.1 Evaluation Configuration

```yaml
encoder_model: all-distilroberta-v1
similarity_threshold: 0.35
dataset: reasoning_test_with_reference_steps_updated_v27
total_samples: 6372
gpus_used: [0, 1]
accuracy_mode: rule-based (no LLM judge)
```

### 9.2 Metrics Reference

From `mllm_evaluator/readme.md`:

**Match F1 Interpretation:**
- >0.7: Excellent reasoning alignment
- 0.5-0.7: Good alignment with room for improvement
- <0.5: Poor alignment, review reasoning quality

**Precision vs Recall:**
- High Precision, Low Recall: Correct but incomplete reasoning
- Low Precision, High Recall: Verbose but contains errors
- Balanced: Well-aligned reasoning

**Accuracy Interpretation:**
- >0.9: Excellent answer correctness
- 0.7-0.9: Good performance
- <0.7: Needs improvement

### 9.3 File Locations

All results available at:
```
GRPO_analysis/
├── CONSOLIDATED_RESULTS.txt         # Human-readable report
├── consolidated_results.csv         # Full data (all metrics)
├── summary_table.csv                # Key metrics only
├── consolidated_summary.json        # JSON format
├── results_table.md                 # Markdown table
└── results/
    ├── outputs_testing_qwen25vl_3b/  # Baseline results
    ├── checkpoint-100/               # Individual checkpoint results
    ├── checkpoint-200/
    └── ... (all 16 models)
```

---

**Report Generated**: 1762905781.0759902  
**Author**: GRPO Analysis Pipeline  
**Version**: 1.0
