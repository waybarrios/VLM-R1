# Composite vs Answer-Only Prediction Comparison

## 1. Overall Accuracy - Nearly Identical

| Metric | Composite GRPO | Answer-Only GRPO |
|--------|---------------|------------------|
| Total Samples | 6,372 | 6,372 |
| Correct | 2,862 | 2,861 |
| **Accuracy** | **44.92%** | **44.90%** |

Both methods achieve virtually identical answer accuracy despite fundamentally different training signals.

## 2. Reasoning Generation - Major Differences

| Metric | Composite GRPO | Answer-Only GRPO |
|--------|---------------|------------------|
| Avg Steps | **3.00** | 3.85 |
| Std Dev Steps | **1.00** | 2.40 |
| Max Steps | 19 | 50 |
| Zero-step responses | **85** | 816 |

**Key Finding**: Answer-only produces **10x more "insufficient information" responses** (816 vs 85). Without the reasoning reward, the model is more hesitant when uncertain.

## 3. Reasoning Quality - Precision vs Recall Tradeoff

| Metric | Composite GRPO | Answer-Only GRPO |
|--------|---------------|------------------|
| Avg Precision | **98.31%** | 80.17% |
| Avg Recall | 28.39% | **31.25%** |
| Avg Similarity | **0.9867** | 0.3367 |
| High Precision (>95%) | **98.0%** | 66.5% |

**Key Finding**: Composite GRPO produces **much more precise reasoning** - when it generates a step, it almost always matches the reference. Answer-only generates more steps but with lower precision.

## 4. Agreement/Disagreement Patterns

| Pattern | Count | Percentage |
|---------|-------|------------|
| Both Correct | 1,908 | 29.9% |
| Composite Only Correct | 954 | 15.0% |
| Answer-Only Only Correct | 953 | 15.0% |
| Both Wrong | 2,557 | 40.1% |

**Key Finding**: Despite identical overall accuracy, they **disagree on ~30% of samples**. Each model is uniquely correct on ~950 samples where the other fails.

## 5. Qualitative Reasoning Differences

### Composite GRPO (with reasoning reward)
- Shorter, focused reasoning (2-3 steps)
- Direct statements: *"The traffic light is green. The light is green."*
- Confident, rarely says "insufficient information"
- High precision but lower coverage

### Answer-Only GRPO (no reasoning reward)
- Longer, more varied reasoning (3-5+ steps)
- Descriptive observations: *"Noted the road. Observed the intersection. The light appears red."*
- More hesitant, often says "insufficient information"
- More exploratory but less aligned with reference

## 6. Training Configuration Difference

**Composite GRPO:**
```bash
--reward_funcs "format" "accuracy" "reasoning"
--reward_weights 3.0 1.0 3.0
```

**Answer-Only GRPO:**
```bash
--reward_funcs "format" "accuracy"
--reward_weights 2.0 3.0
```

## Key Conclusions

1. **Accuracy is independent of reasoning reward**: The reasoning reward doesn't improve final answer accuracy.

2. **Reasoning style differs dramatically**: Composite produces more reference-aligned reasoning (98% precision vs 80%), but this doesn't translate to better answers.

3. **Complementary strengths**: Each method succeeds on ~15% of samples where the other fails - **ensemble potential**.

4. **Confidence calibration differs**: Composite is more confident; Answer-only more conservative.

5. **The reasoning reward shapes HOW models reason, not WHAT they conclude** - suggesting the reference reasoning may not represent the optimal reasoning path.

## Implications for CRYSTAL

The 30% disagreement and complementary correctness patterns suggest:
- **The reasoning reward IS shaping behavior** (different reasoning styles)
- **But word overlap doesn't capture reasoning quality well**
- **A better reasoning reward could potentially unlock both precision AND accuracy gains**
