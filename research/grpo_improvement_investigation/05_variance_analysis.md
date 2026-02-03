# Training Variance Analysis: Why Composite GRPO is Unstable

## 1. Quantitative Variance Comparison

### Answer-Only Training (Accuracy Progression)

| Step | Accuracy | Trend |
|------|----------|-------|
| 150  | 37.99%   | ↗ |
| 300  | 42.56%   | ↗ |
| 600  | 44.07%   | ↗ |
| 900  | 43.11%   | ↘ (minor) |
| 1200 | 44.32%   | ↗ |
| 1400 | **44.90%** | ↗ |
| 1500 | 44.30%   | ↘ (minor) |

**Variance:** σ = 2.3% (monotonic improvement with minor fluctuations)

### Composite Training (Accuracy Progression)

| Step | Accuracy | Trend |
|------|----------|-------|
| 100  | 30.30%   | - |
| 200  | 35.66%   | ↗ |
| 300  | 28.91%   | **↘ DROP** |
| 400  | 30.41%   | ↗ |
| 500  | 30.30%   | → |
| 600  | **26.98%** | **↘ LOWEST** |
| 700  | 38.92%   | **↗ RECOVERY** |
| 1400 | **44.92%** | ↗ |

**Variance:** σ = 6.8% (high oscillation, especially steps 300-700)

## 2. Root Causes of Higher Variance

### A. Multi-Objective Optimization Conflict

**Composite reward has three competing signals:**
```
reward_funcs: "format" "accuracy" "reasoning"
reward_weights: 2.0     3.0       1.0
```

**Answer-only has only two aligned signals:**
```
reward_funcs: "format" "accuracy"
reward_weights: 2.0     3.0
```

The reasoning reward creates a **trade-off**:
- More reasoning steps → improves recall (and reasoning F1)
- More verbose outputs → can hurt accuracy ("reasons itself into errors")

### B. The Checkpoint-600 Anomaly

At checkpoint-600, composite GRPO collapsed:

| Metric | Checkpoint-600 (Composite) | Checkpoint-600 (Answer-Only) |
|--------|---------------------------|------------------------------|
| Accuracy | **26.98%** | 44.07% |
| Avg Steps | **1.93** | 4.00 |
| Match F1 | 0.3045 | 0.4361 |
| Precision | 92.66% | 78.99% |
| Recall | **18.89%** | 31.64% |

**What happened:**
1. Model learned that fewer steps → higher precision (92.66%)
2. But this **severely hurt recall** (18.89% vs 35.90% at step-300)
3. Conservative behavior cascaded to hurt answer accuracy

### C. Step Count Instability Pattern

| Checkpoint | Composite Steps | Answer-Only Steps |
|------------|-----------------|-------------------|
| 300        | 3.77            | 3.62              |
| 600        | **1.93**        | 4.00              |
| 900        | 3.55            | 3.39              |
| 1200       | 3.55            | 3.81              |
| 1500       | 2.85            | 3.78              |

Answer-only maintains consistent verbosity (3.4-4.0 steps).
Composite swings from 1.93 to 3.77.

### D. Reasoning Reward Creates Mode-Switching

The `vqa_reasoning_reward` function uses word-overlap F1:
```python
f1, matched_pred, matched_ref = best_match_f1(predicted_steps, ref_steps_cleaned, threshold=0.45)
reward = f1
```

This creates a **local optimum trap**:
- **High precision path**: Generate few, safe steps (checkpoint-600)
- **High recall path**: Generate many steps (checkpoint-300)

The model switches between these modes, causing variance spikes.

## 3. Why Answer-Only Avoids This Variance

1. **Single objective alignment**: Format + accuracy both benefit from clear, correct answers
2. **No verbosity penalty**: No incentive to reduce reasoning steps
3. **Implicit reasoning emerges**: Model still generates 3.6-4.0 steps without explicit reward, but doesn't get "punished" for exploratory reasoning

## 4. The "Bad Checkpoint" Pattern

### Checkpoints with Collapsed Behavior (low accuracy, few steps)
- checkpoint-100: 30.30% acc, 1.00 steps
- checkpoint-400: 30.41% acc, 2.09 steps
- checkpoint-600: **26.98% acc, 1.93 steps** (worst)
- checkpoint-1300: 35.00% acc, 2.53 steps

### Checkpoints with Recovered Behavior (better accuracy, more steps)
- checkpoint-300: 28.91% acc, 3.77 steps (best reasoning F1)
- checkpoint-700: 38.92% acc, 3.82 steps
- checkpoint-1100: 39.66% acc, 3.76 steps

This oscillation doesn't exist in answer-only training.

## 5. Key Conclusions

1. **Composite GRPO variance is ~3x higher** (6.8% vs 2.3%)

2. **Checkpoint-600 is the clearest "bad checkpoint"** - over-optimized for precision

3. **Answer-only achieves same final accuracy** with much smoother dynamics

4. **The reasoning reward creates instability** by incentivizing mode-switching between conservative and verbose

5. **The problem is the REWARD SIGNAL, not the concept** - a better reasoning reward could maintain stability while improving reasoning quality

## Implications for Improvement

The variance analysis suggests:
- **Don't remove reasoning reward** - it does shape behavior
- **Make the reward signal less noisy** - semantic similarity instead of word overlap
- **Avoid mode collapse** - ensure reward doesn't overly penalize exploration
- **Consider reward smoothing** - running average or momentum on reasoning reward
