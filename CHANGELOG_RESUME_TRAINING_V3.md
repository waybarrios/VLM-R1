# Changelog - Resume Training v3

**Date**: 2025-11-16
**Modified by**: AI Senior Engineer
**File**: `resume_train_vqa_deepspeed.sh`
**Version**: v2 → v3

---

## Summary

Updated training script to resume from **checkpoint-1100** with optimized hyperparameters to balance accuracy and reasoning quality. Previous runs showed accuracy-reasoning trade-off where late checkpoints (1400/1500) over-optimized for accuracy at the expense of reasoning depth.

---

## Changes Made

### 1. Checkpoint Selection
```bash
# OLD: checkpoint-300 (good reasoning but low accuracy)
CHECKPOINT_PATH="${PROJECT_ROOT}/output/.../checkpoint-300"

# NEW: checkpoint-1100 (best balance)
CHECKPOINT_PATH="${PROJECT_ROOT}/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1100"
```
**Rationale**: checkpoint-1100 has 39.66% accuracy + F1=0.502, better balance than 1400/1500

### 2. Learning Rate
```bash
# OLD
LEARNING_RATE=3e-6

# NEW
LEARNING_RATE=2e-6
```
**Rationale**: More conservative to prevent over-optimization for accuracy metric

### 3. Number of Generations (CRITICAL CHANGE)
```bash
# OLD
NUM_GENERATIONS=2

# NEW
NUM_GENERATIONS=4
```
**Rationale**:
- Literature recommends 4-8 for VLMs (2 was too low)
- Larger groups = smoother gradients, lower variance
- Better baseline estimation prevents reward collapse

### 4. Batch Configuration
```bash
# OLD
PER_DEVICE_BATCH=4
GRADIENT_ACCUM=2

# NEW
PER_DEVICE_BATCH=2
GRADIENT_ACCUM=4
```
**Rationale**:
- Reduce per-device batch to accommodate 2x memory from num_generations
- Increase gradient_accum to maintain effective batch size (32)
- **Total completions**: 32 × 4 = 128 (was 64)

### 5. Reward Weights (CRITICAL CHANGE)
```bash
# OLD
--reward_weights 2.0 3.0 1.0   # format, accuracy, reasoning

# NEW
--reward_weights 2.0 2.5 1.5   # format, accuracy, reasoning
```
**Rationale**:
- Reduce accuracy weight 3.0→2.5 (was dominating training)
- Increase reasoning weight 1.0→1.5 (+50% boost)
- More balanced optimization

---

## Expected Improvements

1. **More reasoning steps**: Target 3.5-4.0 avg (vs 2.85-3.0 in ckpt-1400/1500)
2. **Better recall**: Target >0.30 (vs 0.27 in ckpt-1500)
3. **Maintained accuracy**: Target >43% (close to ckpt-1400's 44.9%)
4. **Balanced F1**: Target maintain >0.48

---

## Monitoring Metrics

Watch for these during training:
- `avg_pred_steps`: Should stay above 3.5
- `recall`: Should not drop below 0.25
- `match_f1`: Should stay above 0.45
- `accuracy`: Should improve from 39.66% baseline

**Stop if**: Steps drop below 3.0, recall drops below 0.25, or clear overfitting

---

## References

- Analysis: `GRPO_analysis/ANALYSIS_REPORT.md`
- Recommendations: `GRPO_TRAINING_RECOMMENDATIONS.md`
- GRPO literature: Group size 4-8 optimal for VLMs

---

## Progress Report - Checkpoint 1200 (Step 1200, Epoch 0.63)

**Date**: 2025-11-16 (7 hours training)

### Training Metrics (NOT comparable with previous test results)

**Current training metrics at step 1200:**

| Metric | Value | Baseline (ckpt-1100) | Status |
|--------|-------|----------------------|--------|
| **Accuracy (recent)** | 62.07% | 39.66% | ✅ +22.4% |
| **Accuracy (overall)** | 47.96% | 39.66% | ✅ +8.3% |
| **Match F1** | 0.2699 | Unknown* | ⚠️ TBD |
| **Precision** | 0.5285 | Unknown* | ⚠️ TBD |
| **Recall** | 0.1947 | Unknown* | ⚠️ TBD |
| **Avg Steps** | 3.41 | Unknown* | ⚠️ TBD |

\* Baseline training metrics not available (only test metrics exist)

### ⚠️ Important Note on Comparability

**CANNOT compare training vs test metrics directly:**
- Previous experiment metrics (F1: 0.43, Precision: 0.98, etc.) were measured on **TEST SET** (6,372 samples)
- Current metrics are from **TRAINING SET** (different data distribution)
- Training metrics are typically lower/noisier than test metrics
- **Valid comparison requires evaluation on same test set**

### Observations

**Positive:**
- ✅ Accuracy improving (39.66% → 62% recent)
- ✅ Generating more steps (3.41 avg)
- ✅ Training stable, no crashes

**Needs Investigation:**
- ⚠️ Training F1 (0.27) seems low but cannot confirm without test evaluation
- ⚠️ Training precision (0.53) lower than expected
- ⚠️ 15% responses with 0 steps (direct answers)

### Next Steps

**🟡 OPTION 1 (RECOMMENDED)**: Continue to ckpt-1300, then evaluate on test set
- Get full training trajectory
- Compare test metrics: ckpt-1200 vs ckpt-1300 vs previous ckpt-1400
- Make informed decision with valid comparison

**🟢 OPTION 2**: Evaluate ckpt-1200 on test set NOW
- Get immediate valid comparison with previous experiment
- Decide whether to continue or adjust

**🔴 OPTION 3**: Stop and restart with different reward_weights
- Risky without test evaluation to confirm issues
- Current training metrics alone don't justify stopping

### Recommendation

**Continue to checkpoint-1300, then evaluate both ckpt-1200 and ckpt-1300 on test set.** Only with test metrics can we make valid comparison with previous experiment (ckpt-1300: F1=0.38, ckpt-1400: F1=0.43) and decide next steps.
