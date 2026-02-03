# Synthesis: Root Cause Analysis and Proposed Solutions

## Executive Summary

After investigating multi-objective RL, reasoning reward noise, curriculum learning, prediction patterns, and training variance, we have identified the root causes of why composite GRPO fails to improve both accuracy AND reasoning quality.

**The problem is NOT that reasoning rewards are bad. The problem is that the CURRENT reasoning reward signal is too noisy.**

## Root Cause Diagnosis

### The Three Fundamental Problems

#### 1. Noisy Reward Signal (PRIMARY CAUSE)

**Current Implementation:**
```python
# Word overlap with threshold 0.45
f1, matched_pred, matched_ref = best_match_f1(predicted_steps, ref_steps_cleaned, threshold=0.45)
reward = f1
```

**Why It's Noisy:**
- "The light is green" vs "Traffic signal shows green" = LOW match (different words)
- But they are semantically EQUIVALENT
- Research shows: 80.1% answer correctness but only 39.7% process soundness with such metrics
- Word overlap doesn't capture logical/causal reasoning quality

#### 2. Gradient Conflict (SECONDARY CAUSE)

**The Conflict:**
```
Accuracy reward: "Generate the correct answer"
Reasoning reward: "Generate steps that match reference"
```

**Why They Conflict:**
- Sometimes the BEST reasoning path is NOT the reference path
- Model receives contradictory signals
- Optimizing for reference matching can hurt accuracy

#### 3. Mode Oscillation (CONSEQUENCE)

**The Pattern:**
- Model oscillates between:
  - Conservative mode: few steps, high precision, low accuracy (checkpoint-600: 27%)
  - Verbose mode: many steps, high recall, but more errors

- The reward landscape has multiple local optima
- Without stable signal, model switches between modes

## Alignment with CRYSTAL Thesis

**CRYSTAL's Core Claim:** Logical reasoning steps matter for VLM performance.

**Our Finding:** The reasoning reward DOES shape behavior (98% precision vs 80%), but the noisy signal prevents it from improving accuracy.

**Conclusion:** We need a BETTER reasoning reward, not no reasoning reward.

## Proposed Solution: Semantic Reasoning Reward

### Concept

Replace word overlap with neural semantic similarity to capture meaning, not surface form.

### Implementation Approach

```python
# CURRENT (noisy)
def vqa_reasoning_reward_current(predicted_steps, reference_steps):
    f1 = word_overlap_f1(predicted_steps, reference_steps, threshold=0.45)
    return f1

# PROPOSED (semantic)
def vqa_reasoning_reward_semantic(predicted_steps, reference_steps):
    # Use sentence embeddings for semantic similarity
    pred_embeddings = sentence_transformer.encode(predicted_steps)
    ref_embeddings = sentence_transformer.encode(reference_steps)

    # Compute semantic F1 (matching based on cosine similarity)
    semantic_f1 = compute_semantic_match_f1(pred_embeddings, ref_embeddings, threshold=0.7)
    return semantic_f1
```

### Why This Helps

1. **Reduces Noise:** Semantically equivalent steps get high scores even with different words
2. **Maintains Reasoning Signal:** Still rewards good reasoning, aligned with CRYSTAL
3. **Reduces Mode Oscillation:** More stable gradient signal
4. **Preserves Exploration:** Doesn't penalize valid alternative reasoning paths

### Expected Outcomes

| Metric | Current Composite | Proposed Semantic |
|--------|-------------------|-------------------|
| Accuracy | 44.92% | **>46%** (target) |
| Match F1 | 0.43 | **>0.50** (target) |
| Variance | 6.8% | **<4%** (target) |
| Convergence | 1000 steps | **<500 steps** (target) |

## Alternative Approaches Considered

### Option A: Conditional Reasoning Reward
Apply reasoning reward only when accuracy=0.

**Rejected:** Goes against CRYSTAL thesis that reasoning is always important.

### Option B: Ensemble GRPO
Train both and combine predictions.

**Deferred:** Interesting but doesn't improve the fundamental reward signal.

### Option C: Curriculum Staging
Train accuracy first, then add reasoning.

**Complementary:** Could combine with semantic reward for best results.

## Recommended Implementation Plan

### Phase 1: Semantic Reasoning Reward
1. Replace word overlap with SentenceTransformer similarity
2. Use cosine similarity with threshold 0.7
3. Keep same reward weight structure

### Phase 2: Evaluation
1. Train with semantic reward
2. Compare accuracy, F1, and variance
3. Validate CRYSTAL thesis: better reasoning → better accuracy

### Phase 3: Refinement (if needed)
1. Adjust semantic threshold
2. Consider curriculum staging
3. Explore reward smoothing

## Conclusion

**The reasoning reward is valuable and aligned with CRYSTAL's thesis.** The problem is implementation noise, not the concept. By upgrading from word overlap to semantic similarity, we can maintain the importance of reasoning steps while achieving better accuracy and stability.

This represents a novel contribution: **Semantic Process Reward for VLM Reasoning** - demonstrating that reasoning quality signals CAN improve accuracy when properly implemented.
