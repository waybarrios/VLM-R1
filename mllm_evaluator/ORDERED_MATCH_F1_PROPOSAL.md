# Ordered Match F1: Incorporating Step Sequencing into Reasoning Evaluation

**Author:** Wayne Barrios
**Date:** February 2026
**Status:** Proposal
**Addresses:** Reviewer Aho6 — "Match F1 treats reasoning steps as an unordered bag, ignoring logical sequencing"

---

## 1. Problem Statement

Reviewer Aho6 correctly identifies a fundamental limitation of our current Match F1 metric: it evaluates reasoning chains as **unordered bags of steps**, ignoring the logical sequencing that makes a proof valid.

Consider a reference reasoning chain for "What color is the traffic light?":

```
Reference:
  R1: Observe the traffic light in the image
  R2: The light is showing green
  R3: Green means vehicles can proceed
```

Two model predictions:

```
Prediction A (correct order):
  P1: I see a traffic light            → matches R1
  P2: The light appears green           → matches R2
  P3: Cars are allowed to go            → matches R3

Prediction B (scrambled order):
  P1: Cars are allowed to go            → matches R3
  P2: I see a traffic light             → matches R1
  P3: The light appears green           → matches R2
```

Under current Match F1, **both predictions receive identical scores** (F1 = 1.0, 3/3 matched). Yet Prediction B draws its conclusion before establishing the evidence — a logically invalid chain. In visual reasoning, order reflects the cognitive process: observe → identify → conclude. A metric that ignores this cannot distinguish faithful reasoning from post-hoc rationalization.

---

## 2. Current Match F1 Recap

Our existing implementation lives in **`mllm_evaluator/mllm_evaluator.py:MLLMReasoningEvaluator`** (the evaluation-time Match F1 metric used in our ablation study and all reported results):

### Algorithm (default: `all-distilroberta-v1` via SentenceTransformer)

1. **Encode** each step into dense embeddings using SentenceTransformer (`all-distilroberta-v1`)
   - `MLLMReasoningEvaluator.__init__()` loads the model; `_compute_embeddings()` encodes steps
2. **Build cosine similarity matrix** S where S[i,j] = cosine(embed(predicted_step_i), embed(reference_step_j))
   - `_compute_similarity_matrix()` (line 119)
3. **Filter** pairs above threshold (0.40 for distilroberta — threshold-invariant in [0.30, 0.50])
4. **Sort** remaining pairs by descending similarity
5. **Greedy 1-1 matching**: iterate sorted pairs, assign each pair if neither index is already matched
   - `_find_matches()` (line 133)
6. **Compute F1** from precision (matched/predicted) and recall (matched/reference)
   - `evaluate_single()` (line 165)

> **Note on `simple_similarity.py`:** This is a separate, lightweight module used **only during GRPO training** as a reward function. It provides `best_match_f1()` (word overlap) for DeepSpeed-compatible training rewards, and `semantic_match_f1()` as an experimental semantic training reward with fallback. Neither function is used for evaluation-time Match F1 reporting. The evaluator of record is `MLLMReasoningEvaluator`.

### What it captures

- **Content coverage**: Are the right topics mentioned?
- **Precision**: Does the model avoid irrelevant steps?
- **Recall**: Does the model cover all reference steps?

### What it misses

- **Sequencing**: Is the reasoning in a logically valid order?
- **Causal flow**: Do conclusions follow from established premises?

---

## 3. Proposed: Ordered Match F1

We propose **Ordered Match F1**, which augments the existing Match F1 with a **Kendall's Tau order penalty**. The key idea: after identifying *which* steps match, we measure *how well their order is preserved*.

### 3.1 Why Kendall's Tau?

Kendall's Tau-b measures rank correlation between two orderings. Given a set of matched pairs, it counts how many pairs are **concordant** (same relative order in both sequences) versus **discordant** (swapped).

Properties that make it ideal for this task:

| Property | Benefit |
|----------|---------|
| Bounded in [-1, 1] | Easy to combine with F1 |
| Handles ties | Robust when multiple steps match the same reference position |
| Non-parametric | No distributional assumptions on step positions |
| Interpretable | +1 = perfect order, 0 = random, -1 = perfectly reversed |

### 3.2 Mathematical Formulation

**Step 1: Standard Match F1 (unchanged)**

Let P = {p_1, ..., p_m} be predicted steps and R = {r_1, ..., r_n} be reference steps. After greedy matching, we obtain a set of matched pairs:

```
M = {(p_{a_1}, r_{b_1}), (p_{a_2}, r_{b_2}), ..., (p_{a_k}, r_{b_k})}
```

where k = |M| is the number of matches. The standard F1 is:

```
Precision = k / m
Recall    = k / n
F1        = 2 * Precision * Recall / (Precision + Recall)
```

**Step 2: Extract Order Vectors**

From the matched pairs M, extract two index sequences:

```
A = (a_1, a_2, ..., a_k)    — predicted indices, sorted by reference position b
B = (b_1, b_2, ..., b_k)    — reference indices, in same sort order
```

Concretely: sort matched pairs by their reference index b_i, then read off the predicted indices a_i. If the predicted steps appeared in the same order as reference, A will be a monotonically increasing sequence.

**Step 3: Kendall's Tau**

For the sequence A (predicted indices sorted by reference order):

```
Concordant pairs: C = |{(i,j) : i < j and a_i < a_j}|
Discordant pairs: D = |{(i,j) : i < j and a_i > a_j}|

tau = (C - D) / (k * (k-1) / 2)     for k >= 2
tau = 1.0                             for k < 2
```

This yields tau in [-1, 1]. We normalize to [0, 1]:

```
tau_norm = (tau + 1) / 2
```

**Step 4: Ordered Match F1**

```
Ordered_Match_F1 = F1 * ((1 - alpha) + alpha * tau_norm)
```

where alpha in [0, 1] controls order sensitivity:

- **alpha = 0**: Reduces to standard Match F1 (no order penalty)
- **alpha = 0.3**: Mild order preference (recommended default)
- **alpha = 0.5**: Equal weight to content and order
- **alpha = 1.0**: Fully reversed chain gets F1 halved

The `(1 - alpha) + alpha * tau_norm` term is a **linear interpolation** between 1.0 (ignore order) and tau_norm (full order penalty). This ensures:

- Perfect order (tau_norm = 1): no penalty, Ordered_F1 = F1
- Random order (tau_norm = 0.5): mild penalty, Ordered_F1 = F1 * (1 - alpha/2)
- Reversed order (tau_norm = 0): maximum penalty, Ordered_F1 = F1 * (1 - alpha)

---

## 4. Worked Examples

### Example 1: Perfect Order

```
Reference:  R1(observe) → R2(identify) → R3(conclude)
Predicted:  P1(observe) → P2(identify) → P3(conclude)

Matches: (P1,R1), (P2,R2), (P3,R3)
A sorted by ref index: (1, 2, 3) — monotonically increasing
C = 3, D = 0, tau = 1.0, tau_norm = 1.0

F1 = 1.0
Ordered_F1 (alpha=0.3) = 1.0 * (0.7 + 0.3 * 1.0) = 1.0
```

**Result:** No penalty. Correct order is fully rewarded.

### Example 2: Completely Reversed

```
Reference:  R1(observe) → R2(identify) → R3(conclude)
Predicted:  P1(conclude) → P2(identify) → P3(observe)

Matches: (P1,R3), (P2,R2), (P3,R1)
A sorted by ref index: (3, 2, 1) — monotonically decreasing
C = 0, D = 3, tau = -1.0, tau_norm = 0.0

F1 = 1.0
Ordered_F1 (alpha=0.3) = 1.0 * (0.7 + 0.3 * 0.0) = 0.70
```

**Result:** 30% penalty for fully reversed reasoning. The model covered all content but drew conclusions before establishing evidence.

### Example 3: Partial Swap

```
Reference:  R1(observe) → R2(color) → R3(meaning) → R4(conclude)
Predicted:  P1(observe) → P2(meaning) → P3(color) → P4(conclude)

Matches: (P1,R1), (P2,R3), (P3,R2), (P4,R4)
A sorted by ref index: (1, 3, 2, 4)
Pairs: (1,3)C, (1,2)C, (1,4)C, (3,2)D, (3,4)C, (2,4)C
C = 5, D = 1, tau = (5-1)/6 = 0.667, tau_norm = 0.833

F1 = 1.0
Ordered_F1 (alpha=0.3) = 1.0 * (0.7 + 0.3 * 0.833) = 0.95
```

**Result:** Small 5% penalty for a single adjacent swap. The model got the overall structure right but swapped two middle steps.

### Example 4: Partial Match with Order

```
Reference:  R1 → R2 → R3 → R4
Predicted:  P1 → P2 → P3  (only 3 steps)

Matches: (P1,R1), (P2,R3), (P3,R4)
A sorted by ref index: (1, 2, 3) — monotonically increasing
tau = 1.0, tau_norm = 1.0

F1 = 2 * (3/3) * (3/4) / (1 + 0.75) = 0.857
Ordered_F1 (alpha=0.3) = 0.857 * 1.0 = 0.857
```

**Result:** Penalized for missing R2 (via F1 recall), but no order penalty since the matched steps are in correct relative order.

### Example 5: Single Match (Degenerate Case)

```
Reference:  R1 → R2 → R3
Predicted:  P1  (only 1 step)

Matches: (P1, R2)
k = 1, tau = 1.0 by convention

F1 = 2 * (1/1) * (1/3) / (1 + 0.333) = 0.50
Ordered_F1 (alpha=0.3) = 0.50 * 1.0 = 0.50
```

**Result:** With fewer than 2 matched pairs, order is undefined. We default tau = 1.0 (no order penalty), letting F1 handle the coverage penalty alone.

---

## 5. Edge Cases

| Case | Behavior | Rationale |
|------|----------|-----------|
| **0 matches** | F1 = 0, Ordered_F1 = 0 | No content match, order is irrelevant |
| **1 match** | tau = 1.0 (no penalty) | Cannot assess order from a single pair |
| **All matches concordant** | tau_norm = 1.0, no penalty | Perfect order preservation |
| **All matches discordant** | tau_norm = 0.0, max penalty | Fully reversed chain |
| **Duplicate steps** | Greedy matching prevents double-counting | Same as current Match F1 |
| **More predicted than reference** | Precision < 1.0 via F1 | Extra steps penalized through precision, not order |
| **Empty predicted** | F1 = 0 | No steps to evaluate |
| **Empty reference** | F1 = 0 | No reference to compare against |
| **Tied similarities in matching** | Greedy picks highest first | Order assessment uses whatever matching produces |

### Interaction with Greedy Matching

The greedy matching algorithm can produce different match sets depending on similarity scores, which in turn affects the order assessment. This is acceptable because:

1. High-confidence matches (high similarity) are established first
2. Order is assessed only on the matched subset
3. If a match set is "wrong" (e.g., step A matched to the wrong reference), the similarity score would likely be lower anyway

---

## 6. Comparison with Alternatives

### 6.1 Longest Common Subsequence (LCS)

**Approach:** Find the longest subsequence of matched steps that preserves order.

```
LCS_F1 = 2 * |LCS| / (|predicted| + |reference|)
```

| Aspect | LCS | Ordered Match F1 |
|--------|-----|-------------------|
| Order handling | Binary: step is in-order or not | Graded: measures degree of disorder |
| Partial credit | No — out-of-order steps fully ignored | Yes — out-of-order steps still count, with penalty |
| Sensitivity | Harsh on long chains with few swaps | Proportional to number of inversions |
| Relationship to F1 | Replaces F1 entirely | Augments existing F1 |

**Example showing the difference:**

```
Reference:  R1 → R2 → R3 → R4 → R5
Predicted:  P1(R2) → P2(R1) → P3(R3) → P4(R4) → P5(R5)
```

- **LCS:** {R2, R3, R4, R5} → LCS_F1 = 2*4/10 = 0.80
  (R1 is excluded because it appears after R2 in prediction)
- **Ordered Match F1:** All 5 matched, A = (2,1,3,4,5), tau = 0.80, tau_norm = 0.90
  Ordered_F1 = 1.0 * (0.7 + 0.3 * 0.90) = 0.97

LCS is **too harsh** here — one early swap causes a 20% score drop. Ordered Match F1 correctly identifies this as a minor disorder.

### 6.2 Position-Weighted F1

**Approach:** Weight each match by how close the predicted position is to the reference position.

```
weight(p_i, r_j) = 1 - |pos(p_i)/m - pos(r_j)/n|
Position_F1 = weighted_mean(weights)
```

| Aspect | Position-Weighted | Ordered Match F1 |
|--------|-------------------|-------------------|
| Position sensitivity | Absolute position matters | Only relative order matters |
| Length invariance | Sensitive to sequence length differences | Robust to length differences |
| Interpretability | Weights are hard to explain | Tau has clear meaning |
| Edge behavior | Can penalize correct order if lengths differ | No penalty for correct relative order |

**Problem with absolute positions:** If reference has 5 steps and prediction has 3, even perfectly ordered predictions get penalized because absolute positions don't align.

```
Reference:  R1(pos=0.0) → R2(pos=0.25) → R3(pos=0.5) → R4(pos=0.75) → R5(pos=1.0)
Predicted:  P1(pos=0.0) → P2(pos=0.5) → P3(pos=1.0)

Matches: P1→R1, P2→R3, P3→R5 (all in correct order!)
Position weights: |0-0|=0, |0.5-0.5|=0, |1.0-1.0|=0 → perfect (by luck)

But: P1→R1, P2→R2, P3→R3 (also correct order)
Position weights: |0-0|=0, |0.5-0.25|=0.25, |1.0-0.5|=0.50 → penalized!
```

The position-weighted approach conflates "matched the right step" with "matched it at the proportionally right position", which is a different (and less useful) property.

### 6.3 Summary Comparison

| Metric | Pros | Cons | Best For |
|--------|------|------|----------|
| **Match F1** (current) | Simple, interpretable, content-focused | Ignores order entirely | Content coverage only |
| **LCS-based** | Strict order enforcement | Too harsh on minor reorderings, no partial credit | Strict sequential tasks |
| **Position-Weighted** | Captures absolute positioning | Length-sensitive, can penalize correct order | Fixed-length sequences |
| **Ordered Match F1** (proposed) | Graded order penalty, augments F1, length-robust | Adds complexity, alpha needs tuning | General reasoning chains |

---

## 7. Recommended Defaults and Tuning

### Default Configuration

```python
alpha = 0.3              # Mild order preference
encoder = "all-distilroberta-v1"  # Winner of ablation study — ALWAYS default
threshold = 0.35         # Used in final_table evaluation (threshold-invariant in [0.30, 0.50])
```

> **Note:** The encoder and threshold defaults come directly from our **ablation study** (100 experiments across 5 VLM models × 4 encoders × 5 thresholds, evaluated on 6,372 samples each). `all-distilroberta-v1` won **unanimously on all 5 models** with a cross-model average F1 of 0.5204, beating the runner-up (`all-mpnet-base-v2` at 0.4791) by 4.1 percentage points. A key finding is that distilroberta is **threshold-invariant** — it produces identical results across the entire [0.30, 0.50] range — eliminating the need for threshold tuning.
>
> | Model | distilroberta F1 | Runner-up F1 | Gap |
> |-------|-----------------|--------------|-----|
> | Qwen2.5-VL-32B | 0.6525 | 0.6154 (mpnet@0.30) | +3.7% |
> | Gemma3-4B | 0.6179 | 0.5665 (mpnet@0.30) | +5.1% |
> | Gemma3-12B | 0.6049 | 0.5595 (mpnet@0.30) | +4.5% |
> | LLaVA-7B | 0.5120 | 0.4509 (mpnet@0.30) | +6.1% |
> | MiniCPM-V-8B | 0.2149 | 0.2032 (mpnet@0.30) | +1.2% |
>
> Word overlap exists only as a fallback for DeepSpeed multi-GPU training stability. All evaluation — including Ordered Match F1 — must use `all-distilroberta-v1` semantic matching.

### Tuning Guidelines

| alpha | Behavior | Use When |
|-------|----------|----------|
| 0.0 | Standard Match F1 | Order genuinely doesn't matter (e.g., listing features) |
| 0.1–0.2 | Very mild preference | Reasoning is semi-structured, order is a weak signal |
| **0.3** | **Recommended default** | **General visual reasoning (observe → identify → conclude)** |
| 0.5 | Strong order preference | Mathematical proofs, step-by-step derivations |
| 0.7–1.0 | Very strict ordering | Procedural tasks where order is critical |

### Validation Strategy

To select alpha for a given dataset:
1. Annotate a small set (50–100 examples) with human order-quality scores
2. Compute Ordered Match F1 at several alpha values
3. Select the alpha that maximizes correlation with human judgments

---

## 8. Implementation Notes

The implementation would modify **`MLLMReasoningEvaluator`** in `mllm_evaluator.py` to optionally return match indices from `_find_matches()`, then compute Kendall's Tau on those indices. Estimated changes:

- Modify `_find_matches()` to also return the list of matched `(pred_idx, ref_idx)` pairs (currently returns only sets)
- Add `evaluate_single_ordered()` method that calls `evaluate_single()`, extracts match pair indices, computes tau, and applies the order penalty
- Add `alpha: float = 0.3` parameter for order sensitivity
- Use `scipy.stats.kendalltau` for tau computation (no DeepSpeed constraints since this is evaluation-only)

No changes to `simple_similarity.py` or the GRPO training reward functions are needed — this is strictly an **evaluation-time enhancement** to `MLLMReasoningEvaluator`, consistent with our hybrid strategy (simple word-overlap rewards for training, rich semantic evaluation with `all-distilroberta-v1` for reporting).

---

## 9. References

- Kendall, M. G. (1938). "A New Measure of Rank Correlation." *Biometrika*, 30(1/2), 81–93.
- DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."
- Our implementation: `mllm_evaluator/mllm_evaluator.py:MLLMReasoningEvaluator` — the evaluation-time Match F1 metric (accepts any SentenceTransformer encoder, default `all-distilroberta-v1` per ablation)
- Training reward (separate): `mllm_evaluator/simple_similarity.py` — word overlap (`best_match_f1`) for GRPO training stability with DeepSpeed
- Ablation study: `ablation_results/` — 100 experiments (5 VLMs × 4 encoders × 5 thresholds, 6,372 samples each). `all-distilroberta-v1` won unanimously across all models with cross-model avg F1 = 0.5204, and is threshold-invariant in [0.30, 0.50]
