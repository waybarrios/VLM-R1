# Reasoning Reward Noise Analysis

## 1. Limitations of Word Overlap / Token Matching

### Fundamental Problems with Surface-Level Metrics

Traditional metrics like BLEU and ROUGE have three significant limitations:

1. **Semantic Opposition**: They can assign high scores to semantically opposite content
2. **Semantic Equivalence Ignored**: They give low scores to semantically related content due to surface differences
3. **Unintelligible Text**: They can assign high scores to unintelligible text if token overlap is high

### Specific Issues for Reasoning Evaluation

> "Token-level F1 and BERTScore are commonly used to evaluate LLMs on causal QA tasks. However, these metrics assess surface similarity, not semantic equivalence. Semantically incorrect expressions might score well due to shared tokens."

Even **BERTScore** fails for logical/causal reasoning:
- Assigns high similarity to expressions that are semantically disjoint under causal graphs
- Does not satisfy a soundness guarantee
- "Particularly concerning in high-stakes settings"

## 2. How Noisy Are Different Reasoning Reward Signals?

### Sources of Noise

**Corrupted/Mis-specified Rewards**:
> "Reward noise refers to stochasticity, corruption, or systematic bias in the reward signals used during training."

**Monte Carlo Estimation Noise**:
- **False Positives**: High reward to incorrect steps due to subsequent self-correction
- **False Negatives**: Low reward to correct steps that lead to failed trajectories
- MCE-labeled datasets contain significant label noise

### Outcome Reward Models (ORMs) - Noise Characteristics

> "ORMs in RLVR are too coarse-grained to distinguish flawed reasoning within correct answers or valid reasoning within incorrect answers."

Key issues:
- Sparse reward signals (only at final answer)
- Cannot localize intermediate errors
- **80.1% answer correctness but only 39.7% process soundness**

### Process Reward Models (PRMs) - Noise Characteristics

> "While PRMs offer fine-grained guidance, they frequently suffer from inaccuracies and are susceptible to reward hacking."

Key issues:
- **Training data noise**: MC estimation inferior to human annotation
- **LLM-as-judge noise**: "noisy, inconsistent, and less discriminative"
- **Reward hacking**: Models generate verbose/repetitive steps
- **Annotation disagreement**: ~18% of reasoning steps have low agreement

## 3. Alternatives to Word Overlap

### Neural Embedding-Based Metrics

**BERTScore**:
- 0.93 Pearson correlation with human judgments (vs 0.70 for BLEU)
- Better alignment on semantic tasks
- Robust to paraphrasing
- **Limitation**: Still fails for logical/causal reasoning

**BLEURT**:
- Built on BERT with pre-training on randomized Wikipedia changes
- Better at capturing semantic nuances

**UniEval**:
- Unifies evaluation dimensions into Boolean QA framework
- Assesses text from multiple angles

### LLM-as-Judge

Using GPT-4 or other LLMs:
- Shows improved alignment with ground truth
- Introduces new biases (position bias, verbosity bias, self-preference)
- Inconsistent across replications
- Position bias: select "Response B" 60-69% of time

### Symbolic/Formal Verification

- **Symbolic-based approaches**: Use formal logic to verify step validity
- **Computational graph verification**: Verify via formal representations
- **DoVerifier**: Provides soundness guarantees for causal reasoning

## 4. What Makes a Good Reasoning Reward Signal?

Based on research:

1. **Granularity**: Step-level feedback, but avoid noise from poor labeling
2. **Verifiability**: Direct binary feedback from deterministic tools when possible
3. **Noise Robustness**: Either clean human labels, or noise-aware learning methods
4. **Process-Outcome Consistency**: Use PRMs as filters rather than direct rewards
5. **Multi-dimensional Assessment**: Evaluate correctness, informativeness, coherence
6. **Calibration**: Avoid over-affirming correctness
7. **Diverse Verification**: Combine multiple verification strategies

## Implications for Our Current System

Our current `vqa_reasoning_reward` uses:
```python
f1, matched_pred, matched_ref = best_match_f1(predicted_steps, ref_steps_cleaned, threshold=0.45)
reward = f1
```

**Problems**:
- Word overlap doesn't capture semantic equivalence
- Threshold 0.45 is arbitrary
- No semantic understanding of step meaning
- Rewards surface similarity, not logical correctness

**Recommendation**: Replace with semantic embedding similarity (e.g., SentenceTransformer) to reduce noise while maintaining reasoning quality signal.

## Sources

- [Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
- [The Lessons of Developing PRMs](https://arxiv.org/abs/2501.07301)
- [Towards Robust Process Reward Modeling](https://arxiv.org/abs/2601.12748)
- [BERTScore Paper](https://arxiv.org/abs/1904.09675)
- [A Chain-of-Thought Is as Strong as Its Weakest Link (ACL 2024)](https://aclanthology.org/2024.acl-long.254.pdf)
- [Justice or Prejudice? LLM-as-a-Judge Biases](https://arxiv.org/html/2410.02736v1)
