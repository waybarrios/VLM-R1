# Curriculum and Staged Training Research

## 1. Should Accuracy Be Learned Before Reasoning Quality?

The research suggests: **accuracy and reasoning quality should be developed together, but with careful staging**.

### Key Findings

**DeepSeek R1's Approach**: Uses a **simple accuracy-focused reward** (binary correct/incorrect) in initial RL stages, allowing reasoning behaviors to emerge naturally. They avoided LLM-based reward models for reasoning quality due to reward hacking risks.

**The Reward Hacking Problem**: Using PRMs that evaluate reasoning quality during RL can backfire. Models learn to exploit PRMs by generating numerous correct but unnecessary reasoning steps.

**Progressive Reward Shaping (PRS)**: Provides dense, stage-wise learning signals, enabling models to **first master essential capabilities before optimizing for challenging objectives**.

## 2. Optimal Order/Staging for Multi-Objective Training

### The Multi-Objective Challenge

Training involves multiple objectives (accuracy, format, reasoning quality). Research shows these can **interfere with each other**.

**Key Finding**: Staged RL consistently outperforms mixed strategy training by reducing objective interference.

### Recommended Staging Order

1. **Stage 1 - Format/Basic Skills**: Produce parseable outputs
2. **Stage 2 - Accuracy on Easy Tasks**: Build foundational correctness
3. **Stage 3 - Accuracy on Hard Tasks**: Progress to challenging problems
4. **Stage 4 - Alignment/Safety**: Final refinement

## 3. How DeepSeek R1 and Other Models Stage Their Training

### DeepSeek R1: Four-Stage Pipeline

| Stage | Type | Purpose |
|-------|------|---------|
| **1. Cold-Start SFT** | SFT | Collect long CoT examples with human-aligned thinking |
| **2. Reasoning RL** | RL | Rule-based rewards: accuracy + language consistency (NO neural reward models) |
| **3. Rejection Sampling + SFT** | SFT | Synthetic data from best RL outputs |
| **4. Alignment RL** | RL | Helpfulness and harmlessness |

**Critical Design Choice**: DeepSeek avoided LLM-based reward models entirely for reasoning RL.

### Qwen3/QwQ: Four-Stage Pipeline

| Stage | Purpose |
|-------|---------|
| **1. Long CoT Cold Start** | Fine-tune on long chain-of-thought datasets |
| **2. Reasoning RL** | Scale up with rule-based rewards using GRPO |
| **3. Thinking Mode Fusion** | Integrate non-thinking capabilities |
| **4. General RL** | Apply RL across 20+ general tasks |

## 4. Easy-to-Hard Curriculum

### E2H Reasoner Framework

**Key Findings**:
- RL struggles on harder tasks where pre-trained models have low zero-shot performance
- **Easy tasks are important initially, but fading them out is essential**
- Uses probabilistic scheduler that gradually shifts focus from easy to hard
- Small LLMs (1.5B-3B) that fail with vanilla RL can succeed with E2H curriculum

**Theoretical Guarantee**: Learning through curriculum stages requires **fewer total samples**.

### Difficulty-Aware Staged RL Results

| Training Strategy | MATH-500 | AIME-2024 |
|-------------------|----------|-----------|
| Mixed difficulty | Baseline | Baseline |
| **Staged (easy→hard)** | **+5.6%** | **+13.4%** |

## 5. SFT Before RL vs RL-Only

### Current Research Consensus

| Approach | Pros | Cons |
|----------|------|------|
| **SFT → RL** | Stable training, better initialization | Can induce "pseudo reasoning" that limits exploration |
| **RL Only** | More genuine reasoning, better exploration | Training instability, readability issues |
| **Alternating SFT/RL** | Best of both worlds | More complex |

**Key Warning**: High SFT scores don't reliably predict RL gains. Sometimes RL on base models without SFT outperforms RL on SFT-tuned models.

## Implications for CRYSTAL

1. **Don't abandon reasoning reward** - but make it less noisy
2. **Consider staging**: Format → Accuracy → Reasoning quality
3. **Use semantic matching** instead of word overlap for reasoning reward
4. **Curriculum by difficulty** could help with harder reasoning tasks

## Sources

- [DeepSeek-R1 Paper](https://arxiv.org/html/2501.12948v1)
- [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
- [Curriculum RL from Easy to Hard Tasks](https://arxiv.org/html/2506.06632v1)
- [How Difficulty-Aware Staged RL Enhances LLM Reasoning](https://arxiv.org/html/2504.00829v1)
- [Progressive Reward Shaping for Agentic RL](https://arxiv.org/html/2512.07478)
- [The State of RL for LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
