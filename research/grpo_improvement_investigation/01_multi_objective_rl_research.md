# Multi-Objective RL Research: Reward Conflicts in GRPO

## 1. Conflicting Reward Signals: Accuracy vs. Reasoning Quality

### The Core Problem

When training LLMs with composite rewards (e.g., accuracy + reasoning quality), these objectives often conflict. Research shows that **optimizing for one objective can degrade performance on another**, particularly when using simple weighted sum approaches.

### Key Findings

- **Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs)**: PRMs provide feedback at each reasoning step, enabling better credit assignment, while ORMs only score final outputs. However, PRMs require expensive human annotations and "haven't been super successful yet" (Sebastian Raschka).

- **RLVR Tradeoffs**: Research found that "all reasoning paths in the RLVR model are already present in the base model. RLVR training biases the distribution toward rewarded paths, improving sampling efficiency. However, this comes at the cost of reduced scope of reasoning capacity."

## 2. Gradient Behavior with Composite Rewards

### Clipping and Gradient Loss

From Cameron Wolfe's GRPO analysis:

- **Tokens with large importance ratios get clipped**: "Fork tokens" (pivotal reasoning markers like "aha" or "wait") are rare, have low probabilities, and thus get assigned large importance ratios. These crucial tokens "are usually clipped by the GRPO objective, which eliminates their contribution to the policy update."

- **Token-Level vs. Sequence-Level Gradients**: Algorithms without value networks (like RLOO) assign the same sequence-level advantage to every token.

### DAPO's Findings on GRPO Issues

The DAPO paper (ByteDance) identified:
- **Entropy collapse**: The naive GRPO baseline suffers from entropy collapse during training
- **Reward noise**: Training instability from noisy reward signals
- **Clipping limitations**: CISPO was proposed to let clipped tokens still contribute

### PCGrad for Gradient Conflict

PCGrad identifies conflicting gradients by computing cosine similarity between task gradients. When negative (conflicting), it projects each gradient onto the normal plane of the other to prevent destructive interference.

## 3. Best Practices for Combining Multiple Rewards

### DeepSeek R1's Approach

DeepSeek R1 used a two-component reward system:
1. **Accuracy Rewards**: Verify if final answers are correct
2. **Format Rewards**: Incentivize structured chain-of-thought

**Key design choice**: "The reward signal is solely based on the correctness of final predictions against ground-truth answers, without imposing constraints on the reasoning process itself."

### Multi-Objective Alignment Approaches

- **PAMA (Pareto Multi-Objective Alignment)**: Transforms multi-objective RLHF into convex optimization
- **Rewards-in-Context (RiC)**: Three-stage approach with offline multi-reward conditional SFT
- **GAPO (Gradient-Adaptive Policy Optimization)**: Gradient rescaling for better trade-off handling

## 4. Why Composite Rewards Fail

### Key Reasons:

1. **Non-Convex Pareto Frontiers**: Simple weighted sum can only find "supported solutions" on convex portions
2. **Gradient Conflicts**: Gradients can point in opposing directions
3. **Credit Assignment Problems**: ORMs provide sparse feedback only on final outputs
4. **Entropy Collapse**: GRPO can suffer from entropy collapse, reducing exploration
5. **Clipping of Important Tokens**: Key reasoning tokens get clipped from gradient updates
6. **Reward Hacking**: Models find unintended ways to maximize composite rewards
7. **Different Objective Scales**: Without proper normalization, one reward dominates
8. **Reasoning Path Narrowing**: RLVR biases toward rewarded paths at cost of diversity

## Sources

- [The State of RL for LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [GRPO++ Tricks](https://cameronrwolfe.substack.com/p/grpo-tricks)
- [Reward Hacking in RL](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [DAPO Paper](https://arxiv.org/pdf/2503.14476)
- [DeepSeek-R1](https://arxiv.org/html/2501.12948v1)
- [Panacea - NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/89f39d0b3d49a47606a165eefba2778c-Paper-Conference.pdf)
- [PCGrad](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
