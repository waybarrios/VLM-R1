# GRPO Improvement Investigation

**Date:** 2026-02-02
**Branch:** feature/guided-grpo
**Goal:** Investigate why composite GRPO (accuracy + reasoning) doesn't improve over answer-only, and propose novel solutions

## Executive Summary

This investigation explores why the composite reward (accuracy + reasoning F1) fails to improve both accuracy AND reasoning quality simultaneously, despite the CRYSTAL hypothesis that logical reasoning steps matter.

### Key Finding

The composite GRPO achieves identical accuracy (44.9%) to answer-only but with 3x higher training variance. The root cause is a combination of:

1. **Noisy reward signal** - Word overlap F1 doesn't capture semantic equivalence
2. **Gradient conflict** - Accuracy and reasoning rewards can push in opposite directions
3. **Mode oscillation** - Model switches between conservative (few steps) and verbose (many steps) modes

### Proposed Solution Direction

**Semantic Reasoning Reward** - Replace word overlap with neural semantic similarity to reduce noise while maintaining the importance of reasoning quality (aligned with CRYSTAL thesis).

## Research Agents Dispatched

Five parallel investigations were conducted:

1. Multi-objective RL reward conflicts
2. Reasoning reward noise analysis
3. Curriculum and staged training research
4. Composite vs Answer-only prediction comparison
5. Training variance pattern analysis

## Files in This Directory

- `01_multi_objective_rl_research.md` - Gradient conflicts, Pareto optimization, reward hacking
- `02_reasoning_reward_noise.md` - PRM limitations, word overlap issues, alternatives
- `03_curriculum_staged_training.md` - DeepSeek R1 approach, easy-to-hard curriculum
- `04_prediction_comparison.md` - Detailed comparison of composite vs answer-only outputs
- `05_variance_analysis.md` - Why checkpoint-600 collapsed, oscillation patterns
- `06_synthesis_and_hypotheses.md` - Combined findings and proposed solutions
