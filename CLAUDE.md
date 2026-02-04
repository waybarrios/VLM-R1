# VLM-R1 GRPO Training Context

> **Important:** Read the plans folder for pending tasks: `/thayerfs/home/f005d5s/.claude/plans/`

---

## Key Output Folders

| Training | Path | Best Checkpoint | Notes |
|----------|------|-----------------|-------|
| **Composite GRPO** | `output/qwen2.5-vl-3b-vqa-deepspeed-20251116_133650/` | checkpoint-1400 (44.92%) | Mode collapse at step 600 |
| **Answer-Only Baseline** | `output/GRPO_answer_only_baseline_20260129_202515/` | checkpoint-1400 (42.56%) | Stable, no reasoning |
| **Current Optimal** | `output/grpo-optimal-20260202_222633/` | TBD | In progress |

**Analysis reports:** `GRPO_analysis/ANALYSIS_REPORT.md`

---

## Current Status (Feb 3, 2026)

**Training in progress:** `output/grpo-optimal-20260202_222633`
- Step: ~634/7578 (8%)
- GPUs: 0-3 (DeepSpeed ZeRO-3)
- Estimated time remaining: ~120 hours

## What We Did

### Problem
The previous composite GRPO training had two issues:
1. **Mode collapse**: Accuracy dropped from 44.92% to 26.98% at step 600
2. **DeepSpeed + SentenceTransformer incompatibility**: "'weight' must be 2-D" errors when using semantic similarity in multi-GPU training

### Solution: Hybrid Reward Strategy (DeepSeek-R1 inspired)
- **Training:** Use word overlap F1 (simple, stable gradients)
- **Evaluation:** Use semantic similarity via SentenceTransformer (captures paraphrase equivalence)

### Files Modified

1. **`mllm_evaluator/simple_similarity.py`**
   - Added `_get_semantic_model()`: Process-local model caching for DeepSpeed compatibility
   - Added `semantic_match_f1()`: Semantic similarity using SentenceTransformer embeddings + cosine similarity

2. **`src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py`**
   - Added `_semantic_similarity_threshold = 0.70` class variable
   - Added `configure_semantic_reward()` classmethod
   - Added `vqa_reasoning_reward_semantic()` method
   - Updated `select_reward_func()` to include "reasoning_semantic" case

3. **`src/open-r1-multimodal/src/open_r1/grpo_rec.py`**
   - Added `--use_semantic_reasoning_reward` CLI flag
   - Added `--semantic_similarity_threshold` CLI flag

4. **Created `train_grpo_optimal.sh`** - Training script using word overlap for stability

### Branch
`feature/semantic-process-reward` pushed to https://github.com/waybarrios/VLM-R1

## Results Comparison

### At Step 600
| Training | Accuracy | Status |
|----------|----------|--------|
| Previous Composite | 26.98% | Collapsed |
| Answer-Only Baseline | 42.56% | Stable but no reasoning |
| **Current (Optimal)** | **53-60%** | Stable with reasoning |

### Previous Best
- Checkpoint-1400: 44.92% accuracy (before degradation)

### Current Training Metrics (Step 634)
- Format: 93.8%
- Accuracy: 53.7%
- Reasoning: 0.104
- No mode collapse observed

## Expected Outcome
With current stability, we expect final accuracy of **46-48%** or higher, evaluated with semantic F1 post-training.

## Commands

### Monitor training progress
```bash
tail -f /gpudata3/Wayner/VLM-R1/output/grpo-optimal-20260202_222633/training.log
```

### Check GPU usage
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

### Quick metrics check
```bash
tail -100 output/grpo-optimal-20260202_222633/training.log | grep -E "accuracy|reward"
```

## Next Steps
1. Wait for training to complete
2. Evaluate checkpoints with semantic F1: `python inference/evaluate_predictions.py --use_semantic`
3. Compare final results against baselines

## Plans and Documentation

### Active Plans
Read plans folder: `/thayerfs/home/f005d5s/.claude/plans/`
- `peppy-honking-dolphin.md` - Semantic Process Reward (SPR) Implementation Plan

### Analysis Reports
- `GRPO_analysis/ANALYSIS_REPORT.md` - Comprehensive evaluation of all checkpoints
- `GRPO_analysis/consolidated_results.csv` - Raw metrics data

### Research Ideas (Not Yet Implemented)
From the investigation, these could be future improvements:
1. **Curriculum staging**: Train accuracy first, then add reasoning
2. **Reward smoothing**: Reduce noise in reasoning signal
3. **Dynamic weight adjustment**: Increase reasoning weight as training progresses

## For Advisor (Informal Summary)
We're trying a new approach for GRPO training based on how DeepSeek-R1 does it. We use simple word overlap during training (keeps things stable) and semantic similarity only for evaluation after. This fixes the GPU crashes we had before with SentenceTransformer. The previous composite training collapsed badly at step 600, dropping to 27% accuracy, but this one is holding steady around 53-60%. The answer-only baseline gets around 42.5% accuracy but obviously can't reason. We're hoping this hybrid approach gives us the best of both worlds: stable training like answer-only, but with actual reasoning capabilities. If things keep going well, we should beat the previous best of 44.9%.
