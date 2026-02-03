# Guided GRPO (G2RPO) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Guided GRPO that injects reference reasoning steps into rollout trajectories to improve reasoning quality and training stability.

**Architecture:** Modify the GRPO trainer to split each batch into guided (with hints) and unguided samples. Guided samples receive a prefix of reference_steps before generating their response. An adaptive mechanism adjusts guidance length based on training rewards.

**Tech Stack:** PyTorch, TRL (GRPOTrainer), Transformers, DeepSpeed

---

## Background

Based on G2RPO-A paper (arXiv:2508.13023):
- **Guidance Ratio (α):** Fraction of samples that receive hints (0.14-0.25 for math)
- **Guidance Length (ℓ):** Number of reference steps to inject (adaptive)
- **Adaptive Formula:** `ℓ_{k+1} = ℓ_k × (r_k / avg_reward)` - reduce guidance when rewards improve

Current composite GRPO has high variance (σ=6.8%). Guided GRPO should improve stability and reasoning quality.

---

## Task 1: Add Guidance Configuration Parameters

**Files:**
- Modify: `src/open-r1-multimodal/src/open_r1/grpo_rec.py:66-120`

**Step 1: Add new dataclass fields**

Add these fields to `GRPOScriptArguments`:

```python
@dataclass
class GRPOScriptArguments(ScriptArguments):
    # ... existing fields ...

    # Guided GRPO parameters
    use_guided_grpo: bool = field(
        default=False,
        metadata={"help": "Enable Guided GRPO with reference step injection"},
    )
    guidance_ratio: float = field(
        default=0.25,
        metadata={"help": "Fraction of samples to guide (0.0-1.0). Default 0.25 for math tasks"},
    )
    guidance_num_steps: int = field(
        default=3,
        metadata={"help": "Number of reference steps to inject as hints (initial value for adaptive)"},
    )
    adaptive_guidance: bool = field(
        default=True,
        metadata={"help": "Enable adaptive guidance length based on training rewards"},
    )
    guidance_history_window: int = field(
        default=2,
        metadata={"help": "Number of steps to average for adaptive guidance calculation"},
    )
    guidance_min_steps: int = field(
        default=1,
        metadata={"help": "Minimum number of guidance steps"},
    )
    guidance_max_steps: int = field(
        default=8,
        metadata={"help": "Maximum number of guidance steps"},
    )
```

**Step 2: Verify syntax**

Run: `python -c "from open_r1.grpo_rec import GRPOScriptArguments; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/open-r1-multimodal/src/open_r1/grpo_rec.py
git commit -m "feat(guided-grpo): add guidance configuration parameters"
```

---

## Task 2: Create Guided Prompt Generator

**Files:**
- Modify: `src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py`

**Step 1: Add guided prompt method**

Add this method to `Qwen2VLModule` class (after `get_question_template`):

```python
@staticmethod
def create_guided_prompt(question: str, reference_steps: list, num_steps: int) -> str:
    """Create a prompt with injected reference steps as hints.

    Args:
        question: The original question
        reference_steps: List of reference reasoning steps
        num_steps: Number of steps to inject as hints

    Returns:
        Modified question with reasoning hints prepended
    """
    if not reference_steps or num_steps <= 0:
        return question

    # Take first N steps as hints
    hints = reference_steps[:num_steps]

    # Format hints
    hints_text = "\n".join([f"- {step}" for step in hints])

    # Create guided prompt
    guided_prompt = f"""{question}

Here are some reasoning hints to help you:
{hints_text}

Now, complete your analysis and provide the final answer in JSON format with "reasoning_steps" and "answer" fields."""

    return guided_prompt
```

**Step 2: Test the method manually**

```python
# Quick test
from open_r1.vlm_modules.qwen_module import Qwen2VLModule

q = "What is 2+2?"
refs = ["Identify numbers: 2 and 2", "Add them: 2+2=4", "Result is 4"]
result = Qwen2VLModule.create_guided_prompt(q, refs, 2)
print(result)
# Should show question + first 2 hints
```

**Step 3: Commit**

```bash
git add src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py
git commit -m "feat(guided-grpo): add create_guided_prompt method"
```

---

## Task 3: Modify Dataset to Support Guidance Selection

**Files:**
- Modify: `src/open-r1-multimodal/src/open_r1/grpo_rec.py:133-280` (LazySupervisedDataset)

**Step 1: Add guidance flag to dataset items**

Modify `__getitem__` to include reference_steps in returned dict (already present, verify):

```python
def __getitem__(self, i) -> dict:
    # ... existing code ...

    # Ensure reference_steps is always included
    return {
        'image': image,
        'problem': user_question,
        'solution': example['answer'],
        'reference_steps': example.get('reference_steps', []),  # Ensure this exists
        'prompt': prompt,
        'image_file': f"{example.get('source', 'unknown')}_{i}",
        'data_index': i,
    }
```

**Step 2: Verify dataset returns reference_steps**

```python
from open_r1.grpo_rec import LazySupervisedDataset, GRPOScriptArguments

# Create minimal args
args = GRPOScriptArguments(dataset_name="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps")
args.use_huggingface_dataset = True
ds = LazySupervisedDataset(args.dataset_name, args, "{USER_INSTRUCTION}", seed=42)
item = ds[0]
print("Has reference_steps:", 'reference_steps' in item)
print("Num steps:", len(item.get('reference_steps', [])))
```

**Step 3: Commit**

```bash
git add src/open-r1-multimodal/src/open_r1/grpo_rec.py
git commit -m "feat(guided-grpo): ensure dataset includes reference_steps"
```

---

## Task 4: Implement Guidance Injection in Trainer

**Files:**
- Modify: `src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py`

**Step 1: Add guidance state to trainer __init__**

Find the `__init__` method and add guidance state tracking:

```python
def __init__(self, ...):
    # ... existing init code ...

    # Guided GRPO state
    self.use_guided_grpo = getattr(self.args, 'use_guided_grpo', False)
    self.guidance_ratio = getattr(self.args, 'guidance_ratio', 0.25)
    self.guidance_num_steps = getattr(self.args, 'guidance_num_steps', 3)
    self.adaptive_guidance = getattr(self.args, 'adaptive_guidance', True)
    self.guidance_history_window = getattr(self.args, 'guidance_history_window', 2)
    self.guidance_min_steps = getattr(self.args, 'guidance_min_steps', 1)
    self.guidance_max_steps = getattr(self.args, 'guidance_max_steps', 8)

    # Reward history for adaptive guidance
    self.reward_history = []
    self.current_guidance_length = self.guidance_num_steps

    if self.use_guided_grpo:
        print(f"✓ Guided GRPO enabled: ratio={self.guidance_ratio}, steps={self.guidance_num_steps}, adaptive={self.adaptive_guidance}")
```

**Step 2: Add adaptive guidance update method**

```python
def update_adaptive_guidance(self, current_reward: float):
    """Update guidance length based on reward history (G2RPO-A formula)."""
    if not self.adaptive_guidance:
        return

    self.reward_history.append(current_reward)

    # Need at least 2 steps for comparison
    if len(self.reward_history) < 2:
        return

    # Calculate average of recent rewards
    window = min(self.guidance_history_window, len(self.reward_history) - 1)
    recent_avg = sum(self.reward_history[-window-1:-1]) / window

    if recent_avg > 0:
        # G2RPO-A formula: ℓ_{k+1} = ℓ_k × (recent_avg / current_reward)
        # When current > recent: reduce guidance (model improving)
        # When current < recent: increase guidance (model struggling)
        ratio = recent_avg / current_reward if current_reward > 0 else 1.5
        new_length = self.current_guidance_length * ratio

        # Clamp to bounds
        self.current_guidance_length = max(
            self.guidance_min_steps,
            min(self.guidance_max_steps, round(new_length))
        )

        print(f"  Adaptive guidance: reward={current_reward:.3f}, avg={recent_avg:.3f}, new_length={self.current_guidance_length}")
```

**Step 3: Commit**

```bash
git add src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py
git commit -m "feat(guided-grpo): add guidance state and adaptive update"
```

---

## Task 5: Modify Generation to Apply Guidance

**Files:**
- Modify: `src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py`

**Step 1: Find the generation/rollout method**

Locate where prompts are prepared for generation (likely `_generate_completions` or similar).

**Step 2: Add guidance injection logic**

Before generating, split batch into guided/unguided:

```python
def _prepare_guided_batch(self, batch):
    """Prepare batch with guidance injection for some samples."""
    if not self.use_guided_grpo:
        return batch

    from open_r1.vlm_modules.qwen_module import Qwen2VLModule

    batch_size = len(batch['problem'])
    num_guided = int(batch_size * self.guidance_ratio)

    # Randomly select which samples get guidance
    import random
    guided_indices = set(random.sample(range(batch_size), num_guided))

    # Modify problems for guided samples
    modified_problems = []
    guidance_applied = []

    for i in range(batch_size):
        problem = batch['problem'][i]
        ref_steps = batch.get('reference_steps', [None] * batch_size)[i]

        if i in guided_indices and ref_steps:
            # Apply guidance
            modified_problem = Qwen2VLModule.create_guided_prompt(
                problem,
                ref_steps,
                self.current_guidance_length
            )
            modified_problems.append(modified_problem)
            guidance_applied.append(True)
        else:
            modified_problems.append(problem)
            guidance_applied.append(False)

    batch['problem'] = modified_problems
    batch['_guidance_applied'] = guidance_applied

    num_actually_guided = sum(guidance_applied)
    if num_actually_guided > 0:
        print(f"  Guided batch: {num_actually_guided}/{batch_size} samples with {self.current_guidance_length} steps")

    return batch
```

**Step 3: Integrate into training loop**

Find where `compute_loss` or similar is called and:
1. Apply guidance before generation
2. After computing rewards, call `update_adaptive_guidance(mean_reward)`

**Step 4: Commit**

```bash
git add src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py
git commit -m "feat(guided-grpo): implement guidance injection in generation"
```

---

## Task 6: Create Training Script

**Files:**
- Create: `train_guided_grpo.sh`

**Step 1: Create training script**

```bash
#!/bin/bash

# Guided GRPO Training Script
# Based on G2RPO-A (arXiv:2508.13023)

PROJECT_ROOT="/gpudata3/Wayner/VLM-R1"
SRC_DIR="${PROJECT_ROOT}/src/open-r1-multimodal/src"
MLLM_EVALUATOR_DIR="${PROJECT_ROOT}/mllm_evaluator"
DEEPSPEED_CONFIG="${PROJECT_ROOT}/src/open-r1-multimodal/local_scripts/zero3.json"

export PYTHONPATH="${SRC_DIR}:${MLLM_EVALUATOR_DIR}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model and dataset
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="/gpudata3/Wayner/reasoning/reasoning_train_with_reference_steps"
OUTPUT="${PROJECT_ROOT}/output/GRPO_guided_$(date +%Y%m%d_%H%M%S)"

# GPU Configuration
GPU_IDS="0,1,2,3"
export CUDA_VISIBLE_DEVICES=$GPU_IDS
NUM_GPUS=4

# Training hyperparameters (from stable checkpoint-1400 config)
PER_DEVICE_BATCH=4
GRADIENT_ACCUM=2
NUM_GENERATIONS=4  # Increased for better baseline estimation
LEARNING_RATE=3e-6
MAX_STEPS=1500
SAVE_STEPS=100
SEED=42

# Guided GRPO parameters
USE_GUIDED_GRPO=true
GUIDANCE_RATIO=0.25      # 25% of samples get hints
GUIDANCE_NUM_STEPS=3     # Start with 3 reference steps
ADAPTIVE_GUIDANCE=true   # Enable adaptive adjustment
GUIDANCE_MIN_STEPS=1
GUIDANCE_MAX_STEPS=6

# Image processing
MAX_PIXELS=602112
MIN_PIXELS=3136

mkdir -p $OUTPUT
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT}/reward.txt"

echo "========================================"
echo "Guided GRPO Training (G2RPO-A)"
echo "========================================"
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "GPUs: $GPU_IDS"
echo ""
echo "Guided GRPO Settings:"
echo "  - Guidance ratio: $GUIDANCE_RATIO"
echo "  - Initial guidance steps: $GUIDANCE_NUM_STEPS"
echo "  - Adaptive guidance: $ADAPTIVE_GUIDANCE"
echo "  - Step range: $GUIDANCE_MIN_STEPS - $GUIDANCE_MAX_STEPS"
echo "========================================"

accelerate launch \
    --num_processes $NUM_GPUS \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29502 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file $DEEPSPEED_CONFIG \
    --zero3_init_flag true \
    --zero3_save_16bit_model true \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --gradient_clipping 1.0 \
    ${SRC_DIR}/open_r1/grpo_rec.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --reward_weights 2.0 3.0 1.0 \
    --use_guided_grpo $USE_GUIDED_GRPO \
    --guidance_ratio $GUIDANCE_RATIO \
    --guidance_num_steps $GUIDANCE_NUM_STEPS \
    --adaptive_guidance $ADAPTIVE_GUIDANCE \
    --guidance_min_steps $GUIDANCE_MIN_STEPS \
    --guidance_max_steps $GUIDANCE_MAX_STEPS \
    --output_dir $OUTPUT \
    --seed $SEED \
    --shuffle_train_dataset \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $PER_DEVICE_BATCH \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --learning_rate $LEARNING_RATE \
    --num_generations $NUM_GENERATIONS \
    --gradient_checkpointing \
    --logging_steps 2 \
    --save_steps $SAVE_STEPS \
    --max_pixels $MAX_PIXELS \
    --min_pixels $MIN_PIXELS \
    --bf16 \
    --deepspeed $DEEPSPEED_CONFIG \
    2>&1 | tee -a $OUTPUT/training.log

echo "Training completed! Output: $OUTPUT"
```

**Step 2: Make executable and commit**

```bash
chmod +x train_guided_grpo.sh
git add train_guided_grpo.sh
git commit -m "feat(guided-grpo): add training script"
```

---

## Task 7: Integration Testing

**Files:**
- Test manually with small run

**Step 1: Quick validation run**

```bash
# Test with 10 steps only
MAX_STEPS=10 SAVE_STEPS=5 bash train_guided_grpo.sh
```

**Step 2: Verify guidance is being applied**

Check logs for:
- "Guided GRPO enabled" message
- "Guided batch: X/Y samples" during training
- "Adaptive guidance: reward=..." updates

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix(guided-grpo): integration fixes from testing"
```

---

## Task 8: Full Training Run

**Step 1: Launch full training**

```bash
nohup bash train_guided_grpo.sh > guided_grpo_training.log 2>&1 &
echo "Training started. Monitor with: tail -f guided_grpo_training.log"
```

**Step 2: Monitor progress**

```bash
# Watch training progress
tail -f output/GRPO_guided_*/training.log | grep -E "(loss|reward|guidance|Epoch)"
```

**Step 3: Evaluate checkpoints when ready**

```bash
# After training completes, evaluate
for ckpt in 300 600 900 1200 1500; do
    python compute_metrics.py predictions/guided-checkpoint-$ckpt \
        --dataset-path /gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27 \
        --output-dir evaluations/guided_grpo/checkpoint-$ckpt
done
```

---

## Success Criteria

After implementation, compare Guided GRPO vs baselines:

| Metric | Answer-Only | Composite | Guided GRPO (target) |
|--------|-------------|-----------|---------------------|
| Best Accuracy | 44.90% | 44.92% | **>46%** |
| Training Variance | σ=2.3% | σ=6.8% | **<4%** |
| Convergence (42%+) | 300 steps | 1000 steps | **<500 steps** |
| Match F1 | 0.43 | 0.43 | **>0.50** |

---

## Rollback Plan

If Guided GRPO doesn't improve:
1. Check guidance is actually being injected (DEBUG logs)
2. Try different `guidance_ratio` (0.15, 0.35, 0.50)
3. Disable adaptive guidance and use fixed steps
4. Revert to main branch: `git checkout main`
