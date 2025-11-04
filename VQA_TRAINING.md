# VQA Training with GRPO

This guide explains how to train Vision-Language Models on Visual Question Answering (VQA) tasks using GRPO (Group Relative Policy Optimization) with reasoning steps.

## Overview

The VQA training mode uses a JSON output format with reasoning steps and final answers, evaluated using three reward functions:
- **Format Reward**: Validates JSON structure
- **Accuracy Reward**: Evaluates answer correctness
- **Reasoning Reward**: Measures reasoning quality using Match F1

## Dataset Requirements

### HuggingFace Dataset Format

Your dataset should be saved using `datasets.save_to_disk()` with the following structure:

```python
Dataset({
    features: ['image', 'question', 'answer', 'reference_steps', 'choices', 'options'],
    num_rows: N
})
```

**Required fields:**
- `image`: PIL Image object
- `question`: str - The question text
- `answer`: str - Ground truth answer
- `reference_steps`: List[str] - Reference reasoning steps for the answer

**Optional fields:**
- `choices`: List[str] - Multiple choice options (if applicable)
- `options`: List[str] - Alternative format for multiple choice options
- `source`: str - Dataset source identifier

### Example Dataset Entry

```python
{
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x730>,
    'question': 'what is the brand of phone?',
    'answer': 'nokia',
    'reference_steps': [
        'Check for visible text related to brands on the mobile phone in the image.',
        "Identify the brand text 'NOKIA' in close proximity to the mobile phone.",
        'Verify there are no other text labels or logos visible on the mobile phone.',
        'Return the exact answer string.'
    ],
    'choices': [],
    'options': [],
    'source': 'textvqa'
}
```

## Installation

### Prerequisites

```bash
# Install base requirements
pip install transformers torch datasets pillow pyyaml

# Install TRL and GRPO dependencies
pip install trl peft accelerate

# Install mllm_evaluator dependencies
pip install sentence-transformers pandas numpy openai tqdm
```

### Optional: Ollama Setup (for LLM-based accuracy grading)

If you want to use LLM-based semantic verification for free-form text answers:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2
```

**Note**: The accuracy calculator uses `use_llm_grader=False` by default for speed. Enable it in the code if needed.

## Training Commands

### Basic VQA Training

```bash
python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" \
    --output_dir "./output/qwen2vl-vqa-grpo" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --save_steps 500 \
    --max_pixels 12845056 \
    --min_pixels 3136
```

### With Reasoning Reward

To also evaluate reasoning quality (requires reference_steps in dataset):

```bash
python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --output_dir "./output/qwen2vl-vqa-grpo-reasoning" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --save_steps 500 \
    --max_pixels 12845056 \
    --min_pixels 3136
```

### Multi-GPU Training

```bash
accelerate launch --num_processes 4 \
    src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --output_dir "./output/qwen2vl-vqa-grpo" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 500
```

### With DeepSpeed ZeRO-3

```bash
deepspeed --num_gpus 4 \
    src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --output_dir "./output/qwen2vl-vqa-grpo" \
    --deepspeed configs/deepspeed_zero3.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 500
```

## Key Parameters

### Task Configuration

- `--task_type "vqa"`: **Required** - Sets the task to VQA mode with JSON output
- `--use_huggingface_dataset`: **Required** - Enables HuggingFace dataset loading
- `--dataset_name`: Path to your HuggingFace dataset directory

### Reward Functions

Choose from the following reward functions (can use multiple):

- `"format"`: Validates JSON structure ({"reasoning_steps": [], "answer": ""})
- `"accuracy"`: Evaluates answer correctness using accuracy_calculator
- `"reasoning"`: Evaluates reasoning quality using Match F1 (requires reference_steps)

**Recommended combinations:**
- Fast training: `--reward_funcs "format" "accuracy"`
- High quality: `--reward_funcs "format" "accuracy" "reasoning"`

### Model Configuration

- `--model_name_or_path`: Model to fine-tune
  - Qwen: `"Qwen/Qwen2-VL-7B-Instruct"`, `"Qwen/Qwen2.5-VL-7B-Instruct"`
  - InternVL: `"OpenGVLab/InternVL2-8B"`
  - GLM: `"THUDM/glm-4v-9b"`

- `--max_pixels`: Maximum pixels for image processing (default: 12845056)
- `--min_pixels`: Minimum pixels for image processing (default: 3136)

### Training Configuration

- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate (recommended: 1e-5 to 5e-6)
- `--bf16` or `--fp16`: Use mixed precision training

### LoRA/QLoRA (Optional)

To use parameter-efficient fine-tuning:

```bash
python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" \
    --output_dir "./output/qwen2vl-vqa-lora" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules "q_proj" "v_proj" "k_proj" "o_proj" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4
```

## Expected Model Output

The model will generate predictions in JSON format:

```json
{
  "reasoning_steps": [
    "The image shows a bar chart with two groups of bars, each representing a year.",
    "The first group of bars is labeled '2003', and the second group is labeled '2010'.",
    "The question asks for the label of the second group of bars from the left.",
    "The second group of bars is clearly labeled '2010'."
  ],
  "answer": "2010"
}
```

### Reasoning Steps Guidelines

The prompt instructs the model to generate reasoning steps that:
- Are single-clause sentences (≤14 words)
- Contain directly checkable facts or cue-based inferences
- Are anchored to visible cues in the image
- Build towards the final answer
- Avoid chains like "because/therefore"

### Answer Format

The answer should follow these rules:
- **Multiple choice**: Return only the letter (e.g., "B") if options have letters
- **Numeric**: Include units and follow requested precision
- **Text**: Provide exact answer grounded in visible content
- **Ambiguous**: Return "insufficient information" if unclear

## Debugging and Monitoring

### Enable Debug Mode

To log detailed reward information:

```bash
export DEBUG_MODE=true
export LOG_PATH="./logs/training_debug.txt"

python src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path "Qwen/Qwen2-VL-7B-Instruct" \
    --dataset_name "/path/to/your/huggingface/dataset" \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --output_dir "./output/qwen2vl-vqa-grpo"
```

This will create log files:
- `training_debug_format_vqa.txt`: Format reward logs
- `training_debug_accuracy_vqa.txt`: Accuracy reward logs
- `training_debug_reasoning_vqa.txt`: Reasoning reward logs

### Monitor Training

Watch the logs for:
- Format reward: Should quickly reach ~1.0 (model learns JSON format)
- Accuracy reward: Should gradually improve (0.0 to 1.0)
- Reasoning reward: Should gradually improve (Match F1 score)

## Important Considerations

### 1. Dataset Preparation

**Ensure your dataset has:**
- Valid PIL Images (RGB format)
- Non-empty questions and answers
- High-quality reference_steps (if using reasoning reward)
- Consistent formatting for multiple choice options

**Convert your dataset:**
```python
from datasets import Dataset, load_from_disk

# Example conversion
data = {
    'image': list_of_pil_images,
    'question': list_of_questions,
    'answer': list_of_answers,
    'reference_steps': list_of_reference_steps,
    'choices': list_of_choices,
    'options': list_of_options,
}

dataset = Dataset.from_dict(data)
dataset.save_to_disk("/path/to/save/dataset")
```

### 2. Memory Management

**For 7B models:**
- Single GPU (24GB): batch_size=2, gradient_accumulation=8
- Multi-GPU (4x24GB): batch_size=2 per GPU, gradient_accumulation=4
- With DeepSpeed ZeRO-3: batch_size=1, gradient_accumulation=16

**For larger models (13B+):**
- Use DeepSpeed ZeRO-3 or QLoRA
- Reduce max_pixels if OOM occurs

### 3. Training Speed

**Factors affecting speed:**
- Reasoning reward is slower (semantic similarity computation)
- Use `--reward_funcs "format" "accuracy"` for faster training
- Batch size and gradient accumulation affect throughput

**Optimization tips:**
- Use bf16 if supported (`--bf16`)
- Reduce max_pixels if not needed
- Use gradient checkpointing for larger models

### 4. Reward Balancing

If using multiple rewards, they are combined. Monitor each reward separately:
- Format should reach ~1.0 quickly (within first epoch)
- Accuracy should steadily improve
- Reasoning reward may be noisy; consider it a secondary signal

### 5. Prompt Engineering

The VQA prompt is fixed in the code. If you need modifications:
- Edit `qwen_module.py:72-107`
- Ensure `{USER_INSTRUCTION}` placeholder remains
- Maintain JSON schema in instructions

### 6. Evaluation

**During training:**
- Monitor reward trends in logs/tensorboard
- Check sample outputs in debug logs

**After training:**
- Use mllm_evaluator to evaluate on test set
- Compare Match F1 and accuracy metrics

## Troubleshooting

### Issue: "Unsupported file type"

**Solution**: Ensure you use `--use_huggingface_dataset` flag for HuggingFace datasets.

### Issue: Format reward always 0.0

**Cause**: Model not generating valid JSON.

**Solutions:**
- Check model is generating output (debug logs)
- Ensure sufficient training steps
- Try lower learning rate
- Increase format reward weight

### Issue: OOM (Out of Memory)

**Solutions:**
- Reduce batch size
- Increase gradient accumulation
- Reduce max_pixels
- Use DeepSpeed ZeRO-3
- Use QLoRA/LoRA

### Issue: Reasoning reward is 0.0

**Causes:**
- Dataset missing reference_steps
- Model not generating reasoning_steps
- Steps not semantically similar to references

**Solutions:**
- Verify dataset has reference_steps
- Check debug logs for model outputs
- Consider removing reasoning reward initially

### Issue: Training is very slow

**Solutions:**
- Remove reasoning reward (most expensive)
- Increase batch size if memory allows
- Use multiple GPUs
- Reduce dataset size for testing

## Example Training Script

Create `train_vqa.sh`:

```bash
#!/bin/bash

# Configuration
MODEL="Qwen/Qwen2-VL-7B-Instruct"
DATASET="/path/to/your/dataset"
OUTPUT="./output/qwen2vl-vqa-$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=4

# Enable debug logging
export DEBUG_MODE=true
export LOG_PATH="$OUTPUT/training_debug.txt"

# Create output directory
mkdir -p $OUTPUT

# Train with accelerate
accelerate launch --num_processes $NUM_GPUS \
    src/open-r1-multimodal/src/open_r1/grpo_rec.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --use_huggingface_dataset \
    --task_type "vqa" \
    --reward_funcs "format" "accuracy" "reasoning" \
    --output_dir $OUTPUT \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --warmup_steps 100 \
    --max_grad_norm 1.0 \
    --logging_dir $OUTPUT/logs \
    --report_to "tensorboard" \
    2>&1 | tee $OUTPUT/training.log

echo "Training completed! Output saved to: $OUTPUT"
```

Run with:
```bash
chmod +x train_vqa.sh
./train_vqa.sh
```

## References

- **mllm_evaluator**: See `mllm_evaluator/readme.md` for detailed evaluation metrics
- **Original VLM-R1**: Refer to repository README for base model training
- **TRL Documentation**: https://huggingface.co/docs/trl/

## Support

For issues or questions:
1. Check debug logs (`DEBUG_MODE=true`)
2. Verify dataset format matches requirements
3. Review error messages in training logs
4. Consult mllm_evaluator documentation for reward function details
