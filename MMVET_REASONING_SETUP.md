# MMVet Reasoning Setup Guide

## Overview

MMVet (Multi-Modal Veterinary Benchmark) is a challenging benchmark that evaluates vision-language models across 6 core capabilities: recognition (rec), OCR, knowledge (know), generation (gen), spatial reasoning (spat), and mathematics (math).

This guide describes the reasoning-compatible version that uses JSON format with `{"reasoning_steps": [], "answer": ""}`.

## Files Created

### 1. Task Configuration
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/mmvet/mmvet_reasoning.yaml`
- **Key settings**:
  - `task: "mmvet_reasoning"`
  - `max_new_tokens: 1024`
  - `temperature: 0.2` (same as original)
  - `do_sample: false`
  - Uses GPT-4 judge for evaluation (same as original)

### 2. Reasoning Utilities
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/mmvet/utils_reasoning.py`
- **Key functions**:
  - `get_reasoning_system_prompt()`: System prompt maintaining original MMVet philosophy
  - `extract_answer_from_json()`: Extracts answer from JSON response
  - `mmvet_process_results()`: Same GPT-4 evaluation as original
  - `mmvet_aggregate_results()`: Same capability-based aggregation

### 3. Module Registration
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/mmvet/__init__.py`
- Imports both `utils` and `utils_reasoning`

## System Prompt

The reasoning prompt maintains MMVet's original philosophy:

```
You are a vision-language model answering questions about images. Think step by step to provide the best answer.

Return ONLY a JSON object with this schema:
{"reasoning_steps": [], "answer": ""}

Rules for "reasoning_steps":
- Provide 2-5 clear reasoning steps explaining your thought process
- Each step should be concise (≤20 words) but informative
- Focus on relevant visual details, logical deductions, or domain knowledge
- Think step-by-step through the problem before reaching your conclusion

Rules for "answer":
- Provide a complete, accurate final answer to the question
- For open-ended questions: give a thorough explanation (1-3 sentences)
- For math/counting: provide the numerical answer with units if applicable
- For yes/no questions: answer clearly with brief justification
- Be precise and comprehensive - this answer will be evaluated for correctness
```

**Key differences from original MMVet**:
- Original prompt: `"First please perform reasoning, and think step by step to provide best answer to the following question: \n\n"`
- Reasoning version: Maintains the "think step by step" philosophy but adds JSON structure
- **Evaluation**: Uses the exact same GPT-4 judge with the same scoring rubric (0.0 to 1.0)

## Capabilities Evaluated

MMVet breaks down performance across:

### Core Capabilities (6):
1. **rec** (recognition): Visual object recognition
2. **ocr**: Text reading and understanding
3. **know** (knowledge): External knowledge requirements
4. **gen** (generation): Creative/generative tasks
5. **spat** (spatial): Spatial reasoning
6. **math**: Mathematical reasoning

### Detailed Capability Combinations (16):
- Single capabilities: rec, ocr, math
- Dual combinations: rec_gen, rec_know, rec_spat, rec_ocr, spat_ocr, math_ocr, spat_math_ocr, etc.
- Complex combinations: rec_spat_gen_ocr, rec_gen_ocr_know, etc.

## Installation

```bash
cd /gpudata3/Wayner/original/lmms-eval
pip install -e .
```

## Verification

```bash
# Check if task is registered
lmms-eval --tasks list | grep mmvet_reasoning

# Expected output:
# - mmvet_reasoning
```

## Running Evaluation

### Basic Command
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="/path/to/checkpoint" \
  --tasks mmvet_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-mmvet-reasoning
```

### With Environment Variables for GPT-4 Judge
```bash
export OPENAI_API_KEY="your-api-key"
export MODEL_VERSION="gpt-4o-2024-11-20"  # Default GPT-4 model
export API_TYPE="openai"  # or "azure"

# Then run the evaluation command
```

## Output Files

The evaluation generates:
1. **results.json**: Overall scores and per-capability breakdown
2. **samples_{task}.jsonl**: Individual predictions with:
   - `question_id`: Sample identifier
   - `question`: The question text
   - `gt_answer`: Ground truth answer
   - `pred_answer`: Model's extracted answer
   - `raw_response`: Full JSON response with reasoning steps
   - `score`: GPT-4 judge score (0.0 to 1.0)
   - `capabilities`: Capability tags for the question

## Evaluation Methodology

MMVet uses the same evaluation as the original:

1. **GPT-4 Judge**: Compares prediction to ground truth
2. **Scoring**: 0.0 (totally wrong) to 1.0 (totally right) in 0.1 increments
3. **Ground Truth Logic**:
   - `<AND>`: All elements must be present in prediction
   - `<OR>`: Any one element is sufficient
4. **Aggregation**: Average scores across all samples, then by capability

### Example Scoring:
```
Ground truth: -1 <AND> -5
Prediction: x = -1           → Score: 0.5 (only one element)
Prediction: x = -1 or x = -5 → Score: 1.0 (both elements)
```

## Expected Performance

**Original MMVet (no reasoning)**:
- Baseline models typically achieve 30-50%
- Strong models achieve 50-70%

**MMVet Reasoning (GRPO checkpoints)**:
- Should maintain or improve upon baseline performance
- GRPO trained on complex reasoning should excel on knowledge-heavy questions
- May struggle with simple visual recognition if overtrained on reasoning

## Comparison with Other Benchmarks

| Benchmark | Question Type | Evaluation | Best for GRPO? |
|-----------|--------------|------------|----------------|
| V*Bench | Open-ended, complex | Semantic similarity | ✅ Yes |
| MMStar | Multiple-choice | Exact match | ❌ No |
| K12 | Educational | GPT-4 judge | 🤔 Maybe |
| MMVet | Open-ended, diverse | GPT-4 judge | ✅ Yes |
| POPE | Binary yes/no | Exact match | ❌ No |

MMVet should be well-suited for GRPO checkpoints because:
1. Open-ended answers (not restricted to letters)
2. GPT-4 judge allows partial credit
3. Requires complex reasoning across multiple capabilities
4. Similar evaluation style to V*Bench (where GRPO excels)

## Troubleshooting

### Issue: Low scores on recognition tasks
- GRPO may be overtrained on reasoning
- Try checkpoint-1400 instead of 1500

### Issue: JSON parsing errors
- The extraction function has multiple fallback methods
- Check `raw_response` field in samples to debug

### Issue: GPT-4 API errors
- Verify `OPENAI_API_KEY` is set
- Check rate limits
- Script automatically retries with increased temperature

## Notes

- MMVet evaluates 218 samples (smaller than most benchmarks)
- Each sample requires GPT-4 API call (costs money!)
- Evaluation takes ~10-20 minutes depending on API latency
- The original MMVet prompt already encouraged reasoning, so the reasoning version should perform similarly or better
