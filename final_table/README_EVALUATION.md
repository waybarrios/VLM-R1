# CRYSTAL Benchmark - Final Evaluation

## Status: 🟢 RUNNING

**Start Time**: November 12, 2025
**Estimated Duration**: 2-3 hours
**Output Directory**: `/gpudata3/Wayner/VLM-R1/final_table/`

---

## Models Being Evaluated (14 total)

### Qwen Family (5 models)
1. **Qwen2.5-VL-32B** - 32B parameters
2. **Qwen2.5-VL-7B** - 7B parameters
3. **Qwen2.5-VL-3B** - 3B parameters
4. **Qwen3-VL-8B** - 8B parameters
5. **Qwen3-VL-2B** - 2B parameters

### InternVL3.5 Family (5 models)
6. **InternVL3.5-38B** - 38B parameters (largest)
7. **InternVL3.5-8B** - 8B parameters
8. **InternVL3.5-4B** - 4B parameters
9. **InternVL3.5-2B** - 2B parameters
10. **InternVL3.5-1B** - 1B parameters (smallest)

### Gemma Family (2 models)
11. **Gemma3-12B** - 12B parameters
12. **Gemma3-4B** - 4B parameters

### Other Baselines (2 models)
13. **LLaVA-v1.6-7B** - 7B parameters
14. **MiniCPMv2.6-8B** - 8B parameters

---

## Evaluation Configuration

### Dataset
- **Name**: CRYSTAL
- **Path**: `/gpudata3/Wayner/reasoning/reasoning_test_with_reference_steps_updated_v27`
- **Samples**: 6,372 questions
- **Source benchmarks**: MathVision, ScienceQA, RealWorldQA, MMVP, PLOTQA

### Encoder Settings (from Ablation Study)
- **Encoder**: `all-distilroberta-v1` (best performer from paper Section 4.3)
- **Threshold**: `τ = 0.35` (optimal from 100 ablation experiments)
- **Device**: Multi-GPU (4× GPUs: 0, 1, 2, 3)

### Metrics Computed
For each sample:
1. **Accuracy** (final answer correctness)
2. **Match F1** (step-level reasoning quality)
3. **Precision** (fraction of predicted steps that match references)
4. **Recall** (fraction of reference steps covered by predictions)
5. **Avg Similarity** (semantic similarity between step pairs)
6. **Confidence** (match confidence score)
7. **Match Type** (how answer was matched: exact, numeric, choice, llm_verified)

---

## Output Structure

```
final_table/
├── evaluation_log.txt                    # Full execution log
├── README_EVALUATION.md                  # This file
│
├── outputs_testing_qwen25vl_32b_64k/     # Per-model results
│   ├── no_judge_metrics.csv              # Sample-level metrics
│   ├── metrics_summary.json              # Aggregated statistics
│   └── metrics_log.txt                   # Evaluation log
│
├── outputs_testing_qwen25vl_7b/
├── outputs_testing_qwen25vl_3b/
├── outputs_testing_qwen3vl_8b/
├── ... (one folder per model)
│
├── consolidated_results.csv              # Combined table (all models)
├── consolidated_results.tex              # LaTeX table for paper
├── family_summary.txt                    # Per-family statistics
└── results_analysis.txt                  # Analysis and insights
```

---

## How to Monitor Progress

### Option 1: Monitor Script (Recommended)
```bash
bash /gpudata3/Wayner/VLM-R1/monitor_evaluation.sh
```

Shows:
- Current progress (X/14 models)
- Currently processing model
- Success/failure counts
- Recently completed models

### Option 2: Watch Log File
```bash
tail -f /gpudata3/Wayner/VLM-R1/final_table/evaluation_log.txt
```

### Option 3: Check Process Status
```bash
ps aux | grep run_final_evaluation
```

---

## Expected Timeline

### Phase 1: Initial Setup (1-2 minutes)
- ✅ Load dataset from HuggingFace format
- ✅ Convert to internal format (6,372 samples)
- ✅ Load predictions from JSON files

### Phase 2: Per-Model Evaluation (~8-10 min each)
For each of 14 models:
1. Load 6,372 prediction JSON files (~1-2 min)
2. Initialize encoders on 4 GPUs (~30 sec)
3. Compute Match F1 for all samples (~5-6 min with 4 GPUs)
   - Sentence embedding generation
   - Similarity matrix computation
   - Optimal step matching
4. Compute accuracy metrics (~1-2 min)
   - Answer normalization
   - Multiple match strategies (numeric, choice, exact, semantic)
5. Save results (CSV + JSON + log) (~30 sec)

**Total per model**: ~8-10 minutes
**Total for 14 models**: ~2-2.5 hours

### Phase 3: Consolidation (5-10 minutes)
- Generate consolidated table (all models)
- Create LaTeX tables for paper
- Compute family-level statistics
- Generate analysis report

---

## Key Analysis Questions

The analysis script (`analyze_final_results.py`) will answer:

1. **Cherry-Picking Hypothesis**
   - Do all models exhibit precision >> recall?
   - How universal is the asymmetry?

2. **Scaling Trends**
   - Does performance scale with model size?
   - Which family scales best (InternVL vs. Qwen)?
   - Is there a sweet spot for parameter count?

3. **Top Performers**
   - Best overall Match F1
   - Best accuracy
   - Best recall (most comprehensive reasoning)
   - Most efficient (F1 per billion parameters)
   - Best small model (<10B)

4. **Step Generation Patterns**
   - What % of reference steps do models generate?
   - Validation of capacity cliff hypothesis
   - Most verbose vs. most concise models

5. **Consistency Analysis**
   - Which models have lowest variance (F1 std)?
   - Trade-offs between performance and consistency

---

## Paper Integration

### Section 4.2 - Main Results
- Update Table 1 with comprehensive results (14 models)
- Add within-family comparisons (InternVL: 1B → 38B)
- Emphasize precision-recall asymmetry across all models

### New Section 4.X - Scaling Analysis
- Create visualization: parameter count vs. Match F1
- Per-family scaling trends
- Efficiency analysis (F1 per billion params)

### Discussion
- Universal cherry-picking behavior
- Capacity cliff validation
- Implications for future model development

---

## Troubleshooting

### If Evaluation Fails
Check `/gpudata3/Wayner/VLM-R1/final_table/evaluation_log.txt` for errors.

Common issues:
- **CUDA OOM**: Reduce `--num-gpus` parameter
- **Missing predictions**: Check model output directories exist
- **JSON errors**: Validate prediction file format

### If Progress Stalls
```bash
# Check GPU utilization
nvidia-smi

# Check if processes are running
ps aux | grep compute_metrics

# Kill and restart if needed
pkill -f run_final_evaluation
bash run_final_evaluation.sh
```

---

## Contact

For questions or issues:
- Check log files first
- Review this README
- Consult METRICS_GUIDE.md for metric interpretations

---

**Last Updated**: November 12, 2025
**Status**: Evaluation in progress (Model 1/14)
