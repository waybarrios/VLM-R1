# Reasoning Tasks Summary

Este documento resume todos los benchmarks que han sido convertidos al formato reasoning con JSON `{"reasoning_steps": [], "answer": ""}`.

## Tasks Completados

### 1. **V*Bench Reasoning** ✅
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/vstar_bench/`
- **Task name**: `vstar_bench_reasoning`
- **Prompt**: General reasoning para preguntas complejas visuales
- **Performance**: ✅ Excelente con GRPO

### 2. **POPE Reasoning** ✅
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/pope/`
- **Task names**:
  - `pope_reasoning` (full)
  - `pope_random_reasoning`
  - `pope_pop_reasoning`
  - `pope_adv_reasoning`
- **Prompt**: Binary yes/no con normalización
- **Performance**: ❌ Pobre con GRPO (demasiado simple)

### 3. **MMStar Reasoning** ✅
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/mmstar/`
- **Task name**: `mmstar_reasoning`
- **Prompt**: Multiple-choice optimizado, solo letra (A-D)
- **Performance**: ❌ 41% vs 55% baseline (GRPO underperforms)

### 4. **ChartQA Reasoning** ✅
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/chartqa/`
- **Task names**:
  - `chartqa_reasoning`
  - `chartqa_lite_reasoning`
- **Prompt**: Cuantitativo con ±5% tolerance
- **Performance**: 🤔 No evaluado aún

### 5. **K12 Reasoning** ✅
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/k12/`
- **Task name**: `k12_reasoning`
- **Prompt**: Educational con 3-8 pasos detallados
- **Performance**: 🤔 No evaluado aún

### 6. **MMBench EN Dev Reasoning** ✅
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/mmbench/`
- **Task name**: `mmbench_en_dev_reasoning`
- **Prompt**: Multiple-choice **MEJORADO** con ejemplos ✅/❌ claros
- **Extraction mejorada**: Extrae solo la letra (A-E)
- **Performance**: 71.33% vs 78.52% baseline (mejora con prompt nuevo pendiente)

### 7. **MMVet Reasoning** ✅
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/mmvet/`
- **Task name**: `mmvet_reasoning`
- **Prompt**: Open-ended, mantiene filosofía "think step by step"
- **Evaluation**: GPT-4 judge (mismo que original)
- **Performance**: 🤔 No evaluado aún

### 8. **Charades-STA Reasoning** ✅
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/charades_sta/`
- **Task name**: `temporal_grounding_charades_reasoning`
- **Prompt**: Temporal grounding enfocado en **acciones humanas, objetos y actividades diarias**
- **Answer format**: "start_time - end_time" (e.g., "24.3 - 30.4")
- **Performance**: 🤔 No evaluado aún

### 9. **MathVista Testmini Reasoning** ✅
- **Path**: `/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/mathvista/`
- **Task name**: `mathvista_testmini_reasoning`
- **Prompt**: Mathematical reasoning con diagramas, charts y figuras
- **Answer format**: Flexible (letra para MC, número para numeric, texto para open-ended)
- **Performance**: 🤔 No evaluado aún

## Formato JSON Estándar

Todos los tasks usan el mismo formato:

```json
{
  "reasoning_steps": [
    "Step 1: Observe the relevant visual details",
    "Step 2: Apply logic or calculation",
    "Step 3: Reach conclusion"
  ],
  "answer": "B"
}
```

### Variaciones por Task:

| Task | Answer Format | Example |
|------|--------------|---------|
| MMStar, MMBench | Solo letra | `"B"` |
| POPE | yes/no | `"yes"` |
| ChartQA | Número/texto | `"42"` or `"Red"` |
| K12 | Open-ended completo | `"The main character learns honesty"` |
| MMVet | Open-ended completo | `"The person is wearing a red hat"` |
| Charades-STA | Timestamp | `"24.3 - 30.4"` |
| MathVista | Flexible | `"B"` (MC), `"42"` (numeric), `"triangle"` (text) |

## Mejoras de Prompts

### MMBench Prompt Improvement (ÚLTIMA VERSIÓN)
El prompt de MMBench fue significativamente mejorado:

**Antes:**
- Prompt genérico
- Modelo generaba "B. Maryland" o "B: The car is red"

**Ahora:**
- ✅ Ejemplos CORRECTOS con checkmarks
- ❌ 6 ejemplos ESPECÍFICOS de errores comunes
- "READ CAREFULLY - THIS IS CRITICAL" para emphasis
- Recordatorio final: "Answer must be a single letter with no other characters"

**Resultado esperado:**
- Modelo debe generar solo `"B"` en lugar de `"B. Maryland"`
- Mejora esperada: de 71.33% a ~78% (baseline)

## Checkpoints Recomendados

Basado en análisis de CRYSTAL metrics:

| Checkpoint | Accuracy | Match F1 | Precision | Loss | Recomendación |
|------------|----------|----------|-----------|------|---------------|
| 1400 | **44.92%** | 0.4264 | **0.9831** | 0.0652 | ✅ **MEJOR OPCIÓN** |
| 1500 | 42.69% | 0.4305 | 0.9822 | 0.0668 | ⚠️ Usado actualmente |

**Usa checkpoint-1400** en lugar de 1500 para mejores resultados.

## Comandos de Evaluación

### MMBench (con prompt mejorado)
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1400" \
  --tasks mmbench_en_dev_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-mmbench-reasoning-improved-ckpt1400
```

### MMVet
```bash
export OPENAI_API_KEY="your-key"
export MODEL_VERSION="gpt-4o-2024-11-20"

CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1400" \
  --tasks mmvet_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-mmvet-reasoning-ckpt1400
```

### Charades-STA
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1400" \
  --tasks temporal_grounding_charades_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-charades-reasoning-ckpt1400
```

### MathVista Testmini
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="/gpudata3/Wayner/VLM-R1/output/qwen2.5-vl-3b-vqa-deepspeed-20251107_235133/checkpoint-1400" \
  --tasks mathvista_testmini_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-mathvista-reasoning-ckpt1400
```

## Análisis de Performance GRPO

### Tasks donde GRPO funciona bien ✅:
- **V*Bench**: Complex visual reasoning
- **K12** (probablemente): Educational, similar a CRYSTAL
- **MMVet** (probablemente): Open-ended, GPT-4 judge
- **MathVista** (probablemente): Mathematical reasoning

### Tasks donde GRPO underperforms ❌:
- **MMStar**: 41% vs 55% baseline (-14pp)
- **POPE**: Reported as poor
- **MMBench**: 71% vs 78% baseline (-7pp)

**Hipótesis**: GRPO fue entrenado en reasoning complejo (CRYSTAL), entonces:
- ✅ Excels en tareas complejas que requieren razonamiento multi-step
- ❌ Underperforms en tareas simples (binary, multiple-choice simple)

## Próximos Pasos

1. **Re-evaluar MMBench** con el prompt mejorado usando checkpoint-1400
2. **Evaluar MMVet, Charades-STA, MathVista** con checkpoint-1400
3. **Comparar performance** entre checkpoint-1400 y checkpoint-1500
4. **Analizar resultados** para paper

## Archivos de Documentación

- `VSTAR_REASONING_SETUP.md` - V*Bench setup
- `POPE_REASONING_SETUP.md` - POPE setup
- `MMSTAR_REASONING_SETUP.md` - MMStar setup
- `CHARTQA_REASONING_SETUP.md` - ChartQA setup
- `MMVET_REASONING_SETUP.md` - MMVet setup
- `recompute_mmbench_metrics.py` - Script para analizar MMBench
- `test_prompt_simple.py` - Script para verificar prompts

## Contacto

Para problemas o preguntas, revisar los archivos en `/gpudata3/Wayner/VLM-R1/`.
