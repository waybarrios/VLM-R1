# ChartQA con Formato de Reasoning Steps

## 📋 Resumen

He modificado el código de `lmms-eval` para que ChartQA use el **mismo formato JSON** que los checkpoints de GRPO:

```json
{
  "reasoning_steps": ["step 1", "step 2", ...],
  "answer": "42"
}
```

## 🔧 Archivos Creados/Modificados

### 1. **Nuevo Utils** (`utils_reasoning.py`)
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/chartqa/utils_reasoning.py
```

**Cambios principales:**
- ✅ Usa el **mismo system prompt** que `run_simple_vqa.py` (GRPO)
- ✅ Extrae solo el campo `"answer"` del JSON generado
- ✅ Mantiene la función `relaxed_correctness()` original de ChartQA
- ✅ Maneja errores de parsing JSON gracefully
- ✅ Mantiene las 3 métricas: overall, human_split, augmented_split

**Funciones clave:**
```python
get_reasoning_system_prompt()     # System prompt con formato JSON
extract_answer_from_json()        # Extrae "answer" del JSON
relaxed_correctness()             # Tolerancia ±5% para respuestas numéricas
chartqa_process_results()         # Procesa resultados y calcula métricas
chartqa_aggregate_results()       # Agrega scores (mean)
```

### 2. **Nuevas Configuraciones YAML**

#### Task principal:
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/chartqa/chartqa_reasoning.yaml
```

#### Task lite (subset):
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/chartqa/chartqa_lite_reasoning.yaml
```

**Cambios:**
- `max_new_tokens: 512` (aumentado para reasoning steps)
- `temperature: 0` (igual que original)
- `do_sample: False` (igual que original)
- Usa `utils_reasoning` en lugar de `utils`

### 3. **Creado `__init__.py`**
```python
from . import utils
from . import utils_reasoning  # ← Nuevo
```

## 🚀 Uso

### **Opción 1: Task Original (sin reasoning)**
```bash
--tasks chartqa
```
- Usa `max_new_tokens: 16`
- Extrae respuesta directamente

### **Opción 2: Task con Reasoning (NUEVO)**
```bash
--tasks chartqa_reasoning
```
- Usa `max_new_tokens: 512`
- Genera JSON con `reasoning_steps` y `answer`
- Extrae solo el campo `answer` para evaluación

### **Opción 3: Lite version (subset más pequeño)**
```bash
--tasks chartqa_lite_reasoning
```
- Subset reducido del dataset completo
- Útil para pruebas rápidas

## 📝 Ejemplo de Uso

### Con Modelo Base (Qwen2.5-VL-3B)
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="Qwen/Qwen2.5-VL-3B-Instruct" \
  --tasks chartqa_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-chartqa-reasoning
```

### Con Checkpoint GRPO
```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
accelerate launch \
  --num_processes 3 \
  --main_process_port 29600 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="/gpudata3/Wayner/VLM-R1/output/.../checkpoint-300" \
  --tasks chartqa_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-chartqa-grpo-ckpt300
```

### Test rápido con Lite version
```bash
--tasks chartqa_lite_reasoning \
--limit 50
```

## 🔍 Extracción del Answer

El código maneja **4 métodos** de extracción del campo `answer`:

### **Método 1: JSON válido**
```json
{"reasoning_steps": ["step1", "step2"], "answer": "42"}
```
→ Extrae: `"42"`

### **Método 2: JSON con code fences**
```markdown
```json
{"reasoning_steps": [...], "answer": "25%"}
```
```
→ Extrae: `"25%"`

### **Método 3: Regex fallback**
```
Some text... "answer": "3.5" ... more text
```
→ Extrae: `"3.5"`

### **Método 4: Fallback a respuesta completa**
Si todo falla, usa la respuesta completa.

## 📊 Evaluación con Relaxed Correctness

ChartQA usa **relaxed correctness** con tolerancia del ±5%:

### **Para respuestas numéricas:**
```python
# Predicción: "47.5"
# Ground truth: "50"
# Relative change: |47.5 - 50| / |50| = 0.05 (5%)
# Score: 1.0 ✅ (dentro del 5%)

# Predicción: "45"
# Ground truth: "50"
# Relative change: |45 - 50| / |50| = 0.10 (10%)
# Score: 0.0 ❌ (fuera del 5%)
```

### **Para respuestas con porcentajes:**
```python
# Predicción: "47.5%"
# Se convierte a: 0.475
# Luego se compara con tolerancia ±5%
```

### **Para respuestas no numéricas:**
```python
# Predicción: "Yes"
# Ground truth: "yes"
# Score: 1.0 ✅ (case-insensitive exact match)
```

## 📊 Métricas

ChartQA reporta 3 métricas:

1. **relaxed_overall**: Accuracy en todo el dataset
2. **relaxed_human_split**: Accuracy en preguntas generadas por humanos
3. **relaxed_augmented_split**: Accuracy en preguntas generadas automáticamente

El dataset se divide en:
- **Human test**: Preguntas complejas creadas por anotadores humanos
- **Augmented**: Preguntas generadas automáticamente (más simples)

## ⚙️ Compatibilidad

✅ **Compatible con:**
- Checkpoints GRPO entrenados con reasoning steps
- Modelo base Qwen2.5-VL (generará reasoning steps si el prompt lo pide)
- Cualquier modelo que pueda seguir el formato JSON

✅ **Ventajas vs. task original:**
- Genera reasoning steps (útil para debugging)
- Mismo formato que training de GRPO
- Más robusto ante respuestas largas
- Permite analizar calidad de reasoning en charts

## 🧪 Testing

Script de prueba rápido:
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="Qwen/Qwen2.5-VL-3B-Instruct" \
  --tasks chartqa_lite_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-chartqa-test \
  --limit 20
```

## ✅ Verificación

Para verificar que todo funciona:

```bash
# 1. Check tasks disponibles
lmms-eval --tasks list | grep chartqa

# Output esperado:
# chartqa
# chartqa_lite
# chartqa_lite_reasoning  ← Nuevo
# chartqa_reasoning  ← Nuevo

# 2. Ver métricas en results
cat logs-chartqa-test/*/results.json | python -m json.tool
```

## 🔗 Relación con GRPO Inference

| Aspecto | GRPO (`run_simple_vqa.py`) | ChartQA Reasoning |
|---------|----------------------------|-------------------|
| System Prompt | `get_system_prompt()` | `get_reasoning_system_prompt()` ✅ |
| Output Format | `{"reasoning_steps": [], "answer": ""}` | Mismo ✅ |
| JSON Parsing | `parse_and_validate_json()` | `extract_answer_from_json()` ✅ |
| Answer Extraction | Directo de `parsed["answer"]` | Extrae de JSON + fallbacks ✅ |
| Answer Matching | Exact match | Relaxed correctness (±5%) ✅ |
| Max Tokens | 512 | 512 ✅ |
| Temperature | 0 | 0 ✅ |
| Sampling | False | False ✅ |

## 📌 Notas Importantes

1. **Padding Side Warning**: Si ves warning sobre `padding_side`, usa `batch_size=1` o modifica el modelo para usar `padding_side='left'`

2. **System Prompt**: El system prompt está **hardcoded** en `utils_reasoning.py`. Si necesitas cambiarlo, edita la función `get_reasoning_system_prompt()`.

3. **Diferencia con task original**:
   - `chartqa`: Respuesta corta (max 16 tokens), extrae directamente
   - `chartqa_reasoning`: JSON completo (max 512 tokens), extrae `answer` field

4. **Performance**: Usar `batch_size > 1` puede causar problemas con padding. Usa `batch_size=1` para máxima confiabilidad.

5. **Dataset Split**:
   - `chartqa`: Full test set (~9,000 preguntas)
   - `chartqa_lite`: Subset reducido (~100-200 preguntas)

6. **Tolerancia Numérica**: La tolerancia del 5% puede hacer que respuestas cercanas sean correctas, lo cual es apropiado para datos extraídos automáticamente de charts.

## 🐛 Debugging

Si el modelo no genera JSON válido:
```python
# En utils_reasoning.py línea ~110
eval_logger.debug(f"Failed to extract answer from JSON, using raw response")
```

Ver logs con:
```bash
--verbosity DEBUG
```

## 📦 Archivos de Salida

```
logs-chartqa-reasoning/
├── results.json              # Resultados agregados
├── samples_chartqa_reasoning_{model}.jsonl  # Predicciones individuales
└── *_chartqa_reasoning.json                 # Samples con log_samples
```

## 📈 Ejemplo de Resultados Esperados

Para un checkpoint GRPO típico en ChartQA:

```
relaxed_overall:           ~60-70%
relaxed_human_split:       ~50-60%  (más difícil)
relaxed_augmented_split:   ~70-80%  (más fácil)
```

Las preguntas human-generated son típicamente más difíciles porque requieren:
- Razonamiento multi-hop
- Comparaciones complejas
- Cálculos matemáticos
- Comprensión de tendencias

Las preguntas augmented son típicamente más simples porque:
- Lectura directa de valores
- Preguntas de existencia
- Comparaciones simples

## 🎯 Siguiente Paso

Ejecuta el benchmark completo:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
accelerate launch \
  --num_processes 3 \
  --main_process_port 29600 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="<checkpoint_path>" \
  --tasks chartqa_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-chartqa-checkpoint
```

ChartQA es un benchmark importante para evaluar la capacidad de razonamiento visual y cuantitativo sobre gráficas y visualizaciones de datos!
