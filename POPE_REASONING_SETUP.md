# POPE con Formato de Reasoning Steps

## 📋 Resumen

He modificado el código de `lmms-eval` para que POPE use el **mismo formato JSON** que los checkpoints de GRPO:

```json
{
  "reasoning_steps": ["step 1", "step 2", ...],
  "answer": "yes"
}
```

## 🔧 Archivos Creados/Modificados

### 1. **Nuevo Utils** (`utils_reasoning.py`)
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/pope/utils_reasoning.py
```

**Cambios principales:**
- ✅ Usa el **mismo system prompt** que `run_simple_vqa.py` (GRPO)
- ✅ Extrae solo el campo `"answer"` del JSON generado
- ✅ Normaliza respuesta a "yes" o "no"
- ✅ Maneja errores de parsing JSON gracefully
- ✅ Mantiene todas las métricas originales (accuracy, precision, recall, F1, yes_ratio)

**Funciones clave:**
```python
get_reasoning_system_prompt()     # System prompt con formato JSON
extract_answer_from_json()        # Extrae "answer" del JSON
normalize_pope_answer()           # Normaliza a "yes"/"no"
pope_process_results()            # Procesa resultados y calcula métricas
pope_aggregate_accuracy()         # Accuracy
pope_aggregate_precision()        # Precision (TP / (TP + FP))
pope_aggregate_recall()           # Recall (TP / (TP + FN))
pope_aggregate_f1_score()         # F1 score
pope_aggregate_yes_ratio()        # Ratio de respuestas "yes"
```

### 2. **Nuevas Configuraciones YAML**

#### Task principal:
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/pope/pope_reasoning.yaml
```

#### Variants:
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/pope/pope_random_reasoning.yaml
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/pope/pope_pop_reasoning.yaml
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/pope/pope_adv_reasoning.yaml
```

**Cambios:**
- `max_new_tokens: 512` (aumentado para reasoning steps)
- Usa `utils_reasoning` en lugar de `utils`
- Mantiene las 5 métricas originales

### 3. **Creado `__init__.py`**
```python
from . import utils
from . import utils_reasoning  # ← Nuevo
```

## 🚀 Uso

### **Opción 1: Task Original (sin reasoning)**
```bash
--tasks pope_random
```
- Usa `max_new_tokens: 128`
- Extrae respuesta directamente

### **Opción 2: Task con Reasoning (NUEVO)**
```bash
--tasks pope_random_reasoning
```
- Usa `max_new_tokens: 512`
- Genera JSON con `reasoning_steps` y `answer`
- Extrae solo el campo `answer` para evaluación

## 📝 Ejemplo de Uso

### Con Modelo Base (Qwen2.5-VL-3B)
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="Qwen/Qwen2.5-VL-3B-Instruct" \
  --tasks pope_random_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-pope-reasoning
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
  --tasks pope_random_reasoning,pope_pop_reasoning,pope_adv_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-pope-grpo-ckpt300
```

### Correr todas las variantes a la vez
```bash
--tasks pope_random_reasoning,pope_pop_reasoning,pope_adv_reasoning
```

## 🔍 Extracción del Answer

El código maneja **4 métodos** de extracción del campo `answer`:

### **Método 1: JSON válido**
```json
{"reasoning_steps": ["step1", "step2"], "answer": "yes"}
```
→ Extrae: `"yes"`

### **Método 2: JSON con code fences**
```markdown
```json
{"reasoning_steps": [...], "answer": "no"}
```
```
→ Extrae: `"no"`

### **Método 3: Regex fallback**
```
Some text... "answer": "yes" ... more text
```
→ Extrae: `"yes"`

### **Método 4: Fallback a respuesta completa**
Si todo falla, usa la respuesta completa y la normaliza.

## 📊 Normalización de Respuesta

La función `normalize_pope_answer()` convierte cualquier variación a "yes" o "no":

**Yes variations:**
- "yes", "y", "Yes", "YES"
- "true", "correct", "present"

**No variations:**
- "no", "n", "No", "NO"
- "false", "incorrect", "absent", "not present"

**Unclear:**
- Si contiene ambos "yes" y "no" → None (score = 0)
- Si no se puede determinar → None (score = 0)

## 📊 Métricas

Las mismas métricas que el task original:

1. **pope_accuracy**: Overall accuracy (correct / total)
2. **pope_precision**: TP / (TP + FP)
3. **pope_recall**: TP / (TP + FN)
4. **pope_f1_score**: 2 × (precision × recall) / (precision + recall)
5. **pope_yes_ratio**: Ratio de ground truth "yes"

Donde:
- **TP** (True Positive): gt="yes" and pred="yes"
- **FP** (False Positive): gt="no" and pred="yes"
- **FN** (False Negative): gt="yes" and pred="no"
- **TN** (True Negative): gt="no" and pred="no"

## ⚙️ Compatibilidad

✅ **Compatible con:**
- Checkpoints GRPO entrenados con reasoning steps
- Modelo base Qwen2.5-VL (generará reasoning steps si el prompt lo pide)
- Cualquier modelo que pueda seguir el formato JSON

✅ **Ventajas vs. task original:**
- Genera reasoning steps (útil para debugging)
- Mismo formato que training de GRPO
- Más robusto ante respuestas largas
- Permite analizar calidad de reasoning

## ✅ Verificación

Para verificar que todo funciona:

```bash
# 1. Check tasks disponibles
lmms-eval --tasks list | grep pope

# Output esperado:
# pope
# pope_adv
# pope_adv_reasoning  ← Nuevo
# pope_full
# pope_pop
# pope_pop_reasoning  ← Nuevo
# pope_random
# pope_random_reasoning  ← Nuevo
# pope_reasoning  ← Nuevo

# 2. Test rápido (10 samples)
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="Qwen/Qwen2.5-VL-3B-Instruct" \
  --tasks pope_random_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-pope-test \
  --limit 10

# 3. Ver métricas
cat logs-pope-test/*/results.json | python -m json.tool | grep pope_
```

## 🔗 Relación con GRPO Inference

| Aspecto | GRPO (`run_simple_vqa.py`) | POPE Reasoning |
|---------|----------------------------|----------------|
| System Prompt | `get_system_prompt()` | `get_reasoning_system_prompt()` ✅ |
| Output Format | `{"reasoning_steps": [], "answer": ""}` | Mismo ✅ |
| JSON Parsing | `parse_and_validate_json()` | `extract_answer_from_json()` ✅ |
| Answer Extraction | Directo de `parsed["answer"]` | Extrae de JSON + fallbacks ✅ |
| Answer Normalization | N/A | `normalize_pope_answer()` (yes/no) ✅ |
| Max Tokens | 512 | 512 ✅ |
| Métricas | Accuracy, Match F1 | Accuracy, Precision, Recall, F1, Yes Ratio ✅ |

## 📌 Notas Importantes

1. **Padding Side Warning**: Si ves warning sobre `padding_side`, usa `batch_size=1` o modifica el modelo para usar `padding_side='left'`

2. **System Prompt**: El system prompt está **hardcoded** en `utils_reasoning.py`. Si necesitas cambiarlo, edita la función `get_reasoning_system_prompt()`.

3. **Diferencia con task original**:
   - `pope_random`: Respuesta corta, extrae yes/no directamente
   - `pope_random_reasoning`: Genera JSON completo, extrae `answer` field

4. **Performance**: Usar `batch_size > 1` puede causar problemas con padding. Usa `batch_size=1` para máxima confiabilidad.

5. **Variantes disponibles**:
   - **Random**: Objetos negativos seleccionados aleatoriamente
   - **Popular**: Objetos negativos son los más populares en COCO
   - **Adversarial**: Objetos negativos son similares a los positivos (más difícil)

## 🎯 Script de Cómputo de Métricas

Similar al de V* Bench, puedes computar métricas manualmente:

```python
python3 /gpudata3/Wayner/VLM-R1/compute_vstar_accuracy.py \
  <path_to_samples.jsonl>
```

(El mismo script funciona porque ambos usan el mismo formato de métricas en los samples)

## 📦 Archivos de Salida

```
logs-pope-reasoning/
├── results.json              # Resultados agregados
├── samples_pope_*_reasoning_{model}.jsonl  # Predicciones individuales
└── *_pope_*_reasoning.json                 # Samples con log_samples
```

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

## 📈 Ejemplo de Resultados Esperados

Para un checkpoint GRPO típico en POPE Random:

```
pope_accuracy:    ~85-90%
pope_precision:   ~90-95%  (alta precisión = pocos falsos positivos)
pope_recall:      ~80-85%  (recall moderado)
pope_f1_score:    ~85-90%
pope_yes_ratio:   50%       (dataset balanceado)
```

POPE Adversarial suele tener accuracy ~5-10% menor que Random.
