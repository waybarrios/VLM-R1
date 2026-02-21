# MMStar con Formato de Reasoning Steps

## 📋 Resumen

He modificado el código de `lmms-eval` para que MMStar use el **mismo formato JSON** que los checkpoints de GRPO:

```json
{
  "reasoning_steps": ["step 1", "step 2", ...],
  "answer": "A"
}
```

## 🔧 Archivos Creados/Modificados

### 1. **Nuevo Utils** (`utils_reasoning.py`)
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/mmstar/utils_reasoning.py
```

**Cambios principales:**
- ✅ Usa el **mismo system prompt** que `run_simple_vqa.py` (GRPO)
- ✅ Extrae solo el campo `"answer"` del JSON generado
- ✅ Mantiene la función `exact_match()` original de MMStar
- ✅ Maneja errores de parsing JSON gracefully
- ✅ Mantiene todas las métricas por categoría (6 categorías + average)

**Funciones clave:**
```python
get_reasoning_system_prompt()     # System prompt con formato JSON
extract_answer_from_json()        # Extrae "answer" del JSON
extract_answer_letter()           # Extrae letra A/B/C/D de la respuesta
exact_match()                     # Matching original de MMStar
mmstar_process_results()          # Procesa resultados y calcula métricas
mmstar_aggregate_results()        # Agrega por L2 category
```

### 2. **Nueva Configuración YAML**
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/mmstar/mmstar_reasoning.yaml
```

**Cambios:**
- `max_new_tokens: 512` (aumentado para reasoning steps)
- `task: "mmstar_reasoning"` (nuevo nombre)
- Usa `utils_reasoning` en lugar de `utils`

### 3. **Creado `__init__.py`**
```python
from . import utils
from . import ko_utils
from . import utils_reasoning  # ← Nuevo
```

## 🚀 Uso

### **Opción 1: Task Original (sin reasoning)**
```bash
--tasks mmstar
```
- Usa `max_new_tokens: 128`
- Extrae letra directamente de la respuesta

### **Opción 2: Task con Reasoning (NUEVO)**
```bash
--tasks mmstar_reasoning
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
  --tasks mmstar_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-mmstar-reasoning
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
  --tasks mmstar_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-mmstar-grpo-ckpt300
```

## 🔍 Extracción del Answer

El código maneja **4 métodos** de extracción del campo `answer`:

### **Método 1: JSON válido**
```json
{"reasoning_steps": ["step1", "step2"], "answer": "B"}
```
→ Extrae: `"B"`

### **Método 2: JSON con code fences**
```markdown
```json
{"reasoning_steps": [...], "answer": "C"}
```
```
→ Extrae: `"C"`

### **Método 3: Regex fallback**
```
Some text... "answer": "D" ... more text
```
→ Extrae: `"D"`

### **Método 4: Fallback a respuesta completa**
Si todo falla, usa la respuesta completa y busca patrones de letra (A/B/C/D).

## 📊 Categorías de MMStar

MMStar evalúa 6 categorías principales con 18 subcategorías L2:

### **1. Coarse Perception**
- Image scene and topic
- Image style & quality
- Image emotion

### **2. Fine-grained Perception**
- Object counting
- Recognition
- Localization

### **3. Instance Reasoning**
- Single-instance reasoning
- Cross-instance attribute reasoning
- Cross-instance relation reasoning

### **4. Logical Reasoning**
- Code & sequence reasoning
- Diagram reasoning
- Common reasoning

### **5. Science & Technology**
- Biology & chemistry & physics
- Electronics & energy & mechanical eng.
- Geography & earth science & agriculture

### **6. Math**
- Geometry
- Numeric commonsense and calculation
- Statistical reasoning

## 📊 Métricas

Las métricas se reportan por categoría L1 (6 categorías) + average:

1. **coarse perception**: Accuracy en percepción gruesa
2. **fine-grained perception**: Accuracy en percepción fina
3. **instance reasoning**: Accuracy en razonamiento de instancias
4. **logical reasoning**: Accuracy en razonamiento lógico
5. **science & technology**: Accuracy en ciencia y tecnología
6. **math**: Accuracy en matemáticas
7. **average**: Promedio de todas las categorías L2

El score se calcula usando `exact_match()` de MMStar original:
- Matching directo: `"a" == "a"` → 1.0
- Matching con paréntesis: `"(a)" → "a"` → 1.0
- Matching con "option ": `"option a" → "a"` → 1.0
- Matching con "the answer is ": `"the answer is a" → "a"` → 1.0

## ⚙️ Compatibilidad

✅ **Compatible con:**
- Checkpoints GRPO entrenados con reasoning steps
- Modelo base Qwen2.5-VL (generará reasoning steps si el prompt lo pide)
- Cualquier modelo que pueda seguir el formato JSON

✅ **Ventajas vs. task original:**
- Genera reasoning steps (útil para debugging)
- Mismo formato que training de GRPO
- Más robusto ante respuestas largas
- Permite analizar calidad de reasoning por categoría

## 🧪 Testing

Script de prueba rápido:
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="Qwen/Qwen2.5-VL-3B-Instruct" \
  --tasks mmstar_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-mmstar-test \
  --limit 50
```

## ✅ Verificación

Para verificar que todo funciona:

```bash
# 1. Check task disponible
lmms-eval --tasks list | grep mmstar

# Output esperado:
# mmstar
# mmstar_ko
# mmstar_reasoning  ← Nuevo task

# 2. Ver métricas en results
cat logs-mmstar-test/*/results.json | python -m json.tool
```

## 🔗 Relación con GRPO Inference

| Aspecto | GRPO (`run_simple_vqa.py`) | MMStar Reasoning |
|---------|----------------------------|------------------|
| System Prompt | `get_system_prompt()` | `get_reasoning_system_prompt()` ✅ |
| Output Format | `{"reasoning_steps": [], "answer": ""}` | Mismo ✅ |
| JSON Parsing | `parse_and_validate_json()` | `extract_answer_from_json()` ✅ |
| Answer Extraction | Directo de `parsed["answer"]` | Extrae de JSON + fallbacks ✅ |
| Answer Matching | N/A | `exact_match()` de MMStar original ✅ |
| Max Tokens | 512 | 512 ✅ |

## 📌 Notas Importantes

1. **Padding Side Warning**: Si ves warning sobre `padding_side`, usa `batch_size=1` o modifica el modelo para usar `padding_side='left'`

2. **System Prompt**: El system prompt está **hardcoded** en `utils_reasoning.py`. Si necesitas cambiarlo, edita la función `get_reasoning_system_prompt()`.

3. **Diferencia con task original**:
   - `mmstar`: Respuesta corta, extrae letra directamente
   - `mmstar_reasoning`: Genera JSON completo, extrae `answer` field

4. **Performance**: Usar `batch_size > 1` puede causar problemas con padding. Usa `batch_size=1` para máxima confiabilidad.

5. **Dataset Split**: MMStar usa el split `val` (no `test`), con 1,500 preguntas en total.

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
logs-mmstar-reasoning/
├── results.json              # Resultados agregados por categoría
├── samples_mmstar_reasoning_{model}.jsonl  # Predicciones individuales
└── *_mmstar_reasoning.json                 # Samples con log_samples
```

## 📈 Ejemplo de Resultados Esperados

Para un checkpoint GRPO típico en MMStar (1,500 preguntas):

```
coarse perception:        ~55-60%
fine-grained perception:  ~50-55%
instance reasoning:       ~45-50%
logical reasoning:        ~40-45%
science & technology:     ~35-40%
math:                     ~30-35%
average:                  ~42-48%
```

Los resultados se agregan primero por L2 category (18 subcategorías), luego se promedian dentro de cada L1 category (6 categorías), y finalmente se calcula el average global.

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
  --tasks mmstar_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-mmstar-checkpoint
```

¡Los resultados te darán un desglose detallado del rendimiento en las 6 categorías principales de evaluación multimodal!
