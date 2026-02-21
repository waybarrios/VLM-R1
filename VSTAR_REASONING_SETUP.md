# V* Bench con Formato de Reasoning Steps

## 📋 Resumen

He modificado el código de `lmms-eval` para que V* Bench use el **mismo formato JSON** que los checkpoints de GRPO:

```json
{
  "reasoning_steps": ["step 1", "step 2", ...],
  "answer": "A"
}
```

## 🔧 Archivos Creados/Modificados

### 1. **Nuevo Utils** (`utils_reasoning.py`)
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/vstar_bench/utils_reasoning.py
```

**Cambios principales:**
- ✅ Usa el **mismo system prompt** que `run_simple_vqa.py` (GRPO)
- ✅ Extrae solo el campo `"answer"` del JSON generado
- ✅ Maneja errores de parsing JSON gracefully
- ✅ Mantiene compatibilidad con formato de respuesta original

**Funciones clave:**
```python
get_reasoning_system_prompt()     # System prompt con formato JSON
extract_answer_from_json()        # Extrae "answer" del JSON
extract_answer_letter()           # Extrae letra A/B/C/D de la respuesta
vstar_process_results()           # Procesa resultados y calcula accuracy
```

### 2. **Nueva Configuración YAML**
```
/gpudata3/Wayner/original/lmms-eval/lmms_eval/tasks/vstar_bench/vstar_bench_reasoning.yaml
```

**Cambios:**
- `max_new_tokens: 512` (aumentado para reasoning steps)
- `task: "vstar_bench_reasoning"` (nuevo nombre)
- Usa `utils_reasoning` en lugar de `utils`

### 3. **Actualizado `__init__.py`**
```python
from . import utils
from . import utils_reasoning  # ← Nuevo
```

## 🚀 Uso

### **Opción 1: Task Original (sin reasoning)**
```bash
--tasks vstar_bench
```
- Usa `max_new_tokens: 16`
- Extrae letra directamente de la respuesta

### **Opción 2: Task con Reasoning (NUEVO)**
```bash
--tasks vstar_bench_reasoning
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
  --tasks vstar_bench_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-vstar-reasoning
```

### Con Checkpoint GRPO
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch \
  --num_processes 1 \
  --module lmms_eval -- \
  --model qwen2_5_vl \
  --model_args pretrained="/gpudata3/Wayner/VLM-R1/output/.../checkpoint-300" \
  --tasks vstar_bench_reasoning \
  --batch_size 1 \
  --log_samples \
  --output_path logs-vstar-grpo-ckpt300
```

## 🔍 Extracción del Answer

El código maneja **3 métodos** de extracción del campo `answer`:

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

## 📊 Métricas

Las mismas métricas que el task original:
- `vstar_overall_acc`: Accuracy global
- `vstar_{category}_acc`: Accuracy por categoría
  - `direct_attributes`
  - `relative_position`

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

## 🧪 Testing

Script de prueba incluido:
```bash
bash /gpudata3/Wayner/VLM-R1/test_vstar_reasoning.sh
```

Esto ejecuta:
1. Verifica que el task está registrado
2. Corre 10 samples de prueba
3. Guarda resultados en `logs-vstar-reasoning-test/`

## 🐛 Debugging

Si el modelo no genera JSON válido:
```python
# En utils_reasoning.py línea ~187
eval_logger.debug(f"Failed to extract answer from JSON, using raw response")
```

Ver logs con:
```bash
--verbosity DEBUG
```

## 📦 Archivos de Salida

```
logs-vstar-reasoning/
├── results.json              # Resultados agregados
├── samples_vstar_bench_reasoning_{model}.jsonl  # Predicciones individuales
└── *_vstar_reasoning_test.json                  # Samples con log_samples
```

## ✅ Verificación

Para verificar que todo funciona:

```bash
# 1. Check task disponible
lmms-eval --tasks list | grep vstar

# Output esperado:
# vstar_bench
# vstar_bench_reasoning  ← Nuevo task
# vstar_bench_direct_attributes
# vstar_bench_relative_position

# 2. Test rápido (10 samples)
bash test_vstar_reasoning.sh

# 3. Ver accuracy
cat logs-vstar-reasoning-test/results.json | grep -i acc
```

## 🔗 Relación con GRPO Inference

| Aspecto | GRPO (`run_simple_vqa.py`) | V* Bench Reasoning |
|---------|----------------------------|-------------------|
| System Prompt | `get_system_prompt()` | `get_reasoning_system_prompt()` ✅ |
| Output Format | `{"reasoning_steps": [], "answer": ""}` | Mismo ✅ |
| JSON Parsing | `parse_and_validate_json()` | `extract_answer_from_json()` ✅ |
| Answer Extraction | Directo de `parsed["answer"]` | Extrae de JSON + fallbacks ✅ |
| Max Tokens | 512 | 512 ✅ |

## 📌 Notas Importantes

1. **Padding Side Warning**: Si ves warning sobre `padding_side`, usa `batch_size=1` o modifica el modelo para usar `padding_side='left'`

2. **System Prompt**: El system prompt está **hardcoded** en `utils_reasoning.py`. Si necesitas cambiarlo, edita la función `get_reasoning_system_prompt()`.

3. **Diferencia con task original**:
   - `vstar_bench`: Respuesta corta, extrae letra directamente
   - `vstar_bench_reasoning`: Genera JSON completo, extrae `answer` field

4. **Performance**: Usar `batch_size > 1` puede causar problemas con padding. Usa `batch_size=1` para máxima confiabilidad.

## 🎯 Siguiente Paso

Ejecuta el test:
```bash
cd /gpudata3/Wayner/VLM-R1
chmod +x test_vstar_reasoning.sh
./test_vstar_reasoning.sh
```

Si funciona correctamente, ejecuta el benchmark completo sin `--limit`.
