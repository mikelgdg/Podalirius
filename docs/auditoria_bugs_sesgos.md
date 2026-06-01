# Auditoría Triage-MRI — Bugs, Sesgos y Vulnerabilidades

> **Documento generado por el agente Arquitecto de ML tras análisis paralelo de 5 agentes especializados.**
> **Fecha:** 2026-06-01
> **Agentes desplegados:** A1 (data), A2 (models), A3 (training), A4 (eval/scripts/demo), A5 (configs/tests)

---

## Resumen Ejecutivo

Se auditaron **21 archivos de código fuente** (1,840 líneas) más 3 YAMLs de configuración y 5 suites de tests.

| Categoría        | CRITICAL | HIGH | MEDIUM | LOW | Total |
|------------------|----------|------|--------|-----|-------|
| Bugs funcionales | 9        | 16   | 18     | 12  | 55    |
| Sesgos           | 0        | 6    | 4      | 1   | 11    |
| Edge cases       | 0        | 2    | 4      | 5   | 11    |
| Rendimiento      | 0        | 1    | 5      | 3   | 9     |
| Reproducibilidad | 0        | 1    | 2      | 2   | 5     |
| Tests (gaps/falsos) | 3     | 3    | 7      | 6   | 19    |
| Configs muertas  | 0        | 0    | 4      | 7   | 11    |
| **TOTAL**        | **12**   | **29** | **44** | **36** | **121** |

**Hallazgo más grave:** El sistema evalúa el threshold de triaje sobre el **test set** (data leakage) y el algoritmo `find_threshold_for_sensitivity` selecciona el threshold con **peor especificidad** posible, haciendo que virtualmente todos los casos se clasifiquen como "REVISAR" — el sistema de triaje no tria nada.

---

## Hallazgos CRÍTICOS (12)

### C1. `find_threshold_for_sensitivity` selecciona el peor threshold posible
**Archivo:** `src/triagemri/training/metrics.py:22`
**Severidad:** CRITICAL
**Tipo:** Bug lógico

```python
best_idx = np.argmax(recall >= target_sensitivity)  # ← PRIMER True → threshold más bajo
```

`precision_recall_curve` devuelve recall **decreciente** (de 1.0 a 0.0). `np.argmax` devuelve el índice del **primer** `True`, que corresponde al threshold más bajo (~0.05), clasificando casi todo como positivo. Resultado: sensibilidad ~100% pero especificidad ~0%. El sistema envía a REVISAR casi todos los casos — **no tria nada**.

**Fix:** `matches = np.where(recall >= target_sensitivity)[0]; best_idx = matches[-1]`

### C2. Threshold calibrado sobre el test set (data leakage)
**Archivo:** `src/triagemri/evaluation/evaluator.py:201-203`, `scripts/evaluate.py:71-75`  
**Severidad:** CRITICAL  
**Tipo:** Data leakage

`full_evaluation()` ejecuta inferencia sobre el test set y luego llama `compute_thresholds()` sobre `y_true, y_scores` **del mismo test set**. El threshold debe encontrarse en el **validation set** y solo **aplicarse** al test set. Esto infla artificialmente las métricas reportadas.

### C3. `pos_weight` recalculado por mini-batch en vez del dataset completo
**Archivo:** `src/triagemri/training/losses.py:31-34`  
**Severidad:** CRITICAL  
**Tipo:** Bug estadístico

```python
# Se ejecuta CADA batch, no una vez al inicio
num_pos = targets.sum().clamp(min=1)        # batch de 4, ~0 positivos → clamp=1
num_neg = (1.0 - targets).sum().clamp(min=1) # → pos_weight oscila 3.0-4.0 por batch
```

Con `batch_size=4` y 2% de positivos, la mayoría de batches tienen 0 positivos. `pos_weight` fluctúa salvajemente entre batches (3.0, 4.0, NaN-lógico). Debe calcularse **una vez** sobre el dataset de entrenamiento completo.

### C4. PI-CAI study ID extraction: todos los archivos silenciosamente ignorados
**Archivo:** `src/triagemri/data/datasets.py:629`  
**Severidad:** CRITICAL  
**Tipo:** Bug lógico

```python
if len(parts) >= 3:                # Convención {STUDY_ID}_{seq}.nii.gz → 2 parts
    study_id = parts[-2]           # Nunca se ejecuta para nombres válidos
```

La guarda `>=3` descarta todos los archivos que siguen la convención de nombres documentada (2 partes tras split por `_`). El dataset de próstata PI-CAI siempre está vacío.

### C5. NaN en volumen corrompe todo el output de normalización
**Archivo:** `src/triagemri/data/preprocessing.py:98-107`  
**Severidad:** CRITICAL  
**Tipo:** Bug de propagación de NaN

`np.quantile([NaN])` = NaN. Luego `NaN < 1e-8` = `False` (comparación NaN siempre falsa). El guarda de volumen constante se salta. `torch.clamp(volume, NaN, NaN)` + división por `(NaN - NaN)` produce tensor lleno de NaN que se inyecta al entrenamiento.

### C6. `RandFlipd([0,1,2])` — flip 3-ejes simultáneo, no independiente
**Archivo:** `src/triagemri/data/transforms.py:22`  
**Severidad:** CRITICAL  
**Tipo:** Bug de aumento de datos

MONAI `RandFlipd(spatial_axis=[0,1,2])` aplica **una** decisión binaria sobre la unión de ejes. Resultado: o no se flipea nada, o se flipean los 3 ejes a la vez (equivalente a rotación 180°). Nunca se aprende invarianza left-right independiente. El flip superior-inferior además no es clínicamente realista (el cerebro no es simétrico en Z).

### C7. Axial y Sagital intercambiados en la demo clínica
**Archivo:** `src/triagemri/demo/app.py:46-51`  
**Severidad:** CRITICAL  
**Tipo:** Bug de visualización clínica

```python
if axis == "axial":      img = v[:, :, w // 2]   # Plano ZY en mid-X = SAGITAL
elif axis == "sagittal": img = v[d // 2, :, :]   # Plano YX en mid-Z = AXIAL
```

Las etiquetas "Axial" y "Sagittal" están invertidas respecto al plano anatómico real. Un radiólogo vería erróneamente un plano sagital etiquetado como axial.

### C8. `get_attention_heatmap` y `predict()` rotos en modo multi_head
**Archivo:** `src/triagemri/models/triage.py:120-162`  
**Severidad:** CRITICAL  
**Tipo:** Bug de integración

Ambos métodos llaman `self.forward(x)` sin pasar el parámetro `anatomy`. En modo `multi_head`, `forward` lanza `ValueError("anatomy is required for multi_head mode")`. Cualquier modelo multi-anatomía no puede generar heatmaps ni predicciones individuales.

### C9. `load_full_volume_for_display` — intercambio de ejes espaciales
**Archivo:** `src/triagemri/data/preprocessing.py:218-237`  
**Severidad:** CRITICAL  
**Tipo:** Bug de coordenadas

La función `F.interpolate` trata los últimos 3 ejes como `(D, H, W)`, pero la entrada NIfTI mapea como `(W, H, D)`. El eje X y el eje Z se intercambian durante la interpolación, produciendo un volumen visualmente distorsionado para la demo.

### C10. Cache key no incluye `sources` — datos corruptos entre instancias
**Archivo:** `src/triagemri/data/datasets.py:113`  
**Severidad:** CRITICAL  
**Tipo:** Bug de caché

```python
cache_key = str(self.data_root.resolve())  # No incluye self.sources
```

Dos datasets con mismo `data_root` pero diferentes `sources` (ej. `["brats"]` vs `["brats","oasis"]`) comparten caché. El primer escaneo puebla el caché; el segundo reutiliza datos potencialmente incompletos.

### C11. Test: `freeze_backbone` vs `frozen` — falsos positivos en tests
**Archivo:** `tests/test_triage.py:232,258`, `configs/model.yaml:7`  
**Severidad:** CRITICAL  
**Tipo:** Falso positivo en tests

Los tests usan `"frozen": True` pero `build_triage_model` lee `freeze_backbone`. Como el default es `True`, los tests pasan siempre aunque la key esté mal. Poner `"frozen": False` en el test **seguiría pasando** porque se usa el default.

### C12. `run_evaluation` fuerza solo anatomía brain
**Archivo:** `src/triagemri/evaluation/evaluator.py:354`, `scripts/evaluate.py:64`  
**Severidad:** CRITICAL  
**Tipo:** Bug de pipeline

`create_dataloaders(config)` se llama sin `enabled_anatomies`, por defecto `["brain"]`. Un modelo entrenado en brain+prostate se evalúa solo en brain, produciendo métricas incompletas y engañosas.

---

## Hallazgos HIGH (29)

### Datos (7)

| ID | Archivo | Línea | Descripción |
|----|---------|-------|-------------|
| H1 | `datasets.py` | 189-196 | Missing seg → etiquetado normal (0). Debería ser -1 (desconocido). |
| H2 | `datasets.py` | 191/213 | `ValueError` de máscara corrupta no capturado → caso completo descartado sin aviso. |
| H3 | `datasets.py` | 925 | `brain_root = brats_path.parent` hace que `oasis_path` configurado sea ignorado al escanear. |
| H4 | `datasets.py` | 532-538 | Volumen corrupto → `torch.zeros` con label real. Modelo aprende `zeros → clase_del_origen`. |
| H5 | `labels.py` | 219-223 | `extract_label_from_picai`: cualquier valor != "YES" → 0 (normal). "UNKNOWN", "", None → normal. |
| H6 | `labels.py` | 84-103 | `extract_label_from_oasis`: NaN, vacío, columna ausente → 0 (normal). Solo CDR>0.5 → 1. |
| H7 | `labels.py` | 179 | `"d" in path.name` dispara OASIS en cualquier directorio con letra 'd'. Falso positivo masivo. |

### Modelos (6)

| ID | Archivo | Línea | Descripción |
|----|---------|-------|-------------|
| H8 | `triage.py` | 186-257 | `build_triage_model` sin `"config"` en checkpoint → encoder rebuilt con defaults aleatorios. |
| H9 | `encoder.py` | 118-122 | `bare except Exception` traga errores de disco/memoria → pesos aleatorios sin advertencia real. |
| H10 | `triage.py` | 238-254 | Checkpoint loading no maneja prefijo `module.` de DDP → pesos ignorados con `strict=False`. |
| H11 | `triage.py` | 209-221 | Multi-head: anatomías del checkpoint vs config divergen → heads no cargados correctamente. |
| H12 | `triage.py` | 160-161 | `get_attention_heatmap`: `reshape` asume N=cubo perfecto. N=28 → crash. |
| H13 | `encoder.py` | 98,195 | `stage_dims` hardcoded. Si MONAI cambia número de etapas → crash o dimensión incorrecta. |

### Entrenamiento (1)

| ID | Archivo | Línea | Descripción |
|----|---------|-------|-------------|
| H14 | `metrics.py` | 22-24 | Fallback cuando no se alcanza 99% sensibilidad: `recall[0] < 0.99` → `best_idx = len(recall)-1` → threshold=0.0 → sensibilidad reportada 0.0 vs real 1.0 (inconsistencia). |

### Evaluación / Demo / Scripts (9)

| ID | Archivo | Línea | Descripción |
|----|---------|-------|-------------|
| H15 | `evaluate.py` | 78-82 | `f"{results['global'].get('roc_auc', 'N/A'):.4f}"` → crash si métrica ausente. |
| H16 | `train.py` | 184 | `multiprocessing.set_start_method("spawn", force=True)` demasiado tarde (tras imports). |
| H17 | `train.py` | 188,253 | Resume con `--anatomies` incorrectas → entrena silenciosamente en subset. |
| H18 | `demo/app.py` | 130-133 | Grid no cúbico (ej. 4×4×2) → atención se desactiva sin feedback al usuario. |
| H19 | `demo/app.py` | 177-178 | Upload de archivo ignorado si hay un caso del registry seleccionado previamente. |
| H20 | `demo/app.py` | 347,531,574 | `Path(__file__).resolve().parent.parent.parent.parent` → roto en cualquier estructura de paquete distinta. |
| H21 | `triage.py` | 187-188 | `weights_only=True` rompe con checkpoints guardados como `torch.save(model, ...)`. |
| H22 | `demo/app.py` | 528-543 | `case_dir.glob()` sobre `None` por path condicional no cubierto → `AttributeError`. |
| H23 | `evaluator.py` | 266-267 | `json.dump(results, f, default=float)` → Paths, np.int64 convertidos a floats basura. |

### Tests (6)

| ID | Archivo | Línea | Descripción |
|----|---------|-------|-------------|
| H24 | `test_trainer.py` | — | Sin test de integración `pl.Trainer.fit()`. Solo tests unitarios de métodos. |
| H25 | `test_encoder.py` | 36-39 | `assert trainable > 0` → si 1/19.8M params son entrenables, pasa. Debería ser `== total`. |
| H26 | `test_preprocessing.py` | 121 | `pytest.raises(Exception)` captura cualquier excepción, incluso KeyboardInterrupt. |
| H27 | `test_mil.py` | 46-55 | `test_deterministic` no fija seed — pasa por coincidencia (modo eval sin dropout). |
| H28 | `test_triage.py` | 99-109, 138-142 | `get_attention_heatmap` y `predict` sin test en multi-head (crashearían). |
| H29 | `datasets.py` | 839-847 | `_make_loader` itera `ds[i]` para datasets sin `_cases` → carga TODOS los volúmenes en RAM. |

---

## Análisis de Sesgos (11 hallazgos)

### Sesgos de etiquetado (4)

| ID | Severidad | Descripción |
|----|-----------|-------------|
| **B1** | HIGH | Missing segmentation → etiquetado "normal". BraTS sin máscara = normal → contaminación de la clase negativa. |
| **B2** | HIGH | `extract_label_from_picai`: "NOT_APPLICABLE", "", None → label=0 (normal). Silencia casos ambiguos. |
| **B3** | HIGH | `extract_label_from_oasis`: NaN CDR, missing CDR → label=0. Pacientes sin dato clínico tratados como sanos. |
| **B4** | MEDIUM | `ABNORMAL_PATTERNS` tiene precedencia incondicional sobre `NORMAL_PATTERNS`. "no evidence of mass" → abnormal. |

### Sesgos de datos y splits (4)

| ID | Severidad | Descripción |
|----|-----------|-------------|
| **B5** | HIGH | Split labels por mayoría de paciente puede diferir de verdad por sesión. CDR que progresa → mayoría vota. |
| **B6** | MEDIUM | `_limit_brats` subsampling sin estratificar por label. Si hay normales en BraTS (bug B1), balance cambia. |
| **B7** | MEDIUM | Prostate split sin `stratify` → distribuciones de clase no garantizadas entre train/val/test. |
| **B8** | LOW | `split_ratio or` L462: si `test_ratio=0` o `val_ratio=0`, todo el holdout va a un solo split. |

### Sesgos de augmentación (2)

| ID | Severidad | Descripción |
|----|-----------|-------------|
| **B9** | HIGH | `RandFlipd([0,1,2])`: flip superior-inferior no es realista (anatomía cerebral no simétrica en Z). |
| **B10** | MEDIUM | `RandShiftIntensityd` + `RandScaleIntensityd` pueden sacar valores de [0,1] sin re-clamping. |

### Sesgo de métricas (1)

| ID | Severidad | Descripción |
|----|-----------|-------------|
| **B11** | MEDIUM | ECE con binning uniforme en datos desbalanceados → bins altos vacíos → ECE artificialmente bajo. |

---

## Reproducibilidad (5 hallazgos)

| ID | Severidad | Descripción |
|----|-----------|-------------|
| **R1** | HIGH | `pl.Trainer` sin `deterministic=True`. Operaciones CUDA no determinísticas permitidas. |
| **R2** | MEDIUM | `evaluate.py` y `demo.py` nunca fijan seed. Workers sin seed determinista. |
| **R3** | MEDIUM | Split con `torch.Generator()` vs `sklearn` — reproducibilidad cross-platform no garantizada. |
| **R4** | LOW | `WeightedRandomSampler(replacement=True)` — muestras minoritarias repetidas, mayoritarias pueden no verse. |
| **R5** | LOW | Caché en disco (`_brain_cache.json`) sin hash de código → cambios en lógica de escaneo usan caché stale. |

---

## Configs muertas o inconsistentes (11 hallazgos)

| ID | Archivo | Descripción |
|----|---------|-------------|
| D1 | `configs/data.yaml` | `voxel_spacing`, `intensity_percentiles`, `stratify_by`, `processed_dir` — definidos pero nunca leídos. |
| D2 | `configs/data.yaml` | `sequences` por dataset — código usa `_SEQUENCE_PATTERNS` hardcodeadas, no la config. |
| D3 | `configs/model.yaml` | `architecture: "swin_transformer_3d"` — nunca referenciado. |
| D4 | `configs/model.yaml` | `num_attention_branches: 1` — el código no implementa multi-branch. |
| D5 | `configs/train.yaml` | `gradient_accumulation_steps: 1` — funcionalmente desactivado, pero presente (confunde). |
| D6 | `configs/data.yaml` | `balance` solo para brain/BraTS. Próstata y mama sin controles de balance. |
| D7 | `configs/data.yaml` | `max_brats: 1400` comentario dice "50/50" pero no se verifica balance real. |
| D8 | `configs/train.yaml` | `save_top_k: -1` → guarda TODOS los checkpoints (~15-30 GB). |
| D9 | `config.py` | `dict.update()` merge superficial — colisiones de keys entre YAMLs destruyen datos sin aviso. |
| D10 | `config.py` | Sin validación de valores (rangos, keys requeridas). YAML malformado pasa silenciosamente. |
| D11 | `configs/train.yaml` | Sin toggle `deterministic` en config (solo `seed`). |

---

## Plan de Corrección — Delegación Modular por Agentes

El plan se estructura en **5 rondas secuenciales** (hay dependencias entre fixes de data → modelo → training → evaluación). Cada ronda despliega agentes en paralelo sobre archivos disjuntos.

```
RONDA 0: Pre-fixes (infraestructura)
  └── Agente 0: Configs, caché, seeds, validación
      (Archivos: config.py, configs/*.yaml)

RONDA 1: Datos (fundación, bloquea todo)
  ├── Agente 1: Labels + Normalización
  │   (Archivos: labels.py, preprocessing.py)
  └── Agente 2: Datasets + Transforms + Splits
      (Archivos: datasets.py, transforms.py)

RONDA 2: Modelos (depende de Ronda 1)
  └── Agente 3: Modelos completos (encoder, MIL, triage)
      (Archivos: encoder.py, mil.py, triage.py)

RONDA 3: Training (depende de Ronda 2)
  ├── Agente 4: Losses + Métricas
  │   (Archivos: losses.py, metrics.py)
  └── Agente 5: Trainer
      (Archivos: trainer.py)

RONDA 4: Eval + Demo + Scripts (depende de Ronda 3)
  ├── Agente 6: Evaluación
  │   (Archivos: evaluator.py, evaluate.py)
  └── Agente 7: Demo + Train script
      (Archivos: app.py, train.py, demo.py)
```

---

### RONDA 0 — Pre-fixes (1 agente, 30 min)

**Agente 0: Infraestructura**

| # | Fix | Severidad | Archivo | Descripción |
|---|-----|-----------|---------|-------------|
| F0.1 | CRITICAL | `config.py:33` | Deep merge en vez de shallow update. Detectar colisiones de keys. |
| F0.2 | HIGH | `config.py:48-53` | Añadir validación de keys requeridas y rangos de valores al construir Config. |
| F0.3 | HIGH | `config.py:62-64` | `Config.get()` con sentinel para distinguir `None` real de key ausente. |
| F0.4 | MEDIUM | `configs/train.yaml:31` | `save_top_k: 3` (no -1). Añadir `save_last: true`. |
| F0.5 | MEDIUM | `configs/train.yaml` | Añadir `deterministic: true` en config y propagar a `pl.Trainer`. |
| F0.6 | LOW | `configs/*.yaml` | Eliminar claves muertas o documentarlas como "planned". |

**Verificación R0:** `python -c "from triagemri.config import load_config; c = load_config(); print(c.to_dict())"` debe cargar sin errores y mostrar todas las keys esperadas.

---

### RONDA 1 — Datos (2 agentes en paralelo)

**Agente 1: Labels + Normalización**

| # | Fix | Severidad | Archivo | Descripción |
|---|-----|-----------|---------|-------------|
| F1.1 | CRITICAL | `preprocessing.py:98` | `np.quantile` → `np.nanquantile` o detección previa de NaN. Si hay NaN → excepción. |
| F1.2 | CRITICAL | `preprocessing.py:102` | Guarda de volumen constante con `np.isfinite` explícito antes de comparar. |
| F1.3 | CRITICAL | `preprocessing.py:218-237` | Corregir orden de ejes en `F.interpolate`. Documentar mapping NIfTI→tensor. |
| F1.4 | CRITICAL | `preprocessing.py:237` | `crop_offsets` devuelve `(d0, h0, w0)` como documentado, no `(w0, h0, d0)`. |
| F1.5 | HIGH | `labels.py:219-223` | PI-CAI: solo "YES"→1, "NO"→0, resto→-1 (desconocido). |
| F1.6 | HIGH | `labels.py:94-99` | OASIS: NaN/ausente→-1. Solo CDR>0→1, CDR==0→0 explícito. |
| F1.7 | HIGH | `labels.py:179` | `"d" in path.name` → detección por nombre de dataset explícito (ej. `"oasis" in path.name.lower()`). |
| F1.8 | MEDIUM | `labels.py:140-147` | Negación scoping en regex: "no evidence of X" detectado antes que "X". |
| F1.9 | LOW | `labels.py:28-35` | BraTS: añadir umbral mínimo de voxels (`np.sum(data>0) > MIN_VOXELS`). |

**Agente 2: Datasets + Transforms + Splits**

| # | Fix | Severidad | Archivo | Descripción |
|---|-----|-----------|---------|-------------|
| F2.1 | CRITICAL | `datasets.py:629` | PI-CAI: `len(parts) >= 3` → `len(parts) >= 2`. Corregir extracción para `.nii.gz`. |
| F2.2 | CRITICAL | `datasets.py:113` | Cache key incluye `self.sources` y `self.sequences`. Invalidar caché si lógica cambia. |
| F2.3 | CRITICAL | `datasets.py:189-196` | Missing seg → label=-1 y skip (no label=0). |
| F2.4 | HIGH | `datasets.py:191/213` | Capturar `ValueError` de label corrupto → label=-1, skip con log warning. |
| F2.5 | HIGH | `datasets.py:532-538` | Volumen corrupto → label=-1 y skip, NO zeros con label real. |
| F2.6 | HIGH | `datasets.py:925` | `_build_brain_datasets`: usar `oasis_path` configurado para el escaneo, no `brats_path.parent`. |
| F2.7 | HIGH | `datasets.py:694-711` | Prostate split: añadir `stratify` con sklearn como en Brain. |
| F2.8 | CRITICAL | `transforms.py:22` | `RandFlipd([0,1,2])` → 3 llamadas separadas para ejes individuales, o `RandAxisFlipd`. |
| F2.9 | MEDIUM | `transforms.py:42` | Shift/scale intensity sin re-clamp → añadir `ClipIntensityd` o `NormalizeIntensityd`. |
| F2.10 | MEDIUM | `transforms.py:45` | Eliminar `Resized` redundante tras `RandAffined` con `spatial_size`. |
| F2.11 | LOW | `datasets.py:427` | Tie-breaking en majority label → warning si empate, preferir `-1`. |

**Verificación R1:**
```bash
pytest tests/test_preprocessing.py tests/test_datasets.py -v
python -c "
from triagemri.data.datasets import BrainMRIDataset
ds = BrainMRIDataset('data/raw/brain', split='train', sources=['brats'])
print(f'Cases: {len(ds)}, Labels: {sum(1 for i in range(len(ds)) if ds[i][\"label\"]==1)} normal: {sum(1 for i in range(len(ds)) if ds[i][\"label\"]==0)}')
"
```

---

### RONDA 2 — Modelos (1 agente, secuencial)

**Agente 3: Encoder + MIL + TriageModel**

| # | Fix | Severidad | Archivo | Descripción |
|---|-----|-----------|---------|-------------|
| F3.1 | CRITICAL | `triage.py:120-162` | `get_attention_heatmap` y `predict`: aceptar y forwardear parámetro `anatomy`. |
| F3.2 | CRITICAL | `triage.py:186-257` | `build_triage_model`: inferir `embed_dim` del checkpoint si no hay `"config"`. |
| F3.3 | HIGH | `encoder.py:118-122` | `bare except Exception` → solo `(FileNotFoundError, RuntimeError, KeyError)`. Re-raise el resto. |
| F3.4 | HIGH | `triage.py:238-254` | Stripear prefijo `module.` además de `model.` al cargar checkpoint. |
| F3.5 | HIGH | `triage.py:209-221` | Inferir anatomías del checkpoint (`mil_heads.*` keys), no del config. |
| F3.6 | HIGH | `triage.py:160-161` | Validar `N == side**3` antes de reshape. Si no es cubo → warning, devolver grid 1D. |
| F3.7 | HIGH | `encoder.py:98,195` | Calcular `stage_dims` dinámicamente con un forward dummy, no hardcodear. |
| F3.8 | MEDIUM | `encoder.py:67-68,163-168` | `_freeze()` debe llamar `self.swin.eval()` además de `requires_grad_(False)`. |
| F3.9 | MEDIUM | `triage.py:91,102` | Multi-head: `KeyError` → mensaje con anatomías disponibles. |
| F3.10 | LOW | `encoder.py:117` | `weights_only=True` con try/except TypeError para compatibilidad PyTorch <2.0. |

**Verificación R2:**
```bash
pytest tests/test_encoder.py tests/test_mil.py tests/test_triage.py -v
```

---

### RONDA 3 — Training (2 agentes en paralelo)

**Agente 4: Losses + Métricas**

| # | Fix | Severidad | Archivo | Descripción |
|---|-----|-----------|---------|-------------|
| F4.1 | CRITICAL | `metrics.py:22` | `np.argmax` → `np.where(recall >= target_sensitivity)[0][-1]`. |
| F4.2 | CRITICAL | `losses.py:31-34` | `pos_weight` precomputado del dataset completo (inyectado en constructor). |
| F4.3 | HIGH | `metrics.py:22-24` | Fallback de sensibilidad: corregir inconsistencia entre reported y real. |
| F4.4 | MEDIUM | `metrics.py:85` | ECE con binning adaptativo (equal-mass) para datos desbalanceados. |
| F4.5 | MEDIUM | `losses.py:9` | `FocalLoss.alpha`: documentar que α=0.25 down-weight positives. Ofrecer α=0.75 para médicas. |
| F4.6 | MEDIUM | `losses.py:17` | `torch.exp(-bce_loss)` en fp32 incluso bajo AMP para evitar underflow. |
| F4.7 | MEDIUM | `metrics.py:44` | ROC-AUC: try/except para single-class, log warning, devolver NaN. |
| F4.8 | LOW | `losses.py:31-32` | Warning si `num_pos` se clampó (batch sin positivos). |

**Agente 5: Trainer**

| # | Fix | Severidad | Archivo | Descripción |
|---|-----|-----------|---------|-------------|
| F5.1 | HIGH | `trainer.py:103-104` | `self.all_gather()` para `val_preds`/`val_labels` en multi-GPU. |
| F5.2 | HIGH | `trainer.py:120-127` | `finally: self.val_preds.clear(); self.val_labels.clear()`. |
| F5.3 | HIGH | `train.py:184` | `multiprocessing.set_start_method("spawn")` al inicio de `main()` antes de imports. |
| F5.4 | MEDIUM | `trainer.py:166-169` | `T_max` corregido para anealing completo. Excluir biases/LayerNorm de weight decay. |
| F5.5 | MEDIUM | `train.py:239` | Añadir `deterministic=config.training.get("deterministic", False)` a `pl.Trainer`. |

**Verificación R3:**
```bash
pytest tests/test_trainer.py tests/test_metrics.py tests/test_losses.py -v
python -c "
from triagemri.training.metrics import find_threshold_for_sensitivity
import numpy as np
y_true = np.array([0,0,0,0,0,0,0,0,1,1])
y_scores = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
thr, sens, spec = find_threshold_for_sensitivity(y_true, y_scores, target_sensitivity=0.5)
assert sens >= 0.5 and spec > 0, f'Threshold {thr} gives sens={sens}, spec={spec}'
print('OK - threshold optimization correct')
"
```

---

### RONDA 4 — Eval + Demo + Scripts (2 agentes en paralelo)

**Agente 6: Evaluación**

| # | Fix | Severidad | Archivo | Descripción |
|---|-----|-----------|---------|-------------|
| F6.1 | CRITICAL | `evaluator.py:201-203` | `compute_thresholds` sobre validation set, no test set. Añadir `val_loader` a `full_evaluation`. |
| F6.2 | CRITICAL | `evaluator.py:354` | `run_evaluation`: pasar `enabled_anatomies` a `create_dataloaders`. |
| F6.3 | CRITICAL | `evaluate.py:64` | Añadir flag `--anatomies` y propagar a dataloaders. |
| F6.4 | HIGH | `evaluate.py:78-82` | Format-string con fallback a string, no a float format. |
| F6.5 | HIGH | `evaluator.py:266-267` | `json.dump` con serializador tipado, no `default=float`. |

**Agente 7: Demo + Train script**

| # | Fix | Severidad | Archivo | Descripción |
|---|-----|-----------|---------|-------------|
| F7.1 | CRITICAL | `demo/app.py:46-51` | Intercambiar implementaciones de "axial" y "sagittal". |
| F7.2 | HIGH | `demo/app.py:130-133` | Grid no cúbico: mensaje "Attention grid non-cubic — display simplified" en vez de silencio. |
| F7.3 | HIGH | `demo/app.py:177-178` | Upload no debe ser ignorado si hay registry_path. Limpiar registry_path al hacer upload. |
| F7.4 | HIGH | `demo/app.py:347,531,574` | Rutas relativas al project root con `Path(__file__).parents[3]` documentado, o variable de entorno. |
| F7.5 | HIGH | `train.py:188,253` | Al resumir, warning si `--anatomies` difiere del checkpoint. |
| F7.6 | MEDIUM | `demo/app.py:528-543` | Guarda `if case_dir is not None` antes de `case_dir.glob()`. |
| F7.7 | MEDIUM | `demo/app.py:181-182` | Evitar doble carga de NIfTI: cachear volumen preprocesado. |
| F7.8 | LOW | `train.py:231` | Añadir opción `--logger wandb` además de TensorBoard. |

**Verificación R4:**
```bash
pytest tests/ -v  # todos los tests
ruff check src/
python scripts/evaluate.py --checkpoint outputs/default/checkpoints/last.ckpt 2>&1 | head -20
```

---

## Estrategia de Verificación Global

Tras cada ronda, ejecutar:

```bash
# 1. Compilación del paquete
python -c "from triagemri.config import load_config; from triagemri.models.triage import build_triage_model; print('OK')"

# 2. Tests unitarios
pytest tests/ -v

# 3. Linting
ruff check src/

# 4. Smoke test de integración (forward pass)
python -c "
import torch
from triagemri.models.triage import build_triage_model
model = build_triage_model({'encoder': {'checkpoint_path': 'weights/triad_swinb_simmim.pth', 'freeze': True, 'embed_dim': 768, 'depths': [2,2,2,2]}})
out = model(torch.randn(1, 1, 96, 96, 96))
assert out['score'].shape == (1,), f'Bad score shape: {out[\"score\"].shape}'
assert out['attention'].shape[0] == 1, f'Bad attention shape: {out[\"attention\"].shape}'
print('Smoke test OK')
"

# 5. Validación de métricas
python -c "
import numpy as np
from triagemri.training.metrics import find_threshold_for_sensitivity, compute_triage_metrics
y_true = np.array([0]*80 + [1]*20)
np.random.seed(42)
y_scores = np.clip(y_true.astype(float) + np.random.randn(100)*0.3, 0, 1)
thr, sens, spec = find_threshold_for_sensitivity(y_true, y_scores, 0.8)
assert sens >= 0.8, f'Sensitivity {sens} below 0.8'
assert spec > 0.1, f'Specificity {spec} too low — argmax bug not fixed'
metrics = compute_triage_metrics(y_true, y_scores)
assert 'roc_auc' in metrics
print(f'Metrics OK — AUC={metrics[\"roc_auc\"]:.3f}, Thr={thr:.3f}, Sens={sens:.3f}, Spec={spec:.3f}')
"
```

---

## Riesgos y Contingencias

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Fix de PI-CAI requiere datos reales para testear | Alta | Medio | Test unitario con paths sintéticos. Validación final cuando haya datos. |
| Cambio de `pos_weight` a dataset-level puede requerir reentrenar | Alta | Alto | Documentar que modelos previos deben reentrenarse. El bug actual invalida cualquier entrenamiento existente. |
| Fix del threshold puede revelar que el modelo real tiene baja especificidad | Media | Alto | Es esperable. El modelo actual parecía bueno por el bug de argmax. Habrá que iterar. |
| Algunos fixes requieren GPU para verificación | Media | Medio | Priorizar tests unitarios CPU-first. Smoke tests con GPU cuando esté disponible. |
| Cambios en interfaces (nuevos parámetros) rompen tests existentes | Media | Medio | Actualizar INTERFACES.md en cada ronda. Tests se actualizan en la misma ronda que el fix. |

---

## Criterios de Éxito

- [ ] Los 12 hallazgos CRITICAL están corregidos y verificados
- [ ] Los 29 hallazgos HIGH están corregidos o documentados como "accepted risk"
- [ ] `pytest tests/ -v` pasa con 0 fallos
- [ ] `ruff check src/` pasa sin errores
- [ ] Smoke test de integración: forward pass con tensor sintético produce shapes correctos
- [ ] Test de métricas: `find_threshold_for_sensitivity` devuelve threshold con especificidad > 0
- [ ] Test de datos: PI-CAI dataset no está vacío con datos reales
- [ ] La demo muestra "Axial" y "Sagittal" correctamente etiquetados
- [ ] `docs/CHANGELOG.md` registra todos los fixes de la sesión
- [ ] `INTERFACES.md` actualizado si alguna firma cambió

---

*Documento generado como parte de la auditoría de Triage-MRI — 2026-06-01*
*Próximo paso: ejecutar Ronda 0 (Agente 0: Infraestructura)*
