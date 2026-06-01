# Plan de Implementación: Sistema Generalista de Triaje por RM Volumétrica

**Objetivo:** Construir un sistema que demuestre capacidad generalista de triaje (descarte de normalidad) en RM 3D multi-anatomía, usando un encoder fundacional pre-entrenado + aprendizaje débilmente supervisado.

**Meta:** Demo funcional en **6-8 semanas** que clasifique normal vs anormal en ≥3 anatomías distintas (cerebro, próstata, mama), con sensibilidad ≥99%.

---

## Índice

1. [Arquitectura del sistema](#1-arquitectura-del-sistema)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Plan por fases](#3-plan-por-fases)
4. [Dependencias y riesgos](#4-dependencias-y-riesgos)
5. [Datasets](#5-datasets)
6. [Métricas de validación](#6-métricas-de-validación)

---

## 1. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────┐
│                    TRIAGE-VLM                            │
│                                                          │
│  Volumen RM 3D (NIfTI/DICOM)                             │
│         │                                                │
│         ▼                                                │
│  ┌──────────────────────┐                                │
│  │  Preprocesador 3D     │  resample (1mm iso)           │
│  │  (MONAI transforms)   │  normalize intensity          │
│  └────────┬─────────────┘  crop/pad → 96×96×96          │
│           │                                              │
│           ▼                                              │
│  ┌──────────────────────┐                                │
│  │  Encoder Triad        │  Swin-B SimMIM (congelado)    │
│  │  (3D Swin Transformer)│  48 → 768 dim                 │
│  │  131K vols pre-train  │  extrae features por sub-vol  │
│  └────────┬─────────────┘                                │
│           │                                              │
│           ▼                                              │
│  ┌──────────────────────┐                                │
│  │  Attention-MIL        │  Gated attention pooling      │
│  │  (instancias = sub-   │  aprende qué regiones         │
│  │   volúmenes 3D)       │  contribuyen a "anormal"      │
│  └────────┬─────────────┘                                │
│           │                                              │
│           ▼                                              │
│  ┌──────────────────────┐                                │
│  │  Score de Anomalía    │  P(anormal | volumen)         │
│  │  [0, 1]               │  threshold calibrado          │
│  └──────────────────────┘                                │
│                                                          │
│  Output: { score: 0.97, decision: "REVISAR",             │
│            confidence: 0.94, anatomy: "brain" }          │
└─────────────────────────────────────────────────────────┘
```

### Componentes clave:

| Componente | Tecnología | Estado |
|-----------|-----------|--------|
| Encoder 3D | Triad Swin-B SimMIM | Pesos descargables (Google Drive) |
| MIL Classifier | Gated Attention (CLAM-style) | A implementar |
| Preprocesado | MONAI transforms | Librería existente |
| Entrenamiento | PyTorch + PyTorch Lightning | Estándar |
| Evaluación | sklearn, MONAI metrics | Librerías existentes |
| Demo | Gradio / CLI | A implementar |

### Por qué Attention-MIL en vez de GAP (Global Average Pooling):

- GAP colapsa todo el volumen en 1 vector → pierde anomalías focales pequeñas
- MIL trata el volumen como una "bolsa" de sub-volúmenes 3D → el attention aprende a enfocarse en regiones sospechosas
- CLAM demostró que gated attention supera ampliamente a mean pooling en patología digital → mismo principio aplica a RM 3D

---

## 2. Estructura del proyecto

```
triage-mri/
├── pyproject.toml
├── README.md
├── configs/
│   ├── data.yaml              # Paths de datasets, secuencias
│   ├── model.yaml             # Config del encoder + MIL
│   └── train.yaml             # Hiperparámetros de entrenamiento
├── src/
│   └── triagemri/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py         # Dataset classes por anatomía
│       │   ├── transforms.py       # Augmentations 3D (MONAI)
│       │   ├── preprocessing.py    # DICOM/NIfTI → tensor normalizado
│       │   └── labels.py           # Extracción de etiquetas (regex + heurísticas)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── encoder.py          # Carga de Triad, extracción de features
│       │   ├── mil.py              # Attention-MIL (gated + standard)
│       │   ├── triage.py           # Modelo completo Triad + MIL
│       │   └── calibration.py      # Platt scaling, isotonic calibration
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py          # PyTorch Lightning module
│       │   ├── losses.py           # FocalLoss, WeightedBCE, etc.
│       │   └── metrics.py          # NPV, PPV, sensitivity@threshold
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluator.py        # Evaluación multi-anatomía
│       │   └── report.py           # Generación de informes/metrics
│       └── demo/
│           ├── __init__.py
│           └── app.py              # Gradio demo interactiva
├── scripts/
│   ├── download_triad.sh           # Descarga pesos de Triad (Google Drive)
│   ├── download_data.sh            # Descarga datasets (BraTS, PI-CAI, etc.)
│   ├── preprocess.py               # Preprocesado batch de volúmenes
│   ├── extract_features.py         # Precomputar features del encoder
│   ├── train.py                    # Script principal de entrenamiento
│   ├── evaluate.py                 # Evaluación sobre test sets
│   ├── demo.py                     # Lanzar demo
│   └── label_reports.py            # Extraer etiquetas de informes
├── tests/
│   ├── test_encoder.py
│   ├── test_mil.py
│   ├── test_datasets.py
│   └── test_triage.py
└── notebooks/
    ├── 01_preprocessing.ipynb      # Exploración de datos
    ├── 02_feature_extraction.ipynb # Visualización de embeddings
    ├── 03_training_analysis.ipynb  # Curvas de aprendizaje, métricas
    └── 04_evaluation.ipynb         # Análisis de resultados
```

---

## 3. Plan por fases

### Fase 0 — Setup e infraestructura [Semana 1]

**Objetivo:** Todo listo para empezar a entrenar.

| Día | Tarea | Entregable |
|-----|-------|-----------|
| 1 | Crear repo, `pyproject.toml`, entorno conda/pip con PyTorch, MONAI, Lightning | Entorno funcional |
| 1-2 | Descargar pesos Triad Swin-B SimMIM (`scripts/download_triad.sh`) | `weights/triad_swinb_simmim.pth` |
| 2-3 | Descargar BraTS 2023 + OASIS-3 (`scripts/download_data.sh`) | Datos en `data/raw/` |
| 3-4 | Implementar `src/triagemri/data/preprocessing.py`: NIfTI → tensor 96³, normalización intensidad, resample | Tests unitarios pasan |
| 4-5 | Implementar `src/triagemri/models/encoder.py`: carga de pesos Triad, `forward()` para extraer features por sub-volumen | Test: `tensor(1,1,96,96,96) → tensor(B, N, 768)` |
| 5 | Configurar `configs/data.yaml`, `configs/model.yaml`, `configs/train.yaml` | Configs validados |

**Dependencias críticas:**
- GPU con ≥24GB VRAM (A100 recomendada, A5000/RTX 3090 mínimo)
- Acceso a Synapse (BraTS) y NITRC (OASIS-3)

---

### Fase 1 — Prototipo mono-anatomía (cerebro) [Semanas 2-3]

**Objetivo:** Pipeline end-to-end funcionando con una anatomía. Demostrar que la arquitectura aprende.

| Semana | Tarea | Entregable |
|--------|-------|-----------|
| 2 | Implementar `models/mil.py`: Gated Attention MIL (inspirado en CLAM) | Tests unitarios + sanity check con datos sintéticos |
| 2 | Implementar `data/datasets.py`: `BrainMRIDataset` (BraTS + OASIS-3) con split train/val/test (70/15/15 estratificado) | Dataset cargable, inspeccionable |
| 2 | Implementar `data/transforms.py`: augmentaciones 3D (random flip, rotate, intensity jitter, elastic deform) | Visualmente verificadas |
| 2-3 | Implementar `training/trainer.py`: Lightning module con FocalLoss, WeightedRandomSampler, cosine LR | Entrenamiento lanza sin errores |
| 3 | **Entrenar modelo v0**: Triad congelado + MIL head, BraTS+OASIS | Primera curva ROC-AUC |
| 3 | Evaluar: AUC, sensibilidad, especificidad, NPV a 99% sens | Métricas baseline |
| 3 | Iterar hiperparámetros (LR, attention dim, número de sub-volúmenes por bolsa) | Mejora sobre baseline |

**Sub-volúmenes por bolsa (parámetro clave):**
- Volumen 96³ → K sub-volúmenes de 32³ o 48³ (con stride 16-24)
- Demasiados sub-vols = memoria explota
- Demasiado pocos = pérdida de resolución espacial
- Empezar con K=27 (3×3×3 grid de 48³ c/u, stride 24)

**Métricas objetivo Fase 1 (cerebro):**
- ROC-AUC ≥ 0.90
- Sensibilidad ≥ 99% con especificidad ≥ 30% (descarte del 30% de normales sin perder patología)
- NPV ≥ 0.995 al threshold operativo

---

### Fase 2 — Generalización multi-anatomía [Semanas 4-5]

**Objetivo:** Demostrar que el sistema funciona en ≥2 anatomías adicionales sin cambios en el encoder. Esto es lo que prueba el carácter "generalista".

| Semana | Tarea | Entregable |
|--------|-------|-----------|
| 4 | Descargar PI-CAI (próstata, 1,500 casos, Zenodo, inmediato) | `data/raw/prostate/` |
| 4 | Descargar Advanced-MRI-Breast-Lesions (mama, 200 casos, TCIA, inmediato) | `data/raw/breast/` |
| 4 | Implementar `ProstateMRIDataset` + `BreastMRIDataset` en `datasets.py`. Misma interfaz que `BrainMRIDataset` | Tests pasan |
| 4 | Preprocesar próstata y mama (resample, normalizar, crop/pad a 96³) | Volúmenes listos |
| 4 | **Entrenar modelo v1**: mismo encoder congelado, 3 MIL heads independientes (una por anatomía) o entrenamiento secuencial | Modelo multi-anatomía |
| 4-5 | Evaluar en próstata y mama con las mismas métricas que cerebro | Tabla comparativa |
| 5 | **Experimento clave de generalización**: entrenar solo en cerebro → evaluar zero-shot en próstata y mama | Métricas de transferencia |
| 5 | Fine-tuning rápido en próstata/mama si zero-shot es insuficiente | Métricas post-fine-tuning |

**Estrategias de arquitectura multi-anatomía (a decidir según resultados):**

| Estrategia | Ventaja | Desventaja |
|-----------|---------|-----------|
| **A) MIL heads independientes** | Máxima especialización | Más parámetros, no demuestra generalización |
| **B) Una MIL head compartida** | Demuestra generalización real | Puede que no converja bien |
| **C) Head compartida + anatomy token** | Balance: la head aprende "anomalía universal" modulada por anatomía | Más complejo |
| **D) Zero-shot (solo cerebro)** | Máxima demostración de generalización | Probablemente bajo rendimiento |

**Recomendación:** Empezar con **C** (anatomy token). Si no converge, caer a **A**.

**Métricas objetivo Fase 2:**
- Misma ROC-AUC ≥ 0.90 en ≥2 anatomías
- El encoder congelado extrae features útiles para anatomías NO vistas en fine-tuning (zero-shot AUC ≥ 0.70 sería un gran resultado)

---

### Fase 3 — Calibración, fiabilidad y demo [Semanas 6-7]

**Objetivo:** Sistema fiable, calibrado, con demo interactiva.

| Semana | Tarea | Entregable |
|--------|-------|-----------|
| 6 | Implementar `models/calibration.py`: Platt scaling + isotonic regression sobre validation set | Curvas de calibración (ECE < 0.05) |
| 6 | **Ajuste de threshold por anatomía**: buscar threshold que da sensibilidad = 99% en validation set de cada anatomía | Thresholds por anatomía |
| 6 | Evaluación final en test sets (NUNCA vistos): todas las métricas | Informe final de rendimiento |
| 6-7 | Implementar `demo/app.py`: Gradio con upload de NIfTI, visualización, score, decisión | Demo funcional |
| 7 | Análisis de errores: ¿qué casos fallan? ¿patrones? | Notebook de análisis |
| 7 | Documentar: README, comentarios en código, `configs/` explicados | Repo documentado |

**Demo (Gradio):**
```
┌──────────────────────────────────────┐
│  TRIAGE-MRI: Sistema de Triaje RM 3D │
│                                       │
│  [ Upload NIfTI (.nii.gz) ]           │
│                                       │
│  Anatomía: [brain ▼]                  │
│                                       │
│  ┌─ Resultado ──────────────────────┐ │
│  │                                   │ │
│  │  Score de anomalía: 0.97          │ │
│  │  Decisión: ⚠️ REVISAR             │ │
│  │  Confianza: 0.94                  │ │
│  │  Umbral de decisión: 0.42         │ │
│  │  Tiempo de inferencia: 0.8s       │ │
│  │                                   │ │
│  └───────────────────────────────────┘ │
└──────────────────────────────────────┘
```

---

### Fase 4 — Pulido y extras [Semana 8+]

**Solo si las fases anteriores son exitosas.**

| Tarea | Prioridad |
|-------|-----------|
| Añadir más anatomías (rodilla vía OAI si hay acceso) | Media |
| Precomputar features para entrenamiento más rápido | Media |
| Experimentar con fine-tuning parcial del encoder (últimas capas) | Media |
| Añadir LLM para explicabilidad (BioMistral) | Baja |
| Empaquetar como Docker container | Baja |
| Escribir paper / informe técnico | Media |

---

## 4. Dependencias y riesgos

### Dependencias técnicas

| Dependencia | Estado | Riesgo | Mitigación |
|------------|--------|-------|------------|
| Pesos Triad Swin-B SimMIM | ✅ Google Drive | Bajo (descargables) | Backup: 3D Neuro SimCLR o BrainMVP |
| GPU ≥24GB | ⚠️ Necesaria | Medio | Reducir batch size, usar gradient checkpointing |
| Acceso a BraTS (Synapse) | ✅ Inmediato | Bajo | Registro + descarga |
| Acceso a OASIS-3 (NITRC) | ⚠️ 1-3 días | Bajo | Empezar con solo BraTS si se retrasa |
| Acceso a PI-CAI (Zenodo) | ✅ Inmediato | Bajo | Descarga directa |
| Acceso a Breast-MRI (TCIA) | ✅ Inmediato | Bajo | NBIA Data Retriever |

### Riesgos técnicos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|------------|---------|------------|
| Triad no carga correctamente (formato pesos desconocido) | Media | Alto | Inspeccionar checkpoint, contactar autores, usar 3D Neuro SimCLR como plan B |
| MIL no converge con encoder congelado | Baja | Alto | Aumentar dim del MIL head, reducir LR, probar sin attention (mean pooling + MLP) |
| Overfitting por pocos datos de mama (200 casos) | Alta | Medio | Data augmentation agresiva, regularización fuerte, validación cruzada |
| El encoder no generaliza bien entre anatomías (zero-shot malo) | Media | Medio | Esperado — fine-tune rápido por anatomía es aceptable |
| Umbral del 99% sensibilidad requiere descartar muy pocos estudios | Media | Medio | Aceptable si descarta ≥15-20%; priorizar seguridad sobre eficiencia |

---

## 5. Datasets

### Resumen de datasets a usar

| Anatomía | Dataset | N casos | Etiqueta binaria | Acceso | Secuencias |
|----------|---------|---------|-----------------|--------|-----------|
| **Cerebro** | BraTS 2023 | ~4,500 | Tumor (sí/no, derivado de máscara) | Inmediato (Synapse) | T1, T1c, T2, FLAIR |
| **Cerebro** | OASIS-3 | 1,378 | CDR=0 (normal) vs CDR>0 (anormal) | 1-3 días (NITRC) | T1w, T2w, FLAIR |
| **Próstata** | PI-CAI | 1,500 | csPCa (ISUP≥2) sí/no | Inmediato (Zenodo) | T2W, DWI, ADC |
| **Mama** | Adv-MRI-Breast | 200 | Benigno vs Maligno | Inmediato (TCIA) | T1 DCE, T2w |

### Estrategia de etiquetado por dataset:

```python
# BraTS: derivar etiqueta de la presencia de máscara de segmentación
label = 1 if np.any(segmentation_mask > 0) else 0  # 1 = anormal

# OASIS-3: usar Clinical Dementia Rating
label = 0 if cdr == 0.0 else 1  # 0 = normal, 1 = anormal

# PI-CAI: usar case-level diagnosis
label = case_info["clinically_significant"]  # True/False → 1/0

# Breast-MRI: usar histopatología
label = 1 if pathology == "MALIGNANT" else 0
```

### Split estrategia:

```
Train (70%) → Validation (15%) → Test (15%)
Estratificado por label + anatomía (si es multi-anatomía)
Misma anatomía NUNCA comparte pacientes entre splits
```

---

## 6. Métricas de validación

### Métricas primarias (reportar para cada anatomía y global)

| Métrica | Fórmula | Objetivo |
|---------|---------|----------|
| **ROC-AUC** | sklearn `roc_auc_score` | ≥ 0.90 |
| **Sensibilidad @99%** | TP/(TP+FN) fijando threshold para sens=0.99 | ≥ 0.99 |
| **Especificidad @99% sens** | TN/(TN+FP) al mismo threshold | ≥ 0.20 |
| **NPV @99% sens** | TN/(TN+FN) | ≥ 0.995 |
| **PPV** | TP/(TP+FP) | Reportar (no hay objetivo fijo) |

### Métricas secundarias

| Métrica | Descripción |
|---------|------------|
| **PR-AUC** | Precision-Recall AUC (relevante para clases desbalanceadas) |
| **ECE** | Expected Calibration Error (calidad de calibración) |
| **Tiempo de inferencia** | Segundos por volumen en GPU |
| **Tasa de descarte** | % de estudios clasificados como normales al threshold operativo |

### Validación de generalización (clave para el objetivo)

| Experimento | Qué mide |
|------------|----------|
| **Within-anatomy** | Entrenar y evaluar en misma anatomía |
| **Cross-anatomy (zero-shot)** | Entrenar en cerebro → evaluar en próstata/mama |
| **Cross-anatomy (few-shot)** | Fine-tune con 5-10 casos de nueva anatomía |
| **Multi-anatomy** | Entrenar en todas → evaluar en todas |

El experimento **zero-shot cross-anatomy** es el que demuestra generalización real tipo OmniMRI.

---

## 7. Checklist de hitos

- [ ] **Fase 0 — Semana 1:** Entorno funcionando, Triad cargado, datos descargados
- [ ] **Fase 1 — Semana 3:** Modelo cerebro entrenado, AUC ≥ 0.90
- [ ] **Fase 2 — Semana 5:** ≥2 anatomías adicionales funcionando
- [ ] **Fase 3 — Semana 7:** Demo Gradio funcional, informe de evaluación
- [ ] **Fase 4 — Semana 8+:** Pulido, paper, extras

---

## 8. Riesgo de Triad — Plan B inmediato

Si los pesos de Triad no cargan correctamente o el formato es incompatible:

```
Triad falla → 3D Neuro SimCLR (emilykaczmarek/3D-Neuro-SimCLR)
               → True 3D ResNet, MIT license, pesos en GitHub Releases
               → Limitado a cerebro T1-w, pero funcional
               → Tiempo de reemplazo: 1-2 días

3D Neuro SimCLR no sirve → BrainMVP (OpenMEDLab)
               → Multi-paramétrico, CVPR 2025 Highlight
               → Tiempo de reemplazo: 2-3 días

BrainMVP no sirve → MRI-CORE (Duke)
               → 2D slice-wise, pero multi-anatomía
               → Tiempo de reemplazo: 3-4 días
```

**En el peor caso, el retraso máximo por cambio de encoder es 1 semana.** El resto de la arquitectura (MIL head, training pipeline, evaluación) es independiente del encoder.

---

*Plan generado Mayo 2026 — PRUEBA_MRI*
