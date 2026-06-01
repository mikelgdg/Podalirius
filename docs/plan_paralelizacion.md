# Plan de Paralelización — Delegación con Agentes

**Objetivo:** Ejecutar el plan de implementación con máxima concurrencia, evitando conflictos de archivos y minimizando bloqueos por dependencias.

---

## 1. Grafo de dependencias

```
─── Wave 0: INTERFACES (bloquea todo, hacer primero) ───
│
├── Configs (data.yaml, model.yaml, train.yaml)
└── Esquema de interfaces (encoder output shape, MIL I/O, Dataset API)
        │
        ▼
─── Wave 1: FUNDACIONES (todo en paralelo, 5 streams) ───
│
├── Stream A: Encoder
│   ├── download_triad.sh          (descarga pesos)
│   └── encoder.py                 (carga Triad, extrae features)
│
├── Stream B: Preprocesado + Datos
│   ├── download_data.sh           (BraTS + OASIS)
│   └── preprocessing.py           (NIfTI → tensor 96³)
│
├── Stream C: Aumentaciones
│   └── transforms.py              (MONAI 3D augmentations)
│
├── Stream D: Métricas + Pérdidas
│   ├── metrics.py                 (NPV, sens, spec, calibración)
│   └── losses.py                  (FocalLoss, WeightedBCE)
│
└── Stream E: Scaffolding
    ├── pyproject.toml             (dependencias)
    └── __init__.py                (estructura de paquetes)
        │
        ▼
─── Wave 2: INTEGRACIÓN (3 streams, dependen de Wave 1) ───
│
├── Stream F: MIL Head ─────────── depende de A (encoder iface) + D (losses)
│   └── mil.py
│
├── Stream G: Dataset ──────────── depende de B (preproc) + C (transforms) + B (data en disco)
│   ├── datasets.py               (BrainMRIDataset)
│   └── labels.py                 (extracción etiquetas)
│
└── Stream H: Modelo completo ──── depende de A + F
    └── triage.py                  (Triad + MIL = TriageModel)
        │
        ▼
─── Wave 3: ENTRENAMIENTO (1 stream, depende de Wave 2) ───
│
└── Stream I: Trainer ──────────── depende de F (MIL) + G (Dataset) + D (metrics/losses)
    └── trainer.py                (Lightning module)
        │
        ▼
─── Wave 4: SCRIPTS + EVALUACIÓN (todo en paralelo) ───
│
├── Stream J: Scripts
│   ├── train.py
│   └── evaluate.py
│
├── Stream K: Demo
│   └── app.py                    (Gradio)
│
└── Stream L: Notebooks
    ├── 01_preprocessing.ipynb
    └── 02_feature_extraction.ipynb
```

---

## 2. Delegación óptima por rondas

### Ronda 0 — INTERFACES (1 agente, 30 min, BLOQUEANTE)

> **Agente 0:** Escribir `configs/data.yaml`, `configs/model.yaml`, `configs/train.yaml` + documentar interfaces en un `INTERFACES.md` que defina:
> - Forma de los tensores en cada etapa: `(B,1,96,96,96) → (B, N_subvols, 768) → (B,1)`
> - API de `BaseMRIDataset`: `__getitem__ → (volume, label, anatomy, metadata)`
> - API de `TriageModel.forward()`: `volume → {"score", "features", "attention_weights"}`
> - Firma de `TriageTrainer`: Lightning module con `training_step`, `validation_step`

**Por qué esto primero:** Si 5 agentes escriben código sin acuerdo en interfaces, las integraciones serán un infierno. 30 minutos aquí ahorran horas después.

### Ronda 1 — FUNDACIONES (4 agentes en paralelo, Día 1)

| Agente | Stream | Archivos | Dependencias | Tiempo |
|--------|--------|----------|-------------|--------|
| **A1** | Encoder | `download_triad.sh`, `models/encoder.py` | Solo las interfaces de Ronda 0 | 2-3h |
| **A2** | Datos + Prepro | `download_data.sh`, `data/preprocessing.py`, `data/labels.py` | Solo las interfaces de Ronda 0 | 3-4h |
| **A3** | Transforms + Métricas + Pérdidas | `data/transforms.py`, `training/metrics.py`, `training/losses.py` | Solo las interfaces de Ronda 0 | 2-3h |
| **A4** | Scaffolding | `pyproject.toml`, todos los `__init__.py`, `configs/*.yaml` (si A0 no los hizo), tests esqueleto | Nada | 1-2h |

**Los 4 agentes tocan archivos DISJUNTOS.** Zero riesgo de conflictos de merge.

**NOTA:** `download_data.sh` y `download_triad.sh` se ejecutan en background mientras los agentes escriben código. Si la descarga falla, los tests lo detectarán.

### Ronda 2 — INTEGRACIÓN (3 agentes en paralelo, Día 2)

| Agente | Stream | Archivos | Dependencias | Tiempo |
|--------|--------|----------|-------------|--------|
| **A5** | MIL Head | `models/mil.py` | `encoder.py` (solo interfaz), `losses.py` | 3-5h |
| **A6** | Datasets | `data/datasets.py` | `preprocessing.py`, `transforms.py`, datos en disco | 3-5h |
| **A7** | Triage completo + Tests | `models/triage.py`, `tests/test_mil.py`, `tests/test_encoder.py` | `encoder.py`, `mil.py` | 2-4h |

**A5 y A6 son independientes entre sí.** A7 depende de ambos pero puede empezar con mocks y tests unitarios mientras A5 y A6 terminan.

### Ronda 3 — ENTRENAMIENTO (2 agentes en paralelo, Día 3)

| Agente | Stream | Archivos | Dependencias | Tiempo |
|--------|--------|----------|-------------|--------|
| **A8** | Trainer | `training/trainer.py`, `scripts/train.py` | TODO lo de Ronda 2 | 3-4h |
| **A9** | Evaluación + Demo | `evaluation/evaluator.py`, `evaluation/report.py`, `demo/app.py` | `triage.py` (mock para demo), `metrics.py` | 3-4h |

**A8 lanza el primer entrenamiento.** A9 construye la demo en paralelo con el modelo mockeado (luego se enchufa el real).

### Ronda 4 — ITERACIÓN Y PULIDO (Días 4-7)

Ya sin agentes — trabajo manual de ML:
- Lanzar entrenamiento, mirar curvas, ajustar hiperparámetros
- Evaluar en validation set
- Iterar (esto NO es paralelizable, es inherentemente secuencial)

### Ronda 5 — MULTI-ANATOMÍA (2-3 agentes en paralelo, Día ~12)

| Agente | Stream | Archivos | Tiempo |
|--------|--------|----------|--------|
| **A10** | Próstata | `download_data.sh` (PI-CAI), extender `datasets.py` con `ProstateMRIDataset` | 2-3h |
| **A11** | Mama | `download_data.sh` (Breast-MRI), extender `datasets.py` con `BreastMRIDataset` | 2-3h |
| **A12** | Multi-anatomía trainer | Actualizar `trainer.py` para multi-head/anatomy token, `scripts/train_multi.py` | 3-4h |

---

## 3. Reglas para agentes (evitar conflictos)

### Regla de oro: 1 archivo = 1 agente por ronda

En cada ronda, los agentes asignados escriben archivos **totalmente disjuntos**. Si dos agentes necesitan modificar el mismo archivo, hay que secuenciarlos o refactorizar.

### Archivos por agente — matriz de propiedad

```
                    A0  A1  A2  A3  A4  A5  A6  A7  A8  A9
configs/data.yaml   ██  ..  ..  ..  ░░  ..  ..  ..  ..  ..
configs/model.yaml  ██  ..  ..  ..  ░░  ..  ..  ..  ..  ..
configs/train.yaml  ██  ..  ..  ..  ░░  ..  ..  ..  ..  ..
models/encoder.py   ..  ██  ..  ..  ..  ..  ..  ░░  ..  ..
models/mil.py       ..  ..  ..  ..  ..  ██  ..  ░░  ..  ..
models/triage.py    ..  ..  ..  ..  ..  ..  ..  ██  ..  ..
data/preproc.py     ..  ..  ██  ..  ..  ..  ░░  ..  ..  ..
data/transforms.py  ..  ..  ..  ██  ..  ..  ░░  ..  ..  ..
data/datasets.py    ..  ..  ..  ..  ..  ..  ██  ..  ░░  ..
data/labels.py      ..  ..  ██  ..  ..  ..  ..  ..  ..  ..
training/metrics.py ..  ..  ..  ██  ..  ..  ..  ..  ..  ░░
training/losses.py  ..  ..  ..  ██  ..  ░░  ..  ..  ..  ..
training/trainer.py ..  ..  ..  ..  ..  ..  ..  ..  ██  ..
evaluation/*.py     ..  ..  ..  ..  ..  ..  ..  ..  ..  ██
demo/app.py         ..  ..  ..  ..  ..  ..  ..  ..  ..  ██
scripts/train.py    ..  ..  ..  ..  ..  ..  ..  ..  ██  ..
scripts/eval.py     ..  ..  ..  ..  ..  ..  ..  ..  ..  ██

██ = escribe este agente
░░ = lee/interactúa pero no escribe
.. = no toca
```

**Cero conflictos de escritura entre agentes de la misma ronda.**

### Qué hacer si un agente necesita modificar algo de otro:

1. Si es un **bug trivial**: el agente lo documenta en su output y sigue con su tarea. Se arregla en una ronda separada de fixes.
2. Si es un **cambio de interfaz necesario**: lo comunica en su output, y se actualizan las interfaces en `INTERFACES.md`. Los demás agentes se lanzan de nuevo con las interfaces actualizadas.
3. Si es un **archivo que dos agentes necesitan crear**: dividir el archivo. Ej: si `datasets.py` crece mucho → `datasets/brain.py`, `datasets/prostate.py`, `datasets/breast.py`.

---

## 4. Plan de ejecución en timeline

```
Día 1 ─────────────────────────────────────
  08:00  Agente 0: Interfaces (30 min) ← BLOQUEANTE
  08:30  Agente 1: Encoder + descarga Triad  ┐
  08:30  Agente 2: Preprocesado + descarga datos │ PARALELO
  08:30  Agente 3: Transforms + Métricas + Losses │ 4 streams
  08:30  Agente 4: Scaffolding + configs       ┘
  12:00  Verificación: ¿compila todo? ¿pasan tests unitarios?
  14:00  Fixes si algo falló (agentes secuenciales para bugs)

Día 2 ─────────────────────────────────────
  08:00  Agente 5: MIL head           ┐
  08:00  Agente 6: BrainMRIDataset    │ PARALELO
  08:00  Agente 7: Triage completo     ┘
  14:00  Verificación: integración MIL + Dataset + Encoder
  16:00  Smoke test: forward pass con batch sintético

Día 3 ─────────────────────────────────────
  08:00  Agente 8: Trainer + train.py     ┐ PARALELO
  08:00  Agente 9: Evaluación + Demo       ┘
  14:00  Verificación: ¿entrena sin errores?
  16:00  Primer entrenamiento real ← LARGAR Y DEJAR CORRIENDO

Días 4-7 ──────────────────────────────────
  Trabajo manual iterativo:
  - Mirar curvas de loss/AUC
  - Ajustar LR, attention dim, batch size
  - Evaluar en validation set
  - (No paralelizable — es I+D secuencial)

Día ~12 ───────────────────────────────────
  08:00  Agente 10: Próstata dataset   ┐
  08:00  Agente 11: Mama dataset       │ PARALELO
  08:00  Agente 12: Multi-anatomía     ┘
```

---

## 5. Estrategia de verificación entre rondas

Al final de cada ronda, antes de lanzar la siguiente:

```bash
# 1. ¿Compila?
python -c "import triagemri" && echo "OK"

# 2. ¿Tests unitarios pasan? (cada agente debe dejar tests)
pytest tests/ -x -q

# 3. ¿Smoke test de integración?
python scripts/smoke_test.py  # forward pass con tensor aleatorio

# 4. ¿Configs válidos?
python -c "from triagemri.config import load_config; load_config()"
```

Si algo falla → ronda de fixes (1-2 agentes secuenciales) antes de continuar.

---

## 6. Lo que NO se debe paralelizar

| Tarea | Por qué no |
|-------|-----------|
| Escribir el mismo archivo desde 2 agentes | Conflicto de merge garantizado |
| Hiperparámetro tuning | Requiere ver resultados del entrenamiento anterior |
| Definir interfaces | Debe ser consensuado primero (por eso la Ronda 0) |
| Evaluación final | Requiere modelo entrenado y thresholds calibrados |
| Descarga de datos + escritura de dataset | Dataset necesita saber formato exacto de datos descargados |

---

## 7. Resumen: máximo paralelismo alcanzable

```
Ronda 0: 1 agente  (30 min)        ← inevitablemente secuencial
Ronda 1: 4 agentes (3-4h)          ← 4x speedup
Ronda 2: 3 agentes (4-5h)          ← 3x speedup
Ronda 3: 2 agentes (3-4h)          ← 2x speedup
Ronda 4: 0 agentes (días)          ← trabajo manual de ML
Ronda 5: 3 agentes (3-4h)          ← 3x speedup

Tiempo total con agentes:    ~3 días hasta primer entrenamiento
Tiempo total secuencial:     ~12 días hasta primer entrenamiento
Speedup:                     ~4x
```

El cuello de botella real no es escribir código — es el **entrenamiento y la iteración de hiperparámetros** (Ronda 4), que es inherentemente secuencial y consume la mayor parte del tiempo del proyecto.

---

*Documento generado como parte de PRUEBA_MRI — Mayo 2026*
