# Evolución del proyecto Triage-MRI

> Reconstrucción cronológica basada en logs de entrenamiento, git history, checkpoints, configs, documentación y timestamps del sistema de archivos.

---

## 1. Contexto y motivación

**Problema:** Los servicios de radiología reciben cientos de RM cerebrales diarias. El 30–40% son normales. Un sistema de triaje automático que descarte con alta sensibilidad los estudios normales reduciría la carga de trabajo del radiólogo sin comprometer la seguridad del paciente.

**Elección del modelo base:** Se auditaron modelos fundacionales disponibles para RM 3D (OmniMRI, Decipher-MR, Swin UNETR, UNETR++, Triad). Triad Swin-B SimMIM fue seleccionado por:
- Pre-entrenado en 131K volúmenes de cerebro/próstata/mama
- Arquitectura Swin Transformer 3D con ventanas (7,7,7)
- Checkpoint público (80 MB) con pesos accesibles
- Soporte multi-secuencia y multi-anatomía desde el diseño original

**Arquitectura:** `Volumen 3D → TriadEncoder (congelado, 19.8M) → AttentionMIL (entrenable, 525K) → Anomaly Score [0,1] → NORMAL / REVISAR`

---

## 2. Fase 0 — Investigación y planificación

**Fecha:** 26 de mayo de 2026 (~08:40–20:30)

Se produjeron todos los documentos de planificación en una sesión intensiva:

| Documento | Propósito |
|-----------|-----------|
| `docs/inicio.md` | Estado del arte: modelos fundacionales para triaje en RM |
| `docs/auditoria_modelos_fundacionales.md` | Por qué Triad y no OmniMRI, Decipher-MR, etc. |
| `docs/plan_implementacion.md` | Plan de 6–8 semanas con fases, anatomías e hitos |
| `docs/plan_paralelizacion.md` | Estrategia de agentes paralelos, matriz de ownership |
| `INTERFACES.md` | Contrato de interfaces entre todos los módulos |
| `MODEL_OVERVIEW.md` | Descripción detallada de la arquitectura |
| `ANALISIS.md` | Análisis de lo construido vs lo planeado, roadmap |

**Decisiones de diseño tomadas este día:**
- Encoder: Triad Swin-B congelado con stage projections entrenables
- MIL: Gated Attention con feature pyramid de 5 stages
- Loss: FocalLoss con gamma=2, alpha=0.25 (luego cambiado a weighted BCE)
- Datos: BraTS 2023 + OASIS-3 para cerebro, PI-CAI para próstata
- Split: 70/15/15 estratificado por paciente y fuente
- Framework: PyTorch Lightning + MONAI + TensorBoard

---

## 3. Fase 1 — Primer entrenamiento (brain-only)

**Fecha:** 27 de mayo de 2026 (09:07–17:06)

**Run:** `run_001`
**Datos:** Solo cerebro (BraTS GLI/MEN/MET/PED + OASIS-3 + IXI + HCP)
**Casos:** ~3,800 total (2,660 train / 578 val / 575 test)
**Config:** batch_size=4, lr=5e-4, AdamW, cosine annealing, weighted BCE

| Época | AUC val | Tiempo |
|-------|---------|--------|
| 0 | 0.909 | 09:07 |
| 1 | 0.989 | 10:11 |
| 2 | 0.992 | 11:29 |
| 3 | 0.992 | 12:51 |
| 4 | 0.993 | 13:53 |
| 5 | 0.993 | 14:59 |
| 6 | **0.994** | 16:05 |
| 7 | 0.993 | 17:06 |

**Duración:** ~8 horas, 1 GPU (RTX 3500 Ada Laptop).
**Resultado:** AUC 0.994 en solo 8 épocas. El modelo aprendía extremadamente rápido, lo que sugería que el encoder Triad ya extraía features muy discriminativas para anomalías cerebrales. Parada temprana por convergencia (no por early stopping — se detuvo manualmente).

**⚠️ Advertencia retrospectiva:** Este resultado estaba inflado por bugs descubiertos después (ver Fase 4). Los AUC reales post-fix son menores.

---

## 4. Fase 2 — Segundo run y primera expansión multi-anatomía

**Fecha:** 28–29 de mayo de 2026

### run_002 (brain-only, 28 mayo)

Segundo run con cerebro. **No fue una reanudación de run_001** — fue un run fresco con split de datos o seed diferente (AUC inicial 0.867 vs 0.909). 9 épocas, mejor AUC 0.989.

### run_brain_prostate (brain + prostate, 28–29 mayo)

**Primer intento multi-anatomía.** Se añadió próstata usando datos PI-CAI. 36 épocas, ~18 horas.

| Hito | Época | AUC val |
|------|-------|---------|
| Inicio | 0 | 0.734 |
| AUC >0.9 | 5 | 0.904 |
| AUC >0.93 | 13 | 0.931 |
| Mejor | 35 | **0.936** |

**Observaciones:**
- El AUC inicial (0.734) es mucho más bajo que brain-only (0.909). Añadir próstata introdujo ruido y complejidad.
- Progreso más lento: 35 épocas para llegar a 0.936 vs 8 épocas para 0.994 en brain-only.
- Oscilaciones frecuentes (ej. 0.924 → 0.888 → 0.913 → 0.906). El entrenamiento multi-anatomía es inestable.
- El modelo nunca alcanzó el rendimiento brain-only, sugiriendo que la próstata requiere más datos o arquitectura específica por anatomía.

---

## 5. Fase 3 — Auditoría de bugs y sesgos

**Fecha:** 30–31 de mayo de 2026

Se realizó una auditoría exhaustiva del código que descubrió **121 problemas** (12 críticos, 29 altos, 44 medios, 36 bajos).

### Bugs críticos encontrados

| ID | Severidad | Descripción | Impacto |
|----|-----------|-------------|---------|
| C1 | CRÍTICO | `find_threshold_for_sensitivity` usaba `np.argmax` que selecciona el **peor** umbral, no el mejor. | Especificidad ~0% — el sistema enviaba todo a REVISAR. No triaba nada. |
| C2 | CRÍTICO | Umbrales de decisión calibrados sobre el conjunto de test (data leakage). | Métricas infladas artificialmente. Los AUC de runs anteriores no son fiables. |
| C3 | CRÍTICO | `pos_weight` calculado por batch en vez de sobre el dataset completo. | Pérdida mal balanceada, sesgo de clase inconsistente entre batches. |
| C8 | ALTO | Ejes axial/sagital intercambiados en la demo Gradio. | Visualización incorrecta de cortes. |
| C12 | ALTO | Claves de configuración muertas en YAMLs. | `save_top_k=-1`, `architecture`, `num_attention_branches` sin efecto real. |

### Otros hallazgos relevantes

- **Train/test leakage:** Pacientes con múltiples sesiones no se excluían correctamente del split.
- **Calibración:** Platt scaling se aplicaba sobre scores ya calibrados (doble calibración).
- **Atención:** Los mapas de atención no se normalizaban correctamente para visualización.
- **Decoder:** El decoder de anomalías 3D estaba implementado pero nunca se probó (siempre deshabilitado en config).

**Conclusión de la auditoría:** Los resultados de runs anteriores (AUC 0.994, 0.989, 0.936) estaban inflados por data leakage. El rendimiento real del modelo es menor. Esto explicaría por qué los runs post-rediseño obtienen AUCs más bajos pero correctos.

---

## 6. Fase 4 — Rediseño y corrección masiva

**Fecha:** 1 de junio de 2026 (08:53–10:22)

**Actividad git:** 26 commits atómicos en rama `restructured` + 1 merge squash a `dev`.

### Cambios estructurales

| Área | Antes | Después |
|------|-------|---------|
| **CLI** | Scripts sueltos en raíz | `src/triagemri/cli/` con entry points registrados en `pyproject.toml` |
| **Serving** | No existía | FastAPI + Pydantic schemas + model loader con singleton |
| **Logging** | Prints ad-hoc | Structured logging con JSONFormatter + run cards YAML |
| **Tests** | 0 tests | 48 tests (encoder, MIL, preprocessing, triage, trainer) |
| **CI/CD** | No existía | GitHub Actions: lint (ruff) + test (pytest 3.10/3.11/3.12) |
| **Docker** | No existía | Dockerfile + docker-compose (train/eval/demo/api) |
| **DVC** | No existía | Pipeline DVC (download → preprocess → train → evaluate) |
| **Docs** | 8 markdowns sueltos | Sphinx con 12 páginas en `docs/source/` |

### Cambios en el modelo

| Componente | Cambio |
|------------|--------|
| `encoder.py` | Añadido soporte LoRA para fine-tuning parcial del backbone |
| `mil.py` | Refactorizado: GatedAttentionMIL + MultiScaleAttentionMIL unificados |
| `triage.py` | Añadido `predict()`, `calibrate()`, `predict_calibrated()`, `get_attention_heatmap()` |
| `decoder.py` | **Nuevo:** AnomalyDecoder 3D con attention gates para heatmaps voxel-level |
| `losses.py` | Corregido `pos_weight` por dataset. Añadido MultiTaskLoss con pseudo-labels |
| `metrics.py` | Corregido `find_threshold_for_sensitivity` (argmax → last match). Añadida calibración ECE. |
| `trainer.py` | Refactorizado TriageLightningModule con soporte multi-task decoder |

### Cambios en datos

| Componente | Cambio |
|------------|--------|
| `datasets.py` | Refactorizado: BrainMRIDataset, ProstateMRIDataset, MultiAnatomyDataset. Corregido patient-level split. |
| `preprocessing.py` | Normalización Z-score por volumen, crop/pad a 96³ |
| `transforms.py` | Añadidas augmentaciones MONAI 3D (RandFlip, RandAffine, RandGaussianNoise) |
| `labels.py` | Extracción de labels desde estructura de directorios y metadatos |

### Corrección del bug C1 (argmax)

```python
# Antes (bug):
best_idx = np.argmax(recall >= target_sensitivity)  # selecciona el PEOR

# Después (fix):
matches = np.where(recall >= target_sensitivity)[0]
best_idx = matches[-1]  # último match = mayor umbral = mayor especificidad
```

**Impacto estimado:** Con el bug, especificidad ~0%. Tras el fix, la especificidad debería estar en 30–50% para cerebro manteniendo sensibilidad 0.99.

---

## 7. Fase 5 — Reentrenamiento post-rediseño

**Fecha:** 1–2 de junio de 2026

Dos runs concurrentes para validar el código corregido:

### PRIMERA_PRUEBA_POST_REDISENO (brain-only)

5 versiones (intentos fallidos + exitosos). Solo la v3 produjo checkpoints.

| Época | AUC val | Notas |
|-------|---------|-------|
| 1 | 0.939 | |
| 3 | 0.951 | |
| 4 | **0.967** | Mejor, last.ckpt |

**Comparación con pre-fix:** AUC 0.967 vs 0.994 (run_001). Caída de 0.027 — atribuible a la eliminación del data leakage (C2) y la corrección del threshold (C1). **Este AUC es más realista y fiable.**

### BRAIN_PROSTATE_V3 (brain + prostate)

2 versiones. Solo la v0 produjo checkpoints.

| Época | AUC val | Notas |
|-------|---------|-------|
| 1 | 0.844 | |
| 2 | **0.880** | Mejor |
| 3 | 0.874 | last.ckpt |

**Comparación con pre-fix:** AUC 0.880 vs 0.936 (run_brain_prostate). Caída de 0.056 — más pronunciada que en brain-only. La próstata sufre más el data leakage porque tiene menos datos y el split paciente-nivel es más crítico.

**⚠️ Ambos runs fueron interrumpidos antes de converger** (4 y 3 épocas respectivamente). Los AUC finales serían mayores con más épocas. Quedan pendientes de reanudación.

---

## 8. Estado actual y métricas comparativas

### Evolución de AUC por run (orden cronológico)

```
run_001 (brain):           ██████████████████████████████████████████████████ 0.994  ← inflado (bugs)
run_002 (brain):           ████████████████████████████████████████████████   0.989  ← inflado (bugs)
run_brain_prostate:        ████████████████████████████████████████████       0.936  ← inflado (bugs)
PRIMERA_PRUEBA (brain):    ██████████████████████████████████████████████     0.967  ← post-fix (incompleto)
BRAIN_PROSTATE_V3:         ████████████████████████████████████████           0.880  ← post-fix (incompleto)
```

### Lo que sabemos y lo que no

| Certeza | Incertidumbre |
|---------|---------------|
| El código pre-fix tenía data leakage que inflaba métricas | ¿Cuánto exactamente? — necesitamos re-evaluar modelos viejos con eval corregido |
| Los bugs C1 (argmax) y C2 (leakage) están corregidos | ¿El modelo post-fix alcanzará >0.95 con suficientes épocas? |
| Brain-only converge en ~10 épocas | ¿Brain+prostate necesita 50+ épocas? |
| La arquitectura shared-head funciona para cerebro | ¿Funciona igual de bien para próstata con solo ~700 casos PI-CAI? |
| El encoder Triad congelado extrae features de calidad | ¿LoRA o unfreezing parcial mejorarían el rendimiento multi-anatomía? |

---

## 9. Lecciones aprendidas

### Técnicas

1. **No calibrar umbrales en test.** Usar solo el conjunto de validación para thresholds y calibración. El test solo se toca una vez al final.

2. **El split paciente-nivel es crítico.** Pacientes con múltiples sesiones (distintos tiempos, mismas patologías) deben ir al mismo split o se produce leakage.

3. **Validar las métricas visualmente.** El bug del argmax pasó desapercibido porque el AUC seguía siendo alto. Solo se detectó al inspeccionar la matriz de confusión (especificidad 0%).

4. **Multi-anatomía es más difícil de lo esperado.** Añadir próstata degrada el rendimiento en cerebro y tarda 4x más en converger. Posiblemente se necesita una arquitectura multi-head en vez de shared-head.

### De proceso

5. **Auditar antes de iterar.** 3 días de entrenamiento produjeron resultados inflados que tuvieron que descartarse. Una auditoría temprana habría ahorrado tiempo.

6. **Versionar configuraciones.** Los runs no guardaban su config en el checkpoint, haciendo imposible reproducir exactamente las condiciones de entrenamiento.

7. **Nombrar runs descriptivamente.** `PRIMERA_PRUEBA_POST_REDISENO` y `BRAIN_PROSTATE_V3` son mejores que `run_001` y `default`, pero un esquema consistente (fecha-anatomías-versión) sería ideal.

---

## 10. Roadmap implícito (inferido de docs y estado del código)

| Fase | Estado | Descripción |
|------|--------|-------------|
| 1. Prototipo mono-anatomía (cerebro) | ✅ Completado | Brain-only con AUC 0.967 post-fix |
| 2. Expansión multi-anatomía (cerebro + próstata) | 🔄 En progreso | BRAIN_PROSTATE_V3 en entrenamiento |
| 3. Añadir mama | ⏳ Pendiente | Datasets definidos en config pero sin datos reales |
| 4. Demo clínica (Gradio) | ✅ Implementado | Visualización 3D + heatmaps de atención |
| 5. API de producción (FastAPI) | ✅ Implementado | Endpoint batch predict + health check |
| 6. CI/CD + tests | ✅ Implementado | GitHub Actions + 48 tests |
| 7. Despliegue Docker | ✅ Implementado | Dockerfile + docker-compose |
| 8. Umbrales tri-zona (NORMAL/INCONCLUSIVE/REVISAR) | ✅ Implementado | `find_inconclusive_thresholds()` |
| 9. Decoder de anomalías 3D | 🔒 Feature-flagged | Implementado pero `decoder.enabled: false` |
| 10. LoRA fine-tuning | 🔒 Feature-flagged | Implementado pero `lora.enabled: false` |

---

*Documento generado el 2 de junio de 2026 a partir de git history, timestamps de archivos, logs de entrenamiento y documentación del repositorio.*
