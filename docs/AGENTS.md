# Triage-MRI — Arquitectura y Workflow para Agentes

> **Para agentes IA:** Este proyecto está documentado. Antes de hacer cualquier análisis del repositorio, lee la documentación existente. No dupliques trabajo de exploración.
>
> - `INTERFACES.md` — Contrato de interfaces entre TODOS los módulos (fuente de verdad)
> - `MODEL_OVERVIEW.md` — Arquitectura del modelo al detalle
> - `ANALISIS.md` — Qué se ha construido vs. lo planeado, roadmap de mejoras
> - `CHEATSHEET.md` — Todos los comandos en un solo lugar
> - `docs/inicio.md` — Investigación SOTA, fundamentos teóricos
> - `docs/auditoria_modelos_fundacionales.md` — Por qué Triad y no otro encoder
> - `docs/plan_implementacion.md` — Plan de fases, arquitectura, dependencias
> - `docs/plan_paralelizacion.md` — Estrategia de agentes en paralelo, matriz de ownership
> - `docs/AGENTS.md` — Este archivo (workflow + principios + verificación)

---

## Workflow al modificar código

1. **Leer documentación existente** antes de tocar nada. No hagas `find`/`grep` masivos; consulta los `.md` relevantes (ver tabla de correspondencia abajo).
2. **Hacer el cambio** siguiendo los principios de abajo (encoder congelado invariante, interfaces como contrato, atención como interpretabilidad).
3. **Verificar que el paquete compila, los tests pasan y el linting es limpio:**
   ```bash
   python -c "from triagemri.config import load_config; print('OK')"   # verifica arranque
   pytest tests/ -v                                                      # 48 tests
   ruff check src/                                                       # linting
   ```
4. **Registrar el cambio en `docs/CHANGELOG.md`** al final de cada sesión. Formato:
   ```markdown
   ## YYYY-MM-DD — Resumen breve de la sesión
   - **Archivos modificados:** `ruta/archivo1`, `ruta/archivo2`
   - **Qué se hizo:** descripción concisa del cambio
   - **Por qué:** motivo del cambio
   - **APIs afectadas:** (si aplica)
   ```
   Si `docs/CHANGELOG.md` no existe, créalo con la primera entrada.

---

## Actualizar la documentación del proyecto

**Regla de oro:** tras cada sesión de cambios, ejecuta `git diff --stat` para ver qué archivos se tocaron. Luego cruza con esta tabla para saber qué `.md` actualizar.

| Si tocaste archivos en... | Actualiza este `.md` |
|---|---|
| `configs/*.yaml` | `INTERFACES.md` (si cambian formas/shapes), `CHEATSHEET.md` (si cambian comandos) |
| `src/triagemri/config.py` | `INTERFACES.md` (sección Config) |
| `src/triagemri/data/datasets.py` | `INTERFACES.md` (sección Dataset) |
| `src/triagemri/data/preprocessing.py` | `INTERFACES.md` (sección Preprocessing) |
| `src/triagemri/data/transforms.py` | `INTERFACES.md` (sección Transforms, si la API cambia) |
| `src/triagemri/data/labels.py` | `INTERFACES.md` (sección Labels, si la API cambia) |
| `src/triagemri/models/encoder.py` | `INTERFACES.md` (sección Encoder), `MODEL_OVERVIEW.md` |
| `src/triagemri/models/mil.py` | `INTERFACES.md` (sección MIL), `MODEL_OVERVIEW.md` |
| `src/triagemri/models/triage.py` | `INTERFACES.md` (sección TriageModel), `MODEL_OVERVIEW.md` |
| `src/triagemri/training/trainer.py` | `INTERFACES.md` (sección Trainer), `CHEATSHEET.md` |
| `src/triagemri/training/losses.py` | `INTERFACES.md` (sección Losses) |
| `src/triagemri/training/metrics.py` | `INTERFACES.md` (sección Métricas) |
| `src/triagemri/evaluation/evaluator.py` | `INTERFACES.md`, `CHEATSHEET.md` |
| `src/triagemri/demo/app.py` | `MODEL_OVERVIEW.md` (sección Demo), `CHEATSHEET.md` |
| `scripts/train.py` | `CHEATSHEET.md`, `MODEL_OVERVIEW.md` |
| `scripts/evaluate.py` | `CHEATSHEET.md` |
| `scripts/demo.py` | `CHEATSHEET.md` |
| `tests/*` | (no requiere doc específica, los tests son autodocumentados) |

**Qué documentar:** no transcribas el código. Explica qué cambio se hizo, por qué, y si modificó el contrato del componente (nuevas formas de tensor, cambio de parámetros, eliminación de exports). Una o dos frases por archivo tocado bastan.

**Si el cambio es nuevo** (módulo, clase, script), documéntalo en `INTERFACES.md` con su sección correspondiente y añade el comando relevante a `CHEATSHEET.md`.

**Si el cambio es de hiperparámetros o configuración**, actualiza `MODEL_OVERVIEW.md` y `configs/*.yaml`.

---

## Ramas y flujo de trabajo

```
main ──────── (producción, intocable)
  └── dev ── (integración, rama base para todo)
       ├── feature/<descripcion>
       ├── fix/<descripcion>
       ├── docs/<descripcion>
       └── chore/<descripcion>
```

- **`main`**: producción. Solo recibe merges desde `dev` vía PR. Nunca se ramifica directamente desde `main`.
- **`dev`**: integración continua. Es la rama base de la que **siempre** se crean las ramas de trabajo. Acumula features, fixes y chores antes de promocionar a `main`.
- **Ramas de trabajo**: `feature/<desc>`, `fix/<desc>`, `docs/<desc>`, `chore/<desc>`. Se crean desde `dev` y se mergean a `dev` vía PR.
- **Al iniciar una tarea nueva**, el agente debe:
  1. `git checkout dev && git pull origin dev`
  2. `git checkout -b feature/<descripcion>` (o `fix/`, `docs/`, `chore/`)
  3. Trabajar, commitear, y al terminar ofrecer PR hacia `dev`.
- **Nunca** crear ramas desde `main` ni mergear directamente a `main`.

---

## Commits

- **Al final de cada sesión, ofrece hacer commit.** Revisa `git diff --stat`, resume los cambios, y pregunta al usuario: *"¿Quieres que haga commit? Propongo: `tipo(scope): descripción`"*. No commitees sin que el usuario confirme.
- **Conventional Commits obligatorio:** `tipo(scope): descripción`
  - Tipos: `feat` (nueva feature), `fix` (bug fix), `chore` (tareas de mantenimiento), `docs` (documentación), `refactor` (refactorización sin cambio funcional), `test` (tests), `style` (formato/linting)
  - Scope: `encoder`, `mil`, `triage`, `data`, `training`, `evaluation`, `demo`, `config`, `scripts`, `docs`
  - Ejemplos: `feat(mil): añadir soporte para multi-head por anatomía`, `fix(data): corregir extracción de labels en PI-CAI`, `docs(interfaces): actualizar firma de load_and_preprocess`
- **Nunca incluyas en commits:** `outputs/`, `weights/` (binarios >10MB), `data/`, `.env`, secretos, `nohup.out`, `*.bak`, `__pycache__/`, `.pytest_cache/`, `notebooks/.ipynb_checkpoints/`.

---

## Estructura del proyecto

```
triage-mri/
├── configs/                              # Configuración YAML (mergeadas en runtime por config.py)
│   ├── data.yaml                         # Paths de datasets, splits, secuencias, balanceo
│   ├── model.yaml                        # Arquitectura: encoder, MIL head
│   └── train.yaml                        # Hiperparámetros: lr, epochs, scheduler, loss
│
├── src/triagemri/                        # Paquete Python principal
│   ├── __init__.py                       # Versión, exports Config
│   ├── config.py                         # Config loader: mergea YAMLs → objeto tipado
│   ├── data/                             # Carga y preprocesado de datos
│   │   ├── datasets.py                   # BrainMRIDataset, MultiAnatomyDataset, create_dataloaders()
│   │   ├── preprocessing.py              # load_nifti, normalize, crop/pad, load_and_preprocess
│   │   ├── transforms.py                 # Aumentaciones 3D MONAI (train/val)
│   │   └── labels.py                     # Extracción de etiquetas por dataset
│   ├── models/                           # Componentes de red neuronal
│   │   ├── encoder.py                    # TriadEncoder: Swin-B 3D congelado + multi-escala + proyección
│   │   ├── mil.py                        # AttentionMIL, GatedAttentionMIL (CLAM-style)
│   │   └── triage.py                     # TriageModel: encoder + MIL, build_triage_model()
│   ├── training/                         # Maquinaria de entrenamiento
│   │   ├── trainer.py                    # TriageLightningModule (PyTorch Lightning wrapper)
│   │   ├── losses.py                     # FocalLoss, WeightedBCE, AsymmetricLoss, get_loss_fn()
│   │   └── metrics.py                    # ROC-AUC, sensibilidad, especificidad, calibración
│   ├── evaluation/                       # Pipeline de evaluación
│   │   └── evaluator.py                  # TriageEvaluator: inferencia, thresholds, reports JSON/MD
│   └── demo/                             # Interfaz web interactiva
│       └── app.py                        # Gradio Blocks: upload NIfTI, 3D viewer, attention heatmaps
│
├── scripts/                              # Puntos de entrada CLI
│   ├── train.py                          # Script principal de entrenamiento
│   ├── evaluate.py                       # Evaluación sobre test set
│   ├── demo.py                           # Lanzar demo Gradio
│   ├── monitor.py                        # Monitor de entrenamiento (TensorBoard)
│   ├── download_triad.sh                 # Descarga pesos Triad Swin-B
│   ├── download_data.sh                  # Instrucciones de descarga de datasets
│   └── view_brats.py                     # Visualizador de casos BraTS
│
├── tests/                                # Tests unitarios (48 tests, pytest)
│   ├── test_encoder.py                   # TriadEncoder: shapes, freeze, proyección
│   ├── test_mil.py                       # AttentionMIL: shapes, atención, determinismo
│   ├── test_preprocessing.py             # Normalización, crop/pad, carga
│   ├── test_triage.py                    # TriageModel: integración, multi-head, predict
│   └── test_trainer.py                   # TriageLightningModule: training_step, optimizador
│
├── docs/                                 # Documentación y planificación
│   ├── inicio.md                         # Investigación SOTA inicial
│   ├── auditoria_modelos_fundacionales.md # Auditoría de modelos disponibles
│   ├── plan_implementacion.md            # Plan de fases y arquitectura
│   ├── plan_paralelizacion.md            # Estrategia de agentes paralelos
│   └── AGENTS.md                        # Este archivo
│
├── outputs/                              # Salidas de entrenamiento (NO COMMITEAR)
│   ├── run_001/                          # Runs individuales con checkpoints y logs
│   ├── evaluation/                       # Reports de evaluación JSON/MD
│   └── saved/                            # Modelos guardados
│
├── weights/                              # Pesos pre-entrenados (NO COMMITEAR)
│   └── triad_swinb_simmim.pth            # Triad Swin-B SimMIM checkpoint
│
├── data/                                 # Datasets (NO COMMITEAR)
│   ├── raw/                              # Datasets originales (brain, prostate, breast)
│   └── processed/                        # Volúmenes preprocesados
│
├── notebooks/                            # Jupyter notebooks (exploración, análisis)
├── pyproject.toml                        # Dependencias, tool config (ruff, black, pytest)
├── README.md                             # Quickstart y visión general
├── INTERFACES.md                         # Contrato de interfaces (fuente de verdad)
├── MODEL_OVERVIEW.md                     # Arquitectura detallada del modelo
├── ANALISIS.md                           # Análisis: construido vs planeado, roadmap
├── CHEATSHEET.md                         # Referencia rápida de comandos
└── prompt.txt                            # Rol y metodología para agentes IA
```

---

## Pipeline de datos

```
NIfTI Volume (.nii.gz)
    │
    ▼
preprocessing.load_and_preprocess()  →  (1, 96, 96, 96) tensor normalizado
    │
    ▼
TriadEncoder.forward()               →  features: (B, 27, 768)
    │                                    (5 etapas Swin multi-escala, concat, proyección)
    ▼
GatedAttentionMIL.forward()          →  logits: (B, 1), attention: (B, 27, 1)
    │                                    (CLAM-style gated attention pooling)
    ▼
torch.sigmoid(logits)                →  score: (B,)  [0, 1]
    │
    ▼
score >= threshold?                  →  "REVISAR" (anomalía) / "NORMAL"
```

---

## Principios arquitectónicos

1. **Encoder congelado como invariante:** El Triad Swin-B SimMIM (19.8M parámetros) permanece siempre en modo `freeze=True`, `requires_grad=False`. Solo se entrena el cabezal MIL (525K). Si se necesita adaptar la entrada (ej. multi-canal T1+T2+FLAIR), se descongela exclusivamente `patch_embed.proj` (~1K params). Cualquier desviación de esta regla debe ser explícitamente autorizada.

2. **Contrato de interfaces sagrado:** `INTERFACES.md` define las formas de tensor, firmas de función y tipos de retorno para cada módulo. Todo agente debe leerlo antes de escribir código. Si un cambio modifica una interfaz, debe actualizarse `INTERFACES.md` en el mismo commit.

3. **Atención como interpretabilidad:** El mapa de atención `(B, 27, 1)` es la única ventana de interpretabilidad del modelo. Cualquier modificación al MIL debe preservar `get_attention_heatmap()` devolviendo una cuadrícula 3×3×3 interpretable.

4. **Supervisión débil:** Las etiquetas se extraen automáticamente de metadatos existentes (máscaras de segmentación, escalas clínicas, CSVs). Nunca se requiere anotación manual pixel a pixel. Si añades un dataset nuevo, implementa su extractor en `data/labels.py`.

5. **Multi-anatomía por diseño:** El sistema soporta cerebro, próstata y mama mediante `MultiAnatomyDataset` y cabezales por anatomía (`nn.ModuleDict` en `triage.py`). El encoder es compartido. Al añadir una anatomía, extiende el dataset y el diccionario de cabezales.

6. **Configuración tipada y mergeada:** `config.py` fusiona `data.yaml`, `model.yaml` y `train.yaml` en un objeto `Config` con acceso por atributos (`config.model.encoder.embed_dim`). Usa siempre `load_config()` en lugar de leer YAMLs manualmente.

---

## Verificación y tests

```bash
# Arranque del paquete
python -c "from triagemri.config import load_config; print('OK')"

# Tests unitarios (48 tests)
pytest tests/ -v

# Test específico de un módulo
pytest tests/test_encoder.py -v
pytest tests/test_mil.py -v
pytest tests/test_preprocessing.py -v
pytest tests/test_triage.py -v
pytest tests/test_trainer.py -v

# Linting
ruff check src/

# Smoke test de integración (forward pass con tensor sintético)
python -c "
import torch
from triagemri.models.triage import build_triage_model
model = build_triage_model({
    'encoder': {
        'checkpoint_path': 'weights/triad_swinb_simmim.pth',
        'freeze': True,
        'embed_dim': 768,
        'depths': [2,2,2,2]
    }
})
x = torch.randn(1, 1, 96, 96, 96)
out = model(x)
print(f'Score: {out[\"score\"].shape}, Attention: {out[\"attention\"].shape}')
"

# Entrenamiento (GPU necesaria)
python scripts/train.py --output_dir outputs/test_run --max_epochs 1

# Evaluación (requiere checkpoint)
python scripts/evaluate.py --checkpoint outputs/default/checkpoints/last.ckpt

# Demo (requiere checkpoint)
python scripts/demo.py --checkpoint outputs/default/checkpoints/last.ckpt --port 7860
```

---

## Convenciones de código

| Aspecto | Convención |
|---------|-----------|
| **Lenguaje** | Python 3.10+ |
| **Formato** | Black (`line-length=100`, `target-version=py310`) |
| **Linting** | Ruff (`line-length=100`, `target-version=py310`) |
| **Type hints** | Obligatorios en todas las funciones y métodos públicos |
| **Docstrings** | Google-style. Solo en funciones públicas. Una línea de descripción + `Args:`/`Returns:` |
| **Imports** | `from __future__ import annotations` en todos los módulos |
| **Nombres** | snake_case para funciones/variables, PascalCase para clases |
| **Path handling** | Siempre `pathlib.Path`, nunca `os.path` |
| **Tensores** | `torch.Tensor`, float32 por defecto, canal-first `(B, C, D, H, W)` |
| **Logging** | PyTorch Lightning `self.log()` dentro del trainer, `logging` stdlib para scripts |
| **Comentarios** | No añadir comentarios inline a menos que se soliciten explícitamente |

---

## Reglas de ownership de archivos

Basado en `docs/plan_paralelizacion.md`, si trabajas con múltiples agentes:

- **1 archivo = 1 agente por ronda.** Los agentes en paralelo tocan archivos disjuntos.
- Si dos agentes necesitan el mismo archivo → secuencializar o refactorizar (dividir el archivo).
- Si un agente encuentra un bug en código de otro agente: lo documenta, no lo corrige directamente (evita conflictos de merge). Se arregla en ronda de fixes.
- **INTERFACES.md** es el punto de sincronización. Si cambia una interfaz, se actualiza y se relanzan los agentes afectados.

---

## Glosario rápido

| Término | Significado |
|---------|------------|
| **SimMIM** | Simple Masked Image Modeling — pre-entrenamiento auto-supervisado del encoder |
| **MIL** | Multiple Instance Learning — clasificación a partir de bolsas de instancias |
| **Triad** | Nombre del encoder Swin-B 3D pre-entrenado (131K volúmenes, cerebro+próstata+mama) |
| **BraTS** | Brain Tumor Segmentation dataset (~600 casos de tumores cerebrales) |
| **OASIS** | Open Access Series of Imaging Studies (envejecimiento normal / Alzheimer) |
| **PI-CAI** | Prostate Imaging: Cancer AI (~1,500 casos de cáncer de próstata) |
| **IXI** | Information eXtraction from Images (~600 cerebros sanos de control) |
| **NIfTI** | Neuroimaging Informatics Technology Initiative — formato de archivo `.nii.gz` |
| **CDR** | Clinical Dementia Rating — escala clínica usada como etiqueta en OASIS |
| **csPCa** | Clinically Significant Prostate Cancer — etiqueta en PI-CAI |
| **ECE** | Expected Calibration Error — métrica de calibración |
| **NPV** | Negative Predictive Value — probabilidad de que un "NORMAL" sea realmente normal |
| **AUC** | Area Under the ROC Curve — métrica global de clasificación |
| **Sensibilidad 99%** | Target de seguridad: el sistema no debe fallar en más del 1% de casos anormales |

---

*Documento generado para Triage-MRI. Mantener actualizado tras cada sesión de cambios significativos.*
