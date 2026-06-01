# Triage-MRI

Sistema generalista de triaje y descarte de normalidad en resonancia magnética 3D, usando modelos fundacionales y aprendizaje débilmente supervisado.

```
Volumen RM 3D → Encoder Triad Swin-B (congelado) → Attention-MIL → Score de anomalía [0,1]
```

## Quickstart

```bash
# Instalar dependencias
pip install -e .[dev]

# Descargar pesos del encoder
bash scripts/download_triad.sh

# Descargar datasets (instrucciones interactivas)
bash scripts/download_data.sh

# Entrenar
python scripts/train.py --config_dir configs --output_dir outputs/run_001

# Evaluar
python scripts/evaluate.py --checkpoint outputs/run_001/checkpoints/best.ckpt

# Demo
python scripts/demo.py --checkpoint outputs/run_001/checkpoints/best.ckpt
```

## Estructura

```
triage-mri/
├── configs/           # YAML de configuración (data, modelo, entrenamiento)
├── src/triagemri/     # Código fuente
│   ├── data/          # Datasets, preprocessing, transforms, labels
│   ├── models/        # TriadEncoder, AttentionMIL, TriageModel
│   ├── training/      # Lightning module, losses, metrics
│   ├── evaluation/    # Evaluador multi-anatomía
│   └── demo/          # Interfaz Gradio
├── scripts/           # Scripts ejecutables (train, eval, demo, downloads)
├── tests/             # Tests unitarios (48 tests)
├── docs/              # Documentación y planificación
└── notebooks/         # Jupyter notebooks
```

## Arquitectura

| Componente | Descripción |
|-----------|------------|
| **Encoder** | Triad Swin-B SimMIM (131K volúmenes RM 3D pre-entrenados, pesos públicos) |
| **MIL Head** | Gated Attention Multiple Instance Learning (CLAM-style) |
| **Entrenamiento** | PyTorch Lightning, Focal Loss, WeightedRandomSampler |
| **Evaluación** | ROC-AUC, NPV, sensibilidad 99%, calibración (ECE), métricas por anatomía |

## Datasets

| Anatomía | Dataset | N casos | Acceso |
|----------|---------|---------|--------|
| Cerebro | BraTS 2023 + OASIS-3 | ~5,800 | Inmediato / 1-3 días |
| Próstata | PI-CAI | ~1,500 | Inmediato (Zenodo) |
| Mama | Advanced-MRI-Breast | ~200 | Inmediato (TCIA) |

## Tests

```bash
pytest tests/ -v     # 48 tests
```

## Documentación

- `docs/inicio.md` — Investigación inicial del SOTA
- `docs/auditoria_modelos_fundacionales.md` — Auditoría completa de modelos disponibles
- `docs/plan_implementacion.md` — Plan de fases, arquitectura, métricas
- `docs/plan_paralelizacion.md` — Estrategia de delegación paralela con agentes
- `INTERFACES.md` — Contrato de interfaces entre módulos
