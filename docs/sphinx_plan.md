# Plan Sphinx — Documentación Triage-MRI

> **Objetivo:** Documentación profesional multi-nivel (recién llegado → experto),
> autogenerada a partir de docstrings, desplegada en ReadTheDocs, integrada en CI.

---

## 1. Marco: Diátaxis (4 cuadrantes)

```
         ┌──────────────────────────────────────┐
         │           orientada al estudio        │
         │                                      │
         │  TUTORIALES          HOW-TO GUIDES   │
         │  (aprender)          (resolver)       │
         │                                      │
  práctico ◄──────────────────────────────────► teórico
         │                                      │
         │  EXPLICACIÓN          REFERENCIA     │
         │  (entender)           (consultar)    │
         │                                      │
         │       orientada al trabajo           │
         └──────────────────────────────────────┘
```

Cada cuadrante responde una pregunta distinta y tiene su propia navegación.
Un usuario puede entrar por cualquiera de los 4 y encontrar su camino.

---

## 2. Estructura de archivos

```
triage-mri/
├── docs/
│   ├── source/                          ← fuente Sphinx
│   │   ├── conf.py                      ← configuración Sphinx
│   │   ├── index.md                     ← landing page
│   │   │
│   │   ├── tutorials/                   ← CUADRANTE 1: Tutoriales
│   │   │   ├── index.md
│   │   │   ├── quickstart.md            ← "5 minutos hasta tu primer triaje"
│   │   │   ├── installation.md          ← pip, Docker, GPU setup
│   │   │   └── first_training.md        ← entrenar tu primer modelo paso a paso
│   │   │
│   │   ├── howto/                       ← CUADRANTE 2: How-to Guides
│   │   │   ├── index.md
│   │   │   ├── add_new_dataset.md       ← añadir anatomía/dataset nuevo
│   │   │   ├── deploy_api.md            ← servir modelo con FastAPI
│   │   │   ├── calibrate_model.md       ← calibrar thresholds y Platt scaling
│   │   │   ├── multi_sequence.md        ← activar modo multi-secuencia
│   │   │   ├── interpret_attention.md   ← usar mapas de atención/heatmaps
│   │   │   ├── docker_training.md       ← entrenar en Docker
│   │   │   └── export_onnx.md           ← exportar a ONNX
│   │   │
│   │   ├── explanation/                 ← CUADRANTE 3: Explicación
│   │   │   ├── index.md
│   │   │   ├── architecture.md          ← diseño del sistema completo
│   │   │   ├── encoder_freeze.md        ← por qué el encoder va congelado
│   │   │   ├── mil_vs_segmentation.md   ← MIL vs segmentación: trade-offs
│   │   │   ├── data_licenses.md         ← situación de licencias y viabilidad
│   │   │   └── design_decisions.md      ← decisiones de diseño y alternativas
│   │   │
│   │   ├── reference/                   ← CUADRANTE 4: Referencia (autogen)
│   │   │   ├── index.md
│   │   │   ├── config.md                ← todas las keys de config
│   │   │   ├── cli.md                   ← comandos CLI (triage-train, etc.)
│   │   │   ├── data.rst                 ← autogen: triagemri.data.*
│   │   │   ├── models.rst               ← autogen: triagemri.models.*
│   │   │   ├── training.rst             ← autogen: triagemri.training.*
│   │   │   ├── evaluation.rst           ← autogen: triagemri.evaluation.*
│   │   │   ├── demo.rst                 ← autogen: triagemri.demo.*
│   │   │   └── serving.rst              ← autogen: triagemri.serving.*
│   │   │
│   │   ├── _static/                     ← CSS personalizado, imágenes
│   │   │   ├── custom.css
│   │   │   └── logo.svg
│   │   │
│   │   └── _templates/                  ← plantillas Sphinx
│   │       └── autosummary/
│   │
│   ├── Makefile                         ← build local (make html)
│   └── make.bat                         ← build Windows
│
├── .readthedocs.yaml                    ← RTD config
└── .github/workflows/docs.yml           ← CI para build de docs
```

---

## 3. Contenido por nivel de usuario

### Nivel 0 — Recién llegado (5 min)
- `tutorials/quickstart.md`
  - `pip install triage-mri`
  - `triage-demo --checkpoint model.ckpt`
  - Subir un .nii.gz → ver score
  - Captura de pantalla de la demo

### Nivel 1 — Usuario que quiere entrenar (30 min)
- `tutorials/installation.md` — requisitos GPU, CUDA, dependencias
- `tutorials/first_training.md` — descargar datos, configurar YAML, lanzar train
- `howto/docker_training.md` — mismo pero en contenedor

### Nivel 2 — Usuario avanzado (integración)
- `howto/deploy_api.md` — Docker Compose con FastAPI
- `howto/multi_sequence.md` — activar T1+T1ce+T2+FLAIR
- `howto/calibrate_model.md` — Platt scaling + thresholds
- `howto/interpret_attention.md` — heatmaps, 3D viewer, pseudo-segmentación

### Nivel 3 — Contribuidor / investigador
- `explanation/architecture.md` — diagramas, flujo de datos
- `explanation/encoder_freeze.md` — principios de diseño
- `explanation/data_licenses.md` — matriz de licencias
- `reference/` — API completa autogenerada

---

## 4. Configuración `conf.py`

Extensiones necesarias:

```python
extensions = [
    "myst_parser",               # Markdown en Sphinx
    "sphinx.ext.autodoc",        # autogen desde docstrings
    "sphinx.ext.autosummary",    # tablas resumen de API
    "sphinx.ext.napoleon",       # Google-style docstrings
    "sphinx.ext.viewcode",       # enlaces al código fuente
    "sphinx.ext.intersphinx",    # enlaces a docs de PyTorch, MONAI
    "sphinx_copybutton",         # botón copiar en code blocks
    "sphinx_design",             # cards, grids, tabs (mejora visual)
    "sphinxcontrib.mermaid",     # diagramas Mermaid inline
    "autodoc2",                  # autodoc moderno (mejor que autodoc)
]

myst_enable_extensions = [
    "colon_fence",               # ::: para admonitions
    "deflist",                   # definition lists
]

html_theme = "furo"              # tema moderno, responsive, dark mode
html_title = "Triage-MRI"
html_logo = "_static/logo.svg"
```

---

## 5. Autogeneración de referencia API

Usando `autodoc2` (moderno, soporta type hints, no requiere `.rst`):

```markdown
<!-- reference/models.md -->
# triagemri.models

```{autodoc2-summary} triagemri.models
```

## Encoder

```{autodoc2-docstring} triagemri.models.encoder
```

## MIL

```{autodoc2-docstring} triagemri.models.mil
```

## Triage Model

```{autodoc2-docstring} triagemri.models.triage
```

## Decoder

```{autodoc2-docstring} triagemri.models.decoder
```
```

Esto genera automáticamente la documentación de todas las clases y funciones
con sus docstrings, type hints, y enlaces al código fuente.

---

## 6. Configuración ReadTheDocs

`.readthedocs.yaml`:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.10"

sphinx:
  configuration: docs/source/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs

formats:
  - pdf
  - epub
```

---

## 7. CI/CD para docs

`.github/workflows/docs.yml`:

```yaml
name: Docs

on:
  push:
    branches: [main, dev]
    paths:
      - "docs/**"
      - "src/triagemri/**"
  pull_request:
    paths:
      - "docs/**"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[docs]"
      - run: sphinx-build -b html docs/source docs/_build -W --keep-going
      - uses: actions/upload-artifact@v4
        with:
          name: docs-html
          path: docs/_build/
```

`-W` convierte warnings en errores → el build falla si una referencia está rota.

---

## 8. Pirámide de navegación en landing page

```markdown
<!-- docs/source/index.md -->
# Triage-MRI

::::{grid} 2
:::{grid-item-card} 🚀 Primeros pasos
:link: tutorials/quickstart
5 minutos para hacer tu primer triaje de RM.
:::
:::{grid-item-card} 🧠 Cómo funciona
:link: explanation/architecture
Arquitectura del sistema, decisiones de diseño.
:::
:::{grid-item-card} 📖 Guías prácticas
:link: howto/index
Cómo desplegar, calibrar, añadir datasets.
:::
:::{grid-item-card} 📚 Referencia API
:link: reference/index
Documentación completa de cada módulo.
:::
::::

---

```{toctree}
:caption: Tutoriales
:hidden:
tutorials/quickstart
tutorials/installation
tutorials/first_training
```

```{toctree}
:caption: Guías
:hidden:
howto/deploy_api
howto/calibrate_model
howto/multi_sequence
howto/interpret_attention
howto/docker_training
howto/add_new_dataset
```

```{toctree}
:caption: Explicación
:hidden:
explanation/architecture
explanation/encoder_freeze
explanation/mil_vs_segmentation
explanation/data_licenses
explanation/design_decisions
```

```{toctree}
:caption: Referencia
:hidden:
reference/config
reference/cli
reference/data
reference/models
reference/training
reference/evaluation
reference/serving
```
```

Con los `:hidden:` los toctrees no se ven en la página pero sí en la sidebar
de navegación de Furo (el tema). La landing page queda limpia con solo las 4
cards de la grid.

---

## 9. Dependencias nuevas en pyproject.toml

```toml
[project.optional-dependencies]
docs = [
    "sphinx>=7.0",
    "furo>=2024",
    "myst-parser>=3.0",
    "autodoc2>=0.5",
    "sphinx-copybutton>=0.5",
    "sphinx-design>=0.5",
    "sphinxcontrib-mermaid>=0.9",
]
```

---

## 10. Plan de implementación (3 agentes)

```
A1: Infraestructura Sphinx + conf.py + tema + CI
    (docs/source/conf.py, .readthedocs.yaml, .github/workflows/docs.yml,
     pyproject.toml [docs deps], docs/Makefile)

A2: Contenido narrativo (tutorials + howto + explanation)
    (docs/source/tutorials/*.md, docs/source/howto/*.md,
     docs/source/explanation/*.md, docs/source/index.md)

A3: Referencia autogenerada + landing + assets
    (docs/source/reference/*.rst, docs/source/_static/,
     verificar docstrings en src/)
```

**A1 y A2/A3 en paralelo** (A1 es infraestructura, A2/A3 es contenido).
A2 y A3 también son independientes entre sí.

---

## 11. Verificación

```bash
# Build local
cd docs && make html

# Sin warnings
sphinx-build -b html docs/source docs/_build -W --keep-going 2>&1 | grep -c WARNING
# Debe ser 0

# Abrir en navegador
open docs/_build/index.html
```
