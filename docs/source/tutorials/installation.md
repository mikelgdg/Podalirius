# Instalación

## Requisitos

- Python 3.10+
- GPU NVIDIA con ≥8 GB VRAM (recomendado)
- CUDA 11.8+ y cuDNN

## pip

```bash
pip install triage-mri
```

Para desarrollo (tests, linting):

```bash
pip install triage-mri[dev]
```

## Docker

```bash
docker compose build
docker compose run --rm train python -c "import torch; print(torch.cuda.is_available())"
```

## Verificar instalación

```bash
python -c "from triagemri.config import load_config; print('OK')"
pytest tests/ -v
```
