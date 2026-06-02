# Entrenar en Docker

```bash
docker compose build
docker compose run --rm train
```

Monta volúmenes para datos, pesos y salidas:

```yaml
volumes:
  - ./data:/app/data
  - ./weights:/app/weights
  - ./outputs:/app/outputs
```

Para cambiar hiperparámetros, edita `configs/train.yaml` localmente.

## DevContainer (VS Code)

Abre el proyecto en VS Code con la extensión Dev Containers.
El entorno incluye Python 3.10+, PyTorch, MONAI, y extensiones.
