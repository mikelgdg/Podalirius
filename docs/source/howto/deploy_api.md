# Desplegar API REST

## Docker Compose

```bash
docker compose up api
```

Endpoints disponibles:

- `POST /predict` — triaje de un volumen
- `POST /batch` — triaje por lotes
- `GET /health` — estado del servidor

## Ejemplo

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@brain_t1ce.nii.gz" \
  -F "anatomy=brain"
```

Respuesta:

```json
{
  "score": 0.87,
  "decision": "REVISAR",
  "threshold": 0.42,
  "inference_time_ms": 120.5
}
```

## Sin Docker

```bash
python scripts/serve.py --checkpoint model.ckpt --port 8000
```
