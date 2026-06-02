# Calibrar thresholds y Platt scaling

## Threshold para 99% sensibilidad

```bash
triage-eval --checkpoint model.ckpt --calibrate
```

Esto:
1. Ejecuta inferencia en validation set
2. Encuentra el threshold que da ≥99% sensibilidad con máxima especificidad
3. Ajusta Platt scaling para calibrar scores

## Zona inconclusive

```bash
triage-eval --checkpoint model.ckpt --calibrate --inconclusive
```

Produce 3 zonas:
- Score < 0.25 → NORMAL
- 0.25 ≤ Score < 0.40 → INCONCLUSIVE (prioridad baja)
- Score ≥ 0.40 → REVISAR

Ajusta los umbrales en `configs/train.yaml`:

```yaml
evaluation:
  inconclusive_range: [0.25, 0.40]
```

## Subgrupos

```bash
triage-eval --checkpoint model.ckpt --subgroup_analysis
```

Métricas desglosadas por dataset de origen (BraTS, OASIS, IXI).
