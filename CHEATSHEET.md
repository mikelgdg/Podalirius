# Triage-MRI Cheatsheet

## Modelos actuales (post-rediseño, Jun 2026)

| Modelo | Anatomías | AUC val | Espec (sens 99%) | Discard | Checkpoint |
|--------|-----------|---------|-------------------|---------|------------|
| **Post-rediseño** | Cerebro | 0.983 | 77.2% | 33.0% | `outputs/PRIMERA_PRUEBA_POST_REDISENO/checkpoints/last.ckpt` |
| **V3** | Cerebro + Próstata | 0.984 | 77.2% | 33.0% | `outputs/BRAIN_PROSTATE_V3/checkpoints/last.ckpt` |

## Entrenamiento

```bash
# Solo cerebro
python scripts/train.py --output_dir outputs/run_nueva --anatomies brain

# Cerebro + próstata
python scripts/train.py --output_dir outputs/run_nueva --anatomies brain prostate

# Resumir desde checkpoint (auto-detecta output_dir)
python scripts/train.py --resume_from outputs/PRIMERA_PRUEBA_POST_REDISENO/checkpoints/last.ckpt

# Cambiar seed
python scripts/train.py --seed 123 --output_dir outputs/run_exp_42
```

## TensorBoard

```bash
# Ver todas las runs juntas
tensorboard --logdir outputs --bind_all --port 6006
```

## Evaluación

```bash
# Post-rediseño (solo cerebro)
python scripts/evaluate.py --checkpoint outputs/PRIMERA_PRUEBA_POST_REDISENO/checkpoints/last.ckpt \
    --calibrate --subgroup_analysis

# V3 (cerebro + próstata)
python scripts/evaluate.py --checkpoint outputs/BRAIN_PROSTATE_V3/checkpoints/last.ckpt \
    --calibrate --subgroup_analysis

# Con sensibilidad distinta a 99%
python scripts/evaluate.py --checkpoint outputs/PRIMERA_PRUEBA_POST_REDISENO/checkpoints/last.ckpt \
    --target_sensitivity 0.95

# Output: outputs/evaluation/evaluation_results.json + evaluation_report.md
```

## Demo (Gradio)

```bash
# Post-rediseño — cerebro
python scripts/demo.py --checkpoint outputs/PRIMERA_PRUEBA_POST_REDISENO/checkpoints/last.ckpt

# V3 — cerebro + próstata
python scripts/demo.py --checkpoint outputs/BRAIN_PROSTATE_V3/checkpoints/last.ckpt --anatomies brain prostate

# Con thresholds de evaluación (zonas NORMAL/INCONCLUSIVE/REVISAR)
python scripts/demo.py --checkpoint outputs/PRIMERA_PRUEBA_POST_REDISENO/checkpoints/last.ckpt \
    --thresholds outputs/evaluation/thresholds.json

# Acceso público (compartir link)
python scripts/demo.py --checkpoint outputs/PRIMERA_PRUEBA_POST_REDISENO/checkpoints/last.ckpt --share

# Puerto y host custom
python scripts/demo.py --checkpoint outputs/BRAIN_PROSTATE_V3/checkpoints/last.ckpt \
    --port 8080 --host 127.0.0.1 --anatomies brain prostate
```

## Tests

```bash
pytest tests/ -v
pytest tests/test_triage.py -v          # Modelo Triage
pytest tests/test_trainer.py -v         # Lightning module
pytest tests/test_encoder.py -v         # Encoder Swin-B
pytest tests/test_mil.py -v             # Attention MIL
pytest tests/test_preprocessing.py -v   # Preprocesado
```

## Debug / Inspección

```bash
# Ver un caso BraTS (abre visualización en navegador)
python scripts/view_brats.py data/raw/brain/brats/BraTS-MEN-00004-000

# Guardar a HTML
python scripts/view_brats.py data/raw/brain/brats/BraTS-MEN-00004-000 --out caso.html

# Monitor de entrenamiento (lee TensorBoard, refresca cada 30s)
python scripts/monitor.py

# Ver logs de una run
cat outputs/run_brain_prostate.log
tail -f outputs/run_brain_prostate/logs/version_0/events.out.tfevents.*
```

## Descarga de datos / pesos

```bash
bash scripts/download_triad.sh                           # Pesos Triad Swin-B
bash scripts/download_data.sh --check                    # Verificar datasets
python scripts/download_brats.py --auth-token TOKEN       # BraTS 2023 (necesita Synapse)
bash scripts/extract_hcp.sh                               # Extraer HCP de zips
```
