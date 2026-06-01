# Triage-MRI Cheatsheet

## Entrenamiento

```bash
# Solo cerebro (default)
python scripts/train.py --output_dir outputs/default

# Cerebro + próstata
python scripts/train.py --output_dir outputs/run_brain_prostate --anatomies brain prostate

# Resumir desde checkpoint
python scripts/train.py --output_dir outputs/run_brain_prostate --resume_from outputs/run_brain_prostate/checkpoints/last.ckpt --anatomies brain prostate

# Cambiar seed y max epochs (editar configs/train.yaml antes)
python scripts/train.py --seed 123 --output_dir outputs/run_exp_42
```

## TensorBoard

```bash
# Ver todas las runs
tensorboard --logdir outputs --bind_all --port 6006

# Solo una run
tensorboard --logdir outputs/run_brain_prostate --bind_all --port 6006
```

## Evaluación

```bash
# Evaluar un checkpoint (métricas en test set)
python scripts/evaluate.py --checkpoint outputs/run_brain_prostate/checkpoints/last.ckpt

# Con sensibilidad distinta a 99%
python scripts/evaluate.py --checkpoint outputs/run_brain_prostate/checkpoints/last.ckpt --target_sensitivity 0.95

# Output: outputs/evaluation/evaluation_results.json + evaluation_report.md
```

## Demo (Gradio)

```bash
# Demo con checkpoint de cerebro+solo
python scripts/demo.py --checkpoint outputs/run_brain_prostate/checkpoints/last.ckpt

# Solo brain + próstata en el dropdown (sin breast)
python scripts/demo.py --checkpoint outputs/run_brain_prostate/checkpoints/last.ckpt --anatomies brain prostate

# Con thresholds de evaluación
python scripts/demo.py --checkpoint outputs/run_brain_prostate/checkpoints/last.ckpt \
    --thresholds outputs/evaluation/thresholds.json

# Acceso público
python scripts/demo.py --checkpoint outputs/run_brain_prostate/checkpoints/last.ckpt --share

# Puerto y host custom
python scripts/demo.py --checkpoint outputs/run_brain_prostate/checkpoints/last.ckpt --port 8080 --host 127.0.0.1
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
