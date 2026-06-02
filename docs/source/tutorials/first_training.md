# Primer entrenamiento

## 1. Preparar datos

Coloca tus datasets en `data/raw/brain/` con la estructura esperada:

```
data/raw/brain/
├── brats/          # BraTS 2023 (opcional)
│   └── BraTS-MEN-00001-000/
├── oasis/          # OASIS-3
├── ixi/            # IXI healthy controls
└── hcp/            # HCP healthy subjects
```

Verifica:

```bash
python scripts/verify_data.py
```

## 2. Configurar

Edita `configs/data.yaml` para apuntar a tus datos.
Edita `configs/train.yaml` para ajustar hiperparámetros.

## 3. Entrenar

```bash
triage-train --output_dir outputs/run_001 --max_epochs 100
```

Progreso en tiempo real en TensorBoard:

```bash
tensorboard --logdir outputs/run_001/logs
```

## 4. Evaluar

```bash
triage-eval --checkpoint outputs/run_001/checkpoints/last.ckpt --calibrate
```

## Modo avanzado

Activar multi-secuencia, decoder, LoRA:

```yaml
# configs/model.yaml
model:
  encoder:
    lora:
      enabled: true
      r: 4
  mil:
    multi_scale: true
  decoder:
    enabled: true
    attention_gates: true
```

Ver [Modo multi-secuencia](../howto/multi_sequence) y [Arquitectura](../explanation/architecture).
