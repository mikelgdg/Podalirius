# CLI — Comandos

## triage-train

Entrena el modelo.

```bash
triage-train [--config_dir CONFIG_DIR] [--output_dir OUTPUT_DIR]
             [--resume_from CHECKPOINT] [--device DEVICE]
             [--seed SEED] [--anatomies ANATOMIES...]
             [--multi_sequence] [--logger {tensorboard,wandb}]
```

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--config_dir` | `configs` | Directorio con YAMLs |
| `--output_dir` | `outputs/default` | Checkpoints y logs |
| `--resume_from` | None | Continuar desde checkpoint |
| `--device` | `cuda` | Dispositivo (cuda / cpu) |
| `--seed` | None | Sobrescribir semilla aleatoria |
| `--anatomies` | `brain` | Anatomías a entrenar |
| `--multi_sequence` | False | Modo multi-canal |
| `--logger` | `tensorboard` | Backend de logging |

## triage-eval

Evalúa un modelo entrenado.

```bash
triage-eval --checkpoint CHECKPOINT [--config_dir CONFIG_DIR]
            [--output_dir OUTPUT_DIR] [--device DEVICE]
            [--target_sensitivity SENS] [--anatomies ANATOMIES...]
            [--calibrate] [--inconclusive] [--subgroup_analysis]
            [--save_attention_maps]
```

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--checkpoint` | *requerido* | Path al checkpoint (.pt/.pth) |
| `--config_dir` | `configs` | Directorio con YAMLs |
| `--output_dir` | `outputs/evaluation` | Resultados de evaluación |
| `--device` | `cuda` | Dispositivo |
| `--target_sensitivity` | `0.99` | Sensibilidad objetivo |
| `--anatomies` | todas | Anatomías a evaluar |
| `--calibrate` | False | Ajustar Platt scaling |
| `--inconclusive` | False | Umbrales de tres zonas |
| `--subgroup_analysis` | False | Métricas por fuente de datos |
| `--save_attention_maps` | False | Exportar mapas de atención NIfTI |

## triage-demo

Lanza la demo interactiva Gradio.

```bash
triage-demo --checkpoint CHECKPOINT [--config_dir CONFIG_DIR]
            [--device DEVICE] [--share] [--port PORT]
            [--host HOST] [--thresholds JSON]
            [--anatomies ANATOMIES...]
```

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--checkpoint` | *requerido* | Path al checkpoint (.pt/.pth) |
| `--config_dir` | `configs` | Directorio con YAMLs |
| `--device` | `cuda` | Dispositivo |
| `--share` | False | Compartir link público Gradio |
| `--port` | `7860` | Puerto |
| `--host` | `0.0.0.0` | Host de bind |
| `--thresholds` | None | JSON con umbrales por anatomía |
| `--anatomies` | todas | Anatomías disponibles |
