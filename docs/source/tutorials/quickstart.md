# Quickstart — 5 minutos

Instala, descarga un checkpoint, y ejecuta la demo.

## 1. Instalación

```bash
pip install triage-mri
```

O desde fuente:

```bash
git clone https://github.com/user/triage-mri
cd triage-mri
pip install -e .
```

## 2. Descarga el encoder

```bash
bash scripts/download_triad.sh
```

Esto descarga `Triad-SwinB-SimMIM.pth` (~80 MB) a `weights/`.

## 3. Entrena (opcional, requiere GPU)

```bash
triage-train --output_dir outputs/quickstart --max_epochs 5
```

## 4. Lanza la demo

```bash
triage-demo --checkpoint outputs/quickstart/checkpoints/last.ckpt --port 7860
```

Abre http://localhost:7860, sube un `.nii.gz`, y obtén un score de triaje.

## 5. Evalúa

```bash
triage-eval --checkpoint outputs/quickstart/checkpoints/last.ckpt
```

## Siguientes pasos

- [Primer entrenamiento](first_training) — entrenamiento completo con tus datos
- [Desplegar API](../howto/deploy_api) — servir modelo como API REST
