# Changelog — Triage-MRI

## 2026-06-01 — Ronda 0: Infraestructura (auditoría bugs)
- **Archivos modificados:** `src/triagemri/config.py`, `configs/train.yaml`, `configs/model.yaml`, `configs/data.yaml`, `scripts/train.py`, `tests/test_preprocessing.py`, `tests/test_triage.py`, `tests/test_trainer.py`
- **Qué se hizo:**
  - `config.py`: Reemplazado `dict.update()` shallow merge por `_deep_merge()` recursivo para evitar colisiones silenciosas entre YAMLs. Añadida validación de schema (`_CONFIG_SCHEMA`, `_CONFIG_VALUE_CHECKS`) con keys requeridas y rangos. Añadido `__contains__` y `keys()` a Config. `Config.get()` ahora distingue `None` real de key ausente mediante sentinel.
  - `train.yaml`: `save_top_k: -1` → `3`. Añadido `deterministic: true`. Eliminado `gradient_accumulation_steps: 1` (dead config).
  - `model.yaml`: Eliminadas keys muertas `architecture` y `num_attention_branches`.
  - `data.yaml`: Keys `processed_dir`, `voxel_spacing`, `intensity_percentiles`, `sequences`, `stratify_by` marcadas como PLANNED.
  - `train.py`: `multiprocessing.set_start_method("spawn")` movido al inicio de `main()` (antes de imports de torch). Añadido `deterministic=` al `pl.Trainer`.
  - Tests: Corregido `"frozen"` → `"freeze_backbone"` en tests. Corregido `test_crop_larger` para respetar smart-crop en eje D.
- **Por qué:** Hallazgos C11, F0.1-F0.6 del plan de auditoría.
- **APIs afectadas:** `load_config()` ahora valida schema. `Config()` soporta `__contains__`.

## 2026-06-01 — Ronda 1: Datos (labels + preprocessing + datasets + transforms)
- **Archivos modificados:** `src/triagemri/data/labels.py`, `src/triagemri/data/preprocessing.py`, `src/triagemri/data/datasets.py`, `src/triagemri/data/transforms.py`
- **Qué se hizo:**
  - `labels.py`: Umbral mínimo de 10 voxels en BraTS. OASIS: NaN/missing CDR → -1 (skip). PI-CAI: solo YES/NO válidos, resto → -1. `"d" in path.name` → `"oasis" in path.name.lower()`. Negación scoping (normales primero). Patrones añadidos (metastasis, hemorrhagic).
  - `preprocessing.py`: NaN/Inf detection antes de `np.quantile`. Guarda de volumen constante con `np.isfinite`. `load_full_volume_for_display`: fix spatial axis swap (transpose antes/después de interpolate). `crop_offsets` devuelve `(d0,h0,w0)`. `validate_file` usa `dataobj` en vez de cargar volumen completo.
  - `datasets.py`: PI-CAI `len(parts)>=3` → `>=2` + manejo de `.nii` en stem. Cache key incluye `sources` y `sequences`. Missing seg → `label=-1` + skip. `ValueError` de seg corrupto capturado. Volumen corrupto → `RuntimeError`. `oasis_path` configurado usado para escaneo. Split de próstata con `stratify`.
  - `transforms.py`: `RandFlipd([0,1,2])` → 3 llamadas independientes por eje. Eliminado `Resized` redundante. Añadido `ScaleIntensityd` clamp tras shift/scale.
- **Por qué:** Hallazgos C4, C5, C6, C9, C10, B1-B10, H1-H7 del plan de auditoría.
## 2026-06-01 — Rondas 2-3-4: Modelos + Training + Eval/Demo/Scripts (auditoría bugs)
- **Archivos modificados:** `src/triagemri/models/encoder.py`, `src/triagemri/models/triage.py`, `src/triagemri/training/losses.py`, `src/triagemri/training/metrics.py`, `src/triagemri/training/trainer.py`, `src/triagemri/evaluation/evaluator.py`, `src/triagemri/demo/app.py`, `scripts/train.py`, `scripts/evaluate.py`
- **Qué se hizo:**
  - `encoder.py`: `bare except Exception` → `(FileNotFoundError, RuntimeError)`. `_freeze()` llama `self.swin.eval()`. `stage_dims` calculado dinámicamente con dummy forward en vez de hardcodeado.
  - `triage.py`: `get_attention_heatmap`/`predict` aceptan parámetro `anatomy` para multi-head. Grid no cúbico → warning + flat. `build_triage_model` infiere `embed_dim` del checkpoint si falta config. Key remapping maneja `module.` (DDP). Multi-head infiere anatomías del checkpoint.
  - `losses.py`: `WeightedBCEWithLogitsLoss` usa `pos_weight` estático del dataset, no por batch. FocalLoss `pt` en fp32. Eliminado alias `"bce"`.
  - `metrics.py` (CRITICAL): `np.argmax` → `matches[-1]` para seleccionar threshold con máxima especificidad. ECE con binning por cuantiles. ROC-AUC single-class → NaN.
  - `trainer.py`: `all_gather()` en multi-GPU. `finally` para limpiar buffers. `T_max` corregido. Bias/LayerNorm excluidos de weight decay. Warmup LR empieza en 0.
  - `evaluator.py`: Thresholds en validation set (no test set → data leakage fix). `run_evaluation` acepta `enabled_anatomies`. `json.dump` con `default=str`.
  - `evaluate.py`: Format-string crash fix. Añadido `--anatomies`. `pl.seed_everything`.
  - `train.py`: Warning si `--anatomies` ≠ checkpoint al resumir. Añadido `--logger`.
  - `demo/app.py` (CRITICAL): Axial/Sagital intercambiados corregidos. Grid no cúbico muestra mensaje. Upload no ignorado. Paths con `TRIAGEMRI_ROOT`. Guard `case_dir is not None`.
- **Por qué:** Hallazgos C1, C2, C3, C7, C8, C12, H8-H29. El argmax bug hacía que el sistema de triaje no triara nada (especificidad ~0%). El data leakage inflaba métricas artificialmente.
- **APIs afectadas:** `find_threshold_for_sensitivity` ahora devuelve threshold óptimo. `WeightedBCEWithLogitsLoss` requiere `pos_weight` en constructor. `full_evaluation` acepta `val_loader`. `get_attention_heatmap`/`predict` aceptan `anatomy`.
