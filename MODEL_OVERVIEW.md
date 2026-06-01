# Triage-MRI — Resumen del Modelo

## Arquitectura

### 1. Encoder (congelado)
- **Swin-B 3D Transformer** pre-entrenado con **SimMIM**
- Input: volumen 3D de `(B, 1, 96, 96, 96)` — un solo canal, secuencia **t1ce** por defecto
- Divide el volumen en una parrilla de **3×3×3 = 27 subvolúmenes** y extrae features multi-escala
- Las 5 etapas del Swin se concatenan y se proyectan a 768 dimensiones
- Output: `(B, 27, 768)` — un embedding por cada región espacial
- **Parámetros: 19.8M — congelados (`frozen: true`)**

### 2. Gated Attention MIL (entrenable)
- Mecanismo de atención con dos ramas paralelas:
  ```
  V = tanh(W_v · H)
  U = sigmoid(W_u · H)
  A = softmax(W · (V ⊙ U))
  Z = Σ(A_i · H_i)
  ```
- `H`: features proyectados a 512 dims
- `attention_dim`: 128
- `dropout`: 0.3
- **Parámetros: 525,698 — entrenables**

### 3. Clasificador
- `Linear(512, 1)` → logit → **sigmoid** → probabilidad de anomalía
- 0 = normal, 1 = anormal (REVISAR)

### Total: 20.3M params / 525K entrenables

---

## ¿Qué es SimMIM?

**Simple Masked Image Modeling**: pre-entrenamiento autosupervisado del encoder.

- Se tapan aleatoriamente ~60% de los parches de volúmenes 3D **normales**
- El modelo reconstruye los píxeles tapados
- No necesita etiquetas, solo volúmenes sanos
- Al reconstruir, aprende la anatomía y textura "normal"
- Cualquier desviación de esa normalidad queda codificada en los features
- El checkpoint `weights/triad_swinb_simmim.pth` se entrenó sobre **150k+ volúmenes** de cerebro, próstata y mama

---

## ¿Qué entrena realmente?

Solo el **cabezal MIL** (attention + classifier). El encoder ya "sabe ver" anatomía normal gracias a SimMIM. El entrenamiento le enseña a **decidir** si lo que ve es patológico.

| Componente     | Parámetros | Entrenable |
|----------------|-----------|------------|
| Swin-B 3D      | 19.8M     | No         |
| Proyección     | ~0        | No         |
| MIL (attention)| 525K      | **Sí**     |
| **Total**      | 20.3M     | 525K       |

---

## Datos de entrenamiento

### Secuencias MRI disponibles (solo se usa t1ce)

| Secuencia | Descripción                                      |
|-----------|--------------------------------------------------|
| **t1**    | T1-weighted — anatomía estructural               |
| **t1ce**  | T1 con contraste (gadolinio) — realza tumores    |
| **t2**    | T2-weighted — edema, inflamación                 |
| **flair** | FLAIR — T2 con supresión de líquido              |

El modelo usa **1 canal** (t1ce), que es la más informativa para tumores cerebrales.

### Datasets (solo cerebro)

| Dataset | Casos | Tipo               |
|---------|-------|--------------------|
| BraTS   | ~600  | Tumores cerebrales |
| OASIS   | ~2500 | Envejecimiento normal / Alzheimer |
| IXI     | 582   | Controles sanos    |

- Split: 70% train / 15% val / 15% test
- Batch size: 4
- Volumen: 96×96×96 voxels normalizado a [0, 1]

---

## Entrenamiento

| Hiperparámetro         | Valor            |
|------------------------|------------------|
| Optimizador            | AdamW            |
| Learning rate          | 5e-4             |
| Weight decay           | 1e-3             |
| Scheduler              | Cosine annealing + warmup (5 epochs) |
| Loss                   | Focal Loss (α=0.25, γ=2.0) |
| Precisión              | 16-mixed (AMP)   |
| Max epochs             | 100              |
| Early stopping         | val/roc_auc, patience=15 |
| GPU                    | RTX 3500 Ada (12 GB) |

---

## Demo (Gradio)

```bash
python scripts/demo.py --checkpoint outputs/run_001/checkpoints/last.ckpt --port 7860
```

La interfaz web (`http://localhost:7860`) permite:
- **Subir** un archivo `.nii.gz`
- Ver **cortes axial, sagital y coronal** del volumen
- Ver el **mapa de atención 3×3**: una cuadrícula que muestra qué región de las 27 tiene más peso
  - Los parches más brillantes indican dónde el modelo "ve" la anomalía
- **Resultado**: NORMAL o REVISAR con el score de confianza

### Atención espacial

La atención `(B, 27, 1)` se reordena a una cuadrícula 3D `(3, 3, 3)`. El slice central de esa cuadrícula se muestra como heatmap 3×3, donde:
- Zonas **oscuras** = atención baja (el modelo ignora esa región)
- Zonas **brillantes** = atención alta (el modelo encuentra anomalía ahí)

---

## Evaluación

```bash
python scripts/evaluate.py --checkpoint outputs/run_001/checkpoints/last.ckpt
```

Calcula sobre el test set (datos nunca vistos):
- **ROC-AUC** global
- **PR-AUC** global
- **Umbral** calibrado al 99% de sensibilidad
