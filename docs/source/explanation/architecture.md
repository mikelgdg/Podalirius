# Arquitectura del sistema

## Pipeline completo

```
NIfTI (.nii.gz) ──► Preprocessing ──► Encoder ──► MIL Head ──► Decoder ──► Score
   1 canal             (1,96,96,96)    (B,N,768)   (B,1) sigmoid   (B,96³)    [0,1]
```

Cada paso está implementado como un módulo independiente con interfaces definidas en `INTERFACES.md`.

## Encoder: Triad Swin-B SimMIM

| Propiedad | Valor |
|-----------|-------|
| Arquitectura | Swin Transformer 3D, variante Base (~19.8M params) |
| Pre-entrenamiento | SimMIM (masked patch reconstruction) sobre 131K volúmenes |
| Volúmenes pre-entrenamiento | Cerebro, próstata, mama (datos clínicos de Emory) |
| Licencia | MIT |
| Input | `(B, 1, 96, 96, 96)` — un canal, secuencia t1ce por defecto |
| Congelado | `requires_grad=False` para todos los parámetros (19.8M) |
| Output por etapa | 5 pirámides de features multi-escala |

El encoder divide el volumen 96³ en una parrilla de parches 3D. Cada parche se tokeniza y se procesa a través de 5 etapas Swin, cada una reduciendo la resolución espacial y duplicando los canales. Las 5 salidas se proyectan a 768 dimensiones y se concatenan, produciendo `(B, 27, 768)` — un embedding de 768 dimensiones para cada uno de los 27 sub-volúmenes de la parrilla 3×3×3.

### Pirámide de features

| Etapa | Resolución espacial | Canales | Dim tras proyección |
|-------|---------------------|---------|---------------------|
| Stage 1 | 48³ → 24³ | 128 | 768 |
| Stage 2 | 24³ → 12³ | 256 | 768 |
| Stage 3 | 12³ → 6³ | 512 | 768 |
| Stage 4 | 6³ → 3³ | 1024 | 768 |
| Stage 5 | 3³ → 3³ | 1024 | 768 |

Cada etapa se promedia espacialmente para alinearse a la grilla 3×3×3, luego se concatena por canal y se proyecta a 768 dimensiones mediante una capa lineal (`Linear(5×C_stage, 768)`). La proyección no es entrenable — usa la primera sub-matriz de cada etapa.

## Gated Attention MIL

El cabezal MIL es el único componente **entrenable** (525,698 parámetros). Implementa el mecanismo CLAM (Lu et al., 2021):

```
H = Linear(768 → 512)(features)          # proyección a espacio de atención
V = tanh(W_v · H)                        # rama de valor
U = sigmoid(W_u · H)                     # rama de puerta
A = softmax(W · (V ⊙ U))                 # pesos de atención normalizados
Z = Σ(A_i · H_i)                         # embedding agregado por atención
logit = Linear(512 → 1)(Z)               # clasificación final
score = sigmoid(logit)                    # probabilidad de anomalía
```

La puerta (`sigmoid`) permite que el mecanismo suprima instancias no informativas, una mejora sobre la atención estándar (Ilse et al., 2018). El dropout (0.3) se aplica tras la proyección y antes del clasificador.

| Hiperparámetro | Valor |
|----------------|-------|
| `input_dim` | 768 |
| `hidden_dim` | 512 |
| `attention_dim` | 128 |
| `dropout` | 0.3 |

### Multi-Scale MIL

Cuando se activa `mil.multi_scale: true`, el modelo extrae features de **dos** resoluciones:
- Stage 3: grilla 6×6×6 (216 instancias)
- Stage 4: grilla 3×3×3 (27 instancias)

Ambas se procesan con cabezales MIL independientes (compartiendo pesos) y sus logits se fusionan con una puerta aprendida:

```
logit_final = α · logit_s3 + (1-α) · logit_s4
```

Esto mejora la resolución espacial efectiva de ~32 mm a ~16 mm.

## AnomalyDecoder

El decoder es opcional y añade ~240K parámetros entrenables. Produce un heatmap de anomalía por voxel de 96³:

| Bloque | Operación | Resolución |
|--------|-----------|------------|
| UpConv 1 | Upsample + Conv3D + AttentionGate | 3³ → 6³ |
| UpConv 2 | Upsample + Conv3D + AttentionGate | 6³ → 12³ |
| UpConv 3 | Upsample + Conv3D + AttentionGate | 12³ → 24³ |
| UpConv 4 | Upsample + Conv3D + AttentionGate | 24³ → 48³ |
| UpConv 5 | Upsample + Conv3D + AttentionGate | 48³ → 96³ |

Los attention gates (Oktay et al., 2018) usan los pesos de atención MIL como señal de guía para suprimir regiones irrelevantes durante el upsampling. El decoder se supervisa débilmente con:

1. **Pérdida de atención**: `KL(avg_pool(heatmap), attention_MIL)` — alinea el heatmap con la atención MIL
2. **BCE por celda**: usa los 27 pesos de atención MIL como pseudo-labels para cada celda de la grilla
3. **Total Variation**: `TV(heatmap)` — suavizado espacial

## LoRA Adapters

Low-Rank Adaptation (Hu et al., 2021) inyecta matrices entrenables de bajo rango en las proyecciones QKV del mecanismo de self-attention del Swin:

```
W' = W + (B · A) / r
```

Donde `A ∈ R^(d×r)`, `B ∈ R^(r×d)`, con rango `r=4`. Esto añade ~3.5K parámetros entrenables por capa de atención, permitiendo adaptación a la tarea sin modificar los pesos del backbone (19.8M). El encoder permanece congelado excepto por las matrices LoRA.

## Pipeline de calibración

Tras el entrenamiento, el evaluador (`triage-eval --calibrate`) ejecuta:

1. **Inferencia** sobre validation set completo
2. **Búsqueda de threshold**: encuentra `t` tal que `sensitivity(t) ≥ 0.99` maximizando `specificity(t)`
3. **Platt scaling**: ajusta una regresión logística `p_calib = sigmoid(a·logit + b)` sobre los logits para calibrar las probabilidades
4. **Zona inconclusive** (opcional): define un rango intermedio `[t_low, t_high]` donde el modelo se abstiene

## Multi-secuencia

Cuando se configura `multi_sequence.enabled: true`, la primera capa `patch_embed.proj` del encoder pasa de `Conv3D(1→128)` a `Conv3D(N→128)`. Los pesos del canal 0 se inicializan desde Triad (t1ce), y los canales adicionales con inicialización Kaiming. Solo `patch_embed.proj` se descongela; el resto del encoder permanece congelado.

## Recuento de parámetros

| Componente | Parámetros | Entrenable |
|------------|-----------|------------|
| Triad Swin-B | 19.8M | No |
| Proyección multi-etapa | ~0 | No |
| MIL Head | 525,698 | Sí |
| AnomalyDecoder | ~240K | Sí (opcional) |
| LoRA adapters | ~3.5K/capa | Sí (opcional) |
| **Total** | **~20.3M** | **~525K-765K** |
