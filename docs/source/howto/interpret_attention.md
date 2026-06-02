# Interpretar mapas de atención y heatmaps

## Mapa de atención 3×3×3

El MIL head produce 27 pesos de atención (grid 3×3×3) que indican
qué regiones del volumen contribuyen más a la decisión.

```bash
triage-eval --checkpoint model.ckpt --save_attention_maps
```

Exporta mapas NIfTI a `outputs/evaluation/attention_maps/`.

## Heatmap de pseudo-segmentación (decoder activado)

Con el decoder, el modelo produce un heatmap 96³ de anomalía por voxel:

```yaml
model:
  decoder:
    enabled: true
```

El heatmap se entrena con:
- Pérdida de atención: KL(attention_MIL, pool(heatmap))
- Pseudo-labels: BCE por celda a partir de pesos de atención
- Suavizado: Total Variation

## Demo interactiva

```bash
triage-demo --checkpoint model.ckpt
```

Visualización 3D con:
- Superficie cerebral (gris)
- GT tumoral si disponible (rojo)
- Atención/heatmap del modelo (amarillo)
- Cortes 2D axial/sagital/coronal con overlay

## Limitaciones

La resolución real del MIL es de ~32 mm (grid 3³). El decoder la mejora
a ~1 mm (voxel). Los bordes son aproximados — no es segmentación real.
