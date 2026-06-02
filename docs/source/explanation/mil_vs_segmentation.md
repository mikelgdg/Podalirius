# MIL vs Segmentación

## ¿Qué es MIL?

**Multiple Instance Learning** clasifica **bolsas** (bags) de instancias, no instancias individuales. En Triage-MRI:

- **Bolsa**: un volumen 3D completo (96×96×96 voxels)
- **Instancias**: 27 sub-volúmenes de la parrilla 3×3×3 (~32×32×32 mm cada uno)
- **Etiqueta de bolsa**: 0 (normal) o 1 (anormal)
- **Etiquetas de instancia**: **desconocidas** — no sabemos qué sub-volúmenes contienen patología

El modelo aprende a asignar pesos de atención a cada instancia. Una bolsa es anormal si al menos una instancia tiene atención alta y features anómalos. Esto es supervisión débil: no necesitamos máscaras de segmentación.

## Limitaciones del MIL estándar

### Resolución espacial baja (~32 mm)

Con 27 instancias en un volumen de 96³ voxels, cada celda de la grilla cubre ~32×32×32 mm de tejido. Una lesión pequeña (<10 mm) puede ocupar solo una fracción de una celda, diluyendo su señal entre tejido normal circundante.

### Atención ≠ Segmentación

Los pesos de atención indican **contribución a la decisión de clasificación**, no presencia de patología. Una región con atención alta puede ser:
- Realmente patológica (el modelo detecta anomalía)
- Un artefacto de imagen que el modelo asocia con patología
- Una región anatómicamente variable que correlaciona con la etiqueta

La atención **no es** una máscara de segmentación. Es una señal de importancia para la tarea de clasificación.

### Sin localización fina

El MIL estándar no puede decir "el tumor está en el lóbulo frontal izquierdo, coordenadas (45, 32, 18)". Solo puede decir "la región superior-derecha-anterior del volumen es sospechosa".

## Multi-Scale MIL: mejora parcial

El **Multi-Scale MIL** extrae features en dos resoluciones:
- Stage 3: grilla 6×6×6 = 216 instancias (~16 mm)
- Stage 4: grilla 3×3×3 = 27 instancias (~32 mm)

La fusión con puerta (`α·logit_s3 + (1-α)·logit_s4`) mejora la resolución efectiva a ~16 mm, pero sigue siendo insuficiente para lesiones pequeñas.

## AnomalyDecoder: hacia ~1 mm

El **AnomalyDecoder** es un decoder ligero (5 bloques UpConv + AttentionGate) que produce un heatmap de 96³ voxels (~1 mm). Se entrena con supervisión débil:

1. **Pérdida de atención**: `KL(avg_pool(heatmap), attention_MIL)` — el heatmap, al promediarse a 3³, debe parecerse a la atención MIL
2. **BCE por celda**: los 27 pesos de atención se usan como pseudo-labels para clasificar cada celda
3. **Total Variation**: suavizado espacial para evitar mapas ruidosos

El decoder **no** produce segmentación real. Sus bordes son aproximados y su resolución es artificial (interpolación). Pero ofrece una localización mucho más fina que el MIL puro.

## ¿Por qué no usar segmentación directamente?

### No hay máscaras para todos los datasets

| Dataset | Tiene máscaras de segmentación |
|---------|-------------------------------|
| BraTS | Sí (tumor, edema, necrosis) |
| OASIS | No (solo etiquetas clínicas) |
| IXI | No (controles sanos) |
| HCP | No (controles sanos) |
| PI-CAI | Sí (lesiones de próstata) |

Menos del 20% de los casos tienen ground truth de segmentación. Entrenar un modelo de segmentación requeriría descartar el 80% de los datos o anotar manualmente, lo cual es prohibitivo.

### La tarea es triaje, no diagnóstico

El objetivo es **descartar normalidad**: identificar volúmenes que el radiólogo puede ignorar con seguridad. No se necesita delinear bordes tumorales con precisión milimétrica. La pregunta binaria "¿hay algo anormal?" es suficiente para el triaje.

### Eficiencia de datos

El MIL con supervisión débil usa todas las muestras disponibles (~3,000), mientras que un enfoque de segmentación solo usaría las que tienen máscaras (~600). Con 5× más datos, el MIL aprende mejor la frontera de decisión normal/anormal.

## Comparativa

| Aspecto | MIL (estándar) | Multi-Scale MIL | AnomalyDecoder |
|---------|---------------|-----------------|----------------|
| Resolución | ~32 mm | ~16 mm | ~1 mm |
| Instancias | 27 | 216 | 884,736 voxels |
| Parámetros | 525K | 525K × 2 | +240K |
| Necesita máscaras | No | No | No |
| Interpretabilidad | 3³ grid | 6³ grid | 96³ heatmap |
| Tipo de salida | Atención | Atención | Heatmap (pseudo-seg) |
