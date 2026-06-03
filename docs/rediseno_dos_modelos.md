# Propuesta de rediseño: Arquitectura de dos modelos

> Documento para evaluación de ingeniería. Basado en 27 épocas de evidencia experimental
> con tres arquitecturas (V3, V4, V5, V6) sobre 3,004 casos brain+prostate.
> Contexto completo en [`docs/evolucion_proyecto.md`](evolucion_proyecto.md).

---

## 1. El problema observado

Tras 4 iteraciones de arquitectura multi-tarea (encoder compartido → MIL + decoder),
la evidencia es concluyente:

| Arquitectura | AUC (época 12) | AUC (época 27) | Params entrenables | ¿Converge? |
|---|---|---|---|---|
| **V3** (solo MIL, sin decoder) | **0.984** | convergido | 5.2M | ✅ 12 épocas |
| V4 (MIL + decoder, pseudo-labels) | 0.880 | estancado | 8.1M | ❌ |
| V5 (MIL + decoder, GT masks, pesos altos) | 0.870 | estancado | 8.1M | ❌ |
| **V6** (MIL + decoder, GT masks, warmup 5ép, pesos 0.1) | 0.894 | **0.893** | 8.1M | ⬆ lento (~0.005/ép) |

**V6 es la mejor versión multi-tarea**, pero sigue 0.09 AUC por detrás de V3 a la misma época.
La causa raíz es física del gradiente: el encoder recibe señales contradictorias del MIL
("clasifica") y del decoder ("segmenta"). Por muchos pesos y warmups que se apliquen,
la competencia en el backbone compartido impone un techo a la clasificación.

V3, sin decoder, dedica el 100% del gradiente del encoder a clasificar. Converge en 12 épocas
a 0.984 AUC con 33% de discard rate manteniendo >99% de sensibilidad.

---

## 2. Propuesta: dos modelos independientes

```
┌─────────────────────────────────────────────────────────┐
│                    Entrada: volumen 96³                  │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐     ┌──────────────────────────┐
│   Modelo A (V3)  │     │      Modelo B (Seg)      │
│   Clasificación  │     │      Segmentación        │
│                  │     │                          │
│ Triad congelado  │     │ Triad congelado          │
│       ↓          │     │       ↓                  │
│ GatedAttention   │     │ Decoder 3D + skip conns │
│   MIL (243 tok)  │     │   (AnomalyDecoder)       │
│       ↓          │     │       ↓                  │
│  score [0,1]     │     │  heatmap 96³ [0,1]      │
│       ↓          │     │                          │
│ NORMAL/REVISAR   │     │  (solo donde hay GT)    │
└──────────────────┘     └──────────────────────────┘
   Entrenado con:            Entrenado con:
   TODOS los casos           SOLO casos con máscara GT
   (brain+prostate)          (BraTS 100%, PI-CAI 52%)
   Loss: weighted_bce        Loss: BCE + Dice
```

### Modelo A — Clasificación pura (hereda de V3)

- **Arquitectura:** Triad Swin-B congelado → GatedAttentionMIL multi-escala (243 tokens) → score
- **Entrenamiento:** Todos los casos (3,004 brain+prostate). Sin decoder, sin segmentación.
- **Loss:** `weighted_bce` (sin competencia de gradiente)
- **Métricas esperadas:** AUC ≥0.98, especificidad ≥75%, discard rate ≥30% a sensibilidad 99%
- **Rol en producción:** Decide NORMAL/REVISAR. Es el modelo de triaje.

### Modelo B — Segmentación supervisada (nuevo)

- **Arquitectura:** Triad Swin-B congelado → AnomalyDecoder 3D con skip connections → heatmap 96³
- **Entrenamiento:** Solo casos con máscara GT real.
  - Cerebro: BraTS (~1,000 casos, 100% con segmentación de tumor)
  - Próstata: PI-CAI positivos con delineación de experto (~220 casos)
- **Loss:** `BCEWithLogitsLoss + soft Dice` sobre el heatmap vs máscara GT
- **Métricas esperadas:** Dice ≥0.6 en tumor, activación <0.05 en normales
- **Rol en producción:** Genera heatmap explicativo cuando el Modelo A dice REVISAR

---

## 3. Justificación técnica

### 3.1 Separación de concerns

El gradiente de clasificación y el de segmentación tienen naturalezas opuestas:

| Señal | Escala | Ruido | Óptimo de convergencia |
|-------|-------|-------|------------------------|
| Clasificación | 1 bit por volumen | Bajo (label clínico fiable) | 10-15 épocas |
| Segmentación | 884K voxels por volumen | Alto (variabilidad inter-anotador, solo 52% de PI-CAI) | 30-50 épocas |

Juntas se estorban. Separadas, cada una converge a su propio óptimo.

### 3.2 Evidencia experimental

- **V3 (solo clasificación):** 0.984 AUC en 12 épocas.
- **V6 (mejor multi-tarea):** 0.893 AUC en 27 épocas, subiendo a ~0.005/época. Proyectado a 50+ épocas para alcanzar 0.95+.
- **Decoder sin MIL (no entrenado aún):** El decoder entrenado solo con GT masks no tiene evidencia experimental propia, pero es la arquitectura estándar de segmentación 3D supervisada (nnU-Net, Swin UNETR) y no hay razón para esperar que falle.

### 3.3 Coste computacional

| | V6 (multi-tarea) | Propuesta (A + B) |
|---|---|---|
| Parámetros por modelo | 26.7M | 23.8M (A) + 3.5M (B) |
| Épocas para converger | 50+ (estimado) | 15 (A) + 40 (B) |
| VRAM por modelo | ~12 GB | ~6 GB (A) + ~6 GB (B) |
| ¿Pueden correr en paralelo? | No (uno solo llena la GPU) | Sí (caben los dos) |

### 3.4 Ventajas operativas

1. **Independencia:** Si el Modelo A falla, el B sigue funcionando (y viceversa). En producción, el pipeline tiene alta disponibilidad.
2. **Actualización granular:** Si se añade un dataset nuevo con máscaras, solo se reentrena B. Si se añade una anatomía nueva sin máscaras, solo se reentrena A.
3. **Explicabilidad delegada:** El Modelo A no necesita ser interpretable. El Modelo B se especializa en explicar.
4. **Validación separada:** Las métricas de clasificación y segmentación no se contaminan mutuamente.

---

## 4. Datos necesarios

### Modelo A (ya validado con V3)

| Anatomía | Fuente | Train | Val | Test |
|----------|--------|-------|-----|------|
| Cerebro | BraTS + OASIS + IXI + HCP | 1,954 | 430 | 430 |
| Próstata | PI-CAI | 1,050 | 225 | 225 |
| **Total** | | **3,004** | **655** | **655** |

### Modelo B (por validar)

| Anatomía | Fuente | Train | Val | Test | Cobertura máscara |
|----------|--------|-------|-----|------|-------------------|
| Cerebro | BraTS (solo positivos) | ~990 | ~207 | ~202 | 100% |
| Próstata | PI-CAI (positivos con delineación) | ~155 | ~33 | ~32 | 52% del total |
| **Total** | | **~1,145** | **~240** | **~234** | |

---

## 5. Riesgos identificados

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Modelo B con pocos datos de próstata (solo 52% con máscara) | Alta | Usar aumento 3D agresivo (RandAffine, RandElastic). Evaluar si whole-gland masks de PI-CAI (disponibles para todos) pueden servir como ROI gating |
| El decoder no generaliza a anatomías sin máscara | Media | Entrenar B solo en cerebro inicialmente. Extender a próstata si los resultados lo justifican |
| Dos modelos = doble mantenimiento | Baja | Comparten encoder congelado. Solo cambia la cabeza. Mismo preprocesado, mismo config loader |
| Latencia en producción (dos forward passes) | Baja | El Modelo B solo se ejecuta si A dice REVISAR (~67% de casos). No se ejecuta para NORMAL |

---

## 6. Plan de implementación

| Fase | Duración | Entregable |
|------|----------|------------|
| 1. Extraer Modelo A de V3 | 1 día | Refactorizar V3 como `TriageClassifier` independiente. Validar que reproduce AUC 0.984 |
| 2. Entrenar Modelo B (cerebro) | 2-3 días | Decoder solo con BraTS. Evaluar Dice en tumor. Visualizar heatmaps |
| 3. Entrenar Modelo B (próstata) | 2-3 días | Extender a PI-CAI con máscaras. Evaluar rendimiento con datos limitados |
| 4. Integrar pipeline | 1 día | API que ejecuta A → si REVISAR → ejecuta B → devuelve score + heatmap |
| 5. Demo unificada | 1 día | Interfaz Gradio con selector de caso, clasificación + heatmap lado a lado |

---

## 7. Conclusión

La arquitectura multi-tarea con encoder compartido (V6) tiene mérito teórico pero está limitada
en la práctica por la competencia de gradiente. La evidencia de 27 épocas muestra que, incluso
con warmup y pesos reducidos, el decoder lastra la clasificación en ~0.09 AUC respecto a V3.

La propuesta de dos modelos independientes elimina este conflicto: cada cabeza recibe el 100%
del gradiente del encoder para su tarea. El coste añadido es mínimo (mismo encoder congelado,
doble entrenamiento secuencial) y las ventajas operativas (independencia, actualización granular,
validación separada) son significativas.

**Recomendación:** Adoptar la arquitectura de dos modelos. El Modelo A (V3) ya está validado
con AUC 0.984. El Modelo B requiere desarrollo pero parte de componentes ya implementados
(AnomalyDecoder, load_mask, MultiTaskLoss adaptado a solo segmentación).

---

*Documento generado el 3 de junio de 2026. Para contexto histórico completo,
ver [`docs/evolucion_proyecto.md`](evolucion_proyecto.md).*
