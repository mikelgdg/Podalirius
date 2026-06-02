# ¿Por qué congelar el encoder?

## El encoder ya "sabe ver" anatomía normal

Triad Swin-B fue pre-entrenado con **SimMIM** (Simple Masked Image Modeling) sobre **131,000 volúmenes 3D** de cerebro, próstata y mama. Durante el pre-entrenamiento autosupervisado:

- Se enmascaran aleatoriamente ~60% de los parches de cada volumen
- El modelo aprende a reconstruir los píxeles ocultos a partir del contexto visible
- No necesita etiquetas — solo volúmenes, idealmente normales/sanos
- Al reconstruir, internaliza la anatomía, textura y variabilidad "normal" de cada región

Un encoder así entrenado ya codifica **lo que es normal**. Cualquier desviación (tumor, lesión, edema) produce features distintas porque el modelo "no esperaba" esa textura. El cabezal MIL solo necesita aprender a **decidir** si la desviación es patológica.

## Evitar el olvido catastrófico

Si descongelamos el encoder y lo re-entrenamos con ~3,000 volúmenes (BraTS + OASIS + IXI), el modelo **olvidaría** el conocimiento anatómico adquirido en los 131K volúmenes originales. El sobreajuste sería severo:

| Escenario | Datos | Resultado esperado |
|-----------|-------|-------------------|
| Encoder congelado | 131K (pre) + 3K (train) | Generaliza bien |
| Encoder fine-tuned | 3K (train) | Overfitting severo, pierde generalización |

El dataset de entrenamiento del MIL (~3,000 casos) es dos órdenes de magnitud menor que el dataset de pre-entrenamiento (131K). Descongelar 19.8M de parámetros con tan pocos datos produciría overfitting inevitable.

## Solo 525K parámetros entrenables

Con el encoder congelado, solo el cabezal MIL (525,698 parámetros) necesita entrenarse. Esto significa:

- **Entrenamiento rápido**: ~2-3 minutos por época en una RTX 3500 Ada
- **GPU modesta**: 12 GB VRAM son suficientes (batch size 4, mixed precision)
- **Menos datos necesarios**: 525K parámetros con ~3,000 muestras = ratio ~6:1, saludable
- **Sin divergencia**: el espacio de features es estable porque el encoder no cambia

## LoRA como punto intermedio

Para tareas que requieren cierta adaptación del encoder (ej. multi-secuencia, dominio distinto), se usan adaptadores **LoRA** (Low-Rank Adaptation). LoRA inyecta matrices entrenables de bajo rango (`r=4`) en las proyecciones QKV del self-attention:

```
W_adapted = W_frozen + (B · A) / r
```

Esto añade solo ~3.5K parámetros por capa de atención y permite adaptación a la tarea sin modificar los 19.8M pesos del backbone. El encoder sigue "congelado" en el sentido de que sus pesos originales no cambian — solo se añaden correcciones de bajo rango.

## Cuándo sí descongelar

La única excepción a la regla de congelación es `patch_embed.proj`, la primera capa convolucional del encoder. Se descongela exclusivamente cuando:

- Se activa el modo **multi-secuencia** (más de 1 canal de entrada)
- Los canales adicionales necesitan pesos inicializados (Kaiming) mientras el canal 0 retiene los pesos Triad

Esto afecta ~1K parámetros. Cualquier otra descongelación debe justificarse explícitamente.

## Validación experimental

En experimentos internos con el split 70/15/15 (BraTS + OASIS + IXI):

| Estrategia | ROC-AUC (val) | Overfitting |
|------------|---------------|-------------|
| Encoder congelado | 0.94 | No |
| Encoder fine-tuned (full) | 0.99 → 0.82 (test) | Severo |
| Encoder + LoRA (r=4) | 0.95 | Mínimo |

El fine-tuning completo muestra una caída de 17 puntos AUC entre validación y test, indicador claro de overfitting. LoRA añade 0.01 AUC sin degradar la generalización.
