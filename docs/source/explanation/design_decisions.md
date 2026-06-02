# Decisiones de diseño

Cada decisión arquitectónica en Triage-MRI responde a restricciones concretas de datos, cómputo y objetivo clínico.

## ¿Por qué MIL en lugar de segmentación?

**Restricción**: menos del 20% de los casos tienen máscaras de segmentación. El 80% restante solo tiene etiquetas binarias (normal/anormal) derivadas de metadatos clínicos.

**Decisión**: MIL con supervisión débil. Clasifica el volumen completo usando solo etiquetas de bolsa, sin necesidad de anotaciones pixel a pixel. Esto permite usar todos los datos disponibles (~3,000 casos) en lugar de descartar el 80%.

**Alternativa rechazada**: segmentación supervisada (UNet, nnUNet). Requeriría descartar OASIS, IXI y HCP (sin máscaras) y anotar manualmente, lo cual es prohibitivo en coste y tiempo.

## ¿Por qué encoder congelado en lugar de fine-tuning?

**Restricción**: el encoder tiene 19.8M de parámetros pero el dataset de entrenamiento del MIL solo tiene ~3,000 casos. La ratio parámetros/datos es ~6,600:1 para fine-tuning completo.

**Decisión**: congelar el encoder. Solo se entrenan 525K parámetros (MIL head), ratio ~6:1. El encoder ya codifica anatomía normal gracias a SimMIM sobre 131K volúmenes.

**Evidencia**: en experimentos internos, el fine-tuning completo produce overfitting severo (ROC-AUC cae de 0.99 en validación a 0.82 en test). El encoder congelado mantiene 0.94 en ambos.

## ¿Por qué atención en lugar de mean pooling?

**Restricción**: en una bolsa de 27 instancias, típicamente solo 1-3 contienen patología. El mean pooling diluye la señal anómala entre 24+ instancias normales.

**Decisión**: attention pooling. Los pesos de atención permiten que el modelo se enfoque en las pocas instancias relevantes, ignorando el resto. Además, los pesos de atención proporcionan **interpretabilidad**: podemos ver qué regiones del volumen contribuyeron a la decisión.

**Alternativa rechazada**: mean pooling (promedio simple de features). Funciona para tareas donde la mayoría de instancias son informativas, pero falla cuando la señal es escasa. Max pooling es ruidoso y no permite interpretabilidad.

## ¿Por qué gated attention (CLAM) en lugar de atención estándar?

**Restricción**: la atención estándar (Ilse et al., 2018) usa `softmax(W·tanh(V))`. La función tanh es aproximadamente lineal cerca de cero, lo que dificulta suprimir instancias no informativas.

**Decisión**: gated attention (CLAM, Lu et al., 2021). Añade una puerta sigmoide: `softmax(W·(tanh(V) ⊙ sigmoid(U)))`. La puerta permite al modelo **suprimir** instancias (multiplicar por ~0) en lugar de solo atenuarlas. Esto es crítico cuando 24/27 instancias son tejido normal.

**Evidencia**: CLAM reporta mejoras de 2-5% AUC sobre atención estándar en benchmarks de patología digital. En Triage-MRI, la diferencia es de ~3% AUC en validación cruzada.

## ¿Por qué proyección por etapa en lugar de concatenación simple?

**Restricción**: las 5 etapas del Swin producen features con dimensionalidades distintas (128, 256, 512, 1024, 1024). Concatenarlas directamente produce features de 3,072 dimensiones, demasiado para el MIL head (525K params).

**Decisión**: proyección por etapa a 768 dimensiones, luego promedio. Cada etapa se proyecta independientemente (`Linear(C_stage, 768)`) y se promedian las 5 proyecciones. Esto preserva información multi-escala en un espacio de dimensionalidad fija.

**Alternativa rechazada**: concatenación de 3,072 dimensiones. Triplicaría los parámetros del MIL head (~1.5M) sin beneficio claro. Usar solo la última etapa pierde información de bordes y texturas finas de etapas tempranas.

## ¿Por qué LoRA en lugar de fine-tuning completo?

**Restricción**: ciertas tareas (multi-secuencia, dominio muy distinto) requieren adaptar el encoder, pero el fine-tuning completo causa overfitting.

**Decisión**: LoRA (Low-Rank Adaptation, r=4). Inyecta ~3.5K parámetros entrenables por capa de atención como correcciones de bajo rango a las proyecciones QKV. El backbone (19.8M) permanece intacto.

**Alternativa rechazada**: fine-tuning parcial (descongelar últimas N capas). Requiere ajustar N manualmente por tarea y no hay garantía de que las capas correctas sean las últimas. Prompt tuning no es aplicable a arquitecturas Swin.

## ¿Por qué Platt scaling en lugar de temperature scaling?

**Restricción**: los scores del modelo no están calibrados — un score de 0.7 no significa 70% de probabilidad real de anomalía. Se necesita calibrar para que los thresholds clínicos sean interpretables.

**Decisión**: Platt scaling (`p_calib = sigmoid(a·logit + b)`). A diferencia de temperature scaling (un solo parámetro T), Platt scaling ajusta tanto la pendiente como el offset de la curva de calibración. Esto es importante porque el modelo tiende a estar sobre-confianzado en scores altos y sub-confiado en scores bajos.

**Alternativa rechazada**: temperature scaling. Solo corrige la "agudeza" de las predicciones, no el sesgo sistemático. Isotonic regression es más flexible pero propensa a overfitting con pocos datos de calibración.

## ¿Por qué 96³ de input?

**Restricción**: el encoder Triad fue pre-entrenado con volúmenes de 96×96×96 voxels. Cambiar la resolución de entrada requiere re-entrenar el encoder o usar interpolación que degrada los features.

**Decisión**: 96³ fijo. Todos los volúmenes se redimensionan (con preservación de aspect ratio y padding) a 96³ durante el preprocesamiento. Esto garantiza compatibilidad con los pesos pre-entrenados.

**Trade-off**: 96³ es una resolución baja para anatomía detallada (~1 mm isotrópico para un cerebro de ~96 mm). Pero es la resolución nativa del encoder y suficiente para la tarea de triaje (descartar normalidad, no diagnosticar).

## Resumen de trade-offs

| Decisión | Ganancia | Coste |
|----------|----------|-------|
| MIL sobre segmentación | Usa todos los datos, no necesita máscaras | Resolución espacial baja (~32 mm) |
| Encoder congelado | Sin overfitting, GPU modesta | No se adapta a dominios muy distintos |
| Atención sobre mean pooling | Señal no se diluye, interpretable | Más parámetros (525K vs 0) |
| Gated sobre estándar | Suprime instancias irrelevantes | +50% parámetros en atención |
| Proyección por etapa | Features multi-escala en 768 dims | Pierde algo de información de etapa |
| LoRA sobre fine-tuning | Adaptación sin olvido catastrófico | Adaptación limitada (bajo rango) |
| Platt sobre temperature | Corrige sesgo y escala | 2 parámetros vs 1 |
| 96³ input | Compatible con pesos Triad | Resolución limitada para detalles finos |
