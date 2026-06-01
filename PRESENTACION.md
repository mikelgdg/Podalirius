# Triage-MRI: IA para filtrar resonancias magnéticas

> Guion para presentación oral — de cero a entenderlo todo

---

## Diapositiva 1: El problema

### Demasiadas resonancias, muy pocos radiólogos

- En España se hacen **~4 millones de resonancias al año**
- Cada una la revisa un radiólogo, corte a corte (150-200 imágenes por estudio)
- El **70-80% son normales** — no hay nada patológico
- Un radiólogo tarda 10-15 minutos en firmar un informe normal
- **Conclusión**: los radiólogos pasan la mayor parte del tiempo confirmando que no hay nada

> *[Imagen sugerida: sala de espera llena, pila de informes]*

---

## Diapositiva 2: La idea

### ¿Y si una IA filtrara primero?

- La IA revisa la resonancia en **menos de 1 segundo**
- Si lo ve normal → se informa automáticamente
- Si ve algo sospechoso → **REVISAR**, lo mandamos al radiólogo

> Es como el portero de una discoteca: no decide quién baila bien, solo quién entra

Resultado: el radiólogo dedica su tiempo solo a los casos que realmente lo necesitan.

---

## Diapositiva 3: ¿Qué es una resonancia magnética?

### Un cubo de datos, no una foto

- Una resonancia es un **volumen 3D**: miles de cortes apilados como un libro
- Cada vóxel (píxel 3D) tiene un valor de intensidad
- Distintas secuencias muestran cosas distintas:
  - **T1**: anatomía, estructuras
  - **T1 con contraste**: tumores (se iluminan)
  - **T2**: líquido, inflamación
  - **FLAIR**: lesiones cerebrales

> *[Imagen sugerida: cubo 3D de un cerebro girando, con un tumor coloreado]*

---

## Diapositiva 4: ¿Cómo "ve" una IA un volumen 3D?

### Trocear para entender

- Imposible procesar 240×240×155 vóxels de golpe
- La IA divide el volumen en una rejilla de **3×3×3 = 27 cubos**
- Cada cubo mide 32×32×32 vóxels
- El modelo analiza los 27 a la vez y decide: "¿hay algo raro en alguno?"

> Es como inspeccionar un piso habitación por habitación en vez de abrir la puerta y gritar

> *[Imagen sugerida: cerebro con rejilla 3×3×3 superpuesta]*

---

## Diapositiva 5: El modelo (I) — El encoder

### Un "cerebro" pre-entrenado que ya sabe ver

- Usamos un **Swin Transformer 3D** (arquitectura puntera en visión artificial)
- Fue entrenado con **SimMIM**: se le enseñaron 150.000 resonancias normales tapando trozos al azar y pidiéndole que los reconstruyera
- **No necesitó etiquetas**: solo aprender "cómo es un cerebro/próstata/mama normal"
- Este encoder tiene **20 millones de parámetros** y nosotros **no lo tocamos** (congelado)
- Produce 27 vectores de 768 números: una "firma" de cada región del cerebro

> *[Imagen sugerida: puzzle incompleto → reconstruido. Analogía de SimMIM]*

---

## Diapositiva 6: El modelo (II) — El cabezal de atención

### Enseñando al modelo a "señalar con el dedo"

- Sobre el encoder ponemos un **mecanismo de atención** (MIL)
- Funciona como un profesor que evalúa a 27 alumnos:
  - Cada región del cerebro levanta la mano diciendo "yo creo que aquí hay algo"
  - El profesor pondera: "a ti te creo más, a ti menos"
  - Suma ponderada → decisión final
- Esto es lo único que entrenamos: **525.000 parámetros** (el 2.5% del total)

> *[Imagen sugerida: 27 regiones con pesos de atención, algunas brillando más]*

---

## Diapositiva 7: Los datos

### 3.680 cerebros, 3 fuentes

| Fuente | Cantidad | ¿Qué es? |
|--------|----------|----------|
| **BraTS** | ~2.400 | Tumores cerebrales (todos anormales) |
| **OASIS** | ~700 | Envejecimiento, Alzheimer (mezcla) |
| **IXI** | ~580 | Controles sanos (todos normales) |

- Split: 70% entrenamiento / 15% validación / 15% test
- 71% anormales, 29% normales
- Cada volumen se recorta al centro (96×96×96), la parte más relevante

> *[Imagen sugerida: gráfico circular de fuentes de datos]*

---

## Diapositiva 8: Entrenamiento

### Cómo aprende la máquina

- **Loss**: Focal Loss — penaliza más los errores en casos difíciles
- **Optimizador**: AdamW (learning rate 5e-4)
- **Precisión**: 16 bits (FP16) para ahorrar memoria
- **GPU**: NVIDIA RTX 3500 Ada (12 GB)
- **Tiempo**: ~45 minutos por epoch, 100 epochs máximo
- **Early stopping**: si no mejora en 15 epochs, para solo

> *[Imagen sugerida: curva de aprendizaje típica]*

---

## Diapositiva 9: Métricas — ¿Cómo sé que funciona?

### No basta con "acertar"

- **AUC (Area Under Curve)**: mide capacidad de separar sanos de enfermos. 1.0 = perfecto, 0.5 = aleatorio
- **Sensibilidad**: de cada 100 enfermos, ¿a cuántos detecta? (queremos >99%)
- **Especificidad**: de cada 100 sanos, ¿a cuántos deja pasar? (que no los mande al radiólogo)
- **Curva ROC**: equilibrio entre sensibilidad y especificidad

> *[Imagen sugerida: curva ROC con AUC anotado]*

---

## Diapositiva 10: Resultados del entrenamiento

### Lo que conseguimos

| Época | AUC | ¿Confianza? | ¿Atención focalizada? |
|-------|-----|-------------|----------------------|
| 0 | 0.969 | Baja | No |
| 1 | 0.985 | Baja | No |
| 2 | 0.991 | Media | Empieza |
| ... | ... | ... | ... |

- **AUC sube rápido**: el modelo discrimina bien desde el principio
- **Confianza y atención** mejoran con más epochs
- A las 8-10 horas de entrenamiento (10 epochs) el modelo ya es utilizable

> *[Imagen sugerida: línea temporal de AUC por epoch]*

---

## Diapositiva 11: Demo en vivo

### Probemos el modelo

1. Subir una resonancia (`.nii.gz`)
2. El modelo la procesa en <1 segundo
3. Resultado: **NORMAL** o **REVISAR**
4. Visualización:
   - Cortes del cerebro (axial, sagital, coronal)
   - **Rojo** = tumor real (lo que hay de verdad)
   - **Amarillo** = dónde mira el modelo
   - Visor 3D rotable

> *[Hacer demo en vivo con caso BraTS-MEN-00004-000]*

---

## Diapositiva 12: Lecciones aprendidas

### Lo que salió mal (y arreglamos)

1. **Segfault en DataLoader**: multiprocessing + NIfTI → usar 0 workers
2. **Batch size 12 → 4**: no cabía en 12 GB de VRAM
3. **Archivos corruptos**: validar al cargar, saltar los rotos
4. **Center-crop vs Resize**: intentamos resize para ver el cerebro entero, pero el encoder no estaba entrenado para eso → volvimos a center crop
5. **Atención plana**: sin suficiente entrenamiento el modelo no enfoca → paciencia, más epochs
6. **Métricas engañosas**: accuracy al 99% sensibilidad ≠ accuracy real → añadimos acc@0.5

> Cada error enseñó algo. Así se hace investigación.

---

## Diapositiva 13: ¿Qué falta?

### Próximos pasos

- [ ] Entrenar con **próstata y mama** (ya está preparado, faltan datos)
- [ ] Añadir métricas de **calibración** (ECE, Brier score)
- [ ] **Evaluación externa**: probar en datos de otro hospital (generalización)
- [ ] **Explicabilidad**: que el modelo no solo diga "anormal", sino por qué
- [ ] **Despliegue**: integración con PACS (sistema de gestión de imágenes médicas)

---

## Diapositiva 14: Resumen

### En una frase

> Un Swin Transformer pre-entrenado que mira 27 regiones del cerebro, aprende a señalar anomalías, y filtra el 80% de resonancias normales para que el radiólogo solo vea las importantes.

- **Tecnología**: Swin-B 3D + Attention MIL + SimMIM
- **Datos**: 3.680 cerebros (BraTS, OASIS, IXI)
- **Métrica**: AUC ~0.991 en época 2, mejorando
- **Demo**: interfaz web funcional con visor 3D

---

## Anexo: Glosario para la audiencia

| Término | Explicación simple |
|---------|-------------------|
| **MRI** | Resonancia magnética: imán gigante que saca fotos 3D del interior del cuerpo |
| **Vóxel** | Píxel 3D. Un cubito de tejido de ~1 mm³ |
| **NIfTI** | Formato de archivo para guardar resonancias (.nii.gz) |
| **AUC** | Mide si el modelo ordena bien: sanos puntuación baja, enfermos alta |
| **Encoder** | La parte del modelo que "mira" y extrae características |
| **MIL** | Multiple Instance Learning: decides sobre un conjunto de regiones |
| **Atención** | Mecanismo que da más peso a unas regiones que a otras |
| **SimMIM** | Entrenamiento autosupervisado: reconstruir lo tapado |
| **Grad-CAM / heatmap** | Mapa de colores que muestra dónde "mira" el modelo |
| **Focal Loss** | Función de pérdida que da más importancia a casos difíciles |
| **Sensibilidad** | Porcentaje de enfermos detectados |
| **Especificidad** | Porcentaje de sanos correctamente ignorados |
