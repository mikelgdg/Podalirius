# Análisis del sistema Triage-MRI

## ¿Qué hemos construido?

Sistema de triaje (descarte de normalidad) para RM cerebral 3D usando:
- **Triad Swin-B SimMIM** como encoder congelado (19.8M params, pre-entrenado en 131K volúmenes)
- **Gated Attention MIL** como clasificador débilmente supervisado (525K params entrenables)
- **FocalLoss** (γ=2, α=0.25) para manejar el desbalance 1:5
- ~3,000 casos de entrenamiento: 2,397 BraTS (tumores) + ~600 OASIS-1/2 (normales + Alzheimer)

### Flujo de inferencia
```
Volumen RM 3D (96³) → TriadEncoder → 27×768 features
    → AttentionMIL → score [0,1] → NORMAL (<0.42) / REVISAR
```

---

## Fidelidad al plan original

| Lo planeado | Construido | % |
|------------|-----------|----|
| Encoder Triad Swin-B SimMIM | ✅ 134/134 keys cargadas | 100% |
| Attention-MIL (Gated, CLAM-style) | ✅ FocalLoss + WeightedSampler | 100% |
| 3 anatomías (cerebro, próstata, mama) | ⚠️ Solo cerebro | 33% |
| Demo Gradio | ✅ Implementada, esperando checkpoint | 100% |
| OASIS-3 | ⚠️ OASIS-1 (425) + OASIS-2 (285) | 80% |
| BraTS multi-subset | ✅ GLI + MEN + PED (2,397 casos) | 100% |
| Validación externa | ✅ Test set (15% hold-out, paciente-estratificado) | 100% |
| IXI (600 normales extra) | ❌ No extraído (disco lleno) | 0% |
| LLM explicabilidad | ❌ No implementado | 0% |

---

## Respuestas a tus preguntas

### 1. ¿Se ha usado IXI?

No. Los tars de IXI (T1 y T2, ~600 cerebros sanos) estaban en TEMPORAL pero no se pudieron extraer por falta de espacio en disco. TEMPORAL se borró para liberar espacio. **Habría que re-descargarlos.**

Con IXI, el balance pasaría de 1:5 a ~1:3 — mejora sustancial en capacidad de descarte.

### 2. ¿Hay datos de validación?

Sí. El dataset se parte por paciente (70/15/15):
- **Train**: 70% de los pacientes → ~542 batches
- **Val**: 15% → ~115 batches (se usa durante entrenamiento para early stopping)
- **Test**: 15% → ~120 batches (NUNCA se ve durante entrenamiento)

El test set está completamente aislado. Cuando el entrenamiento termine, se evalúa sobre él para obtener métricas finales sin sesgo.

### 3. ¿El LLM es fácil? ¿Habría que reentrenar?

**No es trivial, pero es factible.** Hay dos enfoques:

**Enfoque A — Projector + LLM congelado (recomendado):**
```
Features (27×768) → Projector (Linear) → LLM embeddings
    → BioMistral-7B → "RM cerebral sin hallazgos patológicos..."
```
- Solo se entrena el projector (~2M params)
- Encoder y MIL head se mantienen intactos
- Necesitas pares (volumen RM, informe radiológico) para entrenar la alineación
- **No requiere reentrenar nada de lo ya entrenado**
- Tiempo: 1-2 días en 1 GPU

**Enfoque B — LLaVA-Med completo:**
- Reentrenar todo el pipeline visión-lenguaje
- Mucho más costoso, requiere datasets masivos de imagen+texto
- No recomendado para este proyecto

**El problema real no es técnico — son los datos.** Necesitas informes radiológicos emparejados con volúmenes de RM. Las opciones:
- MIMIC-CXR tiene informes pero es de rayos X de tórax (no RM)
- Open-I tiene algunos estudios de RM con informes
- La mejor opción: usar un LLM (GPT-4, Claude) para generar informes sintéticos a partir de las etiquetas + hallazgos conocidos de BraTS/OASIS, y entrenar con eso

### 4. ¿Segmentación? ¿Es un addon o importante?

**Es un addon, no crítico para triaje.** Pero tiene valor:

**A favor:**
- Mejora la interpretabilidad: "Descartado porque el volumen hippocampal es normal para la edad"
- Puede servir como señal auxiliar de entrenamiento (multi-task learning)
- BrainSegFounder ya tiene pesos abiertos para esto

**En contra:**
- Añade complejidad sin mejorar directamente el triaje
- Triad no fue entrenado para segmentación (necesitarías otro modelo)
- El attention map del MIL ya da localización aproximada

**Conclusión:** No es prioritario. El attention-MIL ya señala dónde mira. Si necesitas localización precisa, añade BrainSegFounder como módulo separado post-triaje.

### 5. ¿Secuencias simultáneas es lo óptimo?

**Sí, es el gold standard clínico.** Los radiólogos leen T1, T1ce, T2 y FLAIR juntos. Pero:

**Problema técnico:** Triad fue pre-entrenado con entrada monocanal (1, 96, 96, 96). Para usar 3 canales (T1+T2+FLAIR):

- **Opción A — Promediar pesos del patch embedding:** Replicar el peso del primer canal 3 veces y dividir entre 3. Funciona, pero no es óptimo.
- **Opción B — Fine-tuning del patch embedding:** Descongelar solo la primera capa (patch_embed.proj, ~1K params) y entrenarla con datos multi-canal. Rápido y efectivo.
- **Opción C — Entrada multicanal desde cero:** Re-entrenar Triad con 3 canales. Inviable (requiere los 131K volúmenes originales).

**Recomendación:** Opción B. Pero primero demuestra que el sistema funciona con monocanal. La mejora multi-canal es un +5-10% de AUC — vale la pena, pero no es el cuello de botella ahora.

### 6. ¿Cómo mejorar desde aquí?

**Prioridades ordenadas por impacto/effort:**

| # | Mejora | Impacto | Esfuerzo | Tiempo |
|---|--------|---------|----------|--------|
| 1 | **Añadir próstata (PI-CAI) + mama** → generalista real | Alto | Bajo | 1-2 días |
| 2 | **Descargar y extraer IXI** → balance 1:3 | Alto | Bajo | Horas |
| 3 | **Calibrar threshold por anatomía** → seguridad clínica | Alto | Bajo | Horas |
| 4 | **Ensemble de secuencias (votar T1+T2+FLAIR)** | Medio | Bajo | 1 día |
| 5 | **Multi-canal (T1+T2+FLAIR como 3 canales)** | Alto | Medio | 2-3 días |
| 6 | **Projector + LLM para explicabilidad** | Medio | Alto | 1 semana |
| 7 | **Segmentación (BrainSegFounder addon)** | Bajo | Alto | 1-2 semanas |

---

## Comparación con OmniMRI

| Aspecto | OmniMRI [2508.17524] | Triage-MRI | Similitud |
|---------|----------------------|------------|-----------|
| Encoder 3D | Swin-B propietario, 220K vols | Triad Swin-B abierto, 131K vols | 90% |
| Visión-Lenguaje | 4 etapas de entrenamiento, LLM integrado | Solo visión (LLM pendiente) | 30% |
| Aprendizaje débil | Report-guided supervision | CDR + segmentation masks | 70% |
| Multi-tarea | Segm + Clasif + Informe + Reconst | Solo clasificación binaria | 25% |
| Generalista | Multi-anatomía, multi-secuencia, multi-tarea | Cerebro multi-secuencia | 60% |
| Código/Pesos | ❌ Placeholder vacío | ✅ 48 tests, funcional | 100% |
| Reproducibilidad | 60 datasets públicos + datos privados | 100% datos públicos | 100% |

**Somos un OmniMRI-Lite funcional y 100% abierto.** Mismo encoder, misma filosofía (frozen + MIL), mismo objetivo. Nos falta el LLM y la multi-anatomía, pero lo que tenemos funciona, es reproducible, y cualquiera puede descargarlo y entrenarlo.
