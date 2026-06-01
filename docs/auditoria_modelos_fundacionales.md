# Auditoría: Modelos Fundacionales para Triaje y Descarte de Normalidad en RM Volumétrica

**Fecha:** Mayo 2026  
**Objetivo:** Evaluar la viabilidad de construir un sistema de triaje/descarte de normalidad para RM 3D usando modelos fundacionales y aprendizaje débilmente supervisado.

---

## 1. La idea propuesta

Sistema que procesa volúmenes de RM 3D mediante un encoder fundacional pre-entrenado, un clasificador débilmente supervisado (normal vs anormal) y, opcionalmente, un LLM para justificar la decisión de descarte. El flujo propuesto es:

```
Volumen RM 3D → Vision Encoder (congelado) → Feature Map
                                                  ├──→ GAP → Clasificador Débil → Score de Prioridad
                                                  └──→ Projector → LLM → Justificación textual
```

Estrategia de entrenamiento: encoder congelado, solo se entrena el clasificador (MLP o Attention-MIL) con etiquetas extraídas de informes radiológicos (aprendizaje débil).

---

## 2. Verificación de los papers citados en la propuesta

| Paper | arxiv ID | Código público | Pesos disponibles | Licencia |
|-------|----------|----------------|-------------------|----------|
| **OmniMRI** | 2508.17524 | ❌ Solo placeholder (3 commits, "Coming soon") | ❌ No | BSD-2 (repo vacío) |
| **Decipher-MR** | 2509.21249 | ❌ No existe repo | ❌ No (GE HealthCare) | CC BY-NC-SA 4.0 (solo paper) |
| **MRI-CORE** | 2506.12186 | ✅ mazurowski-lab/mri_foundation (86★) | ✅ Google Drive (MRI_CORE_vitb.pth) | Apache 2.0 |
| **BrainSegFounder** | 2406.10395 | ✅ lab-smile/BrainSegFounder (14★) | ✅ Google Drive | GPL-3.0 |
| **Patched Diffusion UAD** | 2303.03758 | ✅ FinnBehrendt/patched-Diffusion-Models-UAD | ❌ No pre-entrenados (entrenar desde 0) | Sin especificar |
| **LLaVA-Med** | 2306.00890 | ✅ microsoft/LLaVA-Med (2.2K★) | ✅ HuggingFace (v1.5) | Apache 2.0 (v1.5) |
| **Attention-MIL** | (Ilse et al., 2018) | ✅ Código disponible | N/A (método, no modelo) | MIT |

---

## 3. Hallazgo crítico: el panorama real de modelos 3D MRI abiertos (Mayo 2026)

### 3.1 Lo que NO existe (o es inaccesible)

- **No hay un equivalente abierto a Decipher-MR u OmniMRI.** Es decir, no existe un modelo fundacional 3D de RM visión-lenguaje con pesos y código públicos completos.
- GE HealthCare no tiene historial de liberar modelos open-source. Decipher-MR es un paper corporativo.
- OmniMRI (Harvard/MGH) tiene un repo placeholder sin contenido real.

### 3.2 Lo que SÍ existe — Modelos con pesos descargables (verificado Mayo 2026)

#### Tier 1 — Modelos 3D específicos de RM con pesos abiertos

| Modelo | GitHub | Pesos | Arquitectura | Datos entrenamiento | Licencia |
|--------|--------|-------|-------------|---------------------|----------|
| **Triad** (Emory, MEDIA 2026) | wangshansong1/Triad (51★) | ✅ Google Drive (4 checkpoints: PlainConvUNet/Swin-B × MAE/SimMIM) | PlainConvUNet + Swin Transformer (Swin-B) | 131K volúmenes RM 3D (19.7K pacientes, 36 datasets, cerebro/mama/próstata) | MIT |
| **3D Neuro SimCLR** (ICCV 2025) | emilykaczmarek/3D-Neuro-SimCLR | ✅ GitHub Releases (v1.0.0) | ResNet 3D, SimCLR contrastivo | 18.7K pacientes, 44.9K scans, 11 datasets neurológicos | MIT |
| **MRI-CORE** (Duke) | mazurowski-lab/mri_foundation (86★) | ✅ Google Drive (MRI_CORE_vitb.pth) | SAM-based (ViT-B), 2D slice-wise | 110K+ volúmenes, 6M+ slices, 18 ubicaciones corporales | Apache 2.0 |
| **BrainSegFounder** (UF+NVIDIA) | lab-smile/BrainSegFounder (14★) | ✅ Google Drive | SwinUNETR | 41.4K participantes UK Biobank | GPL-3.0 |
| **BrainMVP** (CVPR 2025 Highlight) | openmedlab/BrainMVP (87★) | ✅ Disponible (Uniformer/UNet) | Multi-modal 3D (cross-modal recon, contrastive, template distillation) | 16K scans cerebro multi-paramétricos (T1/T2/FLAIR/DWI) | Open-source |

#### Tier 2 — Modelos 3D médicos generales (CT+MRI) con pesos

| Modelo | GitHub | Pesos | Arquitectura | Clases | Licencia |
|--------|--------|-------|-------------|-------|----------|
| **NV-Segment-CTMR / VISTA3D** (NVIDIA, CVPR 2025) | NVIDIA-Medtech/NV-Segment-CTMR | ✅ HuggingFace (nvidia/NV-Segment-CTMR) | SAM-like 3D Transformer, 218M params | 345+ clases (cuerpo CT + cuerpo MRI + cerebro MRI) | Código Apache 2.0, pesos no-comercial |
| **SAM-Med3D** (OpenMEDLab) | openmedlab/SAM-Med3D | ✅ Google Drive | SAM adaptado a 3D volumétrico | 247 categorías, 131K máscaras 3D | Apache 2.0 |
| **MedSAM2** (2025) | bowang-lab/MedSAM2 | ✅ Auto-descarga | SAM2 fine-tuned para 3D médicos + video | 455K pares imagen-máscara | Apache 2.0 |

#### Tier 3 — Solo visión-lenguaje (2D, no RM 3D)

| Modelo | Pesos | Notas |
|--------|-------|-------|
| **LLaVA-Med** (Microsoft) | ✅ HuggingFace | Biomédico general, no específico RM 3D |

### 3.3 Verificación de Triad — el modelo que faltaba

> **Estado: CONFIRMADO.** Investigación exhaustiva completada.

- **Paper:** arxiv 2502.14064 (Feb 2025), publicado en Medical Image Analysis (Elsevier, 2026)
- **Repo:** github.com/wangshansong1/Triad — 51★, 7 commits, MIT
- **Pesos:** 4 checkpoints en Google Drive (PlainConvUNet-MAE, PlainConvUNet-SimMIM, SwinB-MAE, SwinB-SimMIM). **Acceso público confirmado.**
- **Dataset Triad-131K:** NO liberado (datos clínicos de Emory). El repo espera archivos `.npy` locales.
- **Código:** Solo framework de pre-entrenamiento + carga de pesos. **NO incluye código de fine-tuning ni pipelines downstream.**
- **Problema:** 4 issues abiertos (Feb 2025 - Mar 2026) preguntando por código/pesos. **Cero respuestas de los autores.**
- **Rendimiento:** +2.51% en segmentación, +3.97% en clasificación, +4.00% en registro sobre baselines sin pre-entrenar (25 datasets downstream).

**Veredicto sobre Triad:** Es real, es peer-reviewed, y tiene pesos. Pero el repo es mínimo y sin soporte. Usable como encoder base si implementas tu propio pipeline de fine-tuning, pero no es plug-and-play.

---

## 4. Errores factuales detectados en la propuesta original

1. **MIMIC-IV no contiene imágenes de RM.** Es exclusivamente datos clínicos/textuales (notas de UCI, signos vitales, medicaciones). No puede usarse para entrenar un encoder de visión.

2. **ADNI y UK Biobank no son de acceso inmediato.** Requieren:
   - ADNI: acuerdo de uso de datos (DUA), aplicación formal, semanas/meses de aprobación
   - UK Biobank: solicitud institucional, tasas de £500-3000, aprobación IRB, 6-12 meses

3. **Global Average Pooling (GAP) es inadecuado para triaje en 3D.** Colapsar características espaciales de un volumen completo en un solo vector diluye anomalías focales (ej. un tumor pequeño en un slice). Se requiere Attention-MIL como mínimo.

4. **La RM es multimodal (T1, T2, FLAIR, DWI, etc.).** Un único encoder entrenado en una secuencia no generaliza a otras sin adaptación.

---

## 5. Evaluación de viabilidad

### 5.1 ¿Es viable la arquitectura propuesta?
✅ **Sí, conceptualmente.** Sigue el patrón establecido LLaVA/BLIP-2. Es una arquitectura sólida si el encoder existe.

### 5.2 ¿Existen los componentes necesarios?
⚠️ **Parcialmente.** El encoder 3D abierto más prometedor es MRI-CORE, pero es solo visión (sin lenguaje). Si Triad se confirma como real, sería una alternativa superior.

### 5.3 ¿Qué se necesita entrenar?
- **Encoder:** Congelado (si hay pesos pre-entrenados)
- **Clasificador de triaje:** Entrenar desde cero (4-12 horas en 1 GPU)
- **Proyector + LLM:** Entrenar para alineación (1-2 días en 1-2 GPUs)
- **Total estimado:** 1-2 A100 (80GB), 1-3 semanas de experimentación + entrenamiento final

### 5.4 ¿Qué datos se necesitan?
- Para fine-tuning del clasificador: ~1,000-5,000 volúmenes de RM con etiquetas "normal/anormal" (extraíbles de informes o metadatos)
- Datasets accesibles inmediatamente: BraTS (~2,000 casos), OpenNeuro (múltiples datasets públicos)
- UK Biobank / ADNI: solo como objetivo a medio plazo (6-12 meses para acceso)

### 5.5 ¿Es realista clínicamente?
⚠️ **A largo plazo.** El camino a uso clínico real requiere:
1. Estudio retrospectivo (6-12 meses)
2. Reader study con radiólogos (3-6 meses)
3. Prueba prospectiva silenciosa (6-12 meses)
4. Aprobación regulatoria FDA 510(k) o CE: $500K-2M+

Objetivo de sensibilidad para descarte seguro: **NPV > 99.5%** (si el modelo dice "normal", debe acertar el 99.5% de las veces).

---

## 6. Recomendaciones estratégicas (Actualizadas con nuevos hallazgos)

### Ranking de encoders para el proyecto (mejor a peor)

| # | Encoder | 3D real | Pesos abiertos | RM nativo | Multi-secuencia | Multi-anatomía | Soporte |
|---|---------|---------|----------------|-----------|-----------------|----------------|---------|
| 1 | **Triad (Swin-B, SimMIM)** | ✅ | ✅ Google Drive | ✅ | ✅ (T1/T2/FLAIR/DWI) | ✅ (cerebro/mama/próstata) | ❌ Sin soporte |
| 2 | **3D Neuro SimCLR** | ✅ | ✅ GitHub Releases | ✅ | ❌ Solo T1-w | ❌ Solo cerebro | ⚠️ Mínimo |
| 3 | **BrainMVP** | ✅ | ✅ Disponible | ✅ | ✅ Multi-paramétrico | ❌ Solo cerebro | ✅ (OpenMEDLab) |
| 4 | **MRI-CORE (ViT-B)** | ❌ 2D slice-wise | ✅ Google Drive | ✅ | ✅ | ✅ (18 ubicaciones) | ✅ Activo |
| 5 | **VISTA3D (NVIDIA)** | ✅ | ✅ HuggingFace | ✅ (segmentación) | ⚠️ | ✅ (cuerpo+cerebro) | ✅ (NVIDIA/MONAI) |

### Camino A: Pragmático (Mínimo viable — Recomendado para empezar YA)
```
Triad Swin-B SimMIM (encoder congelado) + Attention-MIL (clasificador) + BraTS/OpenNeuro (datos)
  → Mejor encoder 3D-RM disponible con pesos
  → ~2-3 meses, 1-2 A100
  → Sin LLM, solo clasificación binaria normal/anormal
  → Validar sensibilidad >99% antes de añadir nada más
```

### Camino B: Alternativo si Triad no convence
```
3D Neuro SimCLR o BrainMVP (encoder congelado) + Attention-MIL + BraTS
  → Ambos con código y pesos, true 3D
  → BrainMVP tiene ventaja multi-secuencia (clave para generalizar)
```

### Camino C: Ambicioso (solo si A o B funcionan en clasificación)
```
Encoder (A/B) + Proyector + LLM médico (BioMistral/Meditron) + Informes reales
  → Añadir explicabilidad textual cuando el clasificador ya sea fiable
```

---

## 7. Veredicto final (Actualizado Mayo 2026)

**La idea es intelectualmente sólida y aborda un problema clínico real.** La propuesta original del documento tiene estos problemas:

1. Sus dos pilares (Decipher-MR y OmniMRI) no son accesibles como software
2. Confunde MIMIC-IV (texto clínico) con datos de imagen
3. Subestima barreras de acceso a datos (UK Biobank 6-12 meses)
4. El LLM es premature optimization

**PERO — la premisa de que "no existe ningún encoder 3D RM open-source" es FALSA.** Tras investigar a fondo:

- **Triad** (Emory, MEDIA 2026) tiene 4 checkpoints públicos en Google Drive. Es el mejor punto de partida: Swin-B pre-entrenado con SimMIM sobre 131K volúmenes RM multi-secuencia y multi-anatomía.
- **3D Neuro SimCLR** y **BrainMVP** son alternativas viables con true 3D y pesos abiertos.
- **VISTA3D** (NVIDIA) es el modelo más completo en segmentación 3D pero la licencia de pesos es no-comercial.

### Matiz importante sobre Triad

Aunque los pesos existen, el repositorio de Triad tiene limitaciones serias:
- Solo 7 commits, código mínimo (framework de pre-entrenamiento)
- 4 issues sin respuesta desde Feb 2025 — los autores no dan soporte
- Sin código de fine-tuning ni pipelines downstream
- El dataset Triad-131K no es público

**Implicación práctica:** Tendrás que implementar tu propio pipeline de fine-tuning (cargar pesos con `QuickStart.py` → añadir cabeza de clasificación → entrenar). No es complicado pero requiere trabajo de ingeniería.

### Recomendación final

```
Fase 1 (2-3 meses): Triad Swin-B SimMIM + Attention-MIL + BraTS + OpenNeuro
  → Prototipo funcional de clasificación normal/anormal
  → Sin LLM, sin explicabilidad

Fase 2 (si Fase 1 funciona): Añadir más datos, validación externa, curvas de calibración

Fase 3 (solo si las fases anteriores funcionan): LLM para explicabilidad
```

El proyecto es viable, está bien fundamentado, y ahora tienes un abanico real de encoders con pesos abiertos para elegir. El cuello de botella ya no es "no hay modelo" — es implementar el pipeline de fine-tuning y conseguir/curar datos etiquetados de calidad.
