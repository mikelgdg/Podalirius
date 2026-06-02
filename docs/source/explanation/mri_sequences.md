# Secuencias MRI — Guía Completa

## ¿Qué es una secuencia MRI?

Una **secuencia** es una configuración específica del escáner de resonancia magnética
que produce imágenes con **contraste diferente** entre tejidos. Cada secuencia
muestra propiedades físicas distintas del tejido (agua, grasa, hierro, flujo
sanguíneo, difusión de moléculas, realce con contraste intravenoso).

En diagnóstico radiológico **nunca se mira una sola secuencia**. El radiólogo
compara 3-6 secuencias simultáneamente porque:

- Una lesión invisible en T1 puede ser obvia en T2 o FLAIR
- Una captación de contraste sospechosa en T1ce se confirma viendo que en T1
  sin contraste esa zona era normal
- Una isquemia aguda (ictus) es invisible en T1 y T2 pero brilla en DWI

Triage-MRI usa **una sola secuencia por caso** (modo por defecto). Esto es una
limitación importante: el modelo no tiene contexto multi-secuencia para decidir.
El modo multi-secuencia (`multi_sequence: true`) carga hasta 4 secuencias como
canales de entrada.

---

## Catálogo de Secuencias

### T1 — Anatómico (T1-weighted)

- **Qué mide**: Tiempo de relajación longitudinal del tejido
- **Apariencia**: Grasa = brillante (blanca), agua/LCR = oscura (negra),
  sustancia blanca = gris claro, sustancia gris = gris oscuro
- **Buena para**: Anatomía estructural fina, volumen cerebral, atrofia,
  diferenciación sustancia blanca/gris
- **Mala para**: Edema, inflamación, tumores (se confunden con tejido normal)
- **Dataset**: OASIS-3, IXI, HCP usan T1 como secuencia principal
- **En Triage-MRI**: Es la secuencia de fallback preferida para OASIS/IXI/HCP

```
Corte sagital T1 de cerebro sano:
┌─────────────────────┐
│  🌫️ gris claro       │  ← sustancia blanca
│  🌫️ gris oscuro      │  ← sustancia gris (corteza)
│  ⬛ negro profundo    │  ← ventrículos (LCR)
│  ⬜ blanco intenso    │  ← grasa subcutánea
└─────────────────────┘
```

### T1ce — T1 post-contraste (T1 contrast-enhanced)

- **Qué mide**: T1 tras inyectar gadolinio intravenoso
- **Apariencia**: Igual que T1, pero las zonas con **rotura de barrera
  hematoencefálica** (tumores, inflamación activa) captan contraste y se vuelven
  **brillantes**
- **Buena para**: Tumores cerebrales (gliomas, metástasis, meningiomas),
  lesiones inflamatorias activas (esclerosis múltiple), abscesos
- **Mala para**: Tumores de bajo grado sin realce, isquemia aguda, patología
  sin rotura de barrera
- **Dataset**: BraTS usa T1ce como secuencia principal
- **En Triage-MRI**: Es la secuencia preferida para BraTS

```
T1 sin contraste:                    T1ce (mismo corte):
┌─────────────────────┐              ┌─────────────────────┐
│  🌫️ todo homogéneo    │              │  🌫️ zona ⬜ BRILLANTE │ ← tumor realzado
│                      │              │     con gadolinio     │
│  tumor INVISIBLE     │              │  tumor VISIBLE        │
└─────────────────────┘              └─────────────────────┘
```

### ⚠️ Problema de domain shift en Triage-MRI

BraTS se entrena con **T1ce** (post-contraste). OASIS/IXI/HCP usan **T1 sin
contraste**. Son protocolos de adquisición distintos:

| Característica | BraTS (T1ce) | OASIS/IXI/HCP (T1) |
|---------------|-------------|-------------------|
| Contraste IV | Sí (gadolinio) | No |
| Grosor de corte | 1 mm | 1-1.25 mm |
| Campo de visión | Cerebro + edema | Cerebro completo |
| Brillo tumoral | Alto (realce) | Bajo (similar a tejido) |
| Propósito | Protocolo tumoral | Protocolo investigación |

El modelo puede aprender "T1ce = anormal, T1 = normal" en vez de "tumor =
anormal, sano = normal". Es un **proxy espurio**. La validación con
análisis de subgrupos (`--subgroup_analysis`) revelará si esto está pasando.

---

### T2 — Anatómico invertido (T2-weighted)

- **Qué mide**: Tiempo de relajación transversal del tejido
- **Apariencia**: Agua/LCR = brillante (blanca), grasa = brillante pero menos,
  sustancia blanca = gris oscuro. Esencialmente: **el agua brilla**
- **Buena para**: Edema, inflamación, anatomía prostática (zonas periférica vs
  central), lesiones quísticas, hidrocefalia
- **Mala para**: Diferenciación tumoral fina (los tumores son edematosos = brillan igual)
- **Dataset**: PI-CAI (próstata) usa T2W como secuencia principal
- **En Triage-MRI**: Es la secuencia preferida para próstata

```
Corte axial de próstata en T2W:
┌─────────────────────────────┐
│  ⬜ zona periférica (brillante)│ ← donde suelen aparecer tumores
│  🌫️ zona central (oscura)     │
│  ⚫ tumor = área OSCURA       │ ← ¡contraste invertido! El tumor
│    dentro de zona brillante   │   en T2W es más oscuro que el
│                              │   tejido sano circundante
└─────────────────────────────┘
```

### ⚠️ Cerebro (T1ce) vs Próstata (T2W): Contrastes OPUESTOS

Este es el mayor desafío del modo brain+prostate:

| | Cerebro (BraTS) | Próstata (PI-CAI) |
|---|---|---|
| Secuencia | T1ce | T2W |
| Tumor vs sano | Tumor = **BRILLANTE** | Tumor = **OSCURO** |
| Bobina | Head coil | Pelvic coil |
| Orientación | Variable | Axial |
| Tamaño órgano | 1400 cm³ | 25-50 cm³ |

El MIL head tiene que aprender que "anomalía" puede ser **hiperintensa en T1ce**
o **hipointensa en T2W** — dos conceptos físicos opuestos. No es imposible
(el encoder Triad fue pre-entrenado en ambos), pero es inherentemente más
difícil que una sola anatomía.

---

### FLAIR — Fluid Attenuated Inversion Recovery

- **Qué mide**: T2 donde el **agua libre (LCR) se suprime** a negro
- **Apariencia**: LCR = negro, edema periventricular = **muy brillante**
- **Buena para**: Lesiones de sustancia blanca (esclerosis múltiple, enfermedad
  de pequeño vaso, leucoaraiosis), edema peritumoral, gliosis
- **Mala para**: Anatomía fina (borrosa respecto a T1)
- **Dataset**: BraTS incluye FLAIR pero el modo por defecto prefiere T1ce
- **En Triage-MRI**: Secuencia opcional en modo multi-secuencia

```
T2 normal:            FLAIR (mismo corte):
┌──────────────┐      ┌──────────────┐
│  ⬜ LCR blanco│      │  ⬛ LCR negro  │ ← agua suprimida
│  🌫️ edema     │      │  ⬜ edema       │ ← ¡destaca mucho!
│   moderado   │      │   MUY BRILLANTE│   las lesiones son obvias
└──────────────┘      └──────────────┘
```

---

### DWI/ADC — Diffusion-Weighted Imaging (mapa de ADC)

- **Qué mide**: Movimiento browniano (difusión) de moléculas de agua en el
  espacio extracelular
- **DWI (b=1000)**: Agua restringida = **brillante** (hiperintensa). Útil
  para ictus isquémico agudo (minutos tras el evento) y tumores celulares.
- **ADC (mapa)**: Agua restringida = **oscura** (hipointensa). Confirma que
  la hiperintensidad en DWI es restricción real (no artefacto T2).
- **Buena para**: Ictus agudo (gold standard), abscesos vs tumores quísticos,
  celularidad tumoral (grado de agresividad)
- **Dataset**: PI-CAI incluye DWI/ADC para próstata
- **En Triage-MRI**: Secuencia opcional en modo multi-secuencia de próstata

```
DWI (b=1000):         ADC (mapa):
┌──────────────┐      ┌──────────────┐
│  ⬜ zona BRILLANTE│  │  ⚫ misma zona │ ← confirma restricción
│   = restricción  │  │   OSCURA       │   si ADC es oscuro y DWI
│                 │  │                │   es brillante: ictus/tumor
│  ⚫ LCR oscuro   │  │  ⬜ LCR brillante│
└──────────────┘      └──────────────┘
```

---

## Combinaciones que usa Triage-MRI

### Modo por defecto (single-sequence)

| Anatomía | Secuencia preferida | Fallback | Fuente |
|----------|-------------------|----------|--------|
| Cerebro (BraTS) | **T1ce** | T1, T2, FLAIR | `preferred_sequence: "t1ce"` |
| Cerebro (OASIS) | **T1** | — | `preferred_sequence: "t1"` |
| Cerebro (IXI/HCP) | **T1** | — | `preferred_sequence: "t1"` |
| Próstata (PI-CAI) | **T2W** | DWI, ADC | `preferred_sequence: "t2w"` |

### Modo multi-secuencia

```yaml
data:
  multi_sequence:
    enabled: true
    input_sequences: ["t1", "t1ce", "t2", "flair"]  # hasta 4 canales
```

El encoder recibe `(B, 4, 96, 96, 96)` y la primera capa `patch_embed.proj`
se adapta de 1 a 4 canales de entrada, aprendiendo a fusionar las secuencias.

**Ventaja**: El modelo ve toda la información diagnóstica disponible, como un
radiólogo. Puede detectar tumores que son invisibles en T1 pero captan contraste
en T1ce, o lesiones que brillan en FLAIR pero no en T2.

**Desventaja**: Requiere que todos los casos tengan todas las secuencias
configuradas, lo cual es raro en datasets públicos (normalmente tienes 1-2
secuencias por caso, no 4).

---

## Tabla de decisión por tipo de patología

| Patología | Mejor secuencia | ¿Visible en T1? | ¿Visible en T1ce? | ¿Visible en T2? | ¿Visible en FLAIR? |
|-----------|---------------|:---:|:---:|:---:|:---:|
| Tumor cerebral (glioma) | T1ce + FLAIR | ❌ | ✅ | ◆ | ✅ |
| Metástasis cerebral | T1ce | ◆ | ✅ | ◆ | ✅ |
| Esclerosis múltiple | FLAIR | ❌ | ✅ (activa) | ✅ | ✅ |
| Ictus isquémico agudo | DWI | ❌ | ❌ | ❌ (primeras 6h) | ❌ |
| Atrofia / demencia | T1 | ✅ | — | ◆ | ◆ |
| Cáncer de próstata | T2W + DWI | — | — | ✅ | — |
| Edema cerebral | FLAIR | ❌ | ◆ | ✅ | ✅ |
| Hemorragia | T2\* / SWI | ◆ | ◆ | ✅ | ◆ |

✅ = bien visible | ◆ = visible pero no ideal | ❌ = invisible o casi

---

## Impacto en el entrenamiento

### Lo que el modelo PUEDE aprender (según secuencia)

| Secuencia única | Lo que ve | Lo que se le escapa |
|----------------|-----------|-------------------|
| Solo T1 | Anatomía, atrofia | Tumores, edema, inflamación |
| Solo T1ce | Tumores con realce, metástasis | Ictus, tumores sin realce |
| Solo T2 | Edema, anatomía prostática | Diferenciación tumoral fina |
| Solo FLAIR | Lesiones sustancia blanca, edema periventricular | Anatomía fina |
| **Multi-secuencia** | **Todo lo anterior** | Solo patologías que requieren secuencias avanzadas (espectroscopia, perfusión) |

### Por qué el modo multi-secuencia es ideal

Un radiólogo nunca diagnostica con una sola secuencia. El modo multi-secuencia
es el camino hacia un sistema con rendimiento clínico real. El coste es que
requiere datasets con todas las secuencias para cada caso, lo cual limita
qué datos puedes usar.
