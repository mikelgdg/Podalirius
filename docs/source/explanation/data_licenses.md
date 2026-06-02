# Licencias de datos y viabilidad comercial

## Encoder: MIT ✅

El encoder **Triad Swin-B SimMIM** (arxiv 2502.14064) es el componente crítico para la viabilidad comercial. Su situación es favorable:

| Item | Valor |
|------|-------|
| Paper | CC BY 4.0 |
| Código (GitHub) | **MIT** |
| Pesos del modelo | **MIT** (heredan licencia del repositorio) |
| Dataset Triad-131K | No público (datos clínicos de Emory) |
| Uso comercial | **Sí** ✅ |
| Distribución de pesos | **Sí** ✅ |
| Venta / SaaS | **Sí** ✅ |

El encoder **no es el bloqueante** para la comercialización. Confirmado por la insignia de licencia en el repositorio oficial y el paper.

## Datasets de entrenamiento: bloqueantes

Todos los datasets usados para entrenar el cabezal MIL tienen restricciones no comerciales:

| Dataset | Licencia | Comercial | Distribución | SaaS |
|---------|----------|-----------|-------------|------|
| BraTS 2023 | TCIA DUA | No | No | No |
| OASIS-3 | OASIS DUA | No | No | No |
| IXI | CC BY-SA 3.0 | No (SA) | Sí (SA) | Complejo |
| HCP S1200 | HCP DUA | No | No | No |
| PI-CAI | CC BY-NC-SA 4.0 | No (NC) | No (NC) | No (NC) |

La licencia más restrictiva (PI-CAI: CC BY-NC-SA 4.0) es "viral": cualquier modelo entrenado con esos datos hereda la restricción NC (no comercial) y SA (compartir igual). El modelo combinado actual **no puede** venderse, distribuirse comercialmente, ni usarse en un producto SaaS.

## Matriz de compatibilidad de licencias

| Licencia | Vender modelo | Distribuir pesos | Usar en SaaS |
|----------|--------------|-----------------|--------------|
| CC0 | Sí | Sí | Sí |
| CC BY 4.0 | Sí | Sí | Sí |
| CC BY-SA 4.0 | Sí* | Sí* | Sí* |
| CC BY-NC 4.0 | No | No | No |
| CC BY-NC-SA 4.0 | No | No | No |
| TCIA DUA | No | Solo investigación | No |
| DUA (OASIS, HCP) | No | Solo investigación | No |

*SA (ShareAlike) requiere que obras derivadas usen la misma licencia.

## Camino a la viabilidad comercial

### Opción A: Re-entrenamiento con datos permisivos (recomendado, ~3-4 meses)

Reemplazar todos los datasets por alternativas CC0 o CC BY 4.0:

| Anatomía | Actual | Reemplazo | Licencia |
|----------|--------|-----------|----------|
| Tumores cerebrales | BraTS (~600) | UPenn-GBM + LGG-1p19q + IvyGAP | CC BY 4.0 |
| Cerebro normal | OASIS+IXI+HCP (~2,600) | OASIS-1 + UK Biobank | CC BY 4.0 |
| Próstata | PI-CAI (~1,500) | ProstateX + datos hospitalarios | Verificar |
| Mama | MRI-Breast (~200) | DDSM (~2,500) o CMMD (~1,800) | CC0 / CC BY |

Total estimado tras reemplazo: ~4,000-5,000 casos (comparable al setup actual).

### Opción B: Negociación de licencias (~6+ meses, $5K-$50K+)

Contactar a cada proveedor de datos para licencia comercial. Costo y viabilidad varían por dataset.

### Opción C: Despliegue hospitalario interno (~1 mes)

Si el modelo se despliega solo dentro de la red interna de un hospital y nunca se vende o distribuye, muchas restricciones de uso en investigación no aplican al uso clínico interno. Verificar leyes locales y política institucional.

## Estimación de costes

| Concepto | Coste estimado |
|----------|---------------|
| GPU cloud para re-entrenamiento | $500-2,000 |
| Adquisición de datos (licencias) | $0-50,000 |
| Estudio de validación clínica | $10,000-100,000 |
| Certificación regulatoria (FDA 510k) | $50,000-250,000 |
| Revisión legal de licencias | $5,000-15,000 |

## Cronograma

- **Ahora**: confirmar licencia del encoder (hecho ✅), empezar a recolectar datos hospitalarios
- **2-3 meses**: descargar datasets CC0/CC BY, re-entrenar MIL head
- **6+ meses**: estudio de validación clínica (IRB), evaluación de certificación FDA/CE
- **12-18 meses**: producto comercializable

## Conclusión

El modelo actual es un **prototipo de investigación**. Para uso comercial se requiere:

1. Reemplazar datos de entrenamiento por alternativas con licencia permisiva, o
2. Obtener licencias comerciales de cada proveedor de datos
3. Validación clínica y posible certificación regulatoria

El encoder Triad (MIT) es el activo más valioso y ya está listo para uso comercial.
