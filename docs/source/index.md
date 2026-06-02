# Triage-MRI

Normality screening for 3D MRI volumes using a frozen foundational encoder (Triad Swin-B SimMIM, 131K pre-trained volumes) and a trainable Gated Attention MIL head.

::::{grid} 2
:::{grid-item-card} 🚀 Primeros pasos
:link: tutorials/quickstart
5 minutos para tu primer triaje con la demo interactiva.
:::
:::{grid-item-card} 🧠 Cómo funciona
:link: explanation/architecture
Arquitectura del sistema y decisiones de diseño.
:::
:::{grid-item-card} 📖 Guías prácticas
:link: howto/index
Desplegar, calibrar, añadir datasets, multi-secuencia.
:::
:::{grid-item-card} 📚 Referencia API
:link: reference/index
Documentación completa de cada módulo y clase.
:::
::::

---

```{toctree}
:caption: Tutoriales
:hidden:
tutorials/quickstart
tutorials/installation
tutorials/first_training
```

```{toctree}
:caption: Guías
:hidden:
howto/index
howto/deploy_api
howto/calibrate_model
howto/multi_sequence
howto/interpret_attention
howto/docker_training
howto/add_new_dataset
```

```{toctree}
:caption: Explicación
:hidden:
explanation/architecture
explanation/mri_sequences
explanation/encoder_freeze
explanation/mil_vs_segmentation
explanation/data_licenses
explanation/design_decisions
```

```{toctree}
:caption: Referencia
:hidden:
reference/index
reference/config
reference/cli
reference/data
reference/models
reference/training
reference/evaluation
reference/serving
```
