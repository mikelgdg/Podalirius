# Módulo `triagemri.models`

## Encoder

```{eval-rst}
.. autoclass:: triagemri.models.encoder.TriadEncoder
   :members:
   :undoc-members:

.. autoclass:: triagemri.models.encoder.LoRALayer
   :members:
```

## MIL

```{eval-rst}
.. autoclass:: triagemri.models.mil.AttentionMIL
   :members:

.. autoclass:: triagemri.models.mil.GatedAttentionMIL
   :members:

.. autoclass:: triagemri.models.mil.MultiScaleAttentionMIL
   :members:

.. autofunction:: triagemri.models.mil.build_mil_head
.. autofunction:: triagemri.models.mil.build_multiscale_mil
```

## Decoder

```{eval-rst}
.. autoclass:: triagemri.models.decoder.AnomalyDecoder
   :members:

.. autoclass:: triagemri.models.decoder.AttentionGate3D
   :members:

.. autofunction:: triagemri.models.decoder.total_variation_3d
```

## Triage Model

```{eval-rst}
.. autoclass:: triagemri.models.triage.TriageModel
   :members:

.. autofunction:: triagemri.models.triage.build_triage_model
```
