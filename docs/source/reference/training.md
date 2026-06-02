# Módulo `triagemri.training`

## Trainer

```{eval-rst}
.. autoclass:: triagemri.training.trainer.TriageLightningModule
   :members:
```

## Losses

```{eval-rst}
.. autoclass:: triagemri.training.losses.FocalLoss
   :members:

.. autoclass:: triagemri.training.losses.WeightedBCEWithLogitsLoss
   :members:

.. autoclass:: triagemri.training.losses.MultiTaskLoss
   :members:

.. autoclass:: triagemri.training.losses.PseudoLabelLoss
   :members:

.. autofunction:: triagemri.training.losses.get_loss_fn
```

## Metrics

```{eval-rst}
.. automodule:: triagemri.training.metrics
   :members: find_threshold_for_sensitivity, find_inconclusive_thresholds,
             compute_triage_metrics, calibration_metrics,
             compute_segmentation_metrics, compute_metrics_per_anatomy,
             format_metrics_report
```
