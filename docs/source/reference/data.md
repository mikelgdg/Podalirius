# Módulo `triagemri.data`

## Datasets

```{eval-rst}
.. autoclass:: triagemri.data.datasets.BrainMRIDataset
   :members:
   :undoc-members:

.. autoclass:: triagemri.data.datasets.ProstateMRIDataset
   :members:
   :undoc-members:

.. autoclass:: triagemri.data.datasets.MultiAnatomyDataset
   :members:
   :undoc-members:
```

## Preprocesado

```{eval-rst}
.. automodule:: triagemri.data.preprocessing
   :members: load_nifti, load_and_preprocess, load_and_preprocess_multi,
             normalize_intensity, harmonize_intensity, crop_or_pad,
             validate_file, validate_volume
```

## Transforms

```{eval-rst}
.. automodule:: triagemri.data.transforms
   :members: get_train_transforms, get_val_transforms
```

## Labels

```{eval-rst}
.. automodule:: triagemri.data.labels
   :members: extract_label_from_brats, extract_label_from_oasis,
             extract_label_from_picai, extract_label_generic,
             get_label_distribution, LabelExtractor
```
