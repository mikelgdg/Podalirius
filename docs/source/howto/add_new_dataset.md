# Añadir un dataset nuevo

## 1. Crear extractor de etiquetas

En `src/triagemri/data/labels.py`, añade una función:

```python
def extract_label_from_my_dataset(case_path: Path) -> int:
    ...
    return 0  # normal
    return 1  # abnormal
    return -1 # unknown (skip)
```

## 2. Crear Dataset class

En `src/triagemri/data/datasets.py`, añade una clase que herede de `Dataset`
y devuelva `{"volume": (C,96,96,96), "label": 0/1, "anatomy": "...", ...}`.

## 3. Registrar en la factoría

Añade el builder en `_build_*_datasets()` y el caso en `create_dataloaders()`.

## 4. Configurar

```yaml
# configs/data.yaml
data:
  datasets:
    my_anatomy:
      my_dataset:
        path: "data/raw/my_anatomy"
        label_from: "custom"
```

## 5. Verificar

```bash
python -c "
from triagemri.data.datasets import create_dataloaders
from triagemri.config import load_config
config = load_config()
dls = create_dataloaders(config, enabled_anatomies=['my_anatomy'])
print(f'Train: {len(dls[\"train\"].dataset)} samples')
"
```
