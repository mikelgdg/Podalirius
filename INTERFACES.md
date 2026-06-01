# Interfaces del proyecto Triage-MRI

Este documento define las interfaces entre módulos. Todos los agentes deben respetar estas firmas.

---

## Flujo de datos

```
NIfTI/DICOM ──→ Preprocessing ──→ tensor (B,1,96,96,96)
     │                                      │
     │                                      ▼
     │                              Encoder.forward()
     │                              → features: (B, N_subvols, 768)
     │                              → attention_weights: (B, N_subvols, 1)
     │                                      │
     │                                      ▼
     │                              MILHead.forward()
     │                              → logits: (B, 1)
     │                              → attention: (B, N_subvols, 1)
     │                                      │
     │                                      ▼
     │                              TriageModel.forward()
     │                              → {"score": (B,), "logits": (B,1),
     │                                 "attention": (B,N,1), "features": (B,N,768)}
```

---

## 1. Preprocessing

### `preprocessing.py` → `load_and_preprocess(path: str | Path) -> torch.Tensor`

```python
def load_and_preprocess(path: str | Path) -> torch.Tensor:
    """
    Carga un volumen NIfTI/DICOM y lo convierte a tensor normalizado.

    Input:  path al archivo .nii.gz o directorio DICOM
    Output: tensor de forma (1, 96, 96, 96), float32, intensidad normalizada a [0, 1]
    """
    ...

def load_nifti(path: str) -> tuple[torch.Tensor, np.ndarray]:
    """
    Carga NIfTI y devuelve (datos 3D, affine matrix)
    Output: tensor (D,H,W), affine (4,4)
    """
    ...
```

---

## 2. Encoder (Triad Swin-B)

### `encoder.py` → `TriadEncoder`

```python
class TriadEncoder(nn.Module):
    """
    Encoder 3D congelado. Carga pesos de Triad Swin-B SimMIM.
    Input:  volume (B, 1, 96, 96, 96)
    Output: features (B, N_subvols, 768) donde N_subvols = número de parches 3D
    """

    def __init__(self, checkpoint_path: str, freeze: bool = True):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) volumen normalizado
        Returns:
            features: (B, N, 768) características por sub-volumen
        """
        ...

    def extract_subvolume_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide el volumen en sub-volúmenes y extrae features de cada uno.
        Sub-volúmenes: 48³ con stride 24 → N = 27 en volumen 96³
        """
        ...
```

---

## 3. Attention-MIL Head

### `mil.py` → `AttentionMIL`

```python
class AttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning head.
    Inspirado en CLAM (Lu et al., 2021).

    Input:  instance_features (B, N, 768) — features por sub-volumen
    Output: logits (B, 1) + attention_weights (B, N, 1)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512,
                 attention_dim: int = 128, dropout: float = 0.3):
        ...

    def forward(self, instance_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            instance_features: (B, N, input_dim)
        Returns:
            logits: (B, 1) predicción raw (antes de sigmoid)
            attention: (B, N, 1) pesos de atención normalizados (softmax)
        """
        ...
```

---

## 4. Modelo completo

### `triage.py` → `TriageModel`

```python
class TriageModel(nn.Module):
    """
    Modelo completo: Triad Encoder → AttentionMIL → Score.
    """

    def __init__(self, encoder: TriadEncoder, mil_head: AttentionMIL):
        ...

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 1, D, H, W) volumen normalizado
        Returns:
            {
                "score": (B,) probabilidad de anormalidad [0,1],
                "logits": (B, 1) logits raw,
                "attention": (B, N, 1) pesos de atención,
                "features": (B, N, 768) características intermedias
            }
        """
        ...
```

---

## 5. Dataset

### `datasets.py` → `BaseMRIDataset`

```python
class BaseMRIDataset(Dataset):
    """
    Dataset base para RM 3D.
    __getitem__ devuelve (volume, label, anatomy, metadata)
    """

    def __init__(self, data_dir: str, split: str = "train",
                 transform=None, anatomy: str = "brain"):
        ...

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            {
                "volume": torch.Tensor de forma (1, 96, 96, 96),
                "label": torch.Tensor escalar (0=normal, 1=abnormal),
                "anatomy": str ("brain"|"prostate"|"breast"),
                "patient_id": str,
                "sequence": str
            }
        """
        ...

    def __len__(self) -> int: ...
```

---

## 6. Métricas

### `metrics.py`

```python
def compute_triage_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    """
    Args:
        y_true: (N,) etiquetas binarias 0/1
        y_scores: (N,) scores de anomalía [0,1]
    Returns:
        {
            "roc_auc": float,
            "pr_auc": float,
            "sensitivity": float,
            "specificity": float,
            "npv": float,
            "ppv": float,
            "threshold": float,
            "discard_rate": float,
            "ece": float
        }
    """
    ...

def find_threshold_for_sensitivity(y_true, y_scores, target_sensitivity=0.99) -> float:
    """Encuentra el threshold que produce la sensibilidad objetivo."""
    ...

def calibration_metrics(y_true, y_scores, n_bins=10) -> dict:
    """ECE, reliability diagram data."""
    ...
```

---

## 7. Pérdidas

### `losses.py`

```python
class FocalLoss(nn.Module):
    """Focal Loss para clases desbalanceadas."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        ...

class WeightedBCELoss(nn.Module):
    """BCE con pesos por clase."""
    def __init__(self, pos_weight: float = None):
        ...
```

---

## 8. Trainer (Lightning)

### `trainer.py` → `TriageTrainer`

```python
class TriageTrainer(pl.LightningModule):
    def __init__(self, model: TriageModel, loss_fn: nn.Module,
                 lr: float, weight_decay: float, warmup_epochs: int, min_lr: float):
        ...

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        ...

    def validation_step(self, batch, batch_idx) -> dict:
        ...

    def configure_optimizers(self): ...
```

---

## 9. Scripts

### `scripts/train.py`

```python
def main():
    # 1. Cargar configs
    # 2. Inicializar encoder + MIL + TriageModel
    # 3. Crear datasets (train/val)
    # 4. Crear TriageTrainer
    # 5. Entrenar con pl.Trainer
    # 6. Guardar checkpoint
    ...

if __name__ == "__main__":
    main()
```

### `scripts/evaluate.py`

```python
def main():
    # 1. Cargar modelo entrenado
    # 2. Cargar test dataset
    # 3. Evaluar métricas por anatomía y global
    # 4. Encontrar thresholds para 99% sensibilidad
    # 5. Guardar resultados en JSON
    ...

if __name__ == "__main__":
    main()
```

### `scripts/demo.py`

```python
def main():
    # 1. Cargar modelo
    # 2. Lanzar Gradio con upload de NIfTI
    # 3. Mostrar score, decisión, tiempo de inferencia
    ...

if __name__ == "__main__":
    main()
```

---

*Este documento es la fuente de verdad para las interfaces del proyecto.*
*Cualquier agente debe leerlo antes de escribir código.*
