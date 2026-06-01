"""Gradio demo application for Triage-MRI.

Provides :func:`create_demo` (Gradio Blocks) and :func:`launch_demo`
(convenience launcher) for interactive triage of 3D MRI volumes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from triagemri.config import Config, load_config
from triagemri.data.preprocessing import load_and_preprocess, load_full_volume_for_display
from triagemri.models import build_triage_model

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Return the project root directory.

    Uses the ``TRIAGEMRI_ROOT`` environment variable if set, otherwise
    falls back to resolving ``__file__`` four levels up.
    """
    env_root = os.environ.get("TRIAGEMRI_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parent.parent.parent.parent


def _get_center_slice(
    volume: torch.Tensor, axis: str = "axial"
) -> np.ndarray:
    """Extract the centre slice of a 3D volume along *axis*.

    Args:
        volume: Tensor of shape ``(1, D, H, W)`` or ``(D, H, W)``.
        axis: ``"axial"`` | ``"sagittal"`` | ``"coronal"``.

    Returns:
        2-D float32 numpy array (H×W style) scaled to [0, 1] for display.
    """
    v = volume.detach().cpu().numpy()
    while v.ndim > 3:
        v = v[0]
    if v.ndim == 3:
        v = v[0] if v.shape[0] == 1 else v

    d, h, w = v.shape

    if axis == "axial":
        img = v[d // 2, :, :]
    elif axis == "sagittal":
        img = v[:, :, w // 2]
    elif axis == "coronal":
        img = v[:, h // 2, :]
    else:
        raise ValueError(f"Unknown axis: {axis}")

    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn > 1e-8:
        img = (img - mn) / (mx - mn)
    return img


class _DemoModelWrapper:
    """Holds model, thresholds, device for the Gradio callback."""

    def __init__(
        self,
        model: torch.nn.Module,
        thresholds: Dict[str, Dict[str, float]],
        device: str,
        anatomy_mode: str,
        default_anatomy: str = "brain",
    ) -> None:
        self.model = model
        self.thresholds = thresholds
        self.device = device
        self.anatomy_mode = anatomy_mode
        self.default_anatomy = default_anatomy


def _overlay_attention(
    mri_slice: np.ndarray,
    attn_slice: np.ndarray,
    alpha: float = 0.55,
    gt_slice: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Overlay model attention (yellow) and optional GT mask (red) on grayscale MRI.

    Returns uint8 RGB image (H, W, 3).
    """
    mri = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    r, g, b = mri.copy(), mri.copy(), mri.copy()

    if gt_slice is not None:
        gt = gt_slice.astype(np.float32)
        if gt.max() > 0:
            gt = (gt > 0).astype(np.float32)
            blend = gt * 0.55
            r = r * (1 - blend) + blend          # red channel for GT

    attn = attn_slice.astype(np.float32)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    blend_attn = attn * alpha
    r = r * (1 - blend_attn) + blend_attn         # yellow = red + green
    g = g * (1 - blend_attn) + blend_attn

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _predict_single(
    wrapper: _DemoModelWrapper,
    volume: torch.Tensor,
    anatomy: str,
) -> Dict[str, Any]:
    t0 = time.time()
    with torch.no_grad():
        output = wrapper.model(volume)
        score = float(output["score"].item())
    elapsed = time.time() - t0

    anat_thresholds = wrapper.thresholds.get(
        anatomy, wrapper.thresholds.get("global", {})
    )
    threshold = float(anat_thresholds.get("threshold", 0.5))
    decision = "REVISAR" if score >= threshold else "NORMAL"

    attn_3d = None
    attn_warning = None
    if "attention" in output:
        attn = output["attention"].cpu().numpy()
        n_subvols = attn.shape[1]
        grid_side = int(round(n_subvols ** (1 / 3)))
        if grid_side ** 3 == n_subvols:
            attn_3d = attn.reshape(grid_side, grid_side, grid_side)
        else:
            attn_warning = "Attention grid non-cubic — display simplified"

    return {
        "score": score,
        "decision": decision,
        "threshold": threshold,
        "time_s": elapsed,
        "attn_3d": attn_3d,
        "attn_warning": attn_warning,
    }


# ------------------------------------------------------------------
# Gradio callback
# ------------------------------------------------------------------


def _process_upload(
    file_obj: Any,
    case_id: str,
    anatomy: str,
    wrapper: Optional[_DemoModelWrapper],
    original_path: str = "",
) -> Tuple[Any, ...]:
    """Handle upload: preprocess → predict → display.

    Returns a Gradio update tuple for all UI components.
    Returns empty updates on error (no ``gr.Error`` to keep inline).
    """
    if wrapper is None:
        return _error_update("Model not loaded.")

    if file_obj is None:
        return _error_update("No file uploaded.")

    try:
        if isinstance(file_obj, dict):
            file_path = str(file_obj.get("path", file_obj.get("name", "")))
        elif isinstance(file_obj, str):
            file_path = file_obj
        else:
            file_path = str(getattr(file_obj, "name", file_obj))
    except Exception:
        return _error_update("Could not read uploaded file.")

    if original_path and os.path.exists(original_path):
        file_path = original_path

    try:
        volume = load_and_preprocess(file_path)
        volume_display, crop_offsets, crop_shape = load_full_volume_for_display(file_path)
    except Exception as e:
        return _error_update(f"Preprocessing error: {e}")

    if volume.dim() == 3:
        volume = volume.unsqueeze(0)  # (D,H,W) → (C,D,H,W)
    if volume.dim() == 4:
        volume = volume.unsqueeze(0)  # (C,D,H,W) → (B,C,D,H,W)
    volume = volume.to(wrapper.device)

    result = _predict_single(wrapper, volume, anatomy)

    score = result["score"]
    decision = result["decision"]
    threshold = result["threshold"]
    elapsed = result["time_s"]

    is_normal = decision == "NORMAL"

    # Load ground truth segmentation mask for overlay
    # case_id here is the raw ID, but _load_seg_for_case now expects the label_id (with prefix)
    # to use the registry. We need to find the key in the registry.
    case_label_id = None
    for k, v in _CASE_REGISTRY.items():
        if v["case_id"] == case_id:
            case_label_id = k
            break
    
    seg_3d = None
    if case_label_id:
        seg_3d = _load_seg_for_case(case_label_id)
    elif case_id and case_id.strip():
        # Fallback for manual uploads
        seg_3d = _load_seg_for_case(case_id.strip())

    # Create overlay images: full brain display + attention positioned at crop offset
    volume_disp = volume_display.unsqueeze(0).unsqueeze(0)  # (1, 1, 96, 96, 96)
    attn_3d = result.get("attn_3d")
    if attn_3d is not None:
        from scipy.ndimage import zoom as ndzoom
        grid_side = attn_3d.shape[0]
        x0, y0, z0 = crop_offsets
        x_sz, y_sz, z_sz = crop_shape

        attn_3d_raw = attn_3d.copy()

        attn_3d = np.power(attn_3d, 0.5)

        attn_scaled = ndzoom(
            attn_3d.astype(np.float32),
            (x_sz / grid_side, y_sz / grid_side, z_sz / grid_side),
            order=1,
        )
        baseline = 1.0 / (grid_side ** 3)
        attn_scaled[attn_scaled < baseline] = 0.0
        mx = attn_scaled.max()
        if mx > 0:
            attn_scaled /= mx

        attn_full = np.zeros((96, 96, 96), dtype=np.float32)
        attn_full[x0:x0 + x_sz, y0:y0 + y_sz, z0:z0 + z_sz] = attn_scaled
    else:
        attn_full = None
        attn_3d_raw = None

    axial_raw = np.rot90(_get_center_slice(volume_disp, "axial"))
    sagittal_raw = np.rot90(_get_center_slice(volume_disp, "sagittal"))
    coronal_raw = np.rot90(_get_center_slice(volume_disp, "coronal"))

    if attn_full is not None and not is_normal:
        d, h, w = attn_full.shape
        attn_axial = np.rot90(attn_full[:, :, w // 2])
        attn_sagittal = np.rot90(attn_full[d // 2, :, :])
        attn_coronal = np.rot90(attn_full[:, h // 2, :])

        gt_axial = gt_sagittal = gt_coronal = None
        if seg_3d is not None:
            sd, sh, sw = seg_3d.shape
            gt_axial = np.rot90(seg_3d[:, :, sw // 2])
            gt_sagittal = np.rot90(seg_3d[sd // 2, :, :])
            gt_coronal = np.rot90(seg_3d[:, sh // 2, :])

        axial = _overlay_attention(axial_raw, attn_axial, gt_slice=gt_axial)
        sagittal = _overlay_attention(sagittal_raw, attn_sagittal, gt_slice=gt_sagittal)
        coronal = _overlay_attention(coronal_raw, attn_coronal, gt_slice=gt_coronal)
    else:
        axial = (axial_raw * 255).astype(np.uint8)
        sagittal = (sagittal_raw * 255).astype(np.uint8)
        coronal = (coronal_raw * 255).astype(np.uint8)

    info_lines = [
        f"### Resultado: {'NORMAL' if is_normal else 'REVISAR'}",
        f"- **Score:** {score:.4f}",
        f"- **Threshold:** {threshold:.4f}",
        f"- **Inference time:** {elapsed*1000:.1f} ms",
        f"- **Anatomy selected:** {anatomy}",
    ]

    attn_warning = result.get("attn_warning")
    if attn_warning:
        info_lines.append(f"- **Warning:** {attn_warning}")

    if case_id and case_id.strip():
        gt_label = _get_ground_truth(case_id.strip())
        if gt_label is not None:
            gt_str = "Anormal (tumor)" if gt_label == 1 else "Normal"
            pred_label = 1 if not is_normal else 0
            icon = "✔" if gt_label == pred_label else "✘"
            info_lines.insert(1, f"- **GT:** {gt_str}  {icon}")

    return (
        axial,
        sagittal,
        coronal,
        "\n".join(info_lines),
        f"**{decision}**",
        _create_3d_plot(volume_disp, attn_full, seg_3d, crop_offsets, crop_shape, attn_grid=attn_3d_raw, is_normal=is_normal),
        None,  # status
    )


def _extract_case_id_from_upload(file_obj: Any) -> str:
    """Extract BraTS case ID from the uploaded file's original name.

    Example: ``BraTS-MEN-00004-000-t1c.nii.gz`` → ``BraTS-MEN-00004-000``
    """
    orig_name = ""
    if isinstance(file_obj, dict):
        orig_name = file_obj.get("orig_name", file_obj.get("name", ""))
    elif hasattr(file_obj, "orig_name"):
        orig_name = file_obj.orig_name or ""
    elif hasattr(file_obj, "name"):
        orig_name = Path(file_obj.name).name

    if not orig_name or not orig_name.lower().startswith("brats"):
        return ""

    parts = orig_name.split("-")
    if len(parts) >= 4:
        return "-".join(parts[:4])
    return ""


def _error_update(msg: str) -> Tuple[Any, ...]:
    import plotly.graph_objects as go
    blank = (np.zeros((100, 100), dtype=np.float32) * 255).astype(np.uint8)
    empty_plot = go.Figure()
    return (
        blank,
        blank,
        blank,
        f"**Error:** {msg}",
        "ERROR",
        empty_plot,
        f"Error: {msg}",
    )


def _get_ground_truth(case_id: str) -> Optional[int]:
    """Determine ground truth label for a case.

    Checks the registry first (supports brain and prostate), then falls
    back to scanning BraTS segmentation masks.
    Returns 1 (abnormal), 0 (normal), or None if unavailable.
    """
    for entry in _CASE_REGISTRY.values():
        if entry.get("case_id") == case_id:
            return entry.get("label")

    data_root = _get_project_root() / "data" / "raw" / "brain" / "brats"
    case_dir = data_root / case_id
    seg_files = sorted(case_dir.glob("*seg*.nii.gz")) if case_dir.is_dir() else []
    if not seg_files:
        return None

    try:
        import nibabel as nib
        seg = nib.load(str(seg_files[0]))
        data = seg.get_fdata()
        if isinstance(data, np.ndarray) and np.any(data > 0):
            return 1
        return 0
    except Exception:
        return None


def _build_3d_mesh(volume: np.ndarray, level: float, color: str, opacity: float, name: str) -> Dict[str, Any]:
    """Build a Plotly Mesh3d dict from a 3D volume using marching cubes."""
    from skimage.measure import marching_cubes

    v = np.pad(volume.astype(np.float32), pad_width=1, mode="constant", constant_values=0)
    verts, faces, _, _ = marching_cubes(v, level=level)
    verts -= 1.0

    return {
        "type": "mesh3d",
        "x": verts[:, 2].tolist(),
        "y": verts[:, 1].tolist(),
        "z": verts[:, 0].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "color": color,
        "opacity": opacity,
        "name": name,
    }


def _add_voxel_cube(traces: list, x_range: tuple, y_range: tuple, z_range: tuple, weight: float, name: str):
    """Adds a 3D box (6 faces) to the traces list with given opacity."""
    import plotly.graph_objects as go
    if weight < 0.05:
        return
        
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    
    # 8 corners of the box
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    
    # 12 triangles forming the 6 faces
    i = [0, 1, 2, 3, 0, 1, 2, 3, 0, 4, 1, 5]
    j = [1, 2, 3, 0, 4, 5, 6, 7, 4, 7, 5, 6]
    k = [4, 5, 6, 7, 1, 2, 3, 0, 3, 3, 2, 2]
    # (Actually simpler index for Plotly Mesh3d box)
    i = [7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    traces.append(go.Mesh3d(
        x=z, y=y, z=x, # Align with (X,Y,Z) -> (z,y,x) in plotly convention
        i=i, j=j, k=k,
        color="yellow",
        opacity=min(0.6, weight * 1.5), # Boost opacity for visibility
        name=name,
        showlegend=False
    ))

def _draw_grid_wireframe(traces: list, x0: float, y0: float, z0: float, x_sz: float, y_sz: float, z_sz: float, grid_side: int = 3):
    """Adds a 3D wireframe (grid lines) for the attention grid."""
    import plotly.graph_objects as go

    xs = [x0 + i * (x_sz / grid_side) for i in range(grid_side + 1)]
    ys = [y0 + i * (y_sz / grid_side) for i in range(grid_side + 1)]
    zs = [z0 + i * (z_sz / grid_side) for i in range(grid_side + 1)]
    
    line_style = dict(color="rgba(150, 150, 150, 0.3)", width=2)
    
    # Draw lines along each axis
    for x in xs:
        for y in ys:
            traces.append(go.Scatter3d(x=[z0, z0+z_sz], y=[y, y], z=[x, x], mode="lines", line=line_style, showlegend=False))
    for x in xs:
        for z in zs:
            traces.append(go.Scatter3d(x=[z, z], y=[y0, y0+y_sz], z=[x, x], mode="lines", line=line_style, showlegend=False))
    for y in ys:
        for z in zs:
            traces.append(go.Scatter3d(x=[z, z], y=[y, y], z=[x0, x0+x_sz], mode="lines", line=line_style, showlegend=False))

def _create_3d_plot(
    volume_np: torch.Tensor,
    attn_full: Optional[np.ndarray],
    seg_3d: Optional[np.ndarray],
    crop_offsets: Tuple[int, int, int],
    crop_shape: Tuple[int, int, int],
    attn_grid: Optional[np.ndarray] = None,
    is_normal: bool = False,
) -> Any:
    """Build an interactive Plotly 3D figure with brain, GT tumor, and attention grid."""
    import plotly.graph_objects as go

    v = volume_np.squeeze().cpu().numpy().astype(np.float32)
    traces = []

    # Brain surface (X-Ray mode)
    vol_disp = v.copy()
    mn, mx = vol_disp.min(), vol_disp.max()
    if mx - mn > 1e-8:
        vol_disp = (vol_disp - mn) / (mx - mn)
    brain_level = 0.25
    traces.append(_build_3d_mesh(vol_disp, brain_level, "lightgray", 0.05, "Volume"))

    if seg_3d is not None and seg_3d.max() > 0:
        traces.append(_build_3d_mesh(seg_3d, 0.5, "red", 0.45, "GT (tumor)"))

    if attn_grid is not None and not is_normal:
        grid_side = attn_grid.shape[0]
        x0, y0, z0 = crop_offsets
        x_sz, y_sz, z_sz = crop_shape

        _draw_grid_wireframe(traces, x0, y0, z0, x_sz, y_sz, z_sz, grid_side=grid_side)

        dx = x_sz / grid_side
        dy = y_sz / grid_side
        dz = z_sz / grid_side

        attn_flat = attn_grid.flatten()
        n_cells = grid_side ** 3
        baseline = 1.0 / n_cells
        attn_max = attn_flat.max()
        denom = attn_max - baseline if attn_max > baseline else 1.0
        layer_size = grid_side * grid_side

        for idx in range(n_cells):
            w = attn_flat[idx]
            if w <= baseline:
                continue
            norm_w = (w - baseline) / denom
            i = idx // layer_size
            j = (idx % layer_size) // grid_side
            k = idx % grid_side
            _add_voxel_cube(
                traces,
                x_range=(x0 + i * dx, x0 + (i + 1) * dx),
                y_range=(y0 + j * dy, y0 + (j + 1) * dy),
                z_range=(z0 + k * dz, z0 + (k + 1) * dz),
                weight=norm_w,
                name=f"Attn({i},{j},{k}): {w:.4f}",
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def _load_seg_for_case(case_label_id: str) -> Optional[np.ndarray]:
    """Load and resize a segmentation mask for the given case ID string.

    For brain cases: looks for BraTS *seg*.nii.gz.
    For prostate cases: looks for PI-CAI csPCa lesion delineations.
    Returns a (96, 96, 96) binary numpy array, or None if unavailable.
    """
    import torch.nn.functional as F
    import nibabel as nib

    entry = _CASE_REGISTRY.get(case_label_id)
    if not entry:
        if not case_label_id.startswith("BraTS"):
            return None
        root = _get_project_root() / "data" / "raw" / "brain" / "brats"
        case_dir = root / case_label_id
    else:
        case_path = Path(entry["path"])
        case_dir = case_path.parent \
            if case_label_id.startswith("[Brain]") \
            else None
        anatomy = entry.get("anatomy", "brain")

        if anatomy == "prostate":
            return _load_prostate_seg(case_label_id)

    if case_dir is not None:
        seg_files = sorted(case_dir.glob("*seg*.nii.gz")) if case_dir.is_dir() else []
    else:
        seg_files = []
    if not seg_files:
        return None

    try:
        seg = nib.load(str(seg_files[0]))
        data = seg.get_fdata()
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        seg_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        seg_tensor = F.interpolate(seg_tensor, size=(96, 96, 96), mode="nearest")
        return (seg_tensor.squeeze() > 0).float().numpy()
    except Exception:
        return None


def _load_prostate_seg(case_label_id: str) -> Optional[np.ndarray]:
    """Load PI-CAI csPCa lesion mask for a prostate case."""
    import torch.nn.functional as F
    import nibabel as nib

    entry = _CASE_REGISTRY.get(case_label_id)
    if not entry:
        return None
    study_id = entry.get("case_id", "")
    if not study_id:
        return None

    project_root = _get_project_root()
    resampled_dir = project_root / "data" / "raw" / "prostate" / "picai_labels" / \
                    "csPCa_lesion_delineations" / "human_expert" / "resampled"
    if not resampled_dir.is_dir():
        return None

    mask_files = list(resampled_dir.glob(f"*_{study_id}.nii.gz"))
    if not mask_files:
        return None

    try:
        seg = nib.load(str(mask_files[0]))
        data = seg.get_fdata()
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        seg_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        seg_tensor = F.interpolate(seg_tensor, size=(96, 96, 96), mode="nearest")
        return (seg_tensor.squeeze() > 0).float().numpy()
    except Exception:
        return None


# Global registry to store case metadata for the demo
_CASE_REGISTRY: Dict[str, Dict[str, Any]] = {}

def _list_available_cases() -> List[str]:
    """Scan the data directory and return ONLY cases from the official test split.

    Uses the same dataset logic (seed=42, 70/15/15 split) to ensure
    scientific integrity and that we only test on unseen data.
    Includes brain (BraTS/OASIS) and prostate (PI-CAI) cases.
    """
    from triagemri.data.datasets import BrainMRIDataset, ProstateMRIDataset

    global _CASE_REGISTRY
    if _CASE_REGISTRY:
        return sorted(_CASE_REGISTRY.keys())

    project_root = _get_project_root()
    split_kwargs = dict(seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    def _brain_path(case: dict, _vpaths: dict) -> str | None:
        return _vpaths.get("t1ce") or _vpaths.get("t1") or next(iter(_vpaths.values()), None)

    def _prostate_path(case: dict, _vpaths: dict) -> str | None:
        return _vpaths.get("t2w") or next(iter(_vpaths.values()), None)

    # Brain
    try:
        ds = BrainMRIDataset(
            data_root=project_root / "data" / "raw" / "brain",
            split="test",
            sources=("brats", "oasis"),
            **split_kwargs,
        )
        for case in ds._cases:
            prefix = "Positive" if case["label"] == 1 else "Negative"
            display_name = f"[Brain] {prefix}: {case['case_id']}"
            vpaths = case.get("volume_paths", {})
            fpath = _brain_path(case, vpaths)
            if fpath:
                _CASE_REGISTRY[display_name] = {
                    "path": str(Path(fpath).resolve()),
                    "case_id": case["case_id"],
                    "label": case["label"],
                    "anatomy": "brain",
                }
    except Exception as e:
        print(f"Error loading brain test split: {e}")

    # Prostate
    try:
        pi_cai_dir = project_root / "data" / "raw" / "prostate" / "pi_cai"
        labels_dir = pi_cai_dir.parent / "picai_labels"
        ds = ProstateMRIDataset(
            data_root=pi_cai_dir,
            labels_dir=labels_dir,
            split="test",
            preferred_sequence="t2w",
            **split_kwargs,
        )
        for case in ds._cases:
            prefix = "Positive" if case["label"] == 1 else "Negative"
            display_name = f"[Prostate] {prefix}: {case['case_id']}"
            vpaths = case.get("volume_paths", {})
            fpath = _prostate_path(case, vpaths)
            if fpath:
                _CASE_REGISTRY[display_name] = {
                    "path": str(Path(fpath).resolve()),
                    "case_id": case["case_id"],
                    "label": case["label"],
                    "anatomy": "prostate",
                }
    except Exception as e:
        print(f"Error loading prostate test split: {e}")

    return sorted(_CASE_REGISTRY.keys())


def _get_case_file(case_label_id: str) -> Optional[str]:
    """Lookup the absolute path for a given case string from the registry."""
    entry = _CASE_REGISTRY.get(case_label_id)
    return entry["path"] if entry else None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def create_demo(
    model: torch.nn.Module,
    thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    device: str = "cuda",
    anatomy_mode: str = "shared",
    available_anatomies: Optional[List[str]] = None,
) -> Any:
    """Create a Gradio Blocks interface for MRI triage.

    Args:
        model: A trained :class:`~triagemri.models.TriageModel`.
        thresholds: Per-anatomy (and ``"global"``) threshold dict, or
            ``None`` to default to 0.5 everywhere.
        device: Torch device string.
        anatomy_mode: ``"shared"`` | ``"multi_head"``.
        available_anatomies: List of selectable anatomies in the dropdown.

    Returns:
        A ``gradio.Blocks`` instance ready for ``.launch()``.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "gradio is required for the demo.  Install with: pip install gradio"
        )

    if thresholds is None:
        thresholds = {}

    if available_anatomies is None:
        available_anatomies = ["brain", "prostate", "breast"]

    default_anatomy = available_anatomies[0]

    wrapper = _DemoModelWrapper(
        model=model,
        thresholds=thresholds,
        device=device,
        anatomy_mode=anatomy_mode,
        default_anatomy=default_anatomy,
    )

    custom_css = """
    .result-normal { background-color: #e8f5e9 !important; }
    .result-revisar { background-color: #ffebee !important; }
    #result-header { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
    """

    with gr.Blocks(title="Triage-MRI Screening", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Triage-MRI: Normality Screening for 3D MRI

            Upload a NIfTI (`.nii.gz`) or MetaImage (`.mha`) volume or select a test case to triage studies 
            into **NORMAL** (likely no anomalies) or **REVISAR** (needs radiologist review).

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Upload"):
                    file_input = gr.File(
                        label="Upload NIfTI / MetaImage Volume",
                        file_count="single",
                        type="filepath",
                    )
                
                with gr.Tab("Test Dataset"):
                    available_cases = _list_available_cases()
                    case_selector = gr.Dropdown(
                        label="Select Test Case",
                        choices=available_cases,
                        value=None,
                        interactive=True,
                    )
                    load_case_btn = gr.Button("Load Case")

                case_id_input = gr.Textbox(
                    label="Case ID (auto-filled)",
                    placeholder="auto-detected or manually entered",
                )

                anatomy_dropdown = gr.Dropdown(
                    label="Anatomy",
                    choices=available_anatomies,
                    value=default_anatomy,
                    interactive=True,
                )

                registry_path = gr.State("")

                def _on_file_upload(f):
                    return _extract_case_id_from_upload(f), ""

                file_input.upload(
                    fn=_on_file_upload,
                    inputs=[file_input],
                    outputs=[case_id_input, registry_path],
                )

                def _handle_case_selection(label_id):
                    if not label_id:
                        return None, "", "brain", ""
                    fpath = _get_case_file(label_id)
                    actual_id = label_id.split(": ")[1] if ": " in label_id else label_id
                    entry = _CASE_REGISTRY.get(label_id, {})
                    anatomy = entry.get("anatomy", "brain")
                    return fpath, actual_id, anatomy, fpath

                load_case_btn.click(
                    fn=_handle_case_selection,
                    inputs=[case_selector],
                    outputs=[file_input, case_id_input, anatomy_dropdown, registry_path],
                )

                submit_btn = gr.Button("Run Triage", variant="primary", size="lg")

                status_text = gr.Markdown("")

            with gr.Column(scale=2):
                result_panel = gr.Markdown("### Resultado: —", elem_id="result-header")
                decision_text = gr.Markdown("")

                with gr.Accordion("Details", open=True):
                    info_text = gr.Markdown("Awaiting input…")

        gr.Markdown("---")
        gr.Markdown("### Visualización 3D")
        gr.Markdown("*Gris = volumen | Rojo = GT (tumor real) | Amarillo = atención del modelo*")

        plot_3d = gr.Plot(label="3D View")

        gr.Markdown("---")
        gr.Markdown("### Cortes 2D con Heatmap")
        gr.Markdown("*Rojo = GT (tumor real) | Amarillo = atención del modelo*")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Axial**")
                axial_img = gr.Image(label="", height=350)
            with gr.Column():
                gr.Markdown("**Sagittal**")
                sagittal_img = gr.Image(label="", height=350)
            with gr.Column():
                gr.Markdown("**Coronal**")
                coronal_img = gr.Image(label="", height=350)

        submit_btn.click(
            fn=lambda f, c, a, r, w=wrapper: _process_upload(f, c, a, w, r),
            inputs=[file_input, case_id_input, anatomy_dropdown, registry_path],
            outputs=[
                axial_img,
                sagittal_img,
                coronal_img,
                info_text,
                decision_text,
                plot_3d,
                status_text,
            ],
        )

    return demo


def launch_demo(
    checkpoint_path: str,
    config_dir: str = "configs",
    device: str = "cuda",
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    thresholds_file: Optional[str] = None,
    available_anatomies: Optional[List[str]] = None,
) -> Any:
    """Load a model and launch the Gradio demo.

    Args:
        checkpoint_path: Path to ``.pt`` / ``.pth`` model checkpoint.
        config_dir: Directory with config YAML files.
        device: Torch device.
        share: Whether to create a public Gradio link.
        server_name: Host to bind to.
        server_port: Port to listen on.
        thresholds_file: Optional JSON file with per-anatomy thresholds.
        available_anatomies: Override anatomy dropdown choices.

    Returns:
        The running Gradio demo (Blocks instance).
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "gradio is required for the demo.  Install with: pip install gradio"
        )

    print(f"Loading config from: {config_dir}")
    config = load_config(config_dir)

    print(f"Loading model from: {checkpoint_path}")
    model = build_triage_model(checkpoint_path, device=device)
    print(f"Model loaded. Anatomy mode: {model.anatomy_mode}")

    thresholds: Dict[str, Dict[str, float]] = {}
    if thresholds_file and Path(thresholds_file).exists():
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        print(f"Thresholds loaded from: {thresholds_file}")
    else:
        print("No thresholds file provided — using default threshold 0.5.")

    if available_anatomies is None:
        available_anatomies = ["brain", "prostate", "breast"]

    demo = create_demo(
        model=model,
        thresholds=thresholds,
        device=device,
        anatomy_mode=model.anatomy_mode,
        available_anatomies=available_anatomies,
    )

    print(f"\nLaunching demo on http://{server_name}:{server_port}")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
    return demo
