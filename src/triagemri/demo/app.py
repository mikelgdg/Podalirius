"""Gradio demo application for Triage-MRI.

Provides :func:`create_demo` (Gradio Blocks) and :func:`launch_demo`
(convenience launcher) for interactive triage of 3D MRI volumes.
Supports multi-sequence upload, decoder heatmaps, and inconclusive zones.
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

from triagemri.config import load_config
from triagemri.data.preprocessing import (
    load_and_preprocess,
    load_full_volume_for_display,
)
from triagemri.models import build_triage_model

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    env_root = os.environ.get("TRIAGEMRI_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[3]


def _get_center_slice(
    volume: torch.Tensor, axis: str = "axial"
) -> np.ndarray:
    """Extract the centre slice of a 3D volume along *axis*.

    Axial = slice at D/2 (plane perpendicular to depth axis in NIfTI convention).
    Sagittal = slice at W/2 (plane perpendicular to left-right axis).
    Coronal = slice at H/2 (plane perpendicular to anterior-posterior axis).

    Args:
        volume: Tensor of shape ``(1, D, H, W)`` or ``(D, H, W)``.
        axis: ``"axial"``, ``"sagittal"``, or ``"coronal"``.

    Returns:
        2-D float32 numpy array scaled to [0, 1].
    """
    v = volume.detach().cpu().numpy()
    while v.ndim > 3:
        v = v[0]
    if v.ndim == 3:
        v = v[0] if v.shape[0] == 1 else v

    d, h, w = v.shape

    if axis == "axial":
        img = v[:, :, w // 2]
    elif axis == "sagittal":
        img = v[d // 2, :, :]
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
    def __init__(
        self,
        model: torch.nn.Module,
        thresholds: Dict[str, Dict[str, float]],
        device: str,
        anatomy_mode: str,
        default_anatomy: str = "brain",
        inconclusive_range: Optional[Tuple[float, float]] = None,
        seg_model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.model = model
        self.thresholds = thresholds
        self.device = device
        self.anatomy_mode = anatomy_mode
        self.default_anatomy = default_anatomy
        self.inconclusive_range = inconclusive_range
        self.seg_model = seg_model


def _overlay_attention(
    mri_slice: np.ndarray,
    attn_slice: np.ndarray,
    alpha: float = 0.55,
    gt_slice: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Overlay model attention (yellow) and optional GT mask (red) on grayscale MRI."""
    mri = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    r, g, b = mri.copy(), mri.copy(), mri.copy()

    if gt_slice is not None:
        gt = gt_slice.astype(np.float32)
        if gt.max() > 0:
            gt = (gt > 0).astype(np.float32)
            blend = gt * 0.55
            r = r * (1 - blend) + blend

    attn = attn_slice.astype(np.float32)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    blend_attn = attn * alpha
    r = r * (1 - blend_attn) + blend_attn
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

    decision = "REVISAR"
    if wrapper.inconclusive_range is not None:
        low, high = wrapper.inconclusive_range
        if score < low:
            decision = "NORMAL"
        elif score < high:
            decision = "INCONCLUSIVE"
    else:
        if score < threshold:
            decision = "NORMAL"

    attn_3d = None
    attn_warning = None
    heatmap_3d = None

    if "heatmap" in output:
        hm = torch.sigmoid(output["heatmap"]).cpu().numpy()
        heatmap_3d = hm[0, 0]

    # Run segmentation model if available (Model B → heatmap override)
    seg_heatmap_3d = None
    if wrapper.seg_model is not None and decision != "NORMAL":
        with torch.no_grad():
            seg_out = wrapper.seg_model(volume)
            shm = torch.sigmoid(seg_out["heatmap"]).cpu().numpy()
            seg_heatmap_3d = shm[0, 0]

    if "attention" in output:
        attn = output["attention"].cpu().numpy()
        n_subvols = attn.shape[1]
        # Multi-scale: find the largest cubic sub-grid (finest resolution)
        largest_side = 1
        for side in range(round(n_subvols ** (1 / 3)), 1, -1):
            if side ** 3 <= n_subvols:
                largest_side = side
                break
        cubic_n = largest_side ** 3
        if cubic_n > 1:
            # The first cubic_n tokens are the finest scale (stage3 = 6³ = 216)
            attn_3d = attn[:, :cubic_n, :].reshape(largest_side, largest_side, largest_side)
            if n_subvols > cubic_n:
                attn_warning = (
                    f"Multi-scale: showing finest grid ({largest_side}³={cubic_n} "
                    f"of {n_subvols} tokens)"
                )
        else:
            attn_warning = "Attention grid non-cubic — display simplified"

    return {
        "score": score,
        "decision": decision,
        "threshold": threshold,
        "time_s": elapsed,
        "attn_3d": attn_3d,
        "attn_warning": attn_warning,
        "heatmap_3d": heatmap_3d,
        "seg_heatmap_3d": seg_heatmap_3d,
    }


def _process_upload(
    file_obj: Any,
    case_id: str,
    anatomy: str,
    wrapper: Optional[_DemoModelWrapper],
    original_path: str = "",
) -> Tuple[Any, ...]:
    if wrapper is None:
        return _error_update("Model not loaded.")

    if file_obj is None:
        return _error_update("No file uploaded.")

    try:
        if isinstance(file_obj, (list, tuple)):
            file_path = str(file_obj[0]) if file_obj else ""
        elif isinstance(file_obj, dict):
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
        volume_display, crop_offsets, crop_shape = load_full_volume_for_display(
            file_path
        )
    except Exception as e:
        return _error_update(f"Preprocessing error: {e}")

    if volume.dim() == 3:
        volume = volume.unsqueeze(0)
    if volume.dim() == 4:
        volume = volume.unsqueeze(0)
    volume = volume.to(wrapper.device)

    result = _predict_single(wrapper, volume, anatomy)

    score = result["score"]
    decision = result["decision"]
    seg_heatmap_3d = result.get("seg_heatmap_3d")
    heatmap_3d = result.get("heatmap_3d")

    case_label_id = None
    for k, v in _CASE_REGISTRY.items():
        if v["case_id"] == case_id:
            case_label_id = k
            break

    seg_3d = None
    if case_label_id:
        seg_3d = _load_seg_for_case(case_label_id)
    elif case_id and case_id.strip():
        seg_3d = _load_seg_for_case(case_id.strip())

    volume_disp = volume_display.unsqueeze(0).unsqueeze(0)

    if seg_heatmap_3d is not None and not (decision == "NORMAL"):
        # Model B: full 96³ segmentation heatmap — no crop interpolation needed
        attn_full = np.clip(seg_heatmap_3d, 0, 1)
        if attn_full.max() > 0:
            attn_full /= attn_full.max()
    elif heatmap_3d is not None and not (decision == "NORMAL"):
        hm = heatmap_3d.copy()
        hm = np.clip(hm, 0, 1)
        if hm.max() > 0:
            hm /= hm.max()
        attn_full = hm
    else:
        attn_3d = result.get("attn_3d")
        if attn_3d is not None:
            from scipy.ndimage import zoom as ndzoom

            grid_side = attn_3d.shape[0]
            x0, y0, z0 = crop_offsets
            x_sz, y_sz, z_sz = crop_shape

            attn_full_raw = np.power(attn_3d, 0.5)
            attn_scaled = ndzoom(
                attn_full_raw.astype(np.float32),
                (x_sz / grid_side, y_sz / grid_side, z_sz / grid_side),
                order=1,
            )
            baseline = 1.0 / (grid_side ** 3)
            attn_scaled[attn_scaled < baseline] = 0.0
            mx = attn_scaled.max()
            if mx > 0:
                attn_scaled /= mx

            attn_full = np.zeros((96, 96, 96), dtype=np.float32)
            attn_full[
                x0 : x0 + x_sz, y0 : y0 + y_sz, z0 : z0 + z_sz
            ] = attn_scaled
        else:
            attn_full = None

    axial_raw = np.rot90(_get_center_slice(volume_disp, "axial"))
    sagittal_raw = np.rot90(_get_center_slice(volume_disp, "sagittal"))
    coronal_raw = np.rot90(_get_center_slice(volume_disp, "coronal"))

    is_normal = decision == "NORMAL"

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

    decision_emoji = {"NORMAL": "NORMAL", "INCONCLUSIVE": "INCONCLUSIVE", "REVISAR": "REVISAR"}
    info_lines = [
        f"### Resultado: {decision_emoji.get(decision, decision)}",
        f"- **Score:** {score:.4f}",
        f"- **Threshold:** {result['threshold']:.4f}",
        f"- **Inference time:** {result['time_s'] * 1000:.1f} ms",
        f"- **Anatomy selected:** {anatomy}",
    ]

    attn_warning = result.get("attn_warning")
    if attn_warning:
        info_lines.append(f"- **Warning:** {attn_warning}")

    if case_id and case_id.strip():
        gt_label = _get_ground_truth(case_id.strip())
        if gt_label is not None:
            gt_str = "Anormal (tumor)" if gt_label == 1 else "Normal"
            pred_label = 1 if decision == "REVISAR" else 0
            icon = "OK" if gt_label == pred_label else "FAIL"
            info_lines.insert(1, f"- **GT:** {gt_str}  {icon}")
        # Show mask availability
        has_mask = any(
            e.get("has_mask") and e.get("case_id") == case_id.strip()
            for e in _CASE_REGISTRY.values()
        )
        mask_str = "Sí" if has_mask else "No"
        info_lines.insert(2, f"- **Máscara GT:** {mask_str}")

    # Determine attention grid resolution for wireframe
    attn_grid_side = None
    raw_attn = result.get("attn_3d")
    if raw_attn is not None:
        attn_grid_side = raw_attn.shape[0]

    return (
        axial,
        sagittal,
        coronal,
        "\n".join(info_lines),
        f"**{decision}**",
        _create_3d_plot(
            volume_disp,
            attn_full,
            seg_3d,
            crop_offsets,
            crop_shape,
            attn_grid_side=attn_grid_side,
            is_normal=is_normal,
        ),
        None,
    )


def _extract_case_id_from_upload(file_obj: Any) -> str:
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
    return (blank, blank, blank, f"**Error:** {msg}", "ERROR", empty_plot, f"Error: {msg}")


def _get_ground_truth(case_id: str) -> Optional[int]:
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


def _build_3d_mesh(
    volume: np.ndarray, level: float, color: str, opacity: float, name: str
) -> Dict[str, Any]:
    from skimage.measure import marching_cubes

    v = np.pad(
        volume.astype(np.float32), pad_width=1, mode="constant", constant_values=0
    )
    verts, faces, _, _ = marching_cubes(v, level=level)
    verts -= 1.0

    return {
        "type": "mesh3d",
        "x": verts[:, 0].tolist(),
        "y": verts[:, 1].tolist(),
        "z": verts[:, 2].tolist(),
        "i": faces[:, 0].tolist(),
        "j": faces[:, 1].tolist(),
        "k": faces[:, 2].tolist(),
        "color": color,
        "opacity": opacity,
        "name": name,
    }


def _draw_grid_wireframe(traces, x0, y0, z0, x_sz, y_sz, z_sz, grid_side=3):
    import plotly.graph_objects as go

    xs = [x0 + i * (x_sz / grid_side) for i in range(grid_side + 1)]
    ys = [y0 + i * (y_sz / grid_side) for i in range(grid_side + 1)]
    zs = [z0 + i * (z_sz / grid_side) for i in range(grid_side + 1)]
    line_style = dict(color="rgba(150, 150, 150, 0.3)", width=2)
    for x in xs:
        for y in ys:
            traces.append(
                go.Scatter3d(
                    x=[x, x],
                    y=[y, y],
                    z=[z0, z0 + z_sz],
                    mode="lines",
                    line=line_style,
                    showlegend=False,
                )
            )
    for x in xs:
        for z in zs:
            traces.append(
                go.Scatter3d(
                    x=[x, x],
                    y=[y0, y0 + y_sz],
                    z=[z, z],
                    mode="lines",
                    line=line_style,
                    showlegend=False,
                )
            )
    for y in ys:
        for z in zs:
            traces.append(
                go.Scatter3d(
                    x=[x0, x0 + x_sz],
                    y=[y, y],
                    z=[z, z],
                    mode="lines",
                    line=line_style,
                    showlegend=False,
                )
            )


def _create_3d_plot(
    volume_np: torch.Tensor,
    attn_full: Optional[np.ndarray],
    seg_3d: Optional[np.ndarray],
    crop_offsets: Tuple[int, int, int],
    crop_shape: Tuple[int, int, int],
    attn_grid_side: Optional[int] = None,
    is_normal: bool = False,
) -> Any:
    import plotly.graph_objects as go

    v = volume_np.squeeze().cpu().numpy().astype(np.float32)
    traces = []

    vol_disp = v.copy()
    mn, mx = vol_disp.min(), vol_disp.max()
    if mx - mn > 1e-8:
        vol_disp = (vol_disp - mn) / (mx - mn)
    traces.append(_build_3d_mesh(vol_disp, 0.25, "lightgray", 0.05, "Volume"))

    if seg_3d is not None and seg_3d.max() > 0:
        traces.append(_build_3d_mesh(seg_3d, 0.5, "red", 0.45, "GT (tumor)"))

    if not is_normal and attn_full is not None:
        attn_level = 0.3
        if attn_full.max() > attn_level:
            traces.append(
                _build_3d_mesh(attn_full, attn_level, "yellow", 0.25, "Attention")
            )

    if attn_grid_side is not None and attn_grid_side > 1:
        _draw_grid_wireframe(
            traces,
            x0=0, y0=0, z0=0,
            x_sz=96, y_sz=96, z_sz=96,
            grid_side=attn_grid_side,
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
        case_dir = case_path.parent if case_label_id.startswith("[Brain]") else None
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
    import torch.nn.functional as F
    import nibabel as nib

    entry = _CASE_REGISTRY.get(case_label_id)
    if not entry:
        return None
    study_id = entry.get("case_id", "")
    if not study_id:
        return None

    project_root = _get_project_root()
    resampled_dir = (
        project_root
        / "data"
        / "raw"
        / "prostate"
        / "picai_labels"
        / "csPCa_lesion_delineations"
        / "human_expert"
        / "resampled"
    )
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


_CASE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _list_available_cases() -> List[str]:
    from triagemri.data.datasets import BrainMRIDataset, ProstateMRIDataset

    global _CASE_REGISTRY
    if _CASE_REGISTRY:
        return sorted(_CASE_REGISTRY.keys())

    project_root = _get_project_root()
    split_kwargs = dict(seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    try:
        ds = BrainMRIDataset(
            data_root=project_root / "data" / "raw" / "brain",
            split="test",
            sources=("brats", "oasis", "ixi", "hcp"),
            **split_kwargs,
        )
        for case in ds._cases:
            has_mask = case.get("mask_path") is not None
            if case["label"] == 1:
                mask_str = " ✓mask" if has_mask else " ✗no-mask"
            else:
                mask_str = " ✓mask" if has_mask else ""
            prefix = "🟢N" if case["label"] == 0 else "🔴A"
            display_name = f"[Brain] {prefix}: {case['case_id']}{mask_str}"
            vpaths = case.get("volume_paths", {})
            fpath = vpaths.get("t1ce") or vpaths.get("t1") or next(
                iter(vpaths.values()), None
            )
            if fpath:
                _CASE_REGISTRY[display_name] = {
                    "path": str(Path(fpath).resolve()),
                    "case_id": case["case_id"],
                    "label": case["label"],
                    "anatomy": "brain",
                    "has_mask": case.get("mask_path") is not None,
                }
    except Exception as e:
        print(f"Error loading brain test split: {e}")

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
            has_mask = case.get("mask_path") is not None
            if case["label"] == 1:
                mask_str = " ✓mask" if has_mask else " ✗no-mask"
            else:
                mask_str = " ✓mask" if has_mask else ""
            prefix = "🟢N" if case["label"] == 0 else "🔴A"
            display_name = f"[Prostate] {prefix}: {case['case_id']}{mask_str}"
            vpaths = case.get("volume_paths", {})
            fpath = vpaths.get("t2w") or next(iter(vpaths.values()), None)
            if fpath:
                _CASE_REGISTRY[display_name] = {
                    "path": str(Path(fpath).resolve()),
                    "case_id": case["case_id"],
                    "label": case["label"],
                    "anatomy": "prostate",
                    "has_mask": case.get("mask_path") is not None,
                }
    except Exception as e:
        print(f"Error loading prostate test split: {e}")

    return sorted(_CASE_REGISTRY.keys())


def _filter_cases_for_segmentation() -> None:
    """Filter the case registry to only show cases relevant for segmentation eval.

    Keeps: all normals (to verify no hallucination) and abnormals with GT masks
    (to compare heatmap vs ground truth). Removes abnormals without masks.
    """
    global _CASE_REGISTRY
    _CASE_REGISTRY = {
        name: entry
        for name, entry in _CASE_REGISTRY.items()
        if entry["label"] == 0 or entry.get("has_mask", False)
    }


def _get_case_file(case_label_id: str) -> Optional[str]:
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
    inconclusive_range: Optional[Tuple[float, float]] = None,
    seg_model: Optional[torch.nn.Module] = None,
) -> Any:
    import importlib.util

    if importlib.util.find_spec("gradio") is None:
        raise ImportError("gradio is required for the demo.  Install with: pip install gradio")
    import gradio as gr  # noqa: F811

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
        inconclusive_range=inconclusive_range,
        seg_model=seg_model,
    )

    # Build segmentation-filtered case list when seg model is loaded
    _list_available_cases()
    if seg_model is not None:
        _filter_cases_for_segmentation()

    custom_css = """
    .result-normal { background-color: #e8f5e9 !important; }
    .result-revisar { background-color: #ffebee !important; }
    .result-inconclusive { background-color: #fff3e0 !important; }
    #result-header { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
    """

    with gr.Blocks(
        title="Triage-MRI Screening", css=custom_css, theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # Triage-MRI: Normality Screening for 3D MRI

            Upload a NIfTI (`.nii.gz`) or MetaImage (`.mha`) volume or select a test
            case to triage studies into **NORMAL**, **INCONCLUSIVE**, or **REVISAR**.
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

                file_input.upload(
                    fn=lambda f: (_extract_case_id_from_upload(f), ""),
                    inputs=[file_input],
                    outputs=[case_id_input, registry_path],
                )

                def _handle_case_selection(label_id):
                    if not label_id:
                        return None, "", "brain", ""
                    fpath = _get_case_file(label_id)
                    entry = _CASE_REGISTRY.get(label_id, {})
                    actual_id = entry.get("case_id", "")
                    anatomy = entry.get("anatomy", "brain")
                    return fpath, actual_id, anatomy, fpath

                load_case_btn.click(
                    fn=_handle_case_selection,
                    inputs=[case_selector],
                    outputs=[
                        file_input,
                        case_id_input,
                        anatomy_dropdown,
                        registry_path,
                    ],
                )

                submit_btn = gr.Button("Run Triage", variant="primary", size="lg")
                status_text = gr.Markdown("")

            with gr.Column(scale=2):
                gr.Markdown("### Resultado: —", elem_id="result-header")
                decision_text = gr.Markdown("")

                with gr.Accordion("Details", open=True):
                    info_text = gr.Markdown("Awaiting input...")

        gr.Markdown("---")
        gr.Markdown("### Visualización 3D")

        plot_3d = gr.Plot(label="3D View")

        gr.Markdown("---")
        gr.Markdown("### Cortes 2D con Heatmap")
        heatmap_label = "Rojo = GT (tumor real) | Amarillo = Model B (segmentación)" if seg_model is not None else "Rojo = GT (tumor real) | Amarillo = atención del modelo"
        gr.Markdown(f"*{heatmap_label}*")

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
            inputs=[
                file_input,
                case_id_input,
                anatomy_dropdown,
                registry_path,
            ],
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
    inconclusive_range: Optional[Tuple[float, float]] = None,
    seg_checkpoint_path: Optional[str] = None,
) -> Any:
    print(f"Loading config from: {config_dir}")
    config = load_config(config_dir)

    print(f"Loading classifier from: {checkpoint_path}")
    model = build_triage_model(checkpoint_path, device=device)
    print(f"Classifier loaded. Anatomy mode: {model.anatomy_mode}")

    seg_model = None
    if seg_checkpoint_path:
        from triagemri.models.segmentation import build_segmentation_model
        print(f"Loading segmentation model from: {seg_checkpoint_path}")
        seg_model = build_segmentation_model(
            checkpoint_path=config.model.encoder.checkpoint,
            embed_dim=config.model.encoder.embed_dim,
            device=device,
        )
        # Load seg checkpoint weights
        seg_ckpt = torch.load(seg_checkpoint_path, map_location=device, weights_only=True)
        seg_state = seg_ckpt.get("state_dict", seg_ckpt)
        seg_state = {k.removeprefix("model."): v for k, v in seg_state.items()}
        seg_model.load_state_dict(seg_state, strict=False)
        seg_model.eval()
        print("Segmentation model loaded.")

    thresholds: Dict[str, Dict[str, float]] = {}
    if thresholds_file and Path(thresholds_file).exists():
        with open(thresholds_file, "r") as f:
            thresholds = json.load(f)
        print(f"Thresholds loaded from: {thresholds_file}")
    else:
        print("No thresholds file provided — using default threshold 0.5.")

    if available_anatomies is None:
        available_anatomies = ["brain", "prostate", "breast"]

    if inconclusive_range is None:
        inc_cfg = config.get("evaluation", {}).get("inconclusive_range")
        if inc_cfg:
            inconclusive_range = tuple(float(v) for v in inc_cfg)

    demo = create_demo(
        model=model,
        thresholds=thresholds,
        device=device,
        anatomy_mode=model.anatomy_mode,
        available_anatomies=available_anatomies,
        inconclusive_range=inconclusive_range,
        seg_model=seg_model,
    )

    print(f"\nLaunching demo on http://{server_name}:{server_port}")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
    return demo
