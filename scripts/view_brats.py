#!/usr/bin/env python3
"""Interactive slice-by-slice viewer for BraTS MRI.

Usage:
    python scripts/view_brats.py data/raw/brain/brats/BraTS-MEN-00004-000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


SEQUENCES = {
    "t1n": "T1 native",
    "t1c": "T1 contrast",
    "t2w": "T2 weighted",
    "t2f": "FLAIR",
}


def _normalize(vol: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    vol = vol.astype(np.float32)
    vmin = np.percentile(vol, low_pct)
    vmax = np.percentile(vol, high_pct)
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return np.clip((vol - vmin) / (vmax - vmin), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive BraTS slice viewer")
    parser.add_argument("case_dir", help="Path to a BraTS case directory")
    parser.add_argument("--out", default=None, help="Output HTML path")
    parser.add_argument("--key", default=None, help="Sequence to show (t1n/t1c/t2w/t2f, default: first found)")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.is_dir():
        print(f"Error: {case_dir} is not a directory")
        sys.exit(1)

    import nibabel as nib

    files = sorted(case_dir.glob("*.nii.gz"))
    volumes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    seg_data: np.ndarray | None = None

    for f in files:
        name = f.name.lower()
        if "seg" in name:
            seg_data = nib.load(str(f)).get_fdata().astype(np.float32)
            if seg_data.ndim == 4:
                seg_data = seg_data[..., 0]
            seg_data = (seg_data > 0).astype(np.float32)
            continue
        for key in SEQUENCES:
            if key in name:
                vol_nii = nib.load(str(f))
                data = vol_nii.get_fdata().astype(np.float32)
                if data.ndim == 4:
                    data = data[..., 0]
                volumes[key] = (data, _normalize(data))
                break

    if not volumes:
        print("No sequences found in", case_dir)
        sys.exit(1)

    seq_keys = sorted(volumes.keys(), key=lambda k: list(SEQUENCES).index(k) if k in SEQUENCES else 99)
    default_key = args.key if args.key and args.key in volumes else seq_keys[0]
    _, vol_norm = volumes[default_key]
    n_slices = vol_norm.shape[0]

    frames = []
    sliders_steps = []

    for i in range(n_slices):
        img = vol_norm[i, :, :]
        img_rot = np.rot90(img)

        rgb = np.stack([img_rot, img_rot, img_rot], axis=-1)

        if seg_data is not None:
            seg_slice = np.rot90(seg_data[i, :, :])
            mask = seg_slice > 0
            if mask.any():
                blend = mask.astype(np.float32) * 0.45
                rgb[..., 0] = rgb[..., 0] * (1 - blend) + blend
                rgb[..., 1] = rgb[..., 1] * (1 - blend)
                rgb[..., 2] = rgb[..., 2] * (1 - blend)

        frames.append(go.Frame(
            data=[go.Image(z=(rgb * 255).astype(np.uint8))],
            name=str(i),
        ))
        sliders_steps.append({
            "args": [
                [str(i)],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
            ],
            "label": str(i),
            "method": "animate",
        })

    default_img = frames[n_slices // 2].data  # type: ignore[union-attr]
    fig = go.Figure(
        data=default_img,
        frames=frames,
        layout=go.Layout(
            title=f"BraTS: {case_dir.name}  |  {SEQUENCES[default_key]}",
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "fromcurrent": True}],
                    },
                ],
                "x": 0.1,
                "y": 0,
            }],
            sliders=[{
                "active": n_slices // 2,
                "steps": sliders_steps,
                "x": 0.1,
                "y": 0,
                "len": 0.9,
            }],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=10, r=10, t=40, b=60),
            width=600,
            height=600,
        ),
    )

    out_path = Path(args.out) if args.out else Path(f"{case_dir.name}.html")
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_play=False)
    print(f"Saved: {out_path.resolve()}")
    print("Open in browser — use the slider to scroll through slices, ▶ Play for animation.")
    print(f"\nSequences found: {', '.join(SEQUENCES[k] for k in seq_keys)}")
    print("(change --key to view a different sequence)")


if __name__ == "__main__":
    main()
