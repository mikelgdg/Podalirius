#!/usr/bin/env python3
"""
Launch the Triage-MRI Gradio demo.

Usage:
    python scripts/demo.py --checkpoint model.pt
    python scripts/demo.py --checkpoint model.pt --share --port 8080
    python scripts/demo.py --checkpoint model.pt --thresholds results/thresholds.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from triagemri.demo.app import launch_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Triage-MRI interactive Gradio demo."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt/.pth).",
    )
    parser.add_argument(
        "--config_dir",
        default="configs",
        help="Directory with data.yaml, model.yaml, train.yaml (default: configs).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio sharing link.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to listen on (default: 7860).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional JSON file with per-anatomy thresholds "
             "(from evaluator output).",
    )
    parser.add_argument(
        "--anatomies",
        nargs="+",
        default=None,
        help="Override available anatomies in the dropdown "
             "(default: brain prostate breast).",
    )
    args = parser.parse_args()

    launch_demo(
        checkpoint_path=args.checkpoint,
        config_dir=args.config_dir,
        device=args.device,
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        thresholds_file=args.thresholds,
        available_anatomies=args.anatomies,
    )


if __name__ == "__main__":
    main()
