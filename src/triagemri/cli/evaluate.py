#!/usr/bin/env python3
"""
Evaluate a trained Triage-MRI model.

Usage:
    triage-eval --checkpoint outputs/run_001/checkpoints/epoch_50.ckpt
    triage-eval --checkpoint model.pt --calibrate --subgroup_analysis
    triage-eval --checkpoint model.pt --save_attention_maps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl

_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root / "src"))

from triagemri.config import load_config
from triagemri.data.datasets import create_dataloaders
from triagemri.evaluation.evaluator import TriageEvaluator
from triagemri.models import build_triage_model


def main() -> None:
    pl.seed_everything(42)
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Triage-MRI model on the test set."
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
        "--output_dir",
        default="outputs/evaluation",
        help="Directory for evaluation results (default: outputs/evaluation).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--target_sensitivity",
        type=float,
        default=0.99,
        help="Target sensitivity for threshold finding (default: 0.99).",
    )
    parser.add_argument(
        "--anatomies",
        nargs="+",
        choices=["brain", "prostate", "breast"],
        default=None,
        help="Which anatomies to evaluate (default: all).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit Platt scaling on validation set before evaluation.",
    )
    parser.add_argument(
        "--inconclusive",
        action="store_true",
        help="Compute three-zone thresholds (NORMAL/INCONCLUSIVE/REVISAR).",
    )
    parser.add_argument(
        "--subgroup_analysis",
        action="store_true",
        help="Compute metrics per data source (BraTS/OASIS/IXI).",
    )
    parser.add_argument(
        "--save_attention_maps",
        action="store_true",
        help="Export attention maps as NIfTI for qualitative inspection.",
    )
    args = parser.parse_args()

    print(f"Loading config from: {args.config_dir}")
    config = load_config(args.config_dir)

    print(f"Loading model from: {args.checkpoint}")
    model = build_triage_model(args.checkpoint, device=args.device)
    print(f"Model loaded. Anatomy mode: {model.anatomy_mode}")

    print("Creating dataloaders...")
    dataloaders = create_dataloaders(config, enabled_anatomies=args.anatomies)
    test_loader = dataloaders["test"]
    val_loader = dataloaders.get("val")
    print(
        f"Test dataset: {len(test_loader.dataset)} samples, "
        f"{len(test_loader)} batches"
    )

    evaluator = TriageEvaluator(model, device=args.device)
    print("Running evaluation...")
    results = evaluator.full_evaluation(
        test_loader,
        output_dir=args.output_dir,
        target_sensitivity=args.target_sensitivity,
        val_loader=val_loader,
        calibrate=args.calibrate,
        inconclusive=args.inconclusive,
        subgroup_analysis=args.subgroup_analysis,
    )

    if args.save_attention_maps:
        evaluator.generate_attention_report(test_loader, args.output_dir)

    print(f"\nEvaluation complete. Results saved to {args.output_dir}/")
    roc = results.get("global", {}).get("roc_auc")
    print(f"  Global ROC-AUC:  {roc:.4f}" if roc is not None else "  Global ROC-AUC:  N/A")
    pr = results.get("global", {}).get("pr_auc")
    print(f"  Global PR-AUC:   {pr:.4f}" if pr is not None else "  Global PR-AUC:   N/A")

    global_thresh = results.get("thresholds", {}).get("global", {})
    thresh_val = global_thresh.get("threshold")
    if thresh_val is not None:
        print(
            f"  Threshold (global @ {args.target_sensitivity} sens): "
            f"{thresh_val:.4f}"
        )
    else:
        print(f"  Threshold (global @ {args.target_sensitivity} sens): N/A")

    if args.inconclusive:
        inc = results.get("inconclusive_thresholds", {}).get("global", {})
        if inc:
            print(
                f"  Inconclusive zone: "
                f"[{inc.get('thr_normal', 0):.4f}, {inc.get('thr_revisar', 0):.4f}]"
            )


if __name__ == "__main__":
    main()
