#!/usr/bin/env python3
"""
Evaluate a trained Triage-MRI model.

Usage:
    python scripts/evaluate.py --checkpoint outputs/run_001/checkpoints/epoch_50.ckpt
    python scripts/evaluate.py --checkpoint model.pt --config_dir configs --output_dir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
        help="Which anatomies to evaluate (default: all). "
             "Example: --anatomies brain prostate",
    )
    args = parser.parse_args()

    print(f"Loading config from: {args.config_dir}")
    config = load_config(args.config_dir)

    print(f"Loading model from: {args.checkpoint}")
    model = build_triage_model(args.checkpoint, device=args.device)
    print(f"Model loaded. Anatomy mode: {model.anatomy_mode}")

    print("Creating dataloaders…")
    dataloaders = create_dataloaders(config, enabled_anatomies=args.anatomies)
    test_loader = dataloaders["test"]
    print(f"Test dataset: {len(test_loader.dataset)} samples, "
          f"{len(test_loader)} batches")

    evaluator = TriageEvaluator(model, device=args.device)
    print("Running evaluation…")
    results = evaluator.full_evaluation(
        test_loader,
        output_dir=args.output_dir,
        target_sensitivity=args.target_sensitivity,
    )

    print(f"\nEvaluation complete. Results saved to {args.output_dir}/")
    roc = results['global'].get('roc_auc')
    if roc is not None:
        print(f"  Global ROC-AUC:  {roc:.4f}")
    else:
        print(f"  Global ROC-AUC:  N/A")
    pr = results['global'].get('pr_auc')
    if pr is not None:
        print(f"  Global PR-AUC:   {pr:.4f}")
    else:
        print(f"  Global PR-AUC:   N/A")
    global_thresh = results.get("thresholds", {}).get("global", {})
    thresh_val = global_thresh.get('threshold')
    if thresh_val is not None:
        print(f"  Threshold (global @ {args.target_sensitivity} sens): "
              f"{thresh_val:.4f}")
    else:
        print(f"  Threshold (global @ {args.target_sensitivity} sens): N/A")


if __name__ == "__main__":
    main()
