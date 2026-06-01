"""Evaluation pipeline for trained TriageModel instances.

Provides :class:`TriageEvaluator` for running inference, computing
per-anatomy/global metrics, finding optimal thresholds, and generating
reports.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from triagemri.config import Config, load_config
from triagemri.models import build_triage_model
from triagemri.training.metrics import (
    calibration_metrics,
    compute_metrics_per_anatomy,
    compute_triage_metrics,
    find_threshold_for_sensitivity,
    format_metrics_report,
)

logger = logging.getLogger(__name__)


class TriageEvaluator:
    """Evaluates a trained TriageModel on test datasets.

    Computes per-anatomy and global metrics, finds optimal thresholds
    for a target sensitivity (default 99 %), and generates evaluation
    reports in JSON and Markdown formats.

    Args:
        model: A trained :class:`~triagemri.models.TriageModel`.
        device: Torch device string (``"cuda"``, ``"cpu"``, …).
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
        """Run inference on an entire dataset.

        Args:
            dataloader: DataLoader yielding dicts with at least
                ``"volume"``, ``"label"``, and ``"anatomy"``.

        Returns:
            ``(y_true, y_scores, anatomies, metadata)`` where *y_true*
            and *y_scores* are 1-D float arrays, *anatomies* is a list
            of anatomy strings, and *metadata* is a list of per-sample
            metadata dicts.
        """
        all_labels: List[float] = []
        all_scores: List[float] = []
        all_anatomies: List[str] = []
        all_metadata: List[Dict[str, Any]] = []
        inference_times: List[float] = []

        data_iter = iter(dataloader)
        try:
            first_batch = next(data_iter)
        except StopIteration:
            logger.warning("DataLoader yielded 0 batches — returning empty results")
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                [],
                [],
            )

        def _consume_batch(batch):
            volumes = batch["volume"].to(self.device, non_blocking=True)
            labels = batch["label"]

            if isinstance(batch["anatomy"], (list, tuple)):
                batch_anatomies = batch["anatomy"]
            elif isinstance(batch["anatomy"], np.ndarray):
                batch_anatomies = batch["anatomy"].tolist()
            else:
                batch_anatomies = list(batch["anatomy"])

            if isinstance(batch_anatomies[0], (bytes,)):
                batch_anatomies = [a.decode("utf-8") for a in batch_anatomies]

            t0 = time.time()
            output = self.model(volumes)
            elapsed = (time.time() - t0) / len(volumes)

            scores = output["score"].cpu().numpy()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_scores.extend(scores.tolist())
            all_anatomies.extend(batch_anatomies)
            inference_times.extend([elapsed] * len(volumes))

            patient_ids = batch.get("patient_id", None)
            sources = batch.get("source", None)
            for i in range(len(volumes)):
                meta: Dict[str, Any] = {"inference_time_s": elapsed}
                if patient_ids is not None:
                    pid = patient_ids[i]
                    if isinstance(pid, bytes):
                        pid = pid.decode("utf-8")
                    meta["patient_id"] = str(pid)
                if sources is not None:
                    src = sources[i]
                    if isinstance(src, bytes):
                        src = src.decode("utf-8")
                    meta["source"] = str(src)
                all_metadata.append(meta)

        with torch.no_grad():
            _consume_batch(first_batch)
            for batch in data_iter:
                _consume_batch(batch)

        y_true = np.array(all_labels, dtype=np.float32)
        y_scores = np.array(all_scores, dtype=np.float32)

        logger.info(
            "Evaluation complete: %d samples, mean inference %.4fs/sample",
            len(y_true),
            np.mean(inference_times),
        )
        return y_true, y_scores, all_anatomies, all_metadata

    def compute_thresholds(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        anatomies: List[str],
        target_sensitivity: float = 0.99,
    ) -> Dict[str, Dict[str, float]]:
        """Find per-anatomy (and global) thresholds for target sensitivity.

        Args:
            y_true: (N,) binary labels.
            y_scores: (N,) anomaly scores in [0, 1].
            anatomies: List of anatomy strings, same length as *y_true*.
            target_sensitivity: Desired sensitivity (default 0.99).

        Returns:
            Dict mapping anatomy name (or ``"global"``) to
            ``{"threshold": float, "sensitivity": float, "specificity": float}``.
        """
        thresholds: Dict[str, Dict[str, float]] = {}

        unique_anatomies = sorted(set(anatomies))
        for anat in unique_anatomies:
            mask = np.array([a == anat for a in anatomies])
            if mask.sum() == 0:
                continue
            thresh, sens, spec = find_threshold_for_sensitivity(
                y_true[mask], y_scores[mask], target_sensitivity
            )
            thresholds[anat] = {
                "threshold": thresh,
                "sensitivity": sens,
                "specificity": spec,
            }

        thresh, sens, spec = find_threshold_for_sensitivity(
            y_true, y_scores, target_sensitivity
        )
        thresholds["global"] = {
            "threshold": thresh,
            "sensitivity": sens,
            "specificity": spec,
        }

        return thresholds

    def full_evaluation(
        self,
        test_loader: DataLoader,
        output_dir: Optional[str] = None,
        target_sensitivity: float = 0.99,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Complete evaluation pipeline.

        1. Run inference on *test_loader*.
        2. Compute per-anatomy metrics.
        3. Compute global metrics.
        4. Find optimal thresholds (on *val_loader* if provided).
        5. Save results to JSON (if *output_dir* is given).
        6. Print a formatted report.

        Args:
            test_loader: DataLoader for the test set.
            output_dir: Directory to save results.  Created if missing.
            target_sensitivity: Sensitivity target for threshold finding.
            val_loader: Optional DataLoader for a validation set to
                compute thresholds on.  If ``None``, thresholds are
                computed on the test set (WARNING: data leakage!).

        Returns:
            Full results dict with keys ``"metrics"``,
            ``"thresholds"``, ``"calibration"``, ``"metadata"``.
        """
        y_true, y_scores, anatomies, metadata = self.evaluate(test_loader)

        per_anatomy_metrics = self._compute_per_anatomy_from_raw(
            y_true, y_scores, anatomies
        )

        if val_loader is not None:
            val_y_true, val_y_scores, val_anatomies, _ = self.evaluate(val_loader)
            thresholds = self.compute_thresholds(
                val_y_true, val_y_scores, val_anatomies, target_sensitivity
            )
        else:
            logger.warning(
                "No val_loader provided — computing thresholds on test set. "
                "This is DATA LEAKAGE and will inflate reported metrics."
            )
            thresholds = self.compute_thresholds(
                y_true, y_scores, anatomies, target_sensitivity
            )

        calib = calibration_metrics(y_true, y_scores)

        combined: Dict[str, Any] = {
            "evaluation_date": datetime.now().isoformat(),
            "num_samples": int(len(y_true)),
            "num_normal": int((y_true == 0).sum()),
            "num_abnormal": int((y_true == 1).sum()),
            "target_sensitivity": target_sensitivity,
            "per_anatomy": {},
            "global": {},
        }

        for anat, metrics in per_anatomy_metrics.items():
            if anat == "global":
                combined["global"] = metrics
            else:
                combined["per_anatomy"][anat] = metrics

        combined["thresholds"] = thresholds
        combined["calibration"] = calib
        combined["sample_count_per_anatomy"] = {
            anat: int(np.sum(np.array(anatomies) == anat))
            for anat in sorted(set(anatomies))
        }

        report = self.generate_report(combined, output_dir)
        print(report)

        return combined

    def _compute_per_anatomy_from_raw(
        self, y_true: np.ndarray, y_scores: np.ndarray, anatomies: List[str]
    ) -> Dict[str, Any]:
        import pandas as pd

        df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_score": y_scores,
                "anatomy": anatomies,
            }
        )
        return compute_metrics_per_anatomy(df)

    def generate_report(
        self, results: Dict[str, Any], output_dir: Optional[str] = None
    ) -> str:
        """Generate a Markdown evaluation report and optionally save JSON.

        Args:
            results: Dict returned by :meth:`full_evaluation`.
            output_dir: Directory for ``evaluation_report.*`` files.

        Returns:
            Markdown report string.
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            json_path = output_path / "evaluation_results.json"
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info("Saved evaluation JSON to %s", json_path)

        lines: List[str] = []
        lines.append("# Triage-MRI Evaluation Report")
        lines.append("")
        lines.append(f"**Date:** {results.get('evaluation_date', 'unknown')}")
        lines.append(f"**Samples:** {results['num_samples']}  "
                     f"(normal={results['num_normal']}, "
                     f"abnormal={results['num_abnormal']})")
        lines.append(f"**Target sensitivity:** {results['target_sensitivity']}")
        lines.append("")

        threshold_map = results.get("thresholds", {})

        all_metrics: Dict[str, Any] = {}
        for anat, m in results.get("per_anatomy", {}).items():
            all_metrics[anat] = m
        all_metrics["global"] = results.get("global", {})

        report_str = format_metrics_report(all_metrics)
        lines.append(report_str)

        lines.append("---")
        lines.append("")
        lines.append("## Thresholds (target sensitivity)")
        lines.append("")
        lines.append("| Anatomy   | Threshold | Achieved Sensitivity | Specificity |")
        lines.append("|-----------|-----------|----------------------|-------------|")
        for anat in sorted(threshold_map.keys(), key=lambda a: (a != "global", a)):
            t = threshold_map[anat]
            lines.append(
                f"| {anat:<9s} | {t['threshold']:.4f}   | {t['sensitivity']:.4f}             | "
                f"{t['specificity']:.4f}    |"
            )
        lines.append("")

        calib = results.get("calibration", {})
        lines.append("---")
        lines.append("")
        lines.append("## Calibration")
        lines.append(f"- **ECE:** {calib.get('ece', 'N/A'):.4f}")
        lines.append(f"- **Brier score:** {calib.get('brier', 'N/A'):.4f}")

        report = "\n".join(lines)
        if output_dir:
            md_path = output_path / "evaluation_report.md"
            with open(md_path, "w") as f:
                f.write(report)
            logger.info("Saved evaluation report to %s", md_path)

        return report


# ------------------------------------------------------------------
# CLI-friendly entry point
# ------------------------------------------------------------------


def run_evaluation(
    config_path: str = "configs",
    checkpoint: Optional[str] = None,
    output_dir: str = "outputs/evaluation",
    device: str = "cuda",
    target_sensitivity: float = 0.99,
    enabled_anatomies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model from config + checkpoint.

    Args:
        config_path: Directory containing ``data.yaml``, etc.
        checkpoint: Path to ``.pt`` or ``.pth`` checkpoint.
        output_dir: Where to save reports.
        device: Torch device string.
        target_sensitivity: Sensitivity target for threshold finding.
        enabled_anatomies: Optional list of anatomies to restrict
            evaluation to (e.g. ``["brain"]``).

    Returns:
        Full evaluation results dict.
    """
    config = load_config(config_path)

    if checkpoint is None:
        raise ValueError("checkpoint path is required")

    model = build_triage_model(checkpoint, device=device)

    from triagemri.data.datasets import create_dataloaders

    dataloaders = create_dataloaders(config, enabled_anatomies=enabled_anatomies)
    test_loader = dataloaders["test"]

    evaluator = TriageEvaluator(model, device=device)
    results = evaluator.full_evaluation(
        test_loader,
        output_dir=output_dir,
        target_sensitivity=target_sensitivity,
    )
    return results
