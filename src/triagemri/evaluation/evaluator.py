"""Evaluation pipeline for trained TriageModel instances.

Provides :class:`TriageEvaluator` for running inference, computing
per-anatomy/global metrics, finding optimal thresholds, calibration,
and generating reports.
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

from triagemri.config import load_config
from triagemri.models import build_triage_model
from triagemri.training.metrics import (
    calibration_metrics,
    compute_metrics_per_anatomy,
    compute_triage_metrics,
    find_inconclusive_thresholds,
    find_threshold_for_sensitivity,
    format_metrics_report,
)

logger = logging.getLogger(__name__)


class TriageEvaluator:
    """Evaluates a trained TriageModel on test datasets.

    Computes per-anatomy and global metrics, finds optimal thresholds
    for a target sensitivity (default 99 %), performs Platt scaling
    calibration, and generates evaluation reports.

    Args:
        model: A trained :class:`~triagemri.models.TriageModel`.
        device: Torch device string.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def evaluate(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
        """Run inference on an entire dataset.

        Args:
            dataloader: DataLoader yielding dicts with at least
                ``"volume"``, ``"label"``, and ``"anatomy"``.

        Returns:
            ``(y_true, y_scores, anatomies, metadata)``.
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
                batch_anatomies = list(batch["anatomy"])
            elif isinstance(batch["anatomy"], np.ndarray):
                batch_anatomies = batch["anatomy"].tolist()
            else:
                batch_anatomies = list(batch["anatomy"])

            if batch_anatomies and isinstance(batch_anatomies[0], bytes):
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

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------

    def compute_thresholds(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        anatomies: List[str],
        target_sensitivity: float = 0.99,
    ) -> Dict[str, Dict[str, float]]:
        """Find per-anatomy (and global) thresholds for target sensitivity.

        Returns:
            Dict mapping anatomy name (or ``"global"``) to
            ``{"threshold", "sensitivity", "specificity"}``.
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

    def compute_inconclusive_thresholds(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        anatomies: List[str],
        target_sensitivity: float = 0.99,
    ) -> Dict[str, Dict[str, float]]:
        """Compute three-zone thresholds per anatomy."""
        result: Dict[str, Dict[str, float]] = {}
        unique_anatomies = sorted(set(anatomies))
        for anat in unique_anatomies:
            mask = np.array([a == anat for a in anatomies])
            if mask.sum() == 0:
                continue
            result[anat] = find_inconclusive_thresholds(
                y_true[mask], y_scores[mask], target_sensitivity
            )
        result["global"] = find_inconclusive_thresholds(
            y_true, y_scores, target_sensitivity
        )
        return result

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def full_evaluation(
        self,
        test_loader: DataLoader,
        output_dir: Optional[str] = None,
        target_sensitivity: float = 0.99,
        val_loader: Optional[DataLoader] = None,
        calibrate: bool = False,
        inconclusive: bool = False,
        subgroup_analysis: bool = False,
    ) -> Dict[str, Any]:
        """Complete evaluation pipeline.

        1. Run inference on *test_loader*.
        2. Optionally calibrate with Platt scaling on *val_loader*.
        3. Compute per-anatomy and global metrics.
        4. Find optimal thresholds.
        5. Optionally compute inconclusive zones.
        6. Save results to JSON and Markdown.

        Args:
            test_loader: DataLoader for the test set.
            output_dir: Directory to save results.
            target_sensitivity: Sensitivity target for threshold finding.
            val_loader: Optional DataLoader for calibration/thresholds.
            calibrate: If True, fit Platt scaling on val set scores.
            inconclusive: If True, compute three-zone thresholds.
            subgroup_analysis: If True, compute metrics per source.

        Returns:
            Full results dict.
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
            if calibrate:
                self.model.calibrate(val_y_scores, val_y_true)
        else:
            logger.warning(
                "No val_loader provided — computing thresholds on test set. "
                "This is DATA LEAKAGE and will inflate reported metrics."
            )
            thresholds = self.compute_thresholds(
                y_true, y_scores, anatomies, target_sensitivity
            )
            if calibrate:
                self.model.calibrate(y_scores, y_true)

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

        if inconclusive:
            inconclusive_thr = self.compute_inconclusive_thresholds(
                y_true if val_loader is None else val_y_true,
                y_scores if val_loader is None else val_y_scores,
                anatomies if val_loader is None else val_anatomies,
                target_sensitivity,
            )
            combined["inconclusive_thresholds"] = inconclusive_thr

        if subgroup_analysis:
            combined["subgroup"] = self._compute_subgroup_metrics(
                y_true, y_scores, anatomies, metadata
            )

        report = self.generate_report(combined, output_dir)
        print(report)

        return combined

    def calibrate_and_evaluate(
        self,
        train_loader: Optional[DataLoader],
        test_loader: DataLoader,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fit calibration on train/val set, then evaluate on test."""
        if train_loader is not None:
            y_train, y_scores_train, _, _ = self.evaluate(train_loader)
            self.model.calibrate(y_scores_train, y_train)

        return self.full_evaluation(
            test_loader,
            output_dir=output_dir,
            val_loader=train_loader,
            calibrate=True,
        )

    # ------------------------------------------------------------------
    # Attention export
    # ------------------------------------------------------------------

    def generate_attention_report(
        self,
        dataloader: DataLoader,
        output_dir: str,
        max_samples: int = 50,
    ) -> None:
        """Export attention maps as NIfTI for qualitative inspection.

        Args:
            dataloader: DataLoader with cases to export.
            output_dir: Directory to save NIfTI attention maps.
            max_samples: Maximum number of cases to export.
        """
        import nibabel as nib

        output_path = Path(output_dir) / "attention_maps"
        output_path.mkdir(parents=True, exist_ok=True)

        count = 0
        for batch in dataloader:
            if count >= max_samples:
                break
            volumes = batch["volume"].to(self.device)
            patient_ids = batch.get("patient_id", [f"case_{i}" for i in range(len(volumes))])
            output = self.model(volumes)
            attention = output.get("attention")
            if attention is None:
                continue
            for i in range(len(volumes)):
                if count >= max_samples:
                    break
                attn = attention[i].cpu().numpy()
                nii = nib.Nifti1Image(
                    attn.squeeze(),
                    np.eye(4),
                )
                pid = patient_ids[i]
                if isinstance(pid, bytes):
                    pid = pid.decode("utf-8")
                nib.save(nii, str(output_path / f"{count:04d}_{pid}_attention.nii.gz"))
                count += 1

        logger.info("Saved %d attention maps to %s", count, output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_per_anatomy_from_raw(
        self, y_true: np.ndarray, y_scores: np.ndarray, anatomies: List[str]
    ) -> Dict[str, Any]:
        import pandas as pd

        df = pd.DataFrame(
            {"y_true": y_true, "y_score": y_scores, "anatomy": anatomies}
        )
        return compute_metrics_per_anatomy(df)

    def _compute_subgroup_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        anatomies: List[str],
        metadata: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        sources: Dict[str, List[int]] = {}
        for i, meta in enumerate(metadata):
            src = meta.get("source", "unknown")
            if src not in sources:
                sources[src] = []
            sources[src].append(i)

        for src, indices in sources.items():
            idx = np.array(indices)
            if len(np.unique(y_true[idx])) < 2:
                continue
            result[src] = compute_triage_metrics(
                y_true[idx], y_scores[idx]
            )

        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(
        self, results: Dict[str, Any], output_dir: Optional[str] = None
    ) -> str:
        """Generate a Markdown evaluation report and optionally save JSON.

        Returns:
            Markdown report string.
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            json_path = output_path / "evaluation_results.json"

            class _Encoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return str(obj)

            json_path.write_text(json.dumps(results, indent=2, cls=_Encoder))
            logger.info("Saved evaluation JSON to %s", json_path)

        lines: List[str] = []
        lines.append("# Triage-MRI Evaluation Report")
        lines.append("")
        lines.append(f"**Date:** {results.get('evaluation_date', 'unknown')}")
        lines.append(
            f"**Samples:** {results['num_samples']}  "
            f"(normal={results['num_normal']}, "
            f"abnormal={results['num_abnormal']})"
        )
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
        lines.append(
            "| Anatomy   | Threshold | Achieved Sensitivity | Specificity |"
        )
        lines.append(
            "|-----------|-----------|----------------------|-------------|"
        )
        for anat in sorted(
            threshold_map.keys(), key=lambda a: (a != "global", a)
        ):
            t = threshold_map[anat]
            lines.append(
                f"| {anat:<9s} | {t['threshold']:.4f}   | "
                f"{t['sensitivity']:.4f}             | "
                f"{t['specificity']:.4f}    |"
            )
        lines.append("")

        calib = results.get("calibration", {})
        lines.append("---")
        lines.append("")
        lines.append("## Calibration")
        lines.append(f"- **ECE:** {calib.get('ece', 'N/A'):.4f}")
        lines.append(f"- **Brier score:** {calib.get('brier', 'N/A'):.4f}")

        if "subgroup" in results:
            lines.append("")
            lines.append("---")
            lines.append("## Subgroup Analysis (by data source)")
            for src, m in results["subgroup"].items():
                lines.append(f"- **{src}:** AUC={m.get('roc_auc', 'N/A')}, "
                             f"Sens={m.get('sensitivity', 'N/A'):.4f}, "
                             f"Spec={m.get('specificity', 'N/A'):.4f}")

        report = "\n".join(lines)
        if output_dir:
            md_path = output_path / "evaluation_report.md"  # noqa: F821
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
    calibrate: bool = False,
    inconclusive: bool = False,
    subgroup_analysis: bool = False,
    save_attention_maps: bool = False,
) -> Dict[str, Any]:
    """Evaluate a trained model from config + checkpoint.

    Args:
        config_path: Directory containing ``data.yaml``, etc.
        checkpoint: Path to ``.pt`` or ``.pth`` checkpoint.
        output_dir: Where to save reports.
        device: Torch device string.
        target_sensitivity: Sensitivity target.
        enabled_anatomies: Optional list of anatomies.
        calibrate: Fit Platt scaling.
        inconclusive: Compute three-zone thresholds.
        subgroup_analysis: Metrics per data source.
        save_attention_maps: Export NIfTI attention maps.

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
    val_loader = dataloaders.get("val")

    evaluator = TriageEvaluator(model, device=device)
    results = evaluator.full_evaluation(
        test_loader,
        output_dir=output_dir,
        target_sensitivity=target_sensitivity,
        val_loader=val_loader,
        calibrate=calibrate,
        inconclusive=inconclusive,
        subgroup_analysis=subgroup_analysis,
    )

    if save_attention_maps:
        evaluator.generate_attention_report(test_loader, output_dir)

    return results
