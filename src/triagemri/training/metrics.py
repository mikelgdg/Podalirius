"""Evaluation metrics for triage classification and anomaly detection."""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def find_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_sensitivity: float = 0.99,
) -> Tuple[float, float, float]:
    """Find the highest threshold that achieves at least *target_sensitivity*.

    Uses the precision-recall curve, which returns recall sorted
    **decreasing** (from 1.0 down to 0.0).  We pick the **last** index
    where ``recall >= target_sensitivity``, which corresponds to the
    highest threshold — maximising specificity while meeting the
    sensitivity target.

    Args:
        y_true: Binary ground-truth labels.
        y_scores: Predicted anomaly scores in [0, 1].
        target_sensitivity: Minimum acceptable sensitivity.

    Returns:
        ``(threshold, actual_sensitivity, actual_specificity)``.
    """
    _, recall, thresholds = precision_recall_curve(y_true, y_scores)
    thresholds = np.append(thresholds, 0.0)

    matches = np.where(recall >= target_sensitivity)[0]
    if len(matches) > 0:
        best_idx = matches[-1]
        threshold = float(thresholds[best_idx])
        actual_sensitivity = float(recall[best_idx])
    else:
        threshold = 0.0
        actual_sensitivity = 0.0

    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if tn + fp > 0:
        specificity = float(tn / (tn + fp))
    else:
        specificity = float("nan")

    if threshold == 0.0 and tp + fn > 0:
        actual_sensitivity = float(tp / (tp + fn))

    return threshold, actual_sensitivity, specificity


def find_inconclusive_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_sensitivity: float = 0.99,
    margin: float = 0.05,
) -> Dict[str, float]:
    """Find two thresholds defining a triage zone.

    - **thr_revisar**: scores ≥ this value → REVISAR (high confidence abnormal).
    - **thr_inconclusive**: scores below *thr_revisar* but above
      *thr_normal* → INCONCLUSIVE (needs radiologist but not urgent).
    - **thr_normal**: scores ≤ this value → NORMAL.

    The inconclusive zone width is controlled by *margin*.

    Args:
        y_true: Binary labels.
        y_scores: Predicted scores.
        target_sensitivity: Sensitivity target for the REVISAR threshold.
        margin: Fraction of score range reserved for inconclusive zone.

    Returns:
        Dict with ``thr_normal``, ``thr_revisar``, and ``thr_inconclusive_width``.
    """
    thr_revisar, sens, spec = find_threshold_for_sensitivity(
        y_true, y_scores, target_sensitivity
    )
    thr_inconclusive = max(0.0, thr_revisar - margin)
    thr_normal = max(0.0, thr_inconclusive - margin)
    return {
        "thr_normal": thr_normal,
        "thr_revisar": thr_revisar,
        "thr_inconclusive_min": thr_normal,
        "thr_inconclusive_max": thr_revisar,
        "inconclusive_width": thr_revisar - thr_normal,
    }


def compute_triage_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
) -> dict:
    """Compute a standard set of triage metrics.

    Args:
        y_true: Binary ground-truth labels.
        y_scores: Anomaly scores in [0, 1].
        threshold: Decision threshold.  If None, computed via
            :func:`find_threshold_for_sensitivity` at 99%.

    Returns:
        Dict with ``roc_auc``, ``pr_auc``, ``sensitivity``, ``specificity``,
        ``npv``, ``ppv``, ``threshold``, ``discard_rate``, ``f1``,
        ``accuracy``, and ``confusion_matrix``.
    """
    y_true = np.asarray(y_true).flatten()
    y_scores = np.asarray(y_scores).flatten()

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        warnings.warn("Only one class present in y_true; ROC-AUC is undefined")
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_scores))
    except ValueError:
        warnings.warn("Only one class present in y_true; PR-AUC is undefined")
        pr_auc = float("nan")

    if threshold is None:
        threshold, _, _ = find_threshold_for_sensitivity(
            y_true, y_scores, target_sensitivity=0.99
        )

    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv_val = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    discard_rate = tn / len(y_true)

    f1 = float(f1_score(y_true, y_pred))
    accuracy = (tp + tn) / len(y_true)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(npv_val),
        "ppv": float(ppv),
        "threshold": float(threshold),
        "discard_rate": float(discard_rate),
        "f1": f1,
        "accuracy": float(accuracy),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }


def calibration_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration metrics with equal-mass binning.

    Uses quantile-based bins so every bin contains approximately the same
    number of samples — robust for imbalanced datasets.

    Args:
        y_true: Binary labels.
        y_scores: Scores in [0, 1].
        n_bins: Number of bins.

    Returns:
        Dict with ``ece``, ``brier``, ``n_bins``, ``bin_edges``,
        ``bin_centers``, ``bin_accuracies``, ``bin_confidences``,
        ``bin_counts``.
    """
    y_true = np.asarray(y_true).flatten()
    y_scores = np.asarray(y_scores).flatten()

    q_values = np.quantile(y_scores, np.linspace(0, 1, n_bins + 1))
    if len(np.unique(q_values)) < len(q_values):
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bin_edges = q_values
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    accuracies = np.zeros(n_bins)
    confidences = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        in_bin = (y_scores >= bin_edges[i]) & (y_scores < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (y_scores >= bin_edges[i]) & (y_scores <= bin_edges[i + 1])
        counts[i] = np.sum(in_bin)
        if counts[i] > 0:
            accuracies[i] = np.mean(y_true[in_bin])
            confidences[i] = np.mean(y_scores[in_bin])

    ece = np.sum(counts * np.abs(accuracies - confidences)) / max(np.sum(counts), 1)
    brier = float(brier_score_loss(y_true, y_scores))

    return {
        "ece": float(ece),
        "brier": brier,
        "n_bins": n_bins,
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "bin_accuracies": accuracies.tolist(),
        "bin_confidences": confidences.tolist(),
        "bin_counts": counts.tolist(),
    }


def compute_segmentation_metrics(
    heatmap: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute Dice, IoU between binary heatmap and ground truth mask.

    Args:
        heatmap: ``(D, H, W)`` float heatmap.
        gt_mask: ``(D, H, W)`` binary ground truth mask.
        threshold: Binarisation threshold for the heatmap.

    Returns:
        Dict with ``dice``, ``iou``, ``sensitivity``, ``specificity``.
    """
    heatmap = np.asarray(heatmap, dtype=np.float64)
    gt_mask = np.asarray(gt_mask, dtype=np.float64)

    pred_bin = (heatmap >= threshold).astype(np.float64)
    gt_bin = (gt_mask > 0).astype(np.float64)

    intersection = np.sum(pred_bin * gt_bin)
    dice_val = (
        (2.0 * intersection) / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)
    )

    union = np.sum((pred_bin + gt_bin) > 0)
    iou_val = intersection / (union + 1e-8)

    tp = intersection
    fn = np.sum((1 - pred_bin) * gt_bin)
    tn = np.sum((1 - pred_bin) * (1 - gt_bin))
    fp = np.sum(pred_bin * (1 - gt_bin))

    return {
        "dice": float(dice_val),
        "iou": float(iou_val),
        "sensitivity_voxel": float(tp / (tp + fn + 1e-8)),
        "specificity_voxel": float(tn / (tn + fp + 1e-8)),
    }


def compute_metrics_per_anatomy(results_df) -> dict:
    """Compute per-anatomy metrics from a results DataFrame."""
    import pandas as pd

    df = pd.DataFrame(results_df)

    per_anatomy = {}
    for anatomy in df["anatomy"].unique():
        mask = df["anatomy"] == anatomy
        y_true = df.loc[mask, "y_true"].values
        y_score = df.loc[mask, "y_score"].values
        per_anatomy[anatomy] = compute_triage_metrics(y_true, y_score)

    y_true_all = df["y_true"].values
    y_score_all = df["y_score"].values
    per_anatomy["global"] = compute_triage_metrics(y_true_all, y_score_all)

    return per_anatomy


def format_metrics_report(metrics_dict: dict) -> str:
    """Format metrics dict as a human-readable report string."""
    lines = []
    for key, metrics in metrics_dict.items():
        lines.append(f"--- {key} ---")
        lines.append(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
        lines.append(f"  PR-AUC:        {metrics['pr_auc']:.4f}")
        lines.append(f"  Accuracy:      {metrics['accuracy']:.4f}")
        lines.append(f"  Sensitivity:   {metrics['sensitivity']:.4f}")
        lines.append(f"  Specificity:   {metrics['specificity']:.4f}")
        lines.append(f"  PPV:           {metrics['ppv']:.4f}")
        lines.append(f"  NPV:           {metrics['npv']:.4f}")
        lines.append(f"  F1:            {metrics['f1']:.4f}")
        lines.append(f"  Discard Rate:  {metrics['discard_rate']:.4f}")
        lines.append(f"  Threshold:     {metrics['threshold']:.4f}")
        cm = metrics.get("confusion_matrix", {})
        lines.append(
            f"  Confusion:     TN={cm.get('tn', '?')} FP={cm.get('fp', '?')} "
            f"FN={cm.get('fn', '?')} TP={cm.get('tp', '?')}"
        )
        lines.append("")
    return "\n".join(lines)
