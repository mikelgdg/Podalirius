import warnings
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
    brier_score_loss,
)


def find_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_sensitivity: float = 0.99,
) -> Tuple[float, float, float]:
    _, recall, thresholds = precision_recall_curve(y_true, y_scores)
    thresholds = np.append(thresholds, 0.0)

    matches = np.where(recall >= target_sensitivity)[0]
    if len(matches) > 0:
        best_idx = matches[-1]
        threshold = thresholds[best_idx]
        actual_sensitivity = recall[best_idx]
    else:
        best_idx = len(recall) - 1
        threshold = 0.0
        actual_sensitivity = float("nan")

    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = float("nan")

    if threshold == 0.0 and tp + fn > 0:
        actual_sensitivity = tp / (tp + fn)

    return float(threshold), float(actual_sensitivity), float(specificity)


def compute_triage_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
) -> dict:
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
        threshold, _, _ = find_threshold_for_sensitivity(y_true, y_scores, target_sensitivity=0.99)

    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    discard_rate = tn / len(y_true)

    f1 = float(f1_score(y_true, y_pred))
    accuracy = (tp + tn) / len(y_true)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(npv),
        "ppv": float(ppv),
        "threshold": float(threshold),
        "discard_rate": float(discard_rate),
        "f1": f1,
        "accuracy": float(accuracy),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def calibration_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
) -> dict:
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

    ece = np.sum(counts * np.abs(accuracies - confidences)) / np.sum(counts)
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


def compute_metrics_per_anatomy(results_df) -> dict:
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
        lines.append(f"  Confusion:     TN={cm.get('tn', '?')} FP={cm.get('fp', '?')} FN={cm.get('fn', '?')} TP={cm.get('tp', '?')}")
        lines.append("")
    return "\n".join(lines)
