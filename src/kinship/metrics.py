from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


DEFAULT_FAR_TARGETS = (1e-3, 1e-2, 1e-1)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    objective: str
    score: float


def _as_arrays(scores: Iterable[float], labels: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    score_arr = np.asarray(list(scores), dtype=np.float64)
    label_arr = np.asarray(list(labels), dtype=np.int64)
    if score_arr.ndim != 1 or label_arr.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if score_arr.shape[0] != label_arr.shape[0]:
        raise ValueError("scores and labels must have the same length")
    if score_arr.shape[0] == 0:
        raise ValueError("scores and labels must not be empty")
    return score_arr, label_arr


def _prediction_counts(predictions: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    return {
        "samples": int(labels.size),
        "positive_labels": int(np.sum(labels == 1)),
        "negative_labels": int(np.sum(labels == 0)),
        "positive_predictions": int(np.sum(predictions == 1)),
        "negative_predictions": int(np.sum(predictions == 0)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores)
    if unique_scores.size == 1:
        return np.array([float(unique_scores[0])], dtype=np.float64)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.concatenate(
        (
            np.array([unique_scores[0] - 1e-6], dtype=np.float64),
            midpoints,
            np.array([unique_scores[-1] + 1e-6], dtype=np.float64),
        )
    )
    return candidates


def select_threshold(
    scores: Iterable[float],
    labels: Iterable[int],
    objective: str = "balanced_accuracy",
) -> ThresholdSelection:
    score_arr, label_arr = _as_arrays(scores, labels)
    objective = objective.lower()
    if objective not in {"balanced_accuracy", "f1", "eer", "accuracy"}:
        raise ValueError(
            "objective must be one of: balanced_accuracy, f1, eer, accuracy"
        )

    candidates = _candidate_thresholds(score_arr)
    best_threshold = float(candidates[0])
    best_score = float("-inf")

    for threshold in candidates:
        predictions = (score_arr >= threshold).astype(np.int64)
        if objective == "balanced_accuracy":
            score = float(balanced_accuracy_score(label_arr, predictions))
        elif objective == "f1":
            score = float(f1_score(label_arr, predictions, zero_division=0))
        elif objective == "accuracy":
            score = float(accuracy_score(label_arr, predictions))
        else:
            counts = _prediction_counts(predictions, label_arr)
            far = counts["fp"] / max(1, counts["negative_labels"])
            fnr = counts["fn"] / max(1, counts["positive_labels"])
            score = -abs(far - fnr)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    if objective == "eer":
        best_score = -best_score
    return ThresholdSelection(threshold=best_threshold, objective=objective, score=best_score)


def _tar_at_far(
    roc_fpr: np.ndarray,
    roc_tpr: np.ndarray,
    far_targets: Iterable[float],
) -> dict[str, float]:
    payload: dict[str, float] = {}
    for far in far_targets:
        key = f"tar@far={far:g}"
        mask = roc_fpr <= far
        payload[key] = float(np.max(roc_tpr[mask])) if np.any(mask) else 0.0
    return payload


def compute_verification_metrics(
    scores: Iterable[float],
    labels: Iterable[int],
    threshold: float = 0.5,
    far_targets: Iterable[float] = DEFAULT_FAR_TARGETS,
) -> dict:
    score_arr, label_arr = _as_arrays(scores, labels)
    predictions = (score_arr >= threshold).astype(np.int64)
    counts = _prediction_counts(predictions, label_arr)

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(label_arr, score_arr)
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(label_arr, score_arr)
    pr_auc = float(auc(recall_curve, precision_curve))
    roc_auc = float(auc(roc_fpr, roc_tpr))
    fnr = 1.0 - roc_tpr
    eer_index = int(np.argmin(np.abs(roc_fpr - fnr)))
    eer = float((roc_fpr[eer_index] + fnr[eer_index]) / 2.0)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(label_arr, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(label_arr, predictions)),
        "recall": float(recall_score(label_arr, predictions, zero_division=0)),
        "precision": float(precision_score(label_arr, predictions, zero_division=0)),
        "f1_score": float(f1_score(label_arr, predictions, zero_division=0)),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "eer": eer,
        "tar_at_far": _tar_at_far(roc_fpr, roc_tpr, far_targets),
        "counts": counts,
        "scores": list(map(float, score_arr.tolist())),
        "labels": list(map(int, label_arr.tolist())),
        "predictions": list(map(int, predictions.tolist())),
        "precision_curve": list(map(float, precision_curve.tolist())),
        "recall_curve": list(map(float, recall_curve.tolist())),
        "pr_thresholds": list(map(float, pr_thresholds.tolist())),
        "roc_fpr": list(map(float, roc_fpr.tolist())),
        "roc_tpr": list(map(float, roc_tpr.tolist())),
        "roc_thresholds": list(map(float, roc_thresholds.tolist())),
    }
    metrics["auc"] = pr_auc
    metrics["acc"] = metrics["accuracy"]
    metrics["f1-score"] = metrics["f1_score"]
    metrics["precision_curve_best_index"] = int(np.nanargmax(
        np.divide(
            2 * precision_curve * recall_curve,
            precision_curve + recall_curve,
            out=np.zeros_like(precision_curve, dtype=np.float64),
            where=(precision_curve + recall_curve) != 0,
        )
    ))
    return metrics
