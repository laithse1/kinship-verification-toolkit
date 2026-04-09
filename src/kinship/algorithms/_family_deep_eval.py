from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, precision_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _prediction_payload(scores: list[float], labels: list[float], threshold: float = 0.5) -> dict:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    predictions = (scores_arr > threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "scores": list(map(float, scores_arr.tolist())),
        "labels": list(map(int, labels_arr.tolist())),
        "predictions": list(map(int, predictions.tolist())),
        "counts": {
            "samples": int(labels_arr.size),
            "positive_labels": int(labels_arr.sum()),
            "negative_labels": int(labels_arr.size - labels_arr.sum()),
            "positive_predictions": int(predictions.sum()),
            "negative_predictions": int(predictions.size - predictions.sum()),
        },
    }


class KinshipEvaluator:
    def __init__(self, set_name: str, pair: str, log_path: Path, fold: int | None = None):
        plt.ioff()
        self.set_name = set_name
        self.pair = pair
        self.log_path = Path(log_path)
        self.fold = fold
        self.model_scores: list[float] = []
        self.labels: list[float] = []
        self.best_metrics = {
            "acc": -1.0,
            "recall": -1.0,
            "precision": -1.0,
            "f1-score": -1.0,
            "precision_curve": -1,
            "recall_curve": -1,
            "thresholds": -1,
            "auc": -1.0,
        }
        self.metrics_hist = {"acc": [], "recall": [], "precision": [], "f1-score": [], "auc": []}
        self.best_model_scores: list[float] | None = None
        self.best_model_labels: list[float] | None = None

    def reset(self) -> None:
        self.model_scores = []
        self.labels = []

    def add_batch(self, scores: list[float], labels: list[float]) -> None:
        self.model_scores += scores
        self.labels += labels

    def get_metrics(self, target_metric: str = "acc") -> dict:
        probas = np.array(self.model_scores)
        targets = np.array(self.labels)
        predictions = np.zeros_like(probas)
        predictions[probas > 0.5] = 1
        metrics = {
            "acc": float(accuracy_score(targets, predictions)),
            "recall": float(recall_score(targets, predictions, zero_division=0)),
            "precision": float(precision_score(targets, predictions, zero_division=0)),
            "f1-score": float(f1_score(targets, predictions, zero_division=0)),
        }
        precisions, recalls, thresholds = precision_recall_curve(targets, probas)
        metrics["precision_curve"] = precisions
        metrics["recall_curve"] = recalls
        metrics["thresholds"] = thresholds
        metrics["auc"] = float(auc(recalls, precisions))
        if metrics[target_metric] > self.best_metrics[target_metric]:
            self.best_metrics = copy.deepcopy(metrics)
            self.best_model_scores = copy.deepcopy(self.model_scores)
            self.best_model_labels = copy.deepcopy(self.labels)
        for key in self.metrics_hist:
            self.metrics_hist[key].append(metrics[key])
        return metrics

    def save_hist(self) -> None:
        title = f"{self.pair.upper()} {self.set_name} Metrics"
        log_name = f"{self.pair.lower()}_hist_{self.set_name.lower()}"
        if self.fold is not None:
            title += f" Fold {self.fold}"
            log_name += f"_fold_{self.fold}"
        fig = plt.figure()
        plt.title(title)
        plt.plot(self.metrics_hist["acc"], color="tomato", label="Accuracy")
        plt.plot(self.metrics_hist["f1-score"], color="turquoise", label="F1-Score", linestyle="--")
        plt.plot(self.metrics_hist["auc"], color="gold", label="AUC", linestyle=":")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.grid(color="black", linestyle="--", linewidth=1, alpha=0.15)
        fig.savefig(self.log_path / f"{log_name}.png")
        plt.close()
        save_json(self.log_path / f"{log_name}.json", self.metrics_hist)

    def save_best_metrics(self) -> None:
        title = f"{self.pair.upper()} {self.set_name} Precision Recall Curve"
        log_name = f"{self.pair.lower()}_{self.set_name.lower()}"
        if self.fold is not None:
            title += f" Fold {self.fold}"
            log_name += f"_fold_{self.fold}"
        precision_curve = self.best_metrics["precision_curve"]
        recall_curve = self.best_metrics["recall_curve"]
        thresholds = self.best_metrics["thresholds"]
        denom = precision_curve + recall_curve
        fscore = np.divide(
            2 * precision_curve * recall_curve,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom != 0,
        )
        ix = int(np.nanargmax(fscore))
        best_threshold = float(thresholds[ix]) if len(thresholds) > ix else 0.5
        fig = plt.figure()
        plt.plot(recall_curve, precision_curve, color="turquoise", label="PR", linestyle="--")
        plt.scatter(recall_curve[ix], precision_curve[ix], marker="o", color="tomato", label="Best")
        plt.title(f"{title} AUC: {self.best_metrics['auc']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(color="black", linestyle="--", linewidth=1, alpha=0.15)
        fig.savefig(self.log_path / f"{log_name}.png")
        plt.close()
        payload = copy.deepcopy(self.best_metrics)
        payload["best_threshold"] = best_threshold
        payload["precision_curve"] = list(map(float, precision_curve))
        payload["recall_curve"] = list(map(float, recall_curve))
        payload["thresholds"] = list(map(float, thresholds))
        if self.best_model_scores is not None and self.best_model_labels is not None:
            payload["prediction_export"] = _prediction_payload(
                self.best_model_scores,
                self.best_model_labels,
                threshold=0.5,
            )
        save_json(self.log_path / f"{log_name}.json", payload)

    def get_kinface_pair_metrics(self, evaluators: list["KinshipEvaluator"], pair_type: str) -> dict:
        accs, recalls, precisions, f_scores, scores, labels = [], [], [], [], [], []
        for evaluator in evaluators:
            accs.append(evaluator.best_metrics["acc"])
            recalls.append(evaluator.best_metrics["recall"])
            precisions.append(evaluator.best_metrics["precision"])
            f_scores.append(evaluator.best_metrics["f1-score"])
            if evaluator.best_model_scores is not None:
                scores += evaluator.best_model_scores
            if evaluator.best_model_labels is not None:
                labels += evaluator.best_model_labels
        pair_precisions, pair_recalls, pair_thresholds = precision_recall_curve(labels, scores)
        pair_metrics = {
            "acc": float(np.mean(accs)),
            "recall": float(np.mean(recalls)),
            "precision": float(np.mean(precisions)),
            "f1-score": float(np.mean(f_scores)),
            "auc": float(auc(pair_recalls, pair_precisions)),
        }
        denom = pair_precisions + pair_recalls
        fscore = np.divide(
            2 * pair_precisions * pair_recalls,
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom != 0,
        )
        ix = int(np.nanargmax(fscore))
        fig = plt.figure()
        plt.plot(pair_recalls, pair_precisions, color="turquoise", label="PR", linestyle="--")
        plt.scatter(pair_recalls[ix], pair_precisions[ix], marker="o", color="tomato", label="Best")
        plt.title(f"{pair_type.upper()} Precision Recall Curve AUC: {pair_metrics['auc']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(color="black", linestyle="--", linewidth=1, alpha=0.15)
        fig.savefig(self.log_path / f"{pair_type.upper()}.png")
        plt.close()
        pair_metrics["precision_curve"] = list(map(float, pair_precisions))
        pair_metrics["recall_curve"] = list(map(float, pair_recalls))
        pair_metrics["thresholds"] = list(map(float, pair_thresholds))
        pair_metrics["prediction_export"] = _prediction_payload(scores, labels, threshold=0.5)
        save_json(self.log_path / f"{pair_type}.json", pair_metrics)
        return pair_metrics
