from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
import numpy as np

from kinship.metrics import compute_verification_metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


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
            "accuracy": -1.0,
            "balanced_accuracy": -1.0,
            "recall": -1.0,
            "precision": -1.0,
            "f1-score": -1.0,
            "f1_score": -1.0,
            "auc": -1.0,
            "pr_auc": -1.0,
            "roc_auc": -1.0,
            "eer": 1.0,
            "tar_at_far": {},
            "threshold": 0.5,
            "precision_curve": [],
            "recall_curve": [],
            "pr_thresholds": [],
            "roc_fpr": [],
            "roc_tpr": [],
            "roc_thresholds": [],
            "counts": {},
            "predictions": [],
            "scores": [],
            "labels": [],
        }
        self.metrics_hist = {
            "acc": [],
            "balanced_accuracy": [],
            "recall": [],
            "precision": [],
            "f1-score": [],
            "auc": [],
            "roc_auc": [],
            "eer": [],
        }
        self.best_model_scores: list[float] | None = None
        self.best_model_labels: list[float] | None = None

    def reset(self) -> None:
        self.model_scores = []
        self.labels = []

    def add_batch(self, scores: list[float], labels: list[float]) -> None:
        self.model_scores += scores
        self.labels += labels

    def get_metrics(self, target_metric: str = "acc") -> dict:
        metrics = compute_verification_metrics(self.model_scores, self.labels, threshold=0.5)
        metrics["precision_curve"] = metrics["precision_curve"]
        metrics["recall_curve"] = metrics["recall_curve"]
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
        plt.plot(self.metrics_hist["balanced_accuracy"], color="slateblue", label="Balanced Acc.", linestyle="-.")
        plt.plot(self.metrics_hist["f1-score"], color="turquoise", label="F1-Score", linestyle="--")
        plt.plot(self.metrics_hist["auc"], color="gold", label="PR-AUC", linestyle=":")
        plt.plot(self.metrics_hist["roc_auc"], color="forestgreen", label="ROC-AUC", linestyle="-")
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
        precision_curve = np.asarray(self.best_metrics["precision_curve"], dtype=np.float64)
        recall_curve = np.asarray(self.best_metrics["recall_curve"], dtype=np.float64)
        thresholds = np.asarray(self.best_metrics["pr_thresholds"], dtype=np.float64)
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
        plt.title(f"{title} PR-AUC: {self.best_metrics['pr_auc']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(color="black", linestyle="--", linewidth=1, alpha=0.15)
        fig.savefig(self.log_path / f"{log_name}.png")
        plt.close()
        payload = copy.deepcopy(self.best_metrics)
        payload["best_threshold"] = best_threshold
        save_json(self.log_path / f"{log_name}.json", payload)

    def get_kinface_pair_metrics(self, evaluators: list["KinshipEvaluator"], pair_type: str) -> dict:
        scores, labels = [], []
        accs, balanced_accs, recalls, precisions, f_scores, pr_aucs, roc_aucs, eers = [], [], [], [], [], [], [], []
        tar_series: list[dict[str, float]] = []
        for evaluator in evaluators:
            accs.append(float(evaluator.best_metrics["acc"]))
            balanced_accs.append(float(evaluator.best_metrics["balanced_accuracy"]))
            recalls.append(float(evaluator.best_metrics["recall"]))
            precisions.append(float(evaluator.best_metrics["precision"]))
            f_scores.append(float(evaluator.best_metrics["f1-score"]))
            pr_aucs.append(float(evaluator.best_metrics["pr_auc"]))
            roc_aucs.append(float(evaluator.best_metrics["roc_auc"]))
            eers.append(float(evaluator.best_metrics["eer"]))
            tar_series.append({key: float(value) for key, value in evaluator.best_metrics["tar_at_far"].items()})
            if evaluator.best_model_scores is not None:
                scores += evaluator.best_model_scores
            if evaluator.best_model_labels is not None:
                labels += evaluator.best_model_labels
        pair_metrics = compute_verification_metrics(scores, labels, threshold=0.5)
        pair_metrics["acc"] = float(np.mean(accs))
        pair_metrics["accuracy"] = pair_metrics["acc"]
        pair_metrics["balanced_accuracy"] = float(np.mean(balanced_accs))
        pair_metrics["recall"] = float(np.mean(recalls))
        pair_metrics["precision"] = float(np.mean(precisions))
        pair_metrics["f1-score"] = float(np.mean(f_scores))
        pair_metrics["f1_score"] = pair_metrics["f1-score"]
        pair_metrics["auc"] = float(np.mean(pr_aucs))
        pair_metrics["pr_auc"] = pair_metrics["auc"]
        pair_metrics["roc_auc"] = float(np.mean(roc_aucs))
        pair_metrics["eer"] = float(np.mean(eers))
        if tar_series:
            pair_metrics["tar_at_far"] = {
                key: float(np.mean([series[key] for series in tar_series]))
                for key in tar_series[0]
            }
        precision_curve = np.asarray(pair_metrics["precision_curve"], dtype=np.float64)
        recall_curve = np.asarray(pair_metrics["recall_curve"], dtype=np.float64)
        fig = plt.figure()
        plt.plot(recall_curve, precision_curve, color="turquoise", label="PR", linestyle="--")
        plt.title(f"{pair_type.upper()} Precision Recall Curve PR-AUC: {pair_metrics['pr_auc']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(color="black", linestyle="--", linewidth=1, alpha=0.15)
        fig.savefig(self.log_path / f"{pair_type.upper()}.png")
        plt.close()
        save_json(self.log_path / f"{pair_type}.json", pair_metrics)
        return pair_metrics
