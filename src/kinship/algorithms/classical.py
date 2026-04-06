from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from kinship.datasets.kinface import PairRecord, load_kinface_pairs
from kinship.features import extract_pair_chisq_feature, extract_pair_patch_feature


RANDOM_TRAIN_COUNTS = {
    "fs": (109, 109),
    "fd": (93, 94),
    "ms": (81, 81),
    "md": (88, 89),
}


@dataclass
class ClassicalResult:
    method: str
    relation: str
    dataset: str
    fold_scores: list[float]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.fold_scores))


def _build_classifier(method: str, relation: str) -> SVC:
    if method == "random":
        return SVC(kernel="linear")
    if method == "kfold":
        if relation == "fd":
            return SVC(kernel="poly", degree=2, gamma="scale")
        return SVC(kernel="rbf", gamma="scale")
    if method == "chisq":
        return SVC(kernel="linear")
    raise ValueError(f"Unsupported method '{method}'")


def _select_features(method: str) -> Callable[[PairRecord], np.ndarray]:
    if method == "chisq":
        return lambda record: extract_pair_chisq_feature(
            record.parent_path, record.child_path
        )
    return lambda record: extract_pair_patch_feature(record.parent_path, record.child_path)


def _materialize_features(
    records: list[PairRecord],
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    extractor = _select_features(method)
    x = np.stack([extractor(record) for record in records], axis=0)
    y = np.asarray([record.label for record in records], dtype=np.int32)
    return x, y


def _random_split_indices(labels: np.ndarray, relation: str) -> tuple[np.ndarray, np.ndarray]:
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    pos_train, neg_train = RANDOM_TRAIN_COUNTS[relation]
    if len(positives) <= pos_train or len(negatives) <= neg_train:
        indices = np.arange(labels.shape[0])
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.3,
            random_state=42,
            shuffle=True,
            stratify=labels,
        )
        return np.sort(train_idx), np.sort(test_idx)
    train_idx = np.concatenate([positives[:pos_train], negatives[:neg_train]])
    test_idx = np.concatenate([positives[pos_train:], negatives[neg_train:]])
    return train_idx, test_idx


def run_classical_verification(
    relation: str,
    dataset: str = "KinFaceW-I",
    method: str = "random",
    limit: int | None = None,
) -> ClassicalResult:
    if method not in {"random", "kfold", "chisq"}:
        raise ValueError("method must be one of: random, kfold, chisq")

    records = load_kinface_pairs(relation=relation, dataset=dataset)
    if limit is not None:
        positives = [record for record in records if record.label == 1][: limit // 2]
        negatives = [record for record in records if record.label == 0][: limit // 2]
        records = positives + negatives

    x, y = _materialize_features(records, method)

    if method == "random":
        train_idx, test_idx = _random_split_indices(y, relation)
        clf = _build_classifier(method, relation)
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])
        scores = [float(accuracy_score(y[test_idx], pred))]
    else:
        scores = []
        splitter = StratifiedKFold(n_splits=5, shuffle=False)
        for train_idx, test_idx in splitter.split(x, y):
            clf = _build_classifier(method, relation)
            clf.fit(x[train_idx], y[train_idx])
            pred = clf.predict(x[test_idx])
            scores.append(float(accuracy_score(y[test_idx], pred)))

    return ClassicalResult(
        method=method,
        relation=relation,
        dataset=dataset,
        fold_scores=scores,
    )
