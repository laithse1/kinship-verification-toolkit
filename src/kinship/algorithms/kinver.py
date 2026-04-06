from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from kinship.paths import kinver_workspace_root


PAIR_ORDER = ["fd", "fs", "md", "ms"]
FISHER_DIM_KINFACEW_II = {
    "fd": 0.4,
    "fs": 0.1,
    "md": 0.075,
    "ms": 0.1,
}
FISHER_DIM_KINFACEW_I = {
    "fd": 0.4,
    "fs": 0.075,
    "md": 0.1,
    "ms": 0.075,
}
WDIMS_KINFACEW_II = {"fd": 69, "fs": 66, "md": 50, "ms": 62}
WDIMS_KINFACEW_I = {"fd": 66, "fs": 57, "md": 48, "ms": 55}


@dataclass
class KinVerResult:
    dataset: str
    relation: str
    fold_scores: list[float]
    beta_means: list[float]
    n_components: int

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.fold_scores))


def _dataset_dir(dataset: str) -> Path:
    root = kinver_workspace_root()
    direct = root / f"data-{dataset}"
    if direct.exists():
        return direct
    return root / "data" / f"data-{dataset}"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_feature_matrix(dataset: str, relation: str, prefix: str) -> dict[str, np.ndarray]:
    mat = sio.loadmat(_dataset_dir(dataset) / f"{prefix}_{relation}.mat")
    return {
        "ux": np.asarray(mat["ux"]),
        "idxa": np.asarray(mat["idxa"]).ravel().astype(np.int32) - 1,
        "idxb": np.asarray(mat["idxb"]).ravel().astype(np.int32) - 1,
        "fold": np.asarray(mat["fold"]).ravel().astype(np.int32),
        "matches": np.asarray(mat["matches"]).ravel().astype(np.int32),
    }


def _merge_pairs(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    return np.abs(xa - xb)


def _top_fisher_indices(x: np.ndarray, y: np.ndarray, fraction: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, _ = f_classif(x, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    n_select = max(50, int(round(x.shape[1] * fraction)))
    return np.argsort(scores)[::-1][:n_select]


def _compute_h(
    x: np.ndarray,
    y: np.ndarray,
    n_x: np.ndarray,
    n_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = x.shape[0]
    h1 = np.zeros((dim, dim), dtype=np.float64)
    h2 = np.zeros((dim, dim), dtype=np.float64)
    for i in range(x.shape[1]):
        dif = x[:, [i]] - y[:, n_y[i]]
        h1 += dif @ dif.T
    h1 /= (x.shape[1] * n_y.shape[1])
    for i in range(y.shape[1]):
        dif = x[:, n_x[i]] - y[:, [i]]
        h2 += dif @ dif.T
    h2 /= (y.shape[1] * n_x.shape[1])
    dif = x - y
    h3 = (dif @ dif.T) / x.shape[1]
    return h1, h2, h3


def _knn_indices(x: np.ndarray, k: int) -> np.ndarray:
    neigh = NearestNeighbors(n_neighbors=k + 1, n_jobs=1)
    neigh.fit(x.T)
    return neigh.kneighbors(return_distance=False)[:, 1:]


def _mnrml_train(
    xa_pos: list[np.ndarray],
    xb_pos: list[np.ndarray],
    k: int,
    dim: int,
    n_iter: int,
    q: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    beta = np.ones(len(xa_pos), dtype=np.float64) / len(xa_pos)
    prev_w = None
    for _ in range(n_iter):
        h_total = np.zeros((dim, dim), dtype=np.float64)
        h_parts = []
        neighbor_pairs = []
        for x_feat, y_feat in zip(xa_pos, xb_pos, strict=True):
            n_x = _knn_indices(x_feat, k)
            n_y = _knn_indices(y_feat, k)
            h1, h2, h3 = _compute_h(x_feat, y_feat, n_x, n_y)
            h_parts.append((h1, h2, h3))
            h_total += beta[len(h_parts) - 1] * (h1 + h2 - h3)
            neighbor_pairs.append((n_x, n_y))

        eigvals, eigvecs = np.linalg.eigh(h_total)
        order = np.argsort(eigvals)[::-1][:dim]
        w = eigvecs[:, order]

        denom = []
        for h1, h2, h3 in h_parts:
            score = np.trace(w.T @ (h1 + h2 - h3) @ w)
            score = max(float(score), 1e-12)
            denom.append((1.0 / score) ** (1.0 / (q - 1.0)))
        beta = np.asarray(denom, dtype=np.float64)
        beta /= beta.sum()

        if prev_w is not None and np.linalg.norm(w - prev_w) <= 1e-2:
            break
        prev_w = w
    return w, beta


def run_kinver(
    relation: str,
    dataset: str = "KinFaceW-II",
    use_vggface: bool = True,
    use_vggf: bool = True,
    use_lbp: bool = False,
    use_hog: bool = False,
    use_feature_selection: bool = True,
    use_pca: bool = True,
    use_mnrml: bool = True,
    iterations: int = 4,
    knn: int = 6,
) -> KinVerResult:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    if relation not in PAIR_ORDER:
        raise ValueError(f"Unsupported relation '{relation}'")
    prefixes = []
    if use_vggface:
        prefixes.append("vggFace")
    if use_vggf:
        prefixes.append("vggF")
    if use_lbp:
        prefixes.append("LBP")
    if use_hog:
        prefixes.append("HOG")
    if not prefixes:
        raise ValueError("At least one feature source must be enabled")

    reference = _load_feature_matrix(dataset, relation, prefixes[0])
    idxa = reference["idxa"]
    idxb = reference["idxb"]
    fold = reference["fold"]
    matches = reference["matches"]

    features = []
    for prefix in prefixes:
        item = _load_feature_matrix(dataset, relation, prefix)
        feat = np.asarray(item["ux"], dtype=np.float64)
        if prefix in {"vggFace", "vggF"}:
            feat = _normalize_rows(feat)
        features.append(feat)

    fisher_dims = (
        FISHER_DIM_KINFACEW_I if dataset == "KinFaceW-I" else FISHER_DIM_KINFACEW_II
    )
    wdims = WDIMS_KINFACEW_I if dataset == "KinFaceW-I" else WDIMS_KINFACEW_II
    unique_folds = np.unique(fold)
    projected_by_fold: list[list[np.ndarray]] = []
    beta_by_fold: list[np.ndarray] = []

    if use_feature_selection:
        train_mask = fold != unique_folds[0]
        tr_idxa = idxa[train_mask]
        tr_idxb = idxb[train_mask]
        tr_matches = matches[train_mask]
        for feature_index, feat in enumerate(features):
            merged = _merge_pairs(feat[tr_idxa], feat[tr_idxb])
            ranking = _top_fisher_indices(merged, tr_matches, fisher_dims[relation])
            features[feature_index] = feat[:, ranking]

    for current_fold in unique_folds:
        train_mask = fold != current_fold
        tr_idxa = idxa[train_mask]
        tr_idxb = idxb[train_mask]
        tr_matches = matches[train_mask]
        transformed_features: list[np.ndarray] = []
        xa_pos: list[np.ndarray] = []
        xb_pos: list[np.ndarray] = []

        for feat in features:
            current = feat
            if use_pca:
                train_people = np.vstack([current[tr_idxa], current[tr_idxb]])
                n_components = min(wdims[relation], train_people.shape[0], train_people.shape[1])
                pca = PCA(n_components=n_components, svd_solver="full")
                current = pca.fit_transform(current)
                current = _normalize_rows(current)
            transformed_features.append(current)
            xa_pos.append(current[tr_idxa[tr_matches == 1]].T)
            xb_pos.append(current[tr_idxb[tr_matches == 1]].T)

        if use_mnrml:
            metric_dim = min(transformed_features[0].shape[1], wdims[relation])
            w, beta = _mnrml_train(
                xa_pos=xa_pos,
                xb_pos=xb_pos,
                k=knn,
                dim=metric_dim,
                n_iter=iterations,
            )
            projected_fold = [feat @ w for feat in transformed_features]
        else:
            beta = np.ones(len(transformed_features), dtype=np.float64) / len(
                transformed_features
            )
            projected_fold = transformed_features
        projected_by_fold.append(projected_fold)
        beta_by_fold.append(beta)

    scores: list[float] = []
    for fold_index, current_fold in enumerate(unique_folds):
        train_mask = fold != current_fold
        test_mask = fold == current_fold
        tr_matches = matches[train_mask]
        ts_matches = matches[test_mask]
        tr_idxa = idxa[train_mask]
        tr_idxb = idxb[train_mask]
        ts_idxa = idxa[test_mask]
        ts_idxb = idxb[test_mask]
        blended_score = np.zeros(ts_matches.shape[0], dtype=np.float64)

        for feature_index, feat in enumerate(projected_by_fold[fold_index]):
            train_pairs = _merge_pairs(feat[tr_idxa], feat[tr_idxb])
            test_pairs = _merge_pairs(feat[ts_idxa], feat[ts_idxb])
            clf = SVC(kernel="rbf", gamma="scale")
            clf.fit(train_pairs, tr_matches)
            current_score = clf.decision_function(test_pairs)
            blended_score += beta_by_fold[fold_index][feature_index] * current_score

        pred = (blended_score > 0).astype(np.int32)
        scores.append(float(accuracy_score(ts_matches, pred)))

    beta_means = np.mean(np.stack(beta_by_fold, axis=0), axis=0).tolist()
    n_components = int(projected_by_fold[0][0].shape[1])
    return KinVerResult(
        dataset=dataset,
        relation=relation,
        fold_scores=scores,
        beta_means=beta_means,
        n_components=n_components,
    )
