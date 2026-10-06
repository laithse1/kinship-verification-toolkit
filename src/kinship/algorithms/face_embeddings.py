from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import importlib

import numpy as np
import scipy.io as sio

from kinship.datasets.kinface import load_kinface_official_folds, load_kinface_pairs
from kinship.metrics import DEFAULT_FAR_TARGETS, compute_verification_metrics, select_threshold
from kinship.paths import kinver_workspace_root


PRECOMPUTED_PREFIXES = {
    "vggface-precomputed": "vggFace",
    "vggf-precomputed": "vggF",
}
IMAGE_BACKBONES = {"facenet-vggface2", "facenet-casia"}
ALL_BACKBONES = tuple(list(PRECOMPUTED_PREFIXES) + list(IMAGE_BACKBONES))


@dataclass(frozen=True)
class FaceEmbeddingFoldResult:
    fold: int
    threshold: float
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    eer: float
    tar_at_far: dict[str, float]


@dataclass
class FaceEmbeddingResult:
    dataset: str
    relation: str
    backbone: str
    protocol: str
    score_metric: str
    calibration_objective: str
    embedding_dim: int
    fold_metrics: list[dict]
    mean_accuracy: float
    mean_balanced_accuracy: float
    mean_roc_auc: float
    mean_pr_auc: float
    mean_eer: float
    mean_tar_at_far: dict[str, float]
    sample_count: int
    positive_count: int
    negative_count: int


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


def _load_precomputed_pair_data(dataset: str, relation: str, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    mat = sio.loadmat(_dataset_dir(dataset) / f"{prefix}_{relation}.mat")
    embeddings = _normalize_rows(np.asarray(mat["ux"], dtype=np.float64))
    idxa = np.asarray(mat["idxa"]).ravel().astype(np.int32) - 1
    idxb = np.asarray(mat["idxb"]).ravel().astype(np.int32) - 1
    folds = np.asarray(mat["fold"]).ravel().astype(np.int32)
    labels = np.asarray(mat["matches"]).ravel().astype(np.int32)
    scores = np.sum(embeddings[idxa] * embeddings[idxb], axis=1)
    return scores, labels, folds, embeddings, int(embeddings.shape[1])


class _FaceNetEmbedder:
    def __init__(self, pretrained: str, batch_size: int = 32) -> None:
        try:
            facenet = importlib.import_module("facenet_pytorch")
        except Exception as exc:  # pragma: no cover - dependency gate
            raise RuntimeError(
                "facenet-pytorch is required for image-based face embeddings. "
                "Install the optional deep dependencies first."
            ) from exc
        import torch
        from PIL import Image

        self._torch = torch
        self._Image = Image
        self._batch_size = batch_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = facenet.InceptionResnetV1(pretrained=pretrained).eval().to(self._device)

    def _load_tensor(self, path: Path):
        image = self._Image.open(path).convert("RGB").resize((160, 160))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = self._torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5

    def encode_paths(self, paths: Iterable[Path]) -> dict[Path, np.ndarray]:
        unique_paths = list(dict.fromkeys(Path(path) for path in paths))
        embeddings: dict[Path, np.ndarray] = {}
        with self._torch.no_grad():
            for start in range(0, len(unique_paths), self._batch_size):
                batch_paths = unique_paths[start : start + self._batch_size]
                batch_tensor = self._torch.stack([self._load_tensor(path) for path in batch_paths], dim=0).to(self._device)
                batch_embeddings = self._model(batch_tensor).detach().cpu().numpy()
                batch_embeddings = _normalize_rows(batch_embeddings)
                for path, embedding in zip(batch_paths, batch_embeddings, strict=True):
                    embeddings[path] = embedding.astype(np.float64, copy=False)
        return embeddings


def _load_image_pair_data(dataset: str, relation: str, backbone: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if backbone == "facenet-vggface2":
        embedder = _FaceNetEmbedder(pretrained="vggface2", batch_size=batch_size)
    elif backbone == "facenet-casia":
        embedder = _FaceNetEmbedder(pretrained="casia-webface", batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported image backbone '{backbone}'")

    records = load_kinface_pairs(relation=relation, dataset=dataset)
    folds = np.asarray(load_kinface_official_folds(relation=relation, dataset=dataset), dtype=np.int32)
    labels = np.asarray([record.label for record in records], dtype=np.int32)
    all_paths = [record.parent_path for record in records] + [record.child_path for record in records]
    embeddings = embedder.encode_paths(all_paths)
    parent = np.stack([embeddings[record.parent_path] for record in records], axis=0)
    child = np.stack([embeddings[record.child_path] for record in records], axis=0)
    scores = np.sum(parent * child, axis=1)
    return scores, labels, folds, int(parent.shape[1])


def _mean_tar_at_far(fold_metrics: list[dict]) -> dict[str, float]:
    if not fold_metrics:
        return {}
    keys = fold_metrics[0]["tar_at_far"].keys()
    return {
        key: float(np.mean([float(metric["tar_at_far"][key]) for metric in fold_metrics]))
        for key in keys
    }


def _evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    calibration_objective: str,
    far_targets: Iterable[float],
) -> list[dict]:
    results: list[dict] = []
    for fold in sorted(np.unique(folds).tolist()):
        train_mask = folds != fold
        test_mask = folds == fold
        selection = select_threshold(scores[train_mask], labels[train_mask], objective=calibration_objective)
        metrics = compute_verification_metrics(
            scores[test_mask],
            labels[test_mask],
            threshold=selection.threshold,
            far_targets=far_targets,
        )
        results.append(
            {
                "fold": int(fold),
                "threshold": float(selection.threshold),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "eer": float(metrics["eer"]),
                "tar_at_far": {key: float(value) for key, value in metrics["tar_at_far"].items()},
            }
        )
    return results


def run_face_embedding_verification(
    relation: str,
    dataset: str = "KinFaceW-II",
    backbone: str = "vggface-precomputed",
    calibration_objective: str = "balanced_accuracy",
    far_targets: Iterable[float] = DEFAULT_FAR_TARGETS,
    image_batch_size: int = 32,
) -> FaceEmbeddingResult:
    if backbone not in ALL_BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone}'. Available: {', '.join(ALL_BACKBONES)}")

    if backbone in PRECOMPUTED_PREFIXES:
        scores, labels, folds, embeddings, embedding_dim = _load_precomputed_pair_data(
            dataset=dataset,
            relation=relation,
            prefix=PRECOMPUTED_PREFIXES[backbone],
        )
    else:
        scores, labels, folds, embedding_dim = _load_image_pair_data(
            dataset=dataset,
            relation=relation,
            backbone=backbone,
            batch_size=image_batch_size,
        )
        embeddings = None

    fold_metrics = _evaluate_scores(
        scores=scores,
        labels=labels,
        folds=folds,
        calibration_objective=calibration_objective,
        far_targets=far_targets,
    )
    return FaceEmbeddingResult(
        dataset=dataset,
        relation=relation,
        backbone=backbone,
        protocol="official-5-fold",
        score_metric="cosine",
        calibration_objective=calibration_objective,
        embedding_dim=embedding_dim,
        fold_metrics=fold_metrics,
        mean_accuracy=float(np.mean([metric["accuracy"] for metric in fold_metrics])),
        mean_balanced_accuracy=float(np.mean([metric["balanced_accuracy"] for metric in fold_metrics])),
        mean_roc_auc=float(np.mean([metric["roc_auc"] for metric in fold_metrics])),
        mean_pr_auc=float(np.mean([metric["pr_auc"] for metric in fold_metrics])),
        mean_eer=float(np.mean([metric["eer"] for metric in fold_metrics])),
        mean_tar_at_far=_mean_tar_at_far(fold_metrics),
        sample_count=int(labels.shape[0]),
        positive_count=int(np.sum(labels == 1)),
        negative_count=int(np.sum(labels == 0)),
    )
