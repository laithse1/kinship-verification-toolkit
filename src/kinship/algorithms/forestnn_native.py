from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kinship.datasets.kinface import load_kinface_official_folds, load_kinface_pairs
from kinship.metrics import DEFAULT_FAR_TARGETS, compute_verification_metrics, select_threshold
from kinship.paths import kinver_workspace_root, workspace_root


VIEW_SPECS: tuple[dict[str, object], ...] = (
    {"name": "full", "mode": "crop", "box": (0.00, 0.00, 1.00, 1.00)},
    {"name": "left-eye", "mode": "crop", "box": (0.12, 0.16, 0.44, 0.42)},
    {"name": "right-eye", "mode": "crop", "box": (0.56, 0.16, 0.88, 0.42)},
    {"name": "nose", "mode": "crop", "box": (0.34, 0.28, 0.66, 0.62)},
    {"name": "mouth", "mode": "crop", "box": (0.24, 0.58, 0.76, 0.88)},
    {"name": "mask-left-eye", "mode": "mask", "box": (0.12, 0.16, 0.44, 0.42)},
    {"name": "mask-right-eye", "mode": "mask", "box": (0.56, 0.16, 0.88, 0.42)},
    {"name": "mask-nose", "mode": "mask", "box": (0.34, 0.28, 0.66, 0.62)},
    {"name": "mask-mouth", "mode": "mask", "box": (0.24, 0.58, 0.76, 0.88)},
)

GLOBAL_BACKBONES = {
    "none": (),
    "vggface-precomputed": ("vggFace",),
    "vggf-precomputed": ("vggF",),
    "vggface+vggf-precomputed": ("vggFace", "vggF"),
}
LOCAL_ENCODERS = {"shallow", "residual"}
ALIGNMENT_MODES = {"fixed", "adaptive"}
FUSION_MODELS = {"none", "logreg", "engineered-logreg"}
FUSION_TRAIN_SCOPES = {"val", "trainpool"}
MODEL_SELECTION_METRICS = {"balanced_accuracy", "roc_auc", "pr_auc", "eer"}


@dataclass(frozen=True)
class ForestNNFoldResult:
    fold: int
    threshold: float
    accuracy: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    eer: float
    tar_at_far: dict[str, float]
    train_loss: float
    val_loss: float


@dataclass
class ForestNNResult:
    dataset: str
    relation: str
    model_name: str
    protocol: str
    global_backbone: str
    fusion_model: str
    fusion_train_scope: str
    local_encoder: str
    alignment_mode: str
    view_names: list[str]
    image_size: int
    embedding_dim: int
    hidden_dim: int
    global_feature_dim: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    calibration_objective: str
    model_selection_metric: str
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


def _normalize_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    left = int(round(box[0] * width))
    top = int(round(box[1] * height))
    right = int(round(box[2] * width))
    bottom = int(round(box[3] * height))
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return left, top, min(width, right), min(height, bottom)


def _estimate_face_frame(arr: np.ndarray) -> tuple[int, int, int, int]:
    gy, gx = np.gradient(arr.astype(np.float32))
    energy = np.abs(gx) + np.abs(gy)
    energy += 0.20 * (1.0 - arr)
    energy_sum = float(np.sum(energy))
    if energy_sum <= 1e-8:
        return 0, 0, arr.shape[1], arr.shape[0]

    xs = np.arange(arr.shape[1], dtype=np.float32)
    ys = np.arange(arr.shape[0], dtype=np.float32)
    x_weights = energy.sum(axis=0)
    y_weights = energy.sum(axis=1)
    cx = float(np.sum(xs * x_weights) / np.sum(x_weights))
    cy = float(np.sum(ys * y_weights) / np.sum(y_weights))
    sx = float(np.sqrt(np.sum(((xs - cx) ** 2) * x_weights) / np.sum(x_weights)))
    sy = float(np.sqrt(np.sum(((ys - cy) ** 2) * y_weights) / np.sum(y_weights)))
    half_w = int(np.clip(2.8 * sx, arr.shape[1] * 0.22, arr.shape[1] * 0.48))
    half_h = int(np.clip(2.8 * sy, arr.shape[0] * 0.26, arr.shape[0] * 0.48))
    left = max(0, int(round(cx)) - half_w)
    right = min(arr.shape[1], int(round(cx)) + half_w)
    top = max(0, int(round(cy)) - half_h)
    bottom = min(arr.shape[0], int(round(cy)) + half_h)
    if right - left < arr.shape[1] * 0.25 or bottom - top < arr.shape[0] * 0.25:
        return 0, 0, arr.shape[1], arr.shape[0]
    return left, top, right, bottom


def _project_box_to_frame(
    box: tuple[float, float, float, float],
    frame: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = frame
    width = max(1, right - left)
    height = max(1, bottom - top)
    x1 = left + int(round(box[0] * width))
    y1 = top + int(round(box[1] * height))
    x2 = left + int(round(box[2] * width))
    y2 = top + int(round(box[3] * height))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def _load_face_views(path: Path, image_size: int, alignment_mode: str) -> np.ndarray:
    image = Image.open(path).convert("L").resize((128, 128))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    frame = (0, 0, arr.shape[1], arr.shape[0])
    if alignment_mode == "adaptive":
        frame = _estimate_face_frame(arr)
    views: list[np.ndarray] = []
    for spec in VIEW_SPECS:
        if alignment_mode == "adaptive":
            left, top, right, bottom = _project_box_to_frame(spec["box"], frame)
        else:
            left, top, right, bottom = _normalize_box(spec["box"], arr.shape[1], arr.shape[0])
        if spec["mode"] == "crop":
            patch = arr[top:bottom, left:right]
        else:
            patch = arr.copy()
            fill_value = float(np.mean(patch))
            patch[top:bottom, left:right] = fill_value
        patch_image = Image.fromarray(np.clip(patch * 255.0, 0.0, 255.0).astype(np.uint8))
        patch_resized = patch_image.resize((image_size, image_size))
        patch_array = np.asarray(patch_resized, dtype=np.float32) / 255.0
        patch_array = (patch_array - 0.5) / 0.5
        views.append(patch_array[None, :, :])
    return np.stack(views, axis=0)


def _preload_views(paths: Iterable[Path], image_size: int, alignment_mode: str) -> dict[Path, np.ndarray]:
    return {
        path: _load_face_views(path, image_size=image_size, alignment_mode=alignment_mode)
        for path in dict.fromkeys(paths)
    }


def _kinver_dataset_dir(dataset: str) -> Path:
    root = kinver_workspace_root()
    direct = root / f"data-{dataset}"
    if direct.exists():
        return direct
    return root / "data" / f"data-{dataset}"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_global_pair_embeddings(
    records: list,
    relation: str,
    dataset: str,
    backbone: str,
    selected_indices: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    prefixes = GLOBAL_BACKBONES[backbone]
    if not prefixes:
        return None, None, None, 0

    parent_parts: list[np.ndarray] = []
    child_parts: list[np.ndarray] = []
    cosine_parts: list[np.ndarray] = []
    for prefix in prefixes:
        mat = sio.loadmat(_kinver_dataset_dir(dataset) / f"{prefix}_{relation}.mat")
        ux = _normalize_rows(np.asarray(mat["ux"], dtype=np.float32))
        idxa = np.asarray(mat["idxa"]).ravel().astype(np.int32) - 1
        idxb = np.asarray(mat["idxb"]).ravel().astype(np.int32) - 1
        if selected_indices is not None:
            idxa = idxa[selected_indices]
            idxb = idxb[selected_indices]
        if len(idxa) != len(records):
            raise ValueError(
                f"Precomputed embedding size mismatch for {dataset} {relation} {prefix}: "
                f"{len(idxa)} vs {len(records)}"
            )
        current_parent = ux[idxa]
        current_child = ux[idxb]
        parent_parts.append(current_parent)
        child_parts.append(current_child)
        cosine_parts.append(np.sum(current_parent * current_child, axis=1, keepdims=True))

    parent = np.concatenate(parent_parts, axis=1)
    child = np.concatenate(child_parts, axis=1)
    cosine_scores = np.concatenate(cosine_parts, axis=1)
    return parent, child, cosine_scores, int(parent.shape[1])


class _PairTensorDataset(Dataset):
    def __init__(
        self,
        parent_views: np.ndarray,
        child_views: np.ndarray,
        parent_global: np.ndarray | None,
        child_global: np.ndarray | None,
        labels: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.parent_views = torch.from_numpy(parent_views[indices]).float()
        self.child_views = torch.from_numpy(child_views[indices]).float()
        self.parent_global = (
            torch.from_numpy(parent_global[indices]).float() if parent_global is not None else None
        )
        self.child_global = (
            torch.from_numpy(child_global[indices]).float() if child_global is not None else None
        )
        self.labels = torch.from_numpy(labels[indices]).float()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        parent_global = self.parent_global[index] if self.parent_global is not None else None
        child_global = self.child_global[index] if self.child_global is not None else None
        return self.parent_views[index], self.child_views[index], parent_global, child_global, self.labels[index]


class _PatchEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class _ResidualPatchEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            _ResidualBlock(24),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            _ResidualBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            _ResidualBlock(96),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.projection(x)


class _ForestNNModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        global_feature_dim: int,
        local_encoder: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.global_feature_dim = int(global_feature_dim)
        if local_encoder == "residual":
            self.encoder = _ResidualPatchEncoder(embedding_dim=embedding_dim)
        else:
            self.encoder = _PatchEncoder(embedding_dim=embedding_dim)
        self.view_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.global_projection = (
            nn.Sequential(
                nn.Linear(self.global_feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.10),
            )
            if self.global_feature_dim > 0
            else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 5 + embedding_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        parent_views: torch.Tensor,
        child_views: torch.Tensor,
        parent_global_features: torch.Tensor | None = None,
        child_global_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, view_count = parent_views.shape[:2]
        parent_flat = parent_views.reshape(batch_size * view_count, *parent_views.shape[2:])
        child_flat = child_views.reshape(batch_size * view_count, *child_views.shape[2:])
        parent_embed = self.encoder(parent_flat).reshape(batch_size, view_count, -1)
        child_embed = self.encoder(child_flat).reshape(batch_size, view_count, -1)
        delta = parent_embed - child_embed
        pair_views = torch.cat(
            [
                torch.abs(delta),
                delta.square(),
                parent_embed * child_embed,
                parent_embed + child_embed,
            ],
            dim=-1,
        )
        view_hidden = self.view_mlp(pair_views)
        attention_logits = self.attention(view_hidden).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(view_hidden * attention.unsqueeze(-1), dim=1)
        mean_hidden = view_hidden.mean(dim=1)
        max_hidden = view_hidden.max(dim=1).values
        parent_global = parent_embed.mean(dim=1)
        child_global = child_embed.mean(dim=1)
        global_pair = torch.cat(
            [torch.abs(parent_global - child_global), parent_global * child_global],
            dim=1,
        )
        if self.global_projection is not None and parent_global_features is not None and child_global_features is not None:
            projected_parent = self.global_projection(parent_global_features)
            projected_child = self.global_projection(child_global_features)
            projected_pair = torch.cat(
                [torch.abs(projected_parent - projected_child), projected_parent * projected_child],
                dim=1,
            )
        else:
            projected_pair = torch.zeros(
                (batch_size, self.hidden_dim * 2),
                dtype=max_hidden.dtype,
                device=max_hidden.device,
            )
        logits = self.classifier(
            torch.cat([pooled, mean_hidden, max_hidden, global_pair, projected_pair], dim=1)
        ).squeeze(1)
        return {
            "logits": logits,
            "parent_global": parent_global,
            "child_global": child_global,
            "attention": attention,
        }


def _split_train_val_indices(labels: np.ndarray, rng: np.random.Generator, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for value in (0, 1):
        class_indices = np.flatnonzero(labels == value)
        if len(class_indices) == 0:
            continue
        shuffled = rng.permutation(class_indices)
        if len(shuffled) == 1:
            train_indices.extend(shuffled.tolist())
            continue
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        n_val = min(n_val, len(shuffled) - 1)
        val_indices.extend(shuffled[:n_val].tolist())
        train_indices.extend(shuffled[n_val:].tolist())
    if not train_indices and val_indices:
        train_indices.append(val_indices.pop())
    if not val_indices and train_indices:
        val_indices.append(train_indices.pop())
    return np.asarray(train_indices, dtype=np.int32), np.asarray(val_indices, dtype=np.int32)


def _select_limited_indices(
    folds: np.ndarray,
    limit_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_folds = sorted(np.unique(folds).tolist())
    per_fold = max(1, limit_pairs // max(1, len(unique_folds)))
    selected: list[int] = []
    remaining: list[int] = []
    for fold in unique_folds:
        fold_indices = rng.permutation(np.flatnonzero(folds == fold))
        take = min(len(fold_indices), per_fold)
        selected.extend(fold_indices[:take].tolist())
        remaining.extend(fold_indices[take:].tolist())
    if len(selected) < limit_pairs and remaining:
        extra = rng.permutation(np.asarray(remaining, dtype=np.int32))
        selected.extend(extra[: limit_pairs - len(selected)].tolist())
    return np.asarray(sorted(selected[:limit_pairs]), dtype=np.int32)


def _compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    auxiliary_weight: float,
    margin: float,
) -> torch.Tensor:
    logits = outputs["logits"]
    labels = labels.float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    parent_global = outputs["parent_global"]
    child_global = outputs["child_global"]
    cosine = nn.functional.cosine_similarity(parent_global, child_global, dim=1)

    positive_mask = labels > 0.5
    negative_mask = ~positive_mask

    positive_loss = torch.zeros((), device=logits.device)
    negative_loss = torch.zeros((), device=logits.device)
    if torch.any(positive_mask):
        positive_loss = (1.0 - cosine[positive_mask]).mean()
    if torch.any(negative_mask):
        negative_loss = torch.relu(cosine[negative_mask] - margin).mean()
    return bce + auxiliary_weight * (positive_loss + negative_loss)


def _predict_scores(
    model: _ForestNNModel,
    dataset: _PairTensorDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for parent_views, child_views, parent_global, child_global, batch_labels in dataloader:
            parent_views = parent_views.to(device)
            child_views = child_views.to(device)
            parent_global = parent_global.to(device) if parent_global is not None else None
            child_global = child_global.to(device) if child_global is not None else None
            batch_labels = batch_labels.to(device)
            outputs = model(parent_views, child_views, parent_global, child_global)
            loss = _compute_loss(outputs, batch_labels, auxiliary_weight=0.15, margin=0.35)
            losses.append(float(loss.item()))
            scores.append(torch.sigmoid(outputs["logits"]).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    return (
        np.concatenate(scores, axis=0),
        np.concatenate(labels, axis=0).astype(np.int32),
        float(np.mean(losses)) if losses else 0.0,
    )


def _fit_score_fusion(
    local_scores: np.ndarray,
    global_scores: np.ndarray | None,
    labels: np.ndarray,
    fusion_model: str,
) -> tuple[object | None, np.ndarray]:
    local_scores = np.asarray(local_scores, dtype=np.float64)
    if local_scores.ndim == 1:
        local_scores = local_scores.reshape(-1, 1)
    if global_scores is None or fusion_model == "none":
        return None, local_scores[:, 0]

    global_scores = np.asarray(global_scores, dtype=np.float64)
    if global_scores.ndim == 1:
        global_scores = global_scores.reshape(-1, 1)
    features = _build_fusion_features(local_scores, global_scores, fusion_model=fusion_model)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return None, local_scores[:, 0]
    if fusion_model not in {"logreg", "engineered-logreg"}:
        raise ValueError(f"Unsupported fusion_model '{fusion_model}'")

    model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=0)
    model.fit(features, labels)
    fused_scores = model.predict_proba(features)[:, 1]
    return model, fused_scores


def _apply_score_fusion(
    local_scores: np.ndarray,
    global_scores: np.ndarray | None,
    model: object | None,
    fusion_model: str,
) -> np.ndarray:
    local_scores = np.asarray(local_scores, dtype=np.float64)
    if local_scores.ndim == 1:
        local_scores = local_scores.reshape(-1, 1)
    if model is None or global_scores is None:
        return local_scores[:, 0]
    features = _build_fusion_features(
        local_scores,
        np.asarray(global_scores, dtype=np.float64),
        fusion_model=fusion_model,
    )
    return model.predict_proba(features)[:, 1]


def _build_fusion_features(
    local_scores: np.ndarray,
    global_scores: np.ndarray,
    fusion_model: str,
) -> np.ndarray:
    if local_scores.ndim == 1:
        local_scores = local_scores.reshape(-1, 1)
    if global_scores.ndim == 1:
        global_scores = global_scores.reshape(-1, 1)
    base = np.concatenate([local_scores, global_scores], axis=1)
    if fusion_model == "logreg":
        return base
    if fusion_model != "engineered-logreg":
        raise ValueError(f"Unsupported fusion_model '{fusion_model}'")

    local_mean = np.mean(local_scores, axis=1, keepdims=True)
    local_max = np.max(local_scores, axis=1, keepdims=True)
    local_min = np.min(local_scores, axis=1, keepdims=True)
    local_std = np.std(local_scores, axis=1, keepdims=True)
    local_range = local_max - local_min
    global_mean = np.mean(global_scores, axis=1, keepdims=True)
    global_max = np.max(global_scores, axis=1, keepdims=True)
    global_min = np.min(global_scores, axis=1, keepdims=True)
    global_std = np.std(global_scores, axis=1, keepdims=True)
    local_minus_mean = local_scores - global_mean
    local_times_mean = local_scores * global_mean
    global_range = global_max - global_min
    local_sq = local_scores**2
    globals_sq = global_scores**2
    pairwise = (local_scores[:, :, None] * global_scores[:, None, :]).reshape(local_scores.shape[0], -1)
    return np.concatenate(
        [
            base,
            local_mean,
            local_max,
            local_min,
            local_std,
            local_range,
            global_mean,
            global_max,
            global_min,
            global_std,
            global_range,
            local_minus_mean,
            local_times_mean,
            local_sq,
            globals_sq,
            pairwise,
        ],
        axis=1,
    )


def _mean_tar_at_far(fold_metrics: list[dict]) -> dict[str, float]:
    if not fold_metrics:
        return {}
    keys = fold_metrics[0]["tar_at_far"].keys()
    return {
        key: float(np.mean([float(metric["tar_at_far"][key]) for metric in fold_metrics]))
        for key in keys
    }


def _metric_selection_score(metrics: dict[str, float], metric: str) -> float:
    metric = metric.lower()
    if metric == "eer":
        return -float(metrics["eer"])
    if metric not in MODEL_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported model_selection_metric '{metric}'. Available: {', '.join(sorted(MODEL_SELECTION_METRICS))}"
        )
    return float(metrics[metric])


def _ensure_torch_cache_dir() -> None:
    cache_dir = workspace_root() / "outputs" / "torch-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("TMPDIR", str(cache_dir))
    os.environ.setdefault("TEMP", str(cache_dir))
    os.environ.setdefault("TMP", str(cache_dir))


def run_forestnn(
    relation: str,
    dataset: str = "KinFaceW-II",
    global_backbone: str = "vggface-precomputed",
    fusion_model: str = "engineered-logreg",
    fusion_train_scope: str = "trainpool",
    local_encoder: str = "residual",
    alignment_mode: str = "adaptive",
    image_size: int = 64,
    embedding_dim: int = 64,
    hidden_dim: int = 96,
    batch_size: int = 16,
    num_epochs: int = 6,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    calibration_objective: str = "balanced_accuracy",
    model_selection_metric: str = "roc_auc",
    far_targets: Iterable[float] = DEFAULT_FAR_TARGETS,
    random_state: int = 42,
    limit_pairs: int | None = None,
) -> ForestNNResult:
    torch.set_num_threads(1)
    _ensure_torch_cache_dir()
    if global_backbone not in GLOBAL_BACKBONES:
        raise ValueError(
            f"Unsupported global_backbone '{global_backbone}'. Available: {', '.join(GLOBAL_BACKBONES)}"
        )
    if fusion_model not in FUSION_MODELS:
        raise ValueError(f"fusion_model must be one of: {', '.join(sorted(FUSION_MODELS))}")
    if fusion_train_scope not in FUSION_TRAIN_SCOPES:
        raise ValueError(f"fusion_train_scope must be one of: {', '.join(sorted(FUSION_TRAIN_SCOPES))}")
    if local_encoder not in LOCAL_ENCODERS:
        raise ValueError(f"Unsupported local_encoder '{local_encoder}'. Available: {', '.join(sorted(LOCAL_ENCODERS))}")
    if alignment_mode not in ALIGNMENT_MODES:
        raise ValueError(
            f"Unsupported alignment_mode '{alignment_mode}'. Available: {', '.join(sorted(ALIGNMENT_MODES))}"
        )
    if model_selection_metric not in MODEL_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported model_selection_metric '{model_selection_metric}'. Available: {', '.join(sorted(MODEL_SELECTION_METRICS))}"
        )

    records = load_kinface_pairs(relation=relation, dataset=dataset)
    folds = np.asarray(load_kinface_official_folds(relation=relation, dataset=dataset), dtype=np.int32)
    selected_indices: np.ndarray | None = None
    if limit_pairs is not None:
        limit_pairs = max(10, int(limit_pairs))
        subset_rng = np.random.default_rng(random_state)
        selected_indices = _select_limited_indices(folds=folds, limit_pairs=limit_pairs, rng=subset_rng)
        records = [records[index] for index in selected_indices.tolist()]
        folds = folds[selected_indices]

    labels = np.asarray([record.label for record in records], dtype=np.int32)
    all_paths = [record.parent_path for record in records] + [record.child_path for record in records]
    cached_views = _preload_views(all_paths, image_size=image_size, alignment_mode=alignment_mode)
    parent_views = np.stack([cached_views[record.parent_path] for record in records], axis=0)
    child_views = np.stack([cached_views[record.child_path] for record in records], axis=0)
    parent_global, child_global, global_scores, global_feature_dim = _load_global_pair_embeddings(
        records=records,
        relation=relation,
        dataset=dataset,
        backbone=global_backbone,
        selected_indices=selected_indices,
    )

    fold_metrics: list[dict] = []
    unique_folds = sorted(np.unique(folds).tolist())
    device = torch.device("cpu")

    for fold in unique_folds:
        fold_rng = np.random.default_rng(random_state + int(fold))
        train_pool = np.flatnonzero(folds != fold)
        test_indices = np.flatnonzero(folds == fold)
        train_indices, val_indices = _split_train_val_indices(labels[train_pool], fold_rng, val_fraction)
        train_indices = train_pool[train_indices]
        val_indices = train_pool[val_indices]

        train_dataset = _PairTensorDataset(
            parent_views, child_views, parent_global, child_global, labels, train_indices
        )
        val_dataset = _PairTensorDataset(
            parent_views, child_views, parent_global, child_global, labels, val_indices
        )
        test_dataset = _PairTensorDataset(
            parent_views, child_views, parent_global, child_global, labels, test_indices
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = _ForestNNModel(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            global_feature_dim=global_feature_dim,
            local_encoder=local_encoder,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_state: dict[str, torch.Tensor] | None = None
        best_score = -np.inf
        best_threshold = 0.5
        best_train_loss = 0.0
        best_val_loss = 0.0

        for _epoch in range(num_epochs):
            model.train()
            epoch_losses: list[float] = []
            for parent_batch, child_batch, parent_global_batch, child_global_batch, label_batch in train_loader:
                parent_batch = parent_batch.to(device)
                child_batch = child_batch.to(device)
                parent_global_batch = parent_global_batch.to(device) if parent_global_batch is not None else None
                child_global_batch = child_global_batch.to(device) if child_global_batch is not None else None
                label_batch = label_batch.to(device)
                outputs = model(parent_batch, child_batch, parent_global_batch, child_global_batch)
                loss = _compute_loss(outputs, label_batch, auxiliary_weight=0.15, margin=0.35)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))

            val_local_scores, val_labels, val_loss = _predict_scores(
                model, val_dataset, batch_size=batch_size, device=device
            )
            val_global_scores = global_scores[val_indices] if global_scores is not None else None
            fusion_head, val_scores = _fit_score_fusion(
                local_scores=val_local_scores,
                global_scores=val_global_scores,
                labels=val_labels,
                fusion_model=fusion_model,
            )
            selection = select_threshold(val_scores, val_labels, objective=calibration_objective)
            val_metrics = compute_verification_metrics(
                val_scores,
                val_labels,
                threshold=selection.threshold,
                far_targets=far_targets,
            )
            score = _metric_selection_score(val_metrics, model_selection_metric)
            if score >= best_score:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_threshold = float(selection.threshold)
                best_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                best_val_loss = float(val_loss)

        if best_state is None:
            raise RuntimeError("ForestNN training failed to produce a valid model state")

        model.load_state_dict(best_state)
        test_local_scores, test_labels, _ = _predict_scores(
            model, test_dataset, batch_size=batch_size, device=device
        )
        test_global_scores = global_scores[test_indices] if global_scores is not None else None
        if global_scores is not None:
            if fusion_train_scope == "trainpool":
                refit_dataset = _PairTensorDataset(
                    parent_views, child_views, parent_global, child_global, labels, train_pool
                )
                refit_labels = labels[train_pool]
                refit_global_scores = global_scores[train_pool]
            else:
                refit_dataset = val_dataset
                refit_labels = labels[val_indices]
                refit_global_scores = global_scores[val_indices]
            refit_val_local_scores, refit_val_labels, _ = _predict_scores(
                model, refit_dataset, batch_size=batch_size, device=device
            )
            best_fusion_head, refit_val_scores = _fit_score_fusion(
                local_scores=refit_val_local_scores,
                global_scores=refit_global_scores,
                labels=refit_labels if fusion_train_scope == "trainpool" else refit_val_labels,
                fusion_model=fusion_model,
            )
            best_threshold = float(
                select_threshold(
                    refit_val_scores,
                    refit_labels if fusion_train_scope == "trainpool" else refit_val_labels,
                    objective=calibration_objective,
                ).threshold
            )
        else:
            best_fusion_head = None
        test_scores = _apply_score_fusion(
            local_scores=test_local_scores,
            global_scores=test_global_scores,
            model=best_fusion_head,
            fusion_model=fusion_model,
        )
        metrics = compute_verification_metrics(
            test_scores,
            test_labels,
            threshold=best_threshold,
            far_targets=far_targets,
        )
        fold_metrics.append(
            {
                "fold": int(fold),
                "threshold": float(best_threshold),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "eer": float(metrics["eer"]),
                "tar_at_far": {key: float(value) for key, value in metrics["tar_at_far"].items()},
                "train_loss": float(best_train_loss),
                "val_loss": float(best_val_loss),
            }
        )

    return ForestNNResult(
        dataset=dataset,
        relation=relation,
        model_name="forestnn-native-hybrid" if global_feature_dim > 0 else "forestnn-native",
        protocol="official-5-fold",
        global_backbone=global_backbone,
        fusion_model=fusion_model,
        fusion_train_scope=fusion_train_scope,
        local_encoder=local_encoder,
        alignment_mode=alignment_mode,
        view_names=[str(spec["name"]) for spec in VIEW_SPECS],
        image_size=int(image_size),
        embedding_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
        global_feature_dim=int(global_feature_dim),
        num_epochs=int(num_epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        calibration_objective=calibration_objective,
        model_selection_metric=model_selection_metric,
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
