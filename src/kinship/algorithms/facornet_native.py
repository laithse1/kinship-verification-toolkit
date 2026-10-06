from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kinship.algorithms.forestnn_native import (
    FUSION_MODELS,
    FUSION_TRAIN_SCOPES,
    GLOBAL_BACKBONES,
    MODEL_SELECTION_METRICS,
    _apply_score_fusion,
    _ensure_torch_cache_dir,
    _fit_score_fusion,
    _load_global_pair_embeddings,
    _metric_selection_score,
    _preload_views,
    _select_limited_indices,
    _split_train_val_indices,
)
from kinship.datasets.kinface import load_kinface_official_folds, load_kinface_pairs
from kinship.metrics import DEFAULT_FAR_TARGETS, compute_verification_metrics, select_threshold


FACOR_COMPONENT_SPECS: tuple[dict[str, object], ...] = (
    {"name": "full", "mode": "crop", "box": (0.00, 0.00, 1.00, 1.00)},
    {"name": "brow-eye", "mode": "crop", "box": (0.08, 0.10, 0.92, 0.42)},
    {"name": "nose", "mode": "crop", "box": (0.30, 0.24, 0.70, 0.66)},
    {"name": "mouth", "mode": "crop", "box": (0.18, 0.54, 0.82, 0.90)},
)


@dataclass
class FaCoRNetResult:
    dataset: str
    relation: str
    model_name: str
    protocol: str
    global_backbone: str
    fusion_model: str
    fusion_train_scope: str
    alignment_mode: str
    image_size: int
    embedding_dim: int
    hidden_dim: int
    global_feature_dim: int
    num_heads: int
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


class _FaCoRDataset(Dataset):
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
        return (
            self.parent_views[index],
            self.child_views[index],
            self.parent_global[index] if self.parent_global is not None else None,
            self.child_global[index] if self.child_global is not None else None,
            self.labels[index],
        )


class _ResidualPatchEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(96, embedding_dim), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class _FaCoRNetModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, global_feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.encoder = _ResidualPatchEncoder(embedding_dim=embedding_dim)
        self.parent_to_child = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.10,
        )
        self.child_to_parent = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.10,
        )
        self.component_gate = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
        )
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
            nn.Linear(hidden_dim * 4, hidden_dim),
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

        parent_ctx, parent_attn = self.parent_to_child(parent_embed, child_embed, child_embed)
        child_ctx, child_attn = self.child_to_parent(child_embed, parent_embed, parent_embed)
        fused_components = torch.cat(
            [
                torch.abs(parent_ctx - child_ctx),
                parent_ctx * child_ctx,
                torch.abs(parent_embed - child_embed),
                parent_embed + child_embed,
            ],
            dim=-1,
        )
        gate_logits = self.component_gate(fused_components).squeeze(-1)
        gate_weights = torch.softmax(gate_logits, dim=1)
        gated_summary = torch.sum(fused_components * gate_weights.unsqueeze(-1), dim=1)
        pair_stats = self.pair_projection(gated_summary)
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
                dtype=pair_stats.dtype,
                device=pair_stats.device,
            )
        logits = self.classifier(torch.cat([pair_stats, projected_pair], dim=1)).squeeze(1)
        parent_repr = parent_ctx.mean(dim=1)
        child_repr = child_ctx.mean(dim=1)
        attention = 0.5 * (parent_attn.mean(dim=1) + child_attn.mean(dim=1))
        return {
            "logits": logits,
            "parent_global": parent_repr,
            "child_global": child_repr,
            "attention": attention,
        }


def _compute_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, auxiliary_weight: float, margin: float) -> torch.Tensor:
    logits = outputs["logits"]
    labels = labels.float()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    parent_repr = outputs["parent_global"]
    child_repr = outputs["child_global"]
    cosine = nn.functional.cosine_similarity(parent_repr, child_repr, dim=1)

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
    model: _FaCoRNetModel,
    dataset: _FaCoRDataset,
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


def _mean_tar_at_far(fold_metrics: list[dict]) -> dict[str, float]:
    if not fold_metrics:
        return {}
    keys = fold_metrics[0]["tar_at_far"].keys()
    return {
        key: float(np.mean([float(metric["tar_at_far"][key]) for metric in fold_metrics]))
        for key in keys
    }


def run_facornet(
    relation: str,
    dataset: str = "KinFaceW-II",
    global_backbone: str = "vggface+vggf-precomputed",
    fusion_model: str = "engineered-logreg",
    fusion_train_scope: str = "trainpool",
    alignment_mode: str = "adaptive",
    image_size: int = 64,
    embedding_dim: int = 64,
    hidden_dim: int = 96,
    num_heads: int = 4,
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
) -> FaCoRNetResult:
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
    parent_views = np.stack([cached_views[record.parent_path][: len(FACOR_COMPONENT_SPECS)] for record in records], axis=0)
    child_views = np.stack([cached_views[record.child_path][: len(FACOR_COMPONENT_SPECS)] for record in records], axis=0)
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

        train_dataset = _FaCoRDataset(parent_views, child_views, parent_global, child_global, labels, train_indices)
        val_dataset = _FaCoRDataset(parent_views, child_views, parent_global, child_global, labels, val_indices)
        test_dataset = _FaCoRDataset(parent_views, child_views, parent_global, child_global, labels, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = _FaCoRNetModel(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            global_feature_dim=global_feature_dim,
            num_heads=num_heads,
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

            val_local_scores, val_labels, val_loss = _predict_scores(model, val_dataset, batch_size=batch_size, device=device)
            val_global_scores = global_scores[val_indices] if global_scores is not None else None
            _, val_scores = _fit_score_fusion(
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
            raise RuntimeError("FaCoRNet training failed to produce a valid model state")

        model.load_state_dict(best_state)
        test_local_scores, test_labels, _ = _predict_scores(model, test_dataset, batch_size=batch_size, device=device)
        test_global_scores = global_scores[test_indices] if global_scores is not None else None
        if global_scores is not None:
            if fusion_train_scope == "trainpool":
                refit_dataset = _FaCoRDataset(parent_views, child_views, parent_global, child_global, labels, train_pool)
                refit_labels = labels[train_pool]
                refit_global_scores = global_scores[train_pool]
            else:
                refit_dataset = val_dataset
                refit_labels = labels[val_indices]
                refit_global_scores = global_scores[val_indices]
            refit_local_scores, _, _ = _predict_scores(model, refit_dataset, batch_size=batch_size, device=device)
            best_fusion_head, refit_scores = _fit_score_fusion(
                local_scores=refit_local_scores,
                global_scores=refit_global_scores,
                labels=refit_labels,
                fusion_model=fusion_model,
            )
            best_threshold = float(select_threshold(refit_scores, refit_labels, objective=calibration_objective).threshold)
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

    return FaCoRNetResult(
        dataset=dataset,
        relation=relation,
        model_name="facornet-native-hybrid" if global_feature_dim > 0 else "facornet-native",
        protocol="official-5-fold",
        global_backbone=global_backbone,
        fusion_model=fusion_model,
        fusion_train_scope=fusion_train_scope,
        alignment_mode=alignment_mode,
        image_size=int(image_size),
        embedding_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
        global_feature_dim=int(global_feature_dim),
        num_heads=int(num_heads),
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
