from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import tqdm
from PIL import Image
from torchvision import transforms

from kinship.algorithms._family_deep_data import FIWDataset, KinFaceWLoaderGenerator
from kinship.algorithms._family_deep_eval import KinshipEvaluator
from kinship.algorithms._family_deep_models import (
    KinFaceNet,
    SmallFaceModel,
    SmallSiameseFaceModel,
    VGGFaceMutiChannel,
    VGGFaceSiamese,
)


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("acc", "recall", "precision", "f1-score", "auc", "best_threshold"):
        if key in metrics and isinstance(metrics[key], (int, float, np.floating)):
            result[key] = float(metrics[key])
    return result


@dataclass
class PairEvaluation:
    pair_type: str
    metrics: dict[str, float]
    checkpoint_paths: list[str]


class FamilyDeepTrainer:
    def __init__(
        self,
        model_name: str,
        optimizer_name: str,
        lr: float,
        momentum: float,
        weight_decay: float,
        n_epochs: int,
        dataset: str,
        dataset_path: Path,
        metadata_dir: Path,
        kin_pairs: list[str],
        batch_size: int,
        gpu_id: int,
        logs_dir: Path,
        checkpoints_dir: Path,
        kinfacew_set_name: str = "KinFaceW-II",
        kinfacew_n_folds: int = 5,
        target_metric: str = "acc",
        vgg_weights: str | None = None,
        seed: int = 990411,
    ) -> None:
        self._set_random_seed(seed)
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.dataset = dataset.lower()
        self.dataset_path = Path(dataset_path)
        self.metadata_dir = Path(metadata_dir)
        self.kin_pairs = kin_pairs
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.logs_dir = Path(logs_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.kinfacew_set_name = kinfacew_set_name
        self.kinfacew_n_folds = kinfacew_n_folds
        self.target_metric = target_metric
        self.vgg_weights = vgg_weights
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        if self.dataset not in {"kinfacew", "fiw"}:
            raise ValueError("dataset must be 'kinfacew' or 'fiw'")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.transformer_train, self.transformer_test = self.get_transformers()

    @staticmethod
    def _set_random_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def get_transformers(self):
        if "vgg" in self.model_name:
            train = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.RandomGrayscale(0.3),
                    transforms.RandomRotation([-8, +8]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [129.1863 / 255, 104.7624 / 255, 93.5940 / 255],
                        [1 / 255, 1 / 255, 1 / 255],
                    ),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            test = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [129.1863 / 255, 104.7624 / 255, 93.5940 / 255],
                        [1 / 255, 1 / 255, 1 / 255],
                    ),
                ]
            )
            return train, test

        if "facenet" in self.model_name:
            train = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((160, 160)),
                    transforms.RandomGrayscale(0.3),
                    transforms.RandomRotation([-8, +8]),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            test = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                ]
            )
            return train, test

        train = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.RandomGrayscale(0.3),
                transforms.RandomRotation([-8, +8]),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
            ]
        )
        test = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        return train, test

    def custom_loss(self, emb_a, emb_b, y):
        emb_a = f.normalize(emb_a, dim=1)
        emb_b = f.normalize(emb_b, dim=1)
        hard_negative_indices = []
        for anchor_index, anchor in enumerate(emb_a):
            distances = f.pairwise_distance(anchor.unsqueeze(0).repeat(emb_b.size(0), 1), emb_b)
            distances[anchor_index] = float("inf")
            hard_negative_indices.append(int(torch.argmin(distances).item()))
        emb_n = emb_b[hard_negative_indices]
        dist = f.pairwise_distance(emb_a, emb_b) - f.pairwise_distance(emb_a, emb_n) + 1.0
        dist = f.relu(y * dist)
        return torch.mean(dist)

    def fiw_triplet_loss(
        self,
        parents_embeddings,
        children_embeddings,
        y,
        parents_families,
        children_families,
    ):
        positive_mask = y.detach().cpu().numpy() == 1
        if not np.any(positive_mask):
            return torch.tensor(0.0, device=self.device)
        hard_negative_index = []
        for parent_index, parent_emb in enumerate(parents_embeddings):
            parent_family = parents_families[parent_index]
            family_mask = np.array(
                [1 if parent_family == child_family else 0 for child_family in children_families]
            )
            anchor = parent_emb.unsqueeze(0).repeat(children_embeddings.size(0), 1)
            distances = f.pairwise_distance(anchor, children_embeddings).detach().cpu().numpy()
            distances[family_mask == 1] = float("inf")
            hard_negative_index.append(int(np.argmin(distances)))
        hard_negative_index = np.asarray(hard_negative_index)
        anchors = parents_embeddings[positive_mask]
        positive = children_embeddings[positive_mask]
        negative = children_embeddings[hard_negative_index][positive_mask]
        triplet_loss = f.relu(
            f.pairwise_distance(anchors, positive)
            - f.pairwise_distance(anchors, negative)
            + 2.0
        )
        return torch.mean(triplet_loss)

    def _forward_model(self, model, parent_image, children_image):
        raw = model(parent_image.float(), children_image.float())
        if isinstance(raw, tuple):
            output, parent_features, child_features = raw
        else:
            output = raw
            fallback = torch.flatten(torch.cat((parent_image, children_image), dim=1), 1)
            parent_features = fallback
            child_features = fallback
        return output.squeeze(1), parent_features.squeeze(1), child_features.squeeze(1)

    def train_epoch(self, model, optimizer, criterion, epoch, train_loader, evaluator):
        model.train()
        evaluator.reset()
        for sample in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}"):
            parent_image = sample["parent_image"].to(self.device)
            children_image = sample["children_image"].to(self.device)
            labels = sample["kin"].to(self.device).float()
            optimizer.zero_grad()
            output, p_f, c_f = self._forward_model(model, parent_image, children_image)
            loss = criterion(output, labels) + self.custom_loss(p_f, c_f, labels)
            loss.backward()
            optimizer.step()
            evaluator.add_batch(
                list(torch.sigmoid(output).detach().cpu().numpy()),
                list(labels.detach().cpu().numpy()),
            )
        return evaluator.get_metrics(self.target_metric)

    def val_epoch(self, model, epoch, val_loader, evaluator):
        evaluator.reset()
        model.eval()
        with torch.no_grad():
            for sample in tqdm.tqdm(val_loader, total=len(val_loader), desc=f"Val epoch {epoch}"):
                parent_image = sample["parent_image"].to(self.device)
                children_image = sample["children_image"].to(self.device)
                labels = sample["kin"].to(self.device).float()
                output, _, _ = self._forward_model(model, parent_image, children_image)
                evaluator.add_batch(
                    list(torch.sigmoid(output).detach().cpu().numpy()),
                    list(labels.detach().cpu().numpy()),
                )
        metrics = evaluator.get_metrics(self.target_metric)
        return float(metrics[self.target_metric]), metrics

    def train_epoch_fiw(self, model, optimizer, criterion, epoch, train_loader, evaluator):
        model.train()
        evaluator.reset()
        for sample in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}"):
            parent_image = sample["parent_image"].to(self.device)
            children_image = sample["children_image"].to(self.device)
            labels = sample["kin"].to(self.device).float()
            optimizer.zero_grad()
            output, p_f, c_f = self._forward_model(model, parent_image, children_image)
            triplet_loss = self.fiw_triplet_loss(
                parents_embeddings=p_f,
                children_embeddings=c_f,
                y=labels,
                parents_families=sample["parent_family_id"],
                children_families=sample["children_family_id"],
            )
            loss = criterion(output, labels) + triplet_loss
            loss.backward()
            optimizer.step()
            evaluator.add_batch(
                list(torch.sigmoid(output).detach().cpu().numpy()),
                list(labels.detach().cpu().numpy()),
            )
        return evaluator.get_metrics(self.target_metric)

    def val_epoch_fiw(self, model, epoch, val_loader, evaluator):
        return self.val_epoch(model, epoch, val_loader, evaluator)

    def _checkpoint_path(self, pair_type: str, fold: int | None = None) -> Path:
        suffix = f"_fold_{fold}" if fold is not None else ""
        return self.checkpoints_dir / f"{self.model_name}_{self.dataset}_{pair_type}{suffix}.pth"

    def save_model(self, model, pair_type: str, fold: int | None = None) -> str:
        path = self._checkpoint_path(pair_type, fold)
        torch.save(model.state_dict(), path)
        return str(path)

    def load_model(self):
        if self.model_name == "small_face_model":
            return SmallFaceModel()
        if self.model_name == "small_siamese_face_model":
            return SmallSiameseFaceModel()
        if self.model_name == "vgg_multichannel":
            return VGGFaceMutiChannel(self.vgg_weights)
        if self.model_name == "vgg_siamese":
            return VGGFaceSiamese(self.vgg_weights)
        if self.model_name == "kin_facenet":
            return KinFaceNet()
        raise ValueError(f"Unknown model '{self.model_name}'")

    def load_best_model(self, pair_type: str, fold: int | None = None):
        model = self.load_model()
        checkpoint_path = self._checkpoint_path(pair_type, fold)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return model.to(self.device)

    def load_optimizer(self, model):
        if self.optimizer_name != "SGD":
            raise ValueError(f"Unknown optimizer '{self.optimizer_name}'")
        return optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def load_criterion():
        return nn.BCEWithLogitsLoss()

    def get_color_space_name(self):
        return "bgr" if "vgg" in self.model_name else "rgb"

    def _write_summary(self, name: str, payload: dict) -> None:
        path = self.logs_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def train_kinfacew(self):
        loader_gen = KinFaceWLoaderGenerator(
            dataset_name=self.kinfacew_set_name,
            dataset_path=self.dataset_path,
            color_space_name=self.get_color_space_name(),
        )
        pair_results: dict[str, dict] = {}
        for pair_type in self.kin_pairs:
            fold_evaluators = []
            checkpoint_paths: list[str] = []
            for fold in range(1, self.kinfacew_n_folds + 1):
                test_loader, train_loader = loader_gen.get_data_loader(
                    fold=fold,
                    batch_size=self.batch_size,
                    pair_type=pair_type,
                    train_transformer=self.transformer_train,
                    test_transformer=self.transformer_test,
                    torch_module=torch,
                )
                model = self.load_model().to(self.device)
                optimizer = self.load_optimizer(model)
                criterion = self.load_criterion()
                train_evaluator = KinshipEvaluator("Training", pair_type, self.logs_dir, fold=fold)
                test_evaluator = KinshipEvaluator("Testing", pair_type, self.logs_dir, fold=fold)
                best_score = -1.0
                for epoch in range(1, self.n_epochs + 1):
                    self.train_epoch(model, optimizer, criterion, epoch, train_loader, train_evaluator)
                    score, _ = self.val_epoch(model, epoch, test_loader, test_evaluator)
                    if score > best_score:
                        best_score = score
                        test_evaluator.save_best_metrics()
                        checkpoint_path = self.save_model(model, pair_type, fold)
                    train_evaluator.save_hist()
                    test_evaluator.save_hist()
                fold_evaluators.append(test_evaluator)
                checkpoint_paths.append(checkpoint_path)
            pair_evaluator = KinshipEvaluator("Testing", pair_type, self.logs_dir)
            pair_metrics = pair_evaluator.get_kinface_pair_metrics(fold_evaluators, pair_type)
            pair_results[pair_type] = {
                "metrics": _scalar_metrics(pair_metrics),
                "checkpoint_paths": checkpoint_paths,
            }
        summary = {
            "algorithm": "family-deep",
            "mode": "train",
            "dataset": self.dataset,
            "dataset_variant": self.kinfacew_set_name,
            "model_name": self.model_name,
            "pair_metrics": pair_results,
            "mean_accuracy": float(np.mean([item["metrics"]["acc"] for item in pair_results.values()])),
        }
        self._write_summary("family_deep_train_summary", summary)
        return summary

    def _fiw_loader(self, set_name: str, pair_type: str, transformer):
        dataset = FIWDataset(
            dataset_root=self.dataset_path,
            metadata_dir=self.metadata_dir,
            pair_type=pair_type,
            set_name=set_name,
            transform=transformer,
            color_space=self.get_color_space_name(),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def train_fiw(self):
        pair_results: dict[str, dict] = {}
        for pair_type in self.kin_pairs:
            train_loader = self._fiw_loader("train", pair_type, self.transformer_train)
            test_loader = self._fiw_loader("val", pair_type, self.transformer_test)
            model = self.load_model().to(self.device)
            optimizer = self.load_optimizer(model)
            criterion = self.load_criterion()
            train_evaluator = KinshipEvaluator("Training", pair_type, self.logs_dir)
            test_evaluator = KinshipEvaluator("Testing", pair_type, self.logs_dir)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8)
            best_score = -1.0
            checkpoint_path = ""
            for epoch in range(1, self.n_epochs + 1):
                self.train_epoch_fiw(model, optimizer, criterion, epoch, train_loader, train_evaluator)
                score, _ = self.val_epoch_fiw(model, epoch, test_loader, test_evaluator)
                if score > best_score:
                    best_score = score
                    test_evaluator.save_best_metrics()
                    checkpoint_path = self.save_model(model, pair_type)
                scheduler.step()
                train_evaluator.save_hist()
                test_evaluator.save_hist()
            pair_results[pair_type] = {
                "metrics": _scalar_metrics(test_evaluator.best_metrics),
                "checkpoint_paths": [checkpoint_path],
            }
        summary = {
            "algorithm": "family-deep",
            "mode": "train",
            "dataset": self.dataset,
            "model_name": self.model_name,
            "pair_metrics": pair_results,
            "mean_accuracy": float(np.mean([item["metrics"]["acc"] for item in pair_results.values()])),
        }
        self._write_summary("family_deep_train_summary", summary)
        return summary

    def train(self):
        if self.dataset == "kinfacew":
            return self.train_kinfacew()
        return self.train_fiw()

    def test_kinfacew(self):
        loader_gen = KinFaceWLoaderGenerator(
            dataset_name=self.kinfacew_set_name,
            dataset_path=self.dataset_path,
            color_space_name=self.get_color_space_name(),
        )
        pair_results: dict[str, dict] = {}
        for pair_type in self.kin_pairs:
            fold_metrics = []
            fold_evaluators = []
            for fold in range(1, self.kinfacew_n_folds + 1):
                test_loader, _ = loader_gen.get_data_loader(
                    fold=fold,
                    batch_size=self.batch_size,
                    pair_type=pair_type,
                    train_transformer=self.transformer_train,
                    test_transformer=self.transformer_test,
                    torch_module=torch,
                )
                model = self.load_best_model(pair_type, fold)
                evaluator = KinshipEvaluator("TEST", pair_type, self.logs_dir, fold=fold)
                with torch.no_grad():
                    for sample in tqdm.tqdm(test_loader, total=len(test_loader), desc=f"Test {pair_type} fold {fold}"):
                        parent_image = sample["parent_image"].to(self.device)
                        children_image = sample["children_image"].to(self.device)
                        labels = sample["kin"].to(self.device).float()
                        output, _, _ = self._forward_model(model, parent_image, children_image)
                        evaluator.add_batch(
                            list(torch.sigmoid(output).detach().cpu().numpy()),
                            list(labels.detach().cpu().numpy()),
                        )
                metrics = evaluator.get_metrics(self.target_metric)
                evaluator.save_best_metrics()
                fold_metrics.append(_scalar_metrics(metrics))
                fold_evaluators.append(evaluator)
            pair_evaluator = KinshipEvaluator("TEST", pair_type, self.logs_dir)
            pair_metrics = pair_evaluator.get_kinface_pair_metrics(fold_evaluators, pair_type)
            pair_results[pair_type] = {"metrics": _scalar_metrics(pair_metrics), "fold_metrics": fold_metrics}
        summary = {
            "algorithm": "family-deep",
            "mode": "test",
            "dataset": self.dataset,
            "dataset_variant": self.kinfacew_set_name,
            "model_name": self.model_name,
            "pair_metrics": pair_results,
            "mean_accuracy": float(np.mean([item["metrics"]["acc"] for item in pair_results.values()])),
        }
        self._write_summary("family_deep_test_summary", summary)
        return summary

    def test_fiw(self):
        pair_results: dict[str, dict] = {}
        for pair_type in self.kin_pairs:
            test_loader = self._fiw_loader("val", pair_type, self.transformer_test)
            evaluator = KinshipEvaluator("TEST", pair_type, self.logs_dir)
            model = self.load_best_model(pair_type)
            with torch.no_grad():
                for sample in tqdm.tqdm(test_loader, total=len(test_loader), desc=f"Test {pair_type}"):
                    parent_image = sample["parent_image"].to(self.device)
                    children_image = sample["children_image"].to(self.device)
                    labels = sample["kin"].to(self.device).float()
                    output, _, _ = self._forward_model(model, parent_image, children_image)
                    evaluator.add_batch(
                        list(torch.sigmoid(output).detach().cpu().numpy()),
                        list(labels.detach().cpu().numpy()),
                    )
            metrics = evaluator.get_metrics(self.target_metric)
            evaluator.save_best_metrics()
            pair_results[pair_type] = {"metrics": _scalar_metrics(metrics)}
        summary = {
            "algorithm": "family-deep",
            "mode": "test",
            "dataset": self.dataset,
            "model_name": self.model_name,
            "pair_metrics": pair_results,
            "mean_accuracy": float(np.mean([item["metrics"]["acc"] for item in pair_results.values()])),
        }
        self._write_summary("family_deep_test_summary", summary)
        return summary

    def test(self):
        if self.dataset == "kinfacew":
            return self.test_kinfacew()
        return self.test_fiw()

    def demo(self, img1: str, img2: str, pair_type: str):
        pair_type = pair_type.lower()
        parent_path = Path(img1)
        child_path = Path(img2)
        if not parent_path.exists():
            parent_path = self.dataset_path / "test-faces" / img1
        if not child_path.exists():
            child_path = self.dataset_path / "test-faces" / img2
        parent_image = np.asarray(Image.open(parent_path))
        child_image = np.asarray(Image.open(child_path))
        parent_tensor = self.transformer_test(parent_image).unsqueeze(0).to(self.device)
        child_tensor = self.transformer_test(child_image).unsqueeze(0).to(self.device)
        model = self.load_best_model(pair_type)
        with torch.no_grad():
            output, _, _ = self._forward_model(model, parent_tensor, child_tensor)
            probability = float(torch.sigmoid(output)[0].item())
        result = {
            "algorithm": "family-deep",
            "mode": "demo",
            "dataset": self.dataset,
            "model_name": self.model_name,
            "pair_type": pair_type,
            "probability": probability,
            "is_kin": bool(probability > 0.5),
            "parent_image": str(parent_path),
            "child_image": str(child_path),
        }
        self._write_summary("family_deep_demo_summary", result)
        return result
