from __future__ import annotations

import importlib
from pathlib import Path

from kinship.paths import (
    family_project_root,
    kinface_workspace_root,
    resolve_user_path,
    workspace_root,
)


DEEP_DEPENDENCIES = {
    "base": ["torch", "torchvision", "tqdm", "PIL", "skimage"],
    "kin_facenet": ["facenet_pytorch"],
    "vgg_models": ["torchfile"],
}


def _missing_modules(names: list[str]) -> list[str]:
    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def _required_modules(model_name: str) -> list[str]:
    required = list(DEEP_DEPENDENCIES["base"])
    if model_name == "kin_facenet":
        required += DEEP_DEPENDENCIES["kin_facenet"]
    if model_name in {"vgg_multichannel", "vgg_siamese"}:
        required += DEEP_DEPENDENCIES["vgg_models"]
    return required


def run_family_deep(
    mode: str = "test",
    dataset_name: str = "fiw",
    data_path: str | None = None,
    model_name: str = "kin_facenet",
    gpu: int = 0,
    lr: float = 1e-3,
    bs: int = 40,
    num_epochs: int = 16,
    img1: str | None = None,
    img2: str | None = None,
    pair_type: str = "ms",
    output_dir: str | None = None,
    checkpoints_dir: str | None = None,
    vgg_weights: str | None = None,
) -> dict:
    missing = _missing_modules(_required_modules(model_name))
    if missing:
        raise RuntimeError(
            "family-deep native path is available but missing optional dependencies: "
            + ", ".join(missing)
        )

    dataset_name = dataset_name.lower()
    if dataset_name not in {"fiw", "kinfacew"}:
        raise ValueError("dataset_name must be 'fiw' or 'kinfacew'")
    if mode not in {"train", "test", "demo"}:
        raise ValueError("mode must be one of: train, test, demo")

    from kinship.algorithms._family_deep_runtime import FamilyDeepTrainer

    if data_path is None:
        data_root = kinface_workspace_root() if dataset_name == "kinfacew" else family_project_root()
    else:
        data_root = resolve_user_path(data_path)

    metadata_dir = family_project_root() / "data"
    logs_dir = resolve_user_path(output_dir) if output_dir else workspace_root() / "outputs" / "family-deep-native" / "logs"
    ckpt_dir = resolve_user_path(checkpoints_dir) if checkpoints_dir else workspace_root() / "outputs" / "family-deep-native" / "checkpoints"
    trainer = FamilyDeepTrainer(
        model_name=model_name,
        optimizer_name="SGD",
        lr=lr,
        momentum=0.9,
        weight_decay=0.005,
        n_epochs=num_epochs,
        dataset=dataset_name,
        dataset_path=data_root,
        metadata_dir=metadata_dir,
        kin_pairs=["fd", "ms", "md", "fs"],
        batch_size=bs,
        gpu_id=gpu,
        logs_dir=logs_dir,
        checkpoints_dir=ckpt_dir,
        kinfacew_set_name="KinFaceW-II",
        kinfacew_n_folds=5,
        target_metric="acc",
        vgg_weights=vgg_weights,
    )
    if mode == "train":
        return trainer.train()
    if mode == "test":
        return trainer.test()
    if img1 is None or img2 is None:
        raise ValueError("demo mode requires img1 and img2")
    return trainer.demo(img1, img2, pair_type)
