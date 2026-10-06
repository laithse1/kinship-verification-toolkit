from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio

from kinship.paths import kinface_workspace_root, kinver_workspace_root


RELATION_TO_DIR = {
    "fd": "father-dau",
    "fs": "father-son",
    "md": "mother-dau",
    "ms": "mother-son",
}


@dataclass(frozen=True)
class PairRecord:
    relation: str
    label: int
    parent_path: Path
    child_path: Path
    parent_name: str
    child_name: str


def _matlab_scalar_to_int(value: object) -> int:
    return int(value[0, 0])


def _matlab_scalar_to_str(value: object) -> str:
    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, str):
                return item
        except ValueError:
            pass
    return str(value[0])


def _kinver_dataset_dir(dataset: str) -> Path:
    root = kinver_workspace_root()
    direct = root / f"data-{dataset}"
    if direct.exists():
        return direct
    return root / "data" / f"data-{dataset}"


def load_kinface_pairs(
    relation: str,
    dataset: str = "KinFaceW-I",
    root: Path | None = None,
) -> list[PairRecord]:
    if relation not in RELATION_TO_DIR:
        raise ValueError(f"Unsupported relation '{relation}'")
    root = root or kinface_workspace_root() / dataset
    mat_path = root / "meta_data" / f"{relation}_pairs.mat"
    image_dir = root / "images" / RELATION_TO_DIR[relation]
    pairs = sio.loadmat(mat_path)["pairs"]

    items: list[PairRecord] = []
    for row in pairs:
        label = _matlab_scalar_to_int(row[1])
        parent_name = _matlab_scalar_to_str(row[2])
        child_name = _matlab_scalar_to_str(row[3])
        items.append(
            PairRecord(
                relation=relation,
                label=label,
                parent_path=image_dir / parent_name,
                child_path=image_dir / child_name,
                parent_name=parent_name,
                child_name=child_name,
            )
        )
    return items


def load_kinface_official_folds(
    relation: str,
    dataset: str = "KinFaceW-II",
    prefix: str = "vggFace",
) -> list[int]:
    mat_path = _kinver_dataset_dir(dataset) / f"{prefix}_{relation}.mat"
    payload = sio.loadmat(mat_path)
    folds = np.asarray(payload["fold"]).ravel().astype(np.int32).tolist()
    matches = np.asarray(payload["matches"]).ravel().astype(np.int32).tolist()
    records = load_kinface_pairs(relation=relation, dataset=dataset)
    if len(folds) != len(records):
        raise ValueError(
            f"Official folds length mismatch for {dataset} {relation}: {len(folds)} vs {len(records)}"
        )
    record_labels = [record.label for record in records]
    if record_labels != matches:
        raise ValueError(
            f"Official fold labels do not align with metadata order for {dataset} {relation}"
        )
    return folds


def labels(records: Iterable[PairRecord]) -> list[int]:
    return [record.label for record in records]
