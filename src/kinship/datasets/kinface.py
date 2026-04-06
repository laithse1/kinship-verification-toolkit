from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import scipy.io as sio

from kinship.paths import kinface_workspace_root


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


def labels(records: Iterable[PairRecord]) -> list[int]:
    return [record.label for record in records]

