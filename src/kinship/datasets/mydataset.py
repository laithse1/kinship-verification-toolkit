from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import csv
import json
import random
import re

from kinship.paths import mydataset_root


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORED_DIR_NAMES = {
    ".thumbnails",
    "__macosx",
    "uncropped",
    "families collection_uncropped",
}
IGNORED_FILE_EXTENSIONS = {".zip", ".xls", ".xlsx", ".csv", ".json", ".txt"}


@dataclass(frozen=True)
class MyDatasetImageRecord:
    subset: str
    kinship_group: str
    family_id: str
    person_id: str
    image_path: Path
    relative_path: Path


@dataclass(frozen=True)
class MyDatasetSummary:
    root: str
    subset_count: int
    family_count: int
    person_count: int
    image_count: int
    subsets: list[dict]


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "unknown"


def _infer_kinship_group(subset_name: str) -> str:
    name = subset_name.lower()
    if "identical" in name and "twin" in name:
        return "identical_twins"
    if "tripple" in name or "triple" in name:
        return "triplets"
    if "young" in name or "adult" in name or "middle_aged" in name or "age" in name:
        return "age_variant_kinship"
    if "family" in name or "famliy" in name:
        return "family_kinship"
    if "mydataset_102" in _slugify(name) or "mydataset" in _slugify(name):
        return "family_kinship"
    return "custom_kinship"


def _infer_person_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\d+$", "", stem)
    stem = re.sub(r"[_\-\s]+$", "", stem)
    return _slugify(stem)


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _is_ignored_path(path: Path) -> bool:
    lowered_parts = {_slugify(part) for part in path.parts}
    ignored = {_slugify(name) for name in IGNORED_DIR_NAMES}
    return bool(lowered_parts & ignored)


def _iter_subset_roots(root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.lower() not in IGNORED_DIR_NAMES
        ],
        key=lambda path: path.name.lower(),
    )


def _relative_parts_without_wrappers(subset_root: Path, image_path: Path) -> tuple[str, ...]:
    parts = list(image_path.relative_to(subset_root).parts[:-1])
    while parts and "dataset" in parts[0].lower():
        parts.pop(0)
    return tuple(parts)


def _build_record(root: Path, subset_root: Path, image_path: Path) -> MyDatasetImageRecord:
    subset_name = subset_root.name
    rel_path = image_path.relative_to(root)
    parts = _relative_parts_without_wrappers(subset_root, image_path)
    if len(parts) >= 2:
        family_id = _slugify(parts[-2])
        person_id = _slugify(parts[-1])
    elif len(parts) == 1:
        family_id = _slugify(parts[0])
        person_id = _infer_person_from_filename(image_path.name)
    else:
        family_id = _slugify(subset_name)
        person_id = _infer_person_from_filename(image_path.name)
    return MyDatasetImageRecord(
        subset=subset_name,
        kinship_group=_infer_kinship_group(subset_name),
        family_id=family_id,
        person_id=person_id,
        image_path=image_path,
        relative_path=rel_path,
    )


def scan_mydataset(root: Path | None = None) -> list[MyDatasetImageRecord]:
    root = root or mydataset_root()
    if not root.exists():
        return []
    records: list[MyDatasetImageRecord] = []
    for subset_root in _iter_subset_roots(root):
        for image_path in subset_root.rglob("*"):
            if _is_ignored_path(image_path):
                continue
            if _is_supported_image(image_path):
                records.append(_build_record(root, subset_root, image_path))
    return records


def summarize_mydataset(root: Path | None = None) -> MyDatasetSummary:
    root = root or mydataset_root()
    records = scan_mydataset(root)
    subset_rows: list[dict] = []
    subset_names = sorted({record.subset for record in records})
    for subset_name in subset_names:
        subset_records = [record for record in records if record.subset == subset_name]
        family_keys = {(record.subset, record.family_id) for record in subset_records}
        person_keys = {
            (record.subset, record.family_id, record.person_id) for record in subset_records
        }
        subset_rows.append(
            {
                "subset": subset_name,
                "kinship_group": subset_records[0].kinship_group if subset_records else "unknown",
                "family_count": len(family_keys),
                "person_count": len(person_keys),
                "image_count": len(subset_records),
            }
        )
    return MyDatasetSummary(
        root=str(root),
        subset_count=len(subset_rows),
        family_count=len({(record.subset, record.family_id) for record in records}),
        person_count=len({(record.subset, record.family_id, record.person_id) for record in records}),
        image_count=len(records),
        subsets=subset_rows,
    )


def export_mydataset_summary(
    output_path: Path,
    root: Path | None = None,
) -> Path:
    summary = summarize_mydataset(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "root": summary.root,
                "subset_count": summary.subset_count,
                "family_count": summary.family_count,
                "person_count": summary.person_count,
                "image_count": summary.image_count,
                "subsets": summary.subsets,
            },
            handle,
            indent=2,
        )
    return output_path


def export_mydataset_inventory(
    output_path: Path,
    root: Path | None = None,
) -> Path:
    records = scan_mydataset(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subset", "kinship_group", "family_id", "person_id", "relative_path"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "subset": record.subset,
                    "kinship_group": record.kinship_group,
                    "family_id": record.family_id,
                    "person_id": record.person_id,
                    "relative_path": record.relative_path.as_posix(),
                }
            )
    return output_path


def export_mydataset_pairs(
    output_path: Path,
    root: Path | None = None,
    subset: str | None = None,
    max_positive_pairs_per_person_pair: int = 20,
    negative_ratio: float = 1.0,
    random_state: int = 42,
) -> Path:
    records = scan_mydataset(root)
    if subset is not None:
        records = [record for record in records if _slugify(record.subset) == _slugify(subset)]

    by_family_person: dict[tuple[str, str, str], list[MyDatasetImageRecord]] = {}
    for record in records:
        key = (record.subset, record.family_id, record.person_id)
        by_family_person.setdefault(key, []).append(record)

    families: dict[tuple[str, str], dict[str, list[MyDatasetImageRecord]]] = {}
    for (subset_name, family_id, person_id), items in by_family_person.items():
        families.setdefault((subset_name, family_id), {})[person_id] = items

    positive_rows: list[dict[str, str | int]] = []
    for (subset_name, family_id), people in families.items():
        kinship_group = next(iter(people.values()))[0].kinship_group
        person_names = sorted(people)
        for person_a, person_b in combinations(person_names, 2):
            image_pairs = [
                (img_a, img_b)
                for img_a in people[person_a]
                for img_b in people[person_b]
            ]
            for img_a, img_b in image_pairs[:max_positive_pairs_per_person_pair]:
                positive_rows.append(
                    {
                        "p1": img_a.relative_path.as_posix(),
                        "p2": img_b.relative_path.as_posix(),
                        "ptype": kinship_group,
                        "label": 1,
                        "subset": subset_name,
                        "family_1": family_id,
                        "family_2": family_id,
                        "person_1": person_a,
                        "person_2": person_b,
                    }
                )

    rng = random.Random(random_state)
    family_keys = sorted(families)
    negative_candidates: list[tuple[MyDatasetImageRecord, MyDatasetImageRecord]] = []
    for left_idx, left_family in enumerate(family_keys):
        for right_family in family_keys[left_idx + 1 :]:
            left_people = families[left_family]
            right_people = families[right_family]
            for left_person, left_images in left_people.items():
                for right_person, right_images in right_people.items():
                    negative_candidates.append((left_images[0], right_images[0]))
    rng.shuffle(negative_candidates)
    max_negatives = int(round(len(positive_rows) * negative_ratio))
    negative_rows: list[dict[str, str | int]] = []
    for left_img, right_img in negative_candidates[:max_negatives]:
        negative_rows.append(
            {
                "p1": left_img.relative_path.as_posix(),
                "p2": right_img.relative_path.as_posix(),
                "ptype": "nonkin",
                "label": 0,
                "subset": f"{left_img.subset}__{right_img.subset}",
                "family_1": left_img.family_id,
                "family_2": right_img.family_id,
                "person_1": left_img.person_id,
                "person_2": right_img.person_id,
            }
        )

    rows = positive_rows + negative_rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "p1",
                "p2",
                "ptype",
                "label",
                "subset",
                "family_1",
                "family_2",
                "person_1",
                "person_2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path
