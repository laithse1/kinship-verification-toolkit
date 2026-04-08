from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from kinship.datasets.mydataset import export_mydataset_pairs, summarize_mydataset
from kinship.paths import workspace_root


def _temp_dataset_root(name: str) -> Path:
    path = workspace_root() / "outputs" / "test-temp" / f"{name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_mydataset_summary_and_pair_export() -> None:
    temp_root = _temp_dataset_root("mydataset")
    try:
        dataset_root = temp_root / "mydataset"
        (dataset_root / "Identical_Twins_Dataset" / "Identical_Twins&Tripples_Dataset" / "fam1" / "alice").mkdir(parents=True)
        (dataset_root / "Identical_Twins_Dataset" / "Identical_Twins&Tripples_Dataset" / "fam1" / "beth").mkdir(parents=True)
        (dataset_root / "Identical_Twins_Dataset" / "Identical_Twins&Tripples_Dataset" / "fam2" / "carol").mkdir(parents=True)
        (dataset_root / "Identical_Twins_Dataset" / "Identical_Twins&Tripples_Dataset" / "fam2" / "dina").mkdir(parents=True)
        for relative in [
            "Identical_Twins_Dataset/Identical_Twins&Tripples_Dataset/fam1/alice/a1.jpg",
            "Identical_Twins_Dataset/Identical_Twins&Tripples_Dataset/fam1/beth/b1.jpg",
            "Identical_Twins_Dataset/Identical_Twins&Tripples_Dataset/fam2/carol/c1.jpg",
            "Identical_Twins_Dataset/Identical_Twins&Tripples_Dataset/fam2/dina/d1.jpg",
        ]:
            (dataset_root / relative).write_bytes(b"img")

        summary = summarize_mydataset(dataset_root)
        assert summary.subset_count == 1
        assert summary.family_count == 2
        assert summary.person_count == 4
        assert summary.image_count == 4
        assert summary.subsets[0]["kinship_group"] == "identical_twins"

        pair_path = temp_root / "pairs.csv"
        export_mydataset_pairs(pair_path, root=dataset_root, subset="Identical_Twins_Dataset", random_state=7)
        text = pair_path.read_text(encoding="utf-8")
        assert "label" in text
        assert ",1," in text
        assert ",0," in text
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
