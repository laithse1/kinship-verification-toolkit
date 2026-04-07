from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from scipy.io import loadmat
from skimage import io
from torch.utils.data import Dataset


class FIWDataset(Dataset):
    def __init__(self, dataset_root: Path, metadata_dir: Path, pair_type: str, set_name: str, transform=None, color_space: str = "rgb"):
        self.dataset_root = Path(dataset_root)
        self.metadata_dir = Path(metadata_dir)
        self.pair_type = pair_type
        self.set_name = set_name
        self.transform = transform
        self.color_space = color_space
        self.labels_df = pd.read_csv(self.metadata_dir / f"{self.set_name}-pairs-full.csv")
        self.labels_df = self.labels_df[self.labels_df.ptype == self.pair_type]
        self._path_cache: dict[str, Path] = {}
        self._pid_cache: dict[tuple[str, ...], list[Path]] = {}
        self.skipped_pairs = 0
        self.labels_df = self._drop_unresolved_pairs(self.labels_df)

    def __len__(self) -> int:
        return len(self.labels_df)

    @staticmethod
    def _requested_face_index(image_name: str) -> int:
        match = re.search(r"_face(\d+)$", Path(image_name).stem)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _available_face_index(path: Path) -> int:
        return FIWDataset._requested_face_index(path.name)

    def _legacy_image_path(self, image_name: str) -> Path:
        split = "val" if self.set_name == "test" else self.set_name
        split = "train" if split not in {"train", "val"} else split
        return self.dataset_root / f"{split}-faces" / image_name

    def _fids_candidates(self, image_name: str) -> list[Path]:
        parts = Path(image_name).parts
        if len(parts) < 3:
            return []
        family_id, mid = parts[0], parts[1]
        requested_index = self._requested_face_index(image_name)
        pid = Path(parts[-1]).stem.split("_face", 1)[0]
        cache_key = (family_id, mid, pid)
        if cache_key not in self._pid_cache:
            mid_dir = self.dataset_root / family_id / mid
            family_dir = self.dataset_root / family_id
            if mid_dir.exists():
                matches = sorted(mid_dir.glob(f"{pid}_*.jpg"), key=self._available_face_index)
            elif family_dir.exists():
                matches = sorted(family_dir.glob(f"**/{pid}_*.jpg"), key=self._available_face_index)
            else:
                matches = []
            self._pid_cache[cache_key] = matches
        return sorted(
            self._pid_cache[cache_key],
            key=lambda path: (abs(self._available_face_index(path) - requested_index), self._available_face_index(path), str(path)),
        )

    def get_image_path(self, image_name: str) -> Path:
        if image_name in self._path_cache:
            return self._path_cache[image_name]

        direct_path = self.dataset_root / image_name
        if direct_path.exists():
            self._path_cache[image_name] = direct_path
            return direct_path

        legacy_path = self._legacy_image_path(image_name)
        if legacy_path.exists():
            self._path_cache[image_name] = legacy_path
            return legacy_path

        candidates = self._fids_candidates(image_name)
        if candidates:
            self._path_cache[image_name] = candidates[0]
            return candidates[0]

        raise FileNotFoundError(
            f"Unable to resolve FIW image '{image_name}' under dataset root '{self.dataset_root}'."
        )

    def _drop_unresolved_pairs(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        missing_images: set[str] = set()
        unique_images = pd.unique(labels_df[["p1", "p2"]].to_numpy().ravel())
        for image_name in unique_images:
            try:
                self.get_image_path(str(image_name))
            except FileNotFoundError:
                missing_images.add(str(image_name))
        if not missing_images:
            return labels_df.reset_index(drop=True)
        mask = ~labels_df["p1"].isin(missing_images) & ~labels_df["p2"].isin(missing_images)
        self.skipped_pairs = int((~mask).sum())
        return labels_df.loc[mask].reset_index(drop=True)

    def __getitem__(self, idx: int) -> dict:
        row = self.labels_df.iloc[idx]
        parent_image_path = self.get_image_path(row.p1)
        child_image_path = self.get_image_path(row.p2)
        parent_image = io.imread(parent_image_path)
        child_image = io.imread(child_image_path)
        if self.color_space == "bgr":
            parent_image = parent_image[:, :, ::-1]
            child_image = child_image[:, :, ::-1]
        if self.transform:
            parent_image = self.transform(parent_image)
            child_image = self.transform(child_image)
        return {
            "parent_image": parent_image,
            "children_image": child_image,
            "kin": row.label,
            "parent_image_name": str(parent_image_path),
            "children_image_name": str(child_image_path),
            "parent_family_id": parent_image_path.parts[-3],
            "children_family_id": child_image_path.parts[-3],
        }


class KinFaceDataset(Dataset):
    def __init__(self, labels_df: pd.DataFrame, root_dir: Path, transform=None, color_space: str = "rgb"):
        self.labels_df = labels_df
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.color_space = color_space

    def __len__(self) -> int:
        return len(self.labels_df)

    def get_image_path(self, image_name: str) -> Path:
        prefix = image_name.split("_")[0]
        relation_dir = {"fs": "father-son", "fd": "father-dau", "md": "mother-dau", "ms": "mother-son"}[prefix]
        return self.root_dir / "images" / relation_dir / image_name

    def __getitem__(self, idx: int) -> dict:
        parent_image_path = self.get_image_path(self.labels_df.iloc[idx, 2])
        child_image_path = self.get_image_path(self.labels_df.iloc[idx, 3])
        parent_image = io.imread(parent_image_path)
        child_image = io.imread(child_image_path)
        if self.color_space == "bgr":
            parent_image = parent_image[:, :, ::-1]
            child_image = child_image[:, :, ::-1]
        if self.transform:
            parent_image = self.transform(parent_image)
            child_image = self.transform(child_image)
        return {
            "parent_image": parent_image.double(),
            "children_image": child_image.double(),
            "kin": self.labels_df.iloc[idx, 1],
            "parent_image_name": self.labels_df.iloc[idx, 2],
            "children_image_name": self.labels_df.iloc[idx, 3],
            "fold": self.labels_df.iloc[idx, 0],
        }


class KinFaceWLoaderGenerator:
    def __init__(self, dataset_name: str, dataset_path: Path, color_space_name: str):
        assert dataset_name in {"KinFaceW-I", "KinFaceW-II"}
        self.dataset_name = dataset_name
        self.color_space_name = color_space_name
        self.dataset_path = Path(dataset_path) / self.dataset_name
        self.kin_pairs = pd.concat([
            self.parse_kin_pairs(loadmat(self.dataset_path / "meta_data" / "fs_pairs.mat")),
            self.parse_kin_pairs(loadmat(self.dataset_path / "meta_data" / "fd_pairs.mat")),
            self.parse_kin_pairs(loadmat(self.dataset_path / "meta_data" / "md_pairs.mat")),
            self.parse_kin_pairs(loadmat(self.dataset_path / "meta_data" / "ms_pairs.mat")),
        ])
        self.kin_pairs["pair_type"] = self.kin_pairs["image_1"].apply(lambda x: x.split("_")[0])

    @staticmethod
    def parse_kin_pairs(kin_pair: dict) -> pd.DataFrame:
        result = pd.DataFrame(columns=["fold", "kin", "image_1", "image_2"])
        for pair in kin_pair["pairs"]:
            pair_data = pair.tolist()
            result.loc[len(result)] = [pair_data[0][0][0], pair_data[1][0][0], pair_data[2][0], pair_data[3][0]]
        return result

    def get_data_loader(self, fold: int, batch_size: int, pair_type: str, train_transformer, test_transformer, torch_module):
        kin_pairs = self.kin_pairs[self.kin_pairs["pair_type"] == pair_type]
        test_data_pairs = kin_pairs[kin_pairs["fold"] == fold]
        train_data_pairs = kin_pairs[kin_pairs["fold"] != fold]
        kinfacew_dataset_test = KinFaceDataset(test_data_pairs, self.dataset_path, test_transformer, self.color_space_name)
        kinfacew_dataset_train = KinFaceDataset(train_data_pairs, self.dataset_path, train_transformer, self.color_space_name)
        dataloader_test = torch_module.utils.data.DataLoader(kinfacew_dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
        dataloader_train = torch_module.utils.data.DataLoader(kinfacew_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        return dataloader_test, dataloader_train
