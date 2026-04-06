from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "nri_uniform"
LBP_BINS = 59
PATCH = 4
HOG_BINS = 36


def _chi_square_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x[:, None, :]
    y = y[None, :, :]
    denom = x + y + 1e-12
    return 0.5 * np.sum(((x - y) ** 2) / denom, axis=2)


def _load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    if array.shape[0] % PATCH or array.shape[1] % PATCH:
        raise ValueError(f"Image size must be divisible by {PATCH}: {path}")
    return array


@lru_cache(maxsize=8192)
def _grayscale_patch_features(path_str: str) -> np.ndarray:
    image = Image.open(path_str).convert("L")
    arr = np.asarray(image, dtype=np.uint8)
    features: list[np.ndarray] = []
    for y in range(0, arr.shape[0], PATCH):
        for x in range(0, arr.shape[1], PATCH):
            patch = arr[y : y + PATCH, x : x + PATCH]
            lbp = local_binary_pattern(
                patch, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD
            )
            hist, _ = np.histogram(lbp, bins=np.arange(LBP_BINS + 1), density=False)
            hog_vec = hog(
                patch,
                orientations=9,
                pixels_per_cell=(2, 2),
                cells_per_block=(1, 1),
                feature_vector=True,
            )
            features.append(np.concatenate([hist.astype(np.float32), hog_vec.astype(np.float32)]))
    return np.concatenate(features, axis=0)


@lru_cache(maxsize=8192)
def _rgb_lbp_patch_features(path_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = _load_rgb(Path(path_str))
    per_channel: list[np.ndarray] = []
    for channel_idx in range(3):
        channel = image[:, :, channel_idx]
        features: list[np.ndarray] = []
        for y in range(0, channel.shape[0], PATCH):
            for x in range(0, channel.shape[1], PATCH):
                patch = channel[y : y + PATCH, x : x + PATCH]
                lbp = local_binary_pattern(
                    patch.astype(np.uint8),
                    P=LBP_POINTS,
                    R=LBP_RADIUS,
                    method=LBP_METHOD,
                )
                hist, _ = np.histogram(
                    lbp, bins=np.arange(LBP_BINS + 1), density=False
                )
                features.append(hist.astype(np.float32))
        per_channel.append(np.stack(features, axis=0))
    return tuple(per_channel)  # type: ignore[return-value]


def extract_pair_patch_feature(parent_path: Path, child_path: Path) -> np.ndarray:
    parent = _grayscale_patch_features(str(parent_path))
    child = _grayscale_patch_features(str(child_path))
    return np.concatenate([parent, child], axis=0)


def extract_pair_chisq_feature(parent_path: Path, child_path: Path) -> np.ndarray:
    parent_channels = _rgb_lbp_patch_features(str(parent_path))
    child_channels = _rgb_lbp_patch_features(str(child_path))
    dist_features = []
    for parent, child in zip(parent_channels, child_channels, strict=True):
        dist = _chi_square_distance(parent, child)
        dist_features.append(dist.reshape(-1).astype(np.float32))
    return np.concatenate(dist_features, axis=0)
