from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import scipy.io as sio

from kinship.algorithms import family_deep_native
from kinship.algorithms._family_deep_data import FIWDataset
from kinship.algorithms.family_deep_native import run_family_deep
from kinship.algorithms.gae_native import run_gae
from kinship.paths import workspace_root


def test_family_deep_reports_missing_optional_dependencies(monkeypatch) -> None:
    real_import_module = family_deep_native.importlib.import_module

    def fake_import_module(name: str):
        if name == "facenet_pytorch":
            raise ModuleNotFoundError(name)
        return real_import_module(name)

    monkeypatch.setattr(family_deep_native.importlib, "import_module", fake_import_module)

    try:
        run_family_deep(model_name="kin_facenet")
    except RuntimeError as exc:
        assert "facenet_pytorch" in str(exc)
    else:
        raise AssertionError("Expected run_family_deep to raise for missing optional deps")


def _workspace_temp_dir(name: str) -> Path:
    path = workspace_root() / "outputs" / "test-temp" / f"{name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_native_gae_writes_standard_output() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(18, 12))
    y = x * 0.8 + rng.normal(scale=0.1, size=(18, 12))
    temp_dir = _workspace_temp_dir("gae-standard")
    try:
        input_path = temp_dir / "pairs.mat"
        output_path = temp_dir / "mapped.mat"
        sio.savemat(input_path, {"x": x, "y": y}, oned_as="column")

        result = run_gae(
            input_path=str(input_path),
            output_path=str(output_path),
            variant="standard",
            numfac=6,
            nummap=4,
            numepochs=3,
            verbose=False,
        )

        saved = sio.loadmat(output_path)
        assert output_path.exists()
        assert result.output_path == str(output_path)
        assert saved["wxf"].shape == (12, 6)
        assert saved["wyf"].shape == (12, 6)
        assert saved["whf"].shape == (4, 6)
        assert saved["z_bias"].size == 4
        assert result.reconstruction_error >= 0.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_native_gae_writes_multiview_output() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 10))
    y = x * 0.6 + rng.normal(scale=0.2, size=(20, 10))
    temp_dir = _workspace_temp_dir("gae-multiview")
    try:
        input_path = temp_dir / "pairs.mat"
        output_path = temp_dir / "mapped_multiview.mat"
        sio.savemat(input_path, {"x": x, "y": y}, oned_as="column")

        result = run_gae(
            input_path=str(input_path),
            output_path=str(output_path),
            variant="multiview",
            numfac=8,
            nummap=5,
            numepochs=2,
            verbose=False,
        )

        saved = sio.loadmat(output_path)
        assert output_path.exists()
        assert result.variant == "multiview"
        assert saved["wxf"].shape == (10, 8)
        assert saved["wyf"].shape == (10, 8)
        assert saved["whf"].shape[0] == 5
        assert saved["wpf"].shape == (4, 8)
        assert saved["bmap"].size == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_native_gae_accepts_project_feature_keys() -> None:
    rng = np.random.default_rng(13)
    lfeat = rng.normal(size=(14, 9))
    rfeat = lfeat * 0.5 + rng.normal(scale=0.25, size=(14, 9))
    temp_dir = _workspace_temp_dir("gae-project-keys")
    try:
        input_path = temp_dir / "pairs.mat"
        output_path = temp_dir / "mapped.mat"
        sio.savemat(input_path, {"lfeat": lfeat, "rfeat": rfeat}, oned_as="column")

        result = run_gae(
            input_path=str(input_path),
            output_path=str(output_path),
            variant="standard",
            numfac=5,
            nummap=3,
            numepochs=2,
            verbose=False,
        )

        saved = sio.loadmat(output_path)
        assert result.output_path == str(output_path)
        assert saved["wxf"].shape == (9, 5)
        assert saved["wyf"].shape == (9, 5)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_fiw_dataset_resolves_fids_face_index_mismatches() -> None:
    temp_dir = _workspace_temp_dir("fiw-fids-resolution")
    try:
        dataset_root = temp_dir / "FIDs"
        metadata_dir = temp_dir / "metadata"
        mid_dir = dataset_root / "F0001" / "MID1"
        child_dir = dataset_root / "F0001" / "MID3"
        mid_dir.mkdir(parents=True, exist_ok=True)
        child_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        (mid_dir / "P00001_face2.jpg").write_bytes(b"fake-parent")
        (child_dir / "P00006_face1.jpg").write_bytes(b"fake-child")
        pd.DataFrame(
            [
                {
                    "p1": "F0001/MID1/P00001_face0.jpg",
                    "p2": "F0001/MID3/P00006_face0.jpg",
                    "ptype": "fs",
                    "label": 1,
                }
            ]
        ).to_csv(metadata_dir / "train-pairs-full.csv", index=False)

        dataset = FIWDataset(
            dataset_root=dataset_root,
            metadata_dir=metadata_dir,
            pair_type="fs",
            set_name="train",
        )

        assert dataset.get_image_path("F0001/MID1/P00001_face0.jpg") == mid_dir / "P00001_face2.jpg"
        assert dataset.get_image_path("F0001/MID3/P00006_face0.jpg") == child_dir / "P00006_face1.jpg"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
