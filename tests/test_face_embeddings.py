from __future__ import annotations

from kinship.algorithms.face_embeddings import run_face_embedding_verification
from kinship.configs import load_benchmark_config, load_experiment_config
from kinship.datasets.kinface import load_kinface_official_folds
from kinship.metrics import compute_verification_metrics, select_threshold


def test_select_threshold_balanced_accuracy() -> None:
    selection = select_threshold(
        scores=[0.1, 0.2, 0.8, 0.9],
        labels=[0, 0, 1, 1],
        objective="balanced_accuracy",
    )
    assert 0.2 <= selection.threshold <= 0.8
    assert 0.0 <= selection.score <= 1.0


def test_compute_verification_metrics_outputs_tar_and_eer() -> None:
    metrics = compute_verification_metrics(
        scores=[0.1, 0.2, 0.35, 0.7, 0.8, 0.9],
        labels=[0, 0, 1, 0, 1, 1],
        threshold=0.5,
        far_targets=[0.1, 0.5],
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0
    assert set(metrics["tar_at_far"]) == {"tar@far=0.1", "tar@far=0.5"}


def test_load_kinface_official_folds() -> None:
    folds = load_kinface_official_folds("fs", dataset="KinFaceW-II")
    assert len(folds) == 500
    assert set(folds) == {1, 2, 3, 4, 5}


def test_face_embedding_precomputed_smoke() -> None:
    result = run_face_embedding_verification(
        relation="fs",
        dataset="KinFaceW-II",
        backbone="vggface-precomputed",
        calibration_objective="balanced_accuracy",
        far_targets=[0.01, 0.1],
    )
    assert len(result.fold_metrics) == 5
    assert 0.0 <= result.mean_accuracy <= 1.0
    assert 0.0 <= result.mean_roc_auc <= 1.0
    assert set(result.mean_tar_at_far) == {"tar@far=0.01", "tar@far=0.1"}


def test_load_face_embed_config() -> None:
    config = load_experiment_config("face-embed-vggface-fs")
    assert config.algorithm == "face-embed"
    assert config.parameters["backbone"] == "vggface-precomputed"


def test_load_face_embed_benchmark() -> None:
    config = load_benchmark_config("kinfacew-all-relations-face-embed")
    assert config.name == "kinfacew-all-relations-face-embed"
    assert len(config.experiments) == 4
    assert {experiment.parameters["relation"] for experiment in config.experiments} == {"fd", "fs", "md", "ms"}
