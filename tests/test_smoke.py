from __future__ import annotations

from kinship.algorithms.classical import run_classical_verification
from kinship.algorithms.facornet_native import run_facornet
from kinship.algorithms.forestnn_native import run_forestnn
from kinship.algorithms.kinver import run_kinver
from kinship.algorithms.relation_aware_native import run_relation_aware


def test_classical_random_smoke() -> None:
    result = run_classical_verification(
        relation="fs",
        dataset="KinFaceW-I",
        method="random",
        limit=20,
    )
    assert len(result.fold_scores) == 1
    assert 0.0 <= result.mean_accuracy <= 1.0


def test_classical_kfold_smoke() -> None:
    result = run_classical_verification(
        relation="fs",
        dataset="KinFaceW-I",
        method="kfold",
        limit=20,
    )
    assert len(result.fold_scores) == 5
    assert 0.0 <= result.mean_accuracy <= 1.0


def test_kinver_smoke() -> None:
    result = run_kinver(
        relation="fs",
        dataset="KinFaceW-II",
        use_lbp=False,
        use_hog=False,
        iterations=2,
        knn=3,
    )
    assert len(result.fold_scores) == 5
    assert len(result.beta_means) >= 1
    assert 0.0 <= result.mean_accuracy <= 1.0


def test_forestnn_smoke() -> None:
    result = run_forestnn(
        relation="fs",
        dataset="KinFaceW-II",
        image_size=32,
        embedding_dim=16,
        hidden_dim=24,
        batch_size=8,
        num_epochs=1,
        limit_pairs=60,
    )
    assert len(result.fold_metrics) == 5
    assert result.model_name == "forestnn-native-hybrid"
    assert result.local_encoder == "residual"
    assert result.alignment_mode == "adaptive"
    assert 0.0 <= result.mean_accuracy <= 1.0


def test_facornet_smoke() -> None:
    result = run_facornet(
        relation="fs",
        dataset="KinFaceW-II",
        image_size=32,
        embedding_dim=16,
        hidden_dim=24,
        num_heads=4,
        batch_size=8,
        num_epochs=1,
        limit_pairs=60,
    )
    assert len(result.fold_metrics) == 5
    assert result.model_name == "facornet-native-hybrid"
    assert result.alignment_mode == "adaptive"
    assert 0.0 <= result.mean_accuracy <= 1.0


def test_relation_aware_smoke() -> None:
    result = run_relation_aware(
        relation="fs",
        dataset="KinFaceW-II",
        support_relations="all",
        image_size=32,
        embedding_dim=16,
        hidden_dim=24,
        relation_dim=8,
        num_heads=4,
        batch_size=8,
        num_epochs=1,
        limit_pairs=20,
    )
    assert len(result.fold_metrics) == 5
    assert result.model_name == "relation-aware-native-hybrid"
    assert result.support_relations == "all"
    assert 0.0 <= result.mean_accuracy <= 1.0
