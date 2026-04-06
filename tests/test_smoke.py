from __future__ import annotations

from kinship.algorithms.classical import run_classical_verification
from kinship.algorithms.kinver import run_kinver


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
