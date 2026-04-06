from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from kinship.algorithms.classical import run_classical_verification
from kinship.algorithms.family_deep_native import run_family_deep
from kinship.algorithms.gae_native import run_gae
from kinship.algorithms.kinver import run_kinver


AlgorithmRunner = Callable[..., Any]


ALGORITHM_REGISTRY: dict[str, AlgorithmRunner] = {
    "classical": run_classical_verification,
    "family-deep": run_family_deep,
    "gae": run_gae,
    "kinver": run_kinver,
}


def algorithm_names() -> list[str]:
    return sorted(ALGORITHM_REGISTRY)


def run_algorithm(algorithm: str, parameters: dict[str, Any]) -> Any:
    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {', '.join(algorithm_names())}"
        )
    return ALGORITHM_REGISTRY[algorithm](**parameters)


def serialize_result(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        payload = asdict(result)
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        raise TypeError(f"Unsupported result type: {type(result)!r}")

    if hasattr(result, "mean_accuracy"):
        payload["mean_accuracy"] = float(result.mean_accuracy)
    return payload
