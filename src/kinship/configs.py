from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

from kinship.paths import workspace_root


CONFIG_ROOT = workspace_root() / "configs"
EXPERIMENT_CONFIG_ROOT = CONFIG_ROOT / "experiments"
BENCHMARK_CONFIG_ROOT = CONFIG_ROOT / "benchmarks"


@dataclass
class ExperimentConfig:
    name: str
    algorithm: str
    parameters: dict[str, Any]
    description: str = ""
    tags: list[str] = field(default_factory=list)
    source_path: Path | None = None


@dataclass
class BenchmarkConfig:
    name: str
    experiments: list[ExperimentConfig]
    description: str = ""
    tags: list[str] = field(default_factory=list)
    source_path: Path | None = None


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def experiment_config_paths() -> list[Path]:
    return sorted(EXPERIMENT_CONFIG_ROOT.glob("*.toml"))


def benchmark_config_paths() -> list[Path]:
    return sorted(BENCHMARK_CONFIG_ROOT.glob("*.toml"))


def resolve_experiment_config_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    toml_path = EXPERIMENT_CONFIG_ROOT / f"{name_or_path}.toml"
    if toml_path.exists():
        return toml_path
    raise FileNotFoundError(f"Experiment config not found: {name_or_path}")


def resolve_benchmark_config_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    toml_path = BENCHMARK_CONFIG_ROOT / f"{name_or_path}.toml"
    if toml_path.exists():
        return toml_path
    raise FileNotFoundError(f"Benchmark config not found: {name_or_path}")


def load_experiment_config(path_or_name: str | Path) -> ExperimentConfig:
    path = path_or_name if isinstance(path_or_name, Path) else resolve_experiment_config_path(path_or_name)
    data = _load_toml(path)
    return ExperimentConfig(
        name=data["name"],
        algorithm=data["algorithm"],
        parameters=dict(data.get("parameters", {})),
        description=data.get("description", ""),
        tags=list(data.get("tags", [])),
        source_path=path,
    )


def load_benchmark_config(path_or_name: str | Path) -> BenchmarkConfig:
    path = path_or_name if isinstance(path_or_name, Path) else resolve_benchmark_config_path(path_or_name)
    data = _load_toml(path)
    experiments = [
        ExperimentConfig(
            name=item["name"],
            algorithm=item["algorithm"],
            parameters=dict(item.get("parameters", {})),
            description=item.get("description", ""),
            tags=list(item.get("tags", [])),
            source_path=path,
        )
        for item in data.get("experiments", [])
    ]
    return BenchmarkConfig(
        name=data["name"],
        description=data.get("description", ""),
        tags=list(data.get("tags", [])),
        experiments=experiments,
        source_path=path,
    )

