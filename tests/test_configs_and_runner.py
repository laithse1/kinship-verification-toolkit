from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from kinship.configs import (
    BenchmarkConfig,
    ExperimentConfig,
    load_benchmark_config,
    load_experiment_config,
)
from kinship.paths import workspace_root
from kinship.runner import run_benchmark, run_experiment


def _workspace_temp_dir(name: str) -> Path:
    path = workspace_root() / "outputs" / "test-temp" / f"{name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_load_experiment_config() -> None:
    config = load_experiment_config("classical-fs-kfold-smoke")
    assert config.name == "classical-fs-kfold-smoke"
    assert config.algorithm == "classical"
    assert config.parameters["relation"] == "fs"


def test_load_native_port_experiment_config() -> None:
    config = load_experiment_config("gae-fs-train-p16-standard")
    assert config.algorithm == "gae"
    assert config.parameters["variant"] == "standard"


def test_load_benchmark_config() -> None:
    config = load_benchmark_config("supported-smoke")
    assert config.name == "supported-smoke"
    assert len(config.experiments) == 4
    assert any(experiment.algorithm == "classical" for experiment in config.experiments)


def test_load_native_port_benchmark_config() -> None:
    config = load_benchmark_config("native-ports")
    assert config.name == "native-ports"
    assert len(config.experiments) == 3
    assert {experiment.algorithm for experiment in config.experiments} == {"family-deep", "gae"}


def test_run_experiment_writes_artifacts() -> None:
    output_root = _workspace_temp_dir("experiment")
    config = ExperimentConfig(
        name="tmp-classical-run",
        algorithm="classical",
        parameters={
            "dataset": "KinFaceW-I",
            "relation": "fs",
            "method": "random",
            "limit": 20,
        },
    )
    try:
        result = run_experiment(config, output_root=output_root)
        run_dir = Path(result["run_dir"])
        assert (run_dir / "result.json").exists()
        assert (run_dir / "summary.txt").exists()
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_run_benchmark_writes_summary() -> None:
    output_root = _workspace_temp_dir("benchmark")
    benchmark = BenchmarkConfig(
        name="tmp-benchmark",
        experiments=[
            ExperimentConfig(
                name="tmp-classical-benchmark-run",
                algorithm="classical",
                parameters={
                    "dataset": "KinFaceW-I",
                    "relation": "fs",
                    "method": "kfold",
                    "limit": 20,
                },
            )
        ],
    )
    try:
        result = run_benchmark(benchmark, output_root=output_root)
        run_dir = Path(result["run_dir"])
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "summary.csv").exists()
        assert result["summary_rows"][0]["algorithm"] == "classical"
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
