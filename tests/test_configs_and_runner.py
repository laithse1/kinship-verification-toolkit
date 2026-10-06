from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from kinship.benchmark_report import generate_benchmark_report
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


def test_load_forestnn_experiment_config() -> None:
    config = load_experiment_config("forestnn-fs-smoke")
    assert config.algorithm == "forestnn"
    assert config.parameters["relation"] == "fs"


def test_load_facornet_experiment_config() -> None:
    config = load_experiment_config("facornet-fs-smoke")
    assert config.algorithm == "facornet"
    assert config.parameters["relation"] == "fs"
    assert config.parameters["fusion_train_scope"] == "trainpool"


def test_load_facornet_relation_quality_configs() -> None:
    fd = load_experiment_config("facornet-fd-quality")
    md = load_experiment_config("facornet-md-quality")
    ms = load_experiment_config("facornet-ms-quality")
    assert fd.parameters["relation"] == "fd"
    assert md.parameters["relation"] == "md"
    assert ms.parameters["relation"] == "ms"


def test_load_relation_aware_experiment_config() -> None:
    config = load_experiment_config("relation-aware-fs-smoke")
    assert config.algorithm == "relation-aware"
    assert config.parameters["relation"] == "fs"
    assert config.parameters["support_relations"] == "all"


def test_load_forestnn_quality_experiment_config() -> None:
    config = load_experiment_config("forestnn-fs-quality")
    assert config.algorithm == "forestnn"
    assert config.parameters["global_backbone"] == "vggface+vggf-precomputed"
    assert config.parameters["local_encoder"] == "residual"
    assert config.parameters["alignment_mode"] == "adaptive"
    assert config.parameters["fusion_model"] == "engineered-logreg"


def test_load_forestnn_cpu_practical_experiment_config() -> None:
    config = load_experiment_config("forestnn-fs-cpu-practical")
    assert config.algorithm == "forestnn"
    assert config.parameters["num_epochs"] == 1
    assert config.parameters["limit_pairs"] == 120


def test_load_relation_specific_quality_experiment_configs() -> None:
    fd = load_experiment_config("forestnn-fd-quality")
    md = load_experiment_config("forestnn-md-quality")
    ms = load_experiment_config("forestnn-ms-quality")
    assert fd.parameters["relation"] == "fd"
    assert md.parameters["relation"] == "md"
    assert ms.parameters["relation"] == "ms"


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


def test_load_forestnn_benchmark_config() -> None:
    config = load_benchmark_config("forestnn-smoke")
    assert config.name == "forestnn-smoke"
    assert len(config.experiments) == 4
    assert {experiment.algorithm for experiment in config.experiments} == {"forestnn"}


def test_load_forestnn_alignment_ablation_benchmark_config() -> None:
    config = load_benchmark_config("forestnn-alignment-ablation")
    assert config.name == "forestnn-alignment-ablation"
    assert len(config.experiments) == 2
    assert {experiment.parameters["alignment_mode"] for experiment in config.experiments} == {"fixed", "adaptive"}


def test_load_forestnn_all_relations_quality_benchmark_config() -> None:
    config = load_benchmark_config("forestnn-kinfacew-all-relations-quality")
    assert config.name == "forestnn-kinfacew-all-relations-quality"
    assert len(config.experiments) == 4
    assert {experiment.parameters["relation"] for experiment in config.experiments} == {"fd", "fs", "md", "ms"}


def test_load_forestnn_all_relations_best_known_benchmark_config() -> None:
    config = load_benchmark_config("forestnn-kinfacew-all-relations-best-known")
    assert config.name == "forestnn-kinfacew-all-relations-best-known"
    assert len(config.experiments) == 4
    assert {experiment.parameters["relation"] for experiment in config.experiments} == {"fd", "fs", "md", "ms"}
    assert {experiment.parameters["global_backbone"] for experiment in config.experiments} == {
        "vggface+vggf-precomputed"
    }


def test_load_forestnn_relation_wise_candidates_benchmark_config() -> None:
    config = load_benchmark_config("forestnn-relation-wise-candidates")
    assert config.name == "forestnn-relation-wise-candidates"
    assert len(config.experiments) == 8
    assert {experiment.parameters["relation"] for experiment in config.experiments} == {"fd", "fs", "md", "ms"}


def test_load_modern_kinship_all_relations_benchmark_config() -> None:
    config = load_benchmark_config("modern-kinship-all-relations")
    assert config.name == "modern-kinship-all-relations"
    assert len(config.experiments) == 12
    assert {experiment.algorithm for experiment in config.experiments} == {"forestnn", "facornet", "relation-aware"}


def test_load_best_modern_per_relation_benchmark_config() -> None:
    config = load_benchmark_config("best-modern-per-relation")
    assert config.name == "best-modern-per-relation"
    assert len(config.experiments) == 4
    assert [experiment.algorithm for experiment in config.experiments] == [
        "relation-aware",
        "relation-aware",
        "relation-aware",
        "facornet",
    ]


def test_load_relation_aware_all_relations_benchmark_config() -> None:
    config = load_benchmark_config("relation-aware-kinfacew-all-relations")
    assert config.name == "relation-aware-kinfacew-all-relations"
    assert len(config.experiments) == 4
    assert {experiment.algorithm for experiment in config.experiments} == {"relation-aware"}
    assert {experiment.parameters["relation"] for experiment in config.experiments} == {"fd", "fs", "md", "ms"}


def test_load_relation_aware_ms_tuning_benchmark_config() -> None:
    config = load_benchmark_config("relation-aware-ms-tuning")
    assert config.name == "relation-aware-ms-tuning"
    assert len(config.experiments) == 5
    assert {experiment.algorithm for experiment in config.experiments} == {"relation-aware"}
    assert {experiment.parameters["relation"] for experiment in config.experiments} == {"ms"}


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
        assert (run_dir / "relation_winners.json").exists()
        assert (run_dir / "relation_winners.txt").exists()
        assert (run_dir / "paper_summary.csv").exists()
        assert (run_dir / "paper_summary.md").exists()
        assert result["summary_rows"][0]["algorithm"] == "classical"
        assert result["relation_winners"][0]["relation"] == "fs"
        assert result["paper_summary_rows"][0]["relation"] == "fs"
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_generate_benchmark_report_writes_artifacts() -> None:
    output_root = _workspace_temp_dir("benchmark-report")
    benchmark = BenchmarkConfig(
        name="tmp-report-benchmark",
        experiments=[
            ExperimentConfig(
                name="tmp-classical-report-run",
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
        payload = generate_benchmark_report(Path(result["run_dir"]))
        report_dir = Path(payload["report_dir"])
        assert (report_dir / "leaderboard.csv").exists()
        assert (report_dir / "leaderboard.md").exists()
        assert (report_dir / "figure_scorecard.png").exists()
        assert (report_dir / "figure_ranking_metrics.png").exists()
        assert payload["experiment_count"] == 1
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
