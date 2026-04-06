from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from kinship.configs import BenchmarkConfig, ExperimentConfig
from kinship.registry import run_algorithm, serialize_result
from kinship.reporting import Timer, make_run_dir, write_csv, write_json, write_text


def _format_summary_lines(payload: dict[str, Any]) -> str:
    lines = [
        f"experiment: {payload['experiment']['name']}",
        f"algorithm: {payload['experiment']['algorithm']}",
        f"duration_seconds: {payload['runtime']['duration_seconds']:.3f}",
    ]
    result = payload["result"]
    for key in (
        "dataset",
        "relation",
        "method",
        "mean_accuracy",
        "n_components",
        "variant",
        "reconstruction_error",
        "output_path",
        "model_name",
        "mode",
    ):
        if key in result:
            lines.append(f"{key}: {result[key]}")
    return "\n".join(lines) + "\n"


def run_experiment(
    config: ExperimentConfig,
    output_root: Path | None = None,
) -> dict[str, Any]:
    run_dir = make_run_dir(config.name, output_root=output_root)
    with Timer() as timer:
        result_obj = run_algorithm(config.algorithm, config.parameters)
    result = serialize_result(result_obj)
    payload = {
        "experiment": {
            "name": config.name,
            "description": config.description,
            "algorithm": config.algorithm,
            "tags": config.tags,
            "parameters": config.parameters,
            "source_path": str(config.source_path) if config.source_path else None,
        },
        "runtime": {
            "duration_seconds": timer.elapsed_seconds,
        },
        "result": result,
    }
    write_json(run_dir / "result.json", payload)
    write_text(run_dir / "summary.txt", _format_summary_lines(payload))
    return {
        "run_dir": str(run_dir),
        "payload": payload,
    }


def run_benchmark(
    config: BenchmarkConfig,
    output_root: Path | None = None,
) -> dict[str, Any]:
    benchmark_dir = make_run_dir(config.name, output_root=output_root)
    summary_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []

    for experiment in config.experiments:
        experiment_output_root = benchmark_dir / "runs"
        result = run_experiment(experiment, output_root=experiment_output_root)
        payload = result["payload"]
        run_payloads.append(payload)
        row = {
            "experiment": experiment.name,
            "algorithm": experiment.algorithm,
            "dataset": payload["result"].get("dataset", ""),
            "relation": payload["result"].get("relation", ""),
            "method": payload["result"].get("method", ""),
            "mean_accuracy": payload["result"].get("mean_accuracy", ""),
            "duration_seconds": payload["runtime"]["duration_seconds"],
            "run_dir": result["run_dir"],
        }
        summary_rows.append(row)

    benchmark_payload = {
        "benchmark": {
            "name": config.name,
            "description": config.description,
            "tags": config.tags,
            "source_path": str(config.source_path) if config.source_path else None,
        },
        "experiments": run_payloads,
    }
    write_json(benchmark_dir / "summary.json", benchmark_payload)
    write_csv(benchmark_dir / "summary.csv", summary_rows)
    write_text(
        benchmark_dir / "README.txt",
        "Benchmark outputs:\n- summary.json\n- summary.csv\n- runs/<timestamp>_<experiment>/result.json\n",
    )
    return {
        "run_dir": str(benchmark_dir),
        "summary_rows": summary_rows,
        "payload": benchmark_payload,
    }
