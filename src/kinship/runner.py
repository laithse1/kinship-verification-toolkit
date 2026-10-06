from __future__ import annotations

from pathlib import Path
from typing import Any

from kinship.configs import BenchmarkConfig, ExperimentConfig
from kinship.registry import run_algorithm, serialize_result
from kinship.reporting import Timer, make_run_dir, write_csv, write_json, write_text


SUMMARY_KEYS = (
    "dataset",
    "relation",
    "method",
    "backbone",
    "global_backbone",
    "fusion_model",
    "fusion_train_scope",
    "local_encoder",
    "alignment_mode",
    "protocol",
    "mode",
    "model_name",
    "variant",
    "support_relations",
    "model_selection_metric",
    "ranking_loss_weight",
    "mean_accuracy",
    "mean_balanced_accuracy",
    "mean_roc_auc",
    "mean_pr_auc",
    "mean_eer",
    "n_components",
    "global_feature_dim",
    "num_heads",
    "relation_dim",
    "reconstruction_error",
    "output_path",
)


def _format_summary_lines(payload: dict[str, Any]) -> str:
    lines = [
        f"experiment: {payload['experiment']['name']}",
        f"algorithm: {payload['experiment']['algorithm']}",
        f"duration_seconds: {payload['runtime']['duration_seconds']:.3f}",
    ]
    result = payload["result"]
    for key in SUMMARY_KEYS:
        if key in result:
            lines.append(f"{key}: {result[key]}")
    return "\n".join(lines) + "\n"


def _float_or_none(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def _relation_winners(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        relation = str(row.get("relation", "")).strip()
        if not relation:
            continue
        grouped.setdefault(relation, []).append(row)

    winners: list[dict[str, Any]] = []
    for relation, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                _float_or_none(row.get("mean_roc_auc")) or float("-inf"),
                _float_or_none(row.get("mean_pr_auc")) or float("-inf"),
                -(_float_or_none(row.get("mean_eer")) or float("inf")),
                _float_or_none(row.get("mean_balanced_accuracy")) or float("-inf"),
                _float_or_none(row.get("mean_accuracy")) or float("-inf"),
            ),
            reverse=True,
        )
        best = ranked[0]
        winners.append(
            {
                "relation": relation,
                "winner_experiment": best.get("experiment", ""),
                "algorithm": best.get("algorithm", ""),
                "mean_accuracy": best.get("mean_accuracy", ""),
                "mean_balanced_accuracy": best.get("mean_balanced_accuracy", ""),
                "mean_roc_auc": best.get("mean_roc_auc", ""),
                "mean_pr_auc": best.get("mean_pr_auc", ""),
                "mean_eer": best.get("mean_eer", ""),
                "run_dir": best.get("run_dir", ""),
            }
        )
    return winners


def _format_winner_lines(winners: list[dict[str, Any]]) -> str:
    if not winners:
        return "No relation-wise winners available.\n"
    lines = ["Relation-wise winners:"]
    for item in winners:
        lines.append(
            f"- {item['relation']}: {item['winner_experiment']} "
            f"(bal_acc={item['mean_balanced_accuracy']}, roc_auc={item['mean_roc_auc']}, "
            f"pr_auc={item['mean_pr_auc']}, eer={item['mean_eer']})"
        )
    return "\n".join(lines) + "\n"


def _paper_summary_rows(winners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "relation": item.get("relation", ""),
            "winner_experiment": item.get("winner_experiment", ""),
            "algorithm": item.get("algorithm", ""),
            "mean_accuracy": item.get("mean_accuracy", ""),
            "mean_balanced_accuracy": item.get("mean_balanced_accuracy", ""),
            "mean_roc_auc": item.get("mean_roc_auc", ""),
            "mean_pr_auc": item.get("mean_pr_auc", ""),
            "mean_eer": item.get("mean_eer", ""),
        }
        for item in winners
    ]


def _format_paper_summary_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No paper summary available.\n"
    headers = [
        "Relation",
        "Winner",
        "Accuracy",
        "Balanced Acc.",
        "ROC-AUC",
        "PR-AUC",
        "EER",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("relation", "")),
                    str(row.get("winner_experiment", "")),
                    str(row.get("mean_accuracy", "")),
                    str(row.get("mean_balanced_accuracy", "")),
                    str(row.get("mean_roc_auc", "")),
                    str(row.get("mean_pr_auc", "")),
                    str(row.get("mean_eer", "")),
                ]
            )
            + " |"
        )
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
            "backbone": payload["result"].get("backbone", ""),
            "global_backbone": payload["result"].get("global_backbone", ""),
            "fusion_model": payload["result"].get("fusion_model", ""),
            "fusion_train_scope": payload["result"].get("fusion_train_scope", ""),
            "local_encoder": payload["result"].get("local_encoder", ""),
            "alignment_mode": payload["result"].get("alignment_mode", ""),
            "model_name": payload["result"].get("model_name", ""),
            "support_relations": payload["result"].get("support_relations", ""),
            "mean_accuracy": payload["result"].get("mean_accuracy", ""),
            "mean_balanced_accuracy": payload["result"].get("mean_balanced_accuracy", ""),
            "mean_roc_auc": payload["result"].get("mean_roc_auc", ""),
            "mean_pr_auc": payload["result"].get("mean_pr_auc", ""),
            "mean_eer": payload["result"].get("mean_eer", ""),
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
    winners = _relation_winners(summary_rows)
    paper_summary_rows = _paper_summary_rows(winners)
    benchmark_payload["relation_winners"] = winners
    write_json(benchmark_dir / "summary.json", benchmark_payload)
    write_csv(benchmark_dir / "summary.csv", summary_rows)
    write_json(benchmark_dir / "relation_winners.json", winners)
    write_csv(benchmark_dir / "paper_summary.csv", paper_summary_rows)
    write_text(
        benchmark_dir / "README.txt",
        "Benchmark outputs:\n- summary.json\n- summary.csv\n- relation_winners.json\n- relation_winners.txt\n- paper_summary.csv\n- paper_summary.md\n- runs/<timestamp>_<experiment>/result.json\n",
    )
    write_text(benchmark_dir / "relation_winners.txt", _format_winner_lines(winners))
    write_text(benchmark_dir / "paper_summary.md", _format_paper_summary_markdown(paper_summary_rows))
    return {
        "run_dir": str(benchmark_dir),
        "summary_rows": summary_rows,
        "relation_winners": winners,
        "paper_summary_rows": paper_summary_rows,
        "payload": benchmark_payload,
    }
