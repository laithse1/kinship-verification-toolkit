from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from kinship.reporting import write_csv, write_json, write_text


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _to_float(row.get("mean_roc_auc")),
            _to_float(row.get("mean_pr_auc")),
            -_to_float(row.get("mean_eer")),
            _to_float(row.get("mean_balanced_accuracy")),
            _to_float(row.get("mean_accuracy")),
        ),
        reverse=True,
    )


def _leaderboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = _sort_rows(rows)
    payload: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        payload.append(
            {
                "rank": index,
                "experiment": row.get("experiment", ""),
                "algorithm": row.get("algorithm", ""),
                "relation": row.get("relation", ""),
                "mean_accuracy": row.get("mean_accuracy", ""),
                "mean_balanced_accuracy": row.get("mean_balanced_accuracy", ""),
                "mean_roc_auc": row.get("mean_roc_auc", ""),
                "mean_pr_auc": row.get("mean_pr_auc", ""),
                "mean_eer": row.get("mean_eer", ""),
            }
        )
    return payload


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No benchmark rows available.\n"
    headers = [
        "Rank",
        "Experiment",
        "Algorithm",
        "Relation",
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
                    str(row.get("rank", "")),
                    str(row.get("experiment", "")),
                    str(row.get("algorithm", "")),
                    str(row.get("relation", "")),
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


def _plot_primary_metrics(rows: list[dict[str, Any]], output_path: Path) -> None:
    names = [str(row.get("experiment", "")) for row in rows]
    x = np.arange(len(rows))
    width = 0.18
    metrics = [
        ("Accuracy", [np.nan_to_num(_to_float(row.get("mean_accuracy")), nan=0.0) for row in rows]),
        ("Balanced Acc.", [np.nan_to_num(_to_float(row.get("mean_balanced_accuracy")), nan=0.0) for row in rows]),
        ("ROC-AUC", [np.nan_to_num(_to_float(row.get("mean_roc_auc")), nan=0.0) for row in rows]),
        ("PR-AUC", [np.nan_to_num(_to_float(row.get("mean_pr_auc")), nan=0.0) for row in rows]),
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.35), 4.8))
    for idx, (label, values) in enumerate(metrics):
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=label)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Benchmark Scorecard")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend(ncols=2, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_ranking_metrics(rows: list[dict[str, Any]], output_path: Path) -> None:
    names = [str(row.get("experiment", "")) for row in rows]
    roc = [np.nan_to_num(_to_float(row.get("mean_roc_auc")), nan=0.0) for row in rows]
    pr = [np.nan_to_num(_to_float(row.get("mean_pr_auc")), nan=0.0) for row in rows]
    eer = [np.nan_to_num(_to_float(row.get("mean_eer")), nan=0.0) for row in rows]
    x = np.arange(len(rows))
    fig, ax1 = plt.subplots(figsize=(max(8, len(rows) * 1.35), 4.8))
    ax1.plot(x, roc, marker="o", linewidth=2, label="ROC-AUC")
    ax1.plot(x, pr, marker="s", linewidth=2, label="PR-AUC")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("AUC")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, eer, marker="^", linewidth=2, color="#b24a2f", label="EER")
    ax2.set_ylim(0.0, max(0.4, max(eer) * 1.15))
    ax2.set_ylabel("EER")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower left")
    ax1.set_title("Ranking-Oriented Verification Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_benchmark_report(benchmark_dir: Path) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    summary_csv = benchmark_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found in benchmark directory: {benchmark_dir}")

    rows = _read_summary_rows(summary_csv)
    leaderboard = _leaderboard_rows(rows)
    report_dir = benchmark_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    write_csv(report_dir / "leaderboard.csv", leaderboard)
    write_text(report_dir / "leaderboard.md", _markdown_table(leaderboard))
    _plot_primary_metrics(rows, report_dir / "figure_scorecard.png")
    _plot_ranking_metrics(rows, report_dir / "figure_ranking_metrics.png")

    payload = {
        "benchmark_dir": str(benchmark_dir),
        "report_dir": str(report_dir),
        "experiment_count": len(rows),
        "leaderboard": leaderboard,
        "artifacts": {
            "leaderboard_csv": str(report_dir / "leaderboard.csv"),
            "leaderboard_md": str(report_dir / "leaderboard.md"),
            "figure_scorecard": str(report_dir / "figure_scorecard.png"),
            "figure_ranking_metrics": str(report_dir / "figure_ranking_metrics.png"),
        },
    }
    write_json(report_dir / "report.json", payload)
    return payload
