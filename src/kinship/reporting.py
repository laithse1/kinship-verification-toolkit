from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from kinship.paths import workspace_root


OUTPUT_ROOT = workspace_root() / "outputs"


@dataclass
class RunArtifacts:
    run_dir: Path
    result_path: Path
    metadata_path: Path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "run"


def make_run_dir(name: str, output_root: Path | None = None) -> Path:
    root = output_root or OUTPUT_ROOT
    path = root / f"{utc_timestamp()}_{slugify(name)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class Timer:
    def __enter__(self) -> "Timer":
        self.started = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.ended = perf_counter()
        self.elapsed_seconds = self.ended - self.started

