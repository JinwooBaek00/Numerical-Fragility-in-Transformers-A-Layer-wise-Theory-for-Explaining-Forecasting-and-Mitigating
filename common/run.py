from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import dump_json
from .git_state import collect_git_state
from .summary import render_summary
from .tabular import append_rows, write_rows


REQUIRED_METADATA_FIELDS = (
    "experiment_id",
    "run_id",
    "created_at",
    "git_or_workspace_state",
    "model_name",
    "dataset_name",
    "precision",
    "seed",
    "sequence_length",
    "status",
)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def _iso_timestamp() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _sanitize_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tag).strip("_") or "run"


@dataclass(frozen=True)
class RunPaths:
    experiment_dir: Path
    outputs_dir: Path
    runs_dir: Path
    run_dir: Path
    config_path: Path
    metadata_path: Path
    stdout_path: Path
    summary_path: Path
    metrics_path: Path


class RunContext:
    """Owns the file contract for one experiment run."""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    @property
    def run_id(self) -> str:
        return self.paths.run_dir.name

    def append_stdout(self, text: str) -> Path:
        self.paths.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.stdout_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            if text and not text.endswith("\n"):
                handle.write("\n")
        return self.paths.stdout_path

    def write_metrics(self, metrics: Mapping[str, Any]) -> Path:
        return dump_json(self.paths.metrics_path, metrics)

    def write_summary(self, sections: Mapping[str, object]) -> Path:
        body = render_summary(sections)
        self.paths.summary_path.write_text(body, encoding="utf-8", newline="\n")
        return self.paths.summary_path

    def write_rows(
        self,
        filename: str,
        rows: list[Mapping[str, Any]],
        *,
        fieldnames: list[str] | tuple[str, ...] | None = None,
    ) -> Path:
        return write_rows(self.paths.run_dir / filename, rows, fieldnames=fieldnames)

    def append_rows(
        self,
        filename: str,
        rows: list[Mapping[str, Any]],
        *,
        fieldnames: list[str] | tuple[str, ...] | None = None,
    ) -> Path:
        return append_rows(self.paths.run_dir / filename, rows, fieldnames=fieldnames)

    def update_metadata(self, updates: Mapping[str, Any]) -> Path:
        current = self.read_metadata()
        current.update(dict(updates))
        return dump_json(self.paths.metadata_path, current)

    def read_metadata(self) -> dict[str, Any]:
        import json

        with self.paths.metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"metadata.json must contain an object: {self.paths.metadata_path}")
        return payload

    def mark_completed(self, *, status: str = "completed", extra_metadata: Mapping[str, Any] | None = None) -> Path:
        updates = {"status": status, "completed_at": _iso_timestamp()}
        if extra_metadata:
            updates.update(dict(extra_metadata))
        return self.update_metadata(updates)


def create_run_context(
    experiment_dir: str | Path,
    *,
    short_tag: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    workspace_root: str | Path | None = None,
    run_id: str | None = None,
) -> RunContext:
    experiment_path = _as_path(experiment_dir)
    outputs_dir = experiment_path / "outputs"
    runs_dir = experiment_path / "runs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    tag = _sanitize_tag(short_tag)
    resolved_run_id = run_id or f"{_utc_timestamp()}__{tag}"
    run_dir = runs_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    paths = RunPaths(
        experiment_dir=experiment_path,
        outputs_dir=outputs_dir,
        runs_dir=runs_dir,
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        metadata_path=run_dir / "metadata.json",
        stdout_path=run_dir / "stdout.log",
        summary_path=run_dir / "summary.md",
        metrics_path=run_dir / "metrics.json",
    )
    context = RunContext(paths)

    dump_json(paths.config_path, config)

    root = _as_path(workspace_root) if workspace_root is not None else experiment_path.parent
    default_metadata = {
        "experiment_id": experiment_path.name,
        "run_id": resolved_run_id,
        "created_at": _iso_timestamp(),
        "git_or_workspace_state": collect_git_state(root),
        "model_name": metadata.get("model_name", "unknown"),
        "dataset_name": metadata.get("dataset_name", "unknown"),
        "precision": metadata.get("precision", "unknown"),
        "seed": metadata.get("seed", "unknown"),
        "sequence_length": metadata.get("sequence_length", "unknown"),
        "status": metadata.get("status", "created"),
    }
    merged_metadata = {**default_metadata, **dict(metadata)}
    for field in REQUIRED_METADATA_FIELDS:
        if field not in merged_metadata:
            raise ValueError(f"Missing required metadata field: {field}")
    dump_json(paths.metadata_path, merged_metadata)

    paths.stdout_path.write_text("", encoding="utf-8", newline="\n")
    return context
