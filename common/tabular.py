from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _infer_fieldnames(rows: list[Mapping[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError("Cannot infer CSV fieldnames from empty rows.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: list[str] | tuple[str, ...] | None = None,
) -> Path:
    csv_path = _as_path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = _infer_fieldnames(materialized)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return csv_path


def append_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: list[str] | tuple[str, ...] | None = None,
) -> Path:
    csv_path = _as_path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        return csv_path

    file_exists = csv_path.exists()
    if fieldnames is None:
        if file_exists:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                try:
                    fieldnames = next(reader)
                except StopIteration as exc:
                    raise ValueError(f"Existing CSV has no header: {csv_path}") from exc
        else:
            fieldnames = _infer_fieldnames(materialized)

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(materialized)
    return csv_path


class CsvLogger:
    """Append-only CSV logger with fixed columns."""

    def __init__(self, path: str | Path, fieldnames: list[str] | tuple[str, ...]) -> None:
        self.path = _as_path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Mapping[str, Any]) -> Path:
        return append_rows(self.path, [row], fieldnames=self.fieldnames)

    def log_many(self, rows: Iterable[Mapping[str, Any]]) -> Path:
        return append_rows(self.path, rows, fieldnames=self.fieldnames)
