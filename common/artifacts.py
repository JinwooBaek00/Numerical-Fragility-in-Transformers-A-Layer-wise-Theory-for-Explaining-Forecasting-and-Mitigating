from __future__ import annotations

from pathlib import Path
from shutil import copy2
from typing import Any, Mapping

from .config import dump_json


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _ensure_outputs_dir(outputs_dir: str | Path) -> Path:
    path = _as_path(outputs_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_artifact_to_outputs(
    source_path: str | Path,
    outputs_dir: str | Path,
    *,
    output_name: str | None = None,
) -> Path:
    source = _as_path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    destination_dir = _ensure_outputs_dir(outputs_dir)
    destination = destination_dir / (output_name or source.name)
    copy2(source, destination)
    return destination


def save_text_artifact(outputs_dir: str | Path, name: str, content: str) -> Path:
    destination_dir = _ensure_outputs_dir(outputs_dir)
    path = destination_dir / name
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def save_json_artifact(outputs_dir: str | Path, name: str, payload: Mapping[str, Any] | list[Any]) -> Path:
    destination_dir = _ensure_outputs_dir(outputs_dir)
    return dump_json(destination_dir / name, payload)


def save_matplotlib_figure(
    figure: Any,
    outputs_dir: str | Path,
    name: str,
    *,
    dpi: int = 200,
    close: bool = False,
) -> Path:
    destination_dir = _ensure_outputs_dir(outputs_dir)
    path = destination_dir / name
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.close(figure)
        except ImportError:
            pass
    return path
