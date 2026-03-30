from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file into a plain dictionary."""
    config_path = _as_path(path)
    if config_path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported config format: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a JSON object: {config_path}")
    return data


def dump_json(path: str | Path, payload: Mapping[str, Any] | list[Any], *, indent: int = 2) -> Path:
    """Write JSON with stable formatting and create parents automatically."""
    json_path = _as_path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)
        handle.write("\n")
    return json_path
