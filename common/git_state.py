from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_git_state(cwd: str | Path) -> dict[str, Any]:
    """Collect a lightweight snapshot of the git or workspace state."""
    root = cwd if isinstance(cwd, Path) else Path(cwd)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    commit = _run_git(["rev-parse", "HEAD"], root)
    status_output = _run_git(["status", "--short"], root)

    if branch is None or commit is None:
        return {
            "kind": "workspace",
            "root": str(root.resolve()),
            "git_available": False,
        }

    dirty = bool(status_output)
    return {
        "kind": "git",
        "root": str(root.resolve()),
        "git_available": True,
        "branch": branch,
        "commit": commit,
        "dirty": dirty,
        "status_short": status_output.splitlines() if status_output else [],
    }
