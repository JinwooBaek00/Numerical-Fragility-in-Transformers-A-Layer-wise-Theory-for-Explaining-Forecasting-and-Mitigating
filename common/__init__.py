"""Shared utilities for NFT camera-ready experiments."""

from .artifacts import (
    copy_artifact_to_outputs,
    save_json_artifact,
    save_matplotlib_figure,
    save_text_artifact,
)
from .config import dump_json, load_config
from .git_state import collect_git_state
from .gpt2_manual import (
    manual_attention_forward,
    manual_block_forward,
    manual_forward_with_prefixes,
    manual_patched_forward,
)
from .run import RunContext, create_run_context
from .summary import REQUIRED_SUMMARY_SECTIONS, render_summary, validate_summary_sections
from .tabular import CsvLogger, append_rows, write_rows

__all__ = [
    "CsvLogger",
    "REQUIRED_SUMMARY_SECTIONS",
    "RunContext",
    "append_rows",
    "collect_git_state",
    "copy_artifact_to_outputs",
    "create_run_context",
    "dump_json",
    "load_config",
    "manual_attention_forward",
    "manual_block_forward",
    "manual_forward_with_prefixes",
    "manual_patched_forward",
    "render_summary",
    "save_json_artifact",
    "save_matplotlib_figure",
    "save_text_artifact",
    "validate_summary_sections",
    "write_rows",
]
