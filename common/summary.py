from __future__ import annotations

from collections.abc import Iterable, Mapping


REQUIRED_SUMMARY_SECTIONS = (
    "goal",
    "setup",
    "key_metrics",
    "pass_fail_verdict",
    "anomalies",
    "follow_up",
)


def _normalize_section_body(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        lines = []
        for item in value:
            text = str(item).strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)
    return str(value).strip()


def validate_summary_sections(sections: Mapping[str, object]) -> dict[str, str]:
    """Validate and normalize the required summary sections."""
    normalized: dict[str, str] = {}
    missing = [key for key in REQUIRED_SUMMARY_SECTIONS if key not in sections]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required summary sections: {joined}")

    for key in REQUIRED_SUMMARY_SECTIONS:
        normalized[key] = _normalize_section_body(sections.get(key))
    return normalized


def render_summary(sections: Mapping[str, object]) -> str:
    """Render a summary.md body with the required section order."""
    normalized = validate_summary_sections(sections)
    blocks = []
    for key in REQUIRED_SUMMARY_SECTIONS:
        title = key.replace("_", " ").title()
        body = normalized[key] or "_None_"
        blocks.append(f"## {title}\n\n{body}")
    return "\n\n".join(blocks) + "\n"
