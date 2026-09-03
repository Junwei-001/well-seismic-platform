"""Small data-minimisation helpers for payloads sent to external LLMs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?i)(?<![A-Z0-9_])(?:[A-Z]:[\\/]|\\\\)[^\r\n\"'<>|,;，。；]+"
)
_POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![:/A-Za-z0-9])/(?:[^/\s\"'<>|,;，。；]+/)+[^\r\n\"'<>|,;，。；]*"
)
_PATH_LIKE_NAME = re.compile(r"(?i).+\.[A-Z0-9]{1,12}$")


def issue_local_paths(issue: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect only explicit local-path evidence already attached to an issue."""

    paths: list[str] = []
    for key in ("source", "path", "source_path", "file", "sources", "paths"):
        raw = issue.get(key)
        values: Iterable[Any]
        if isinstance(raw, (list, tuple, set)):
            values = raw
        else:
            values = (raw,)
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            if (
                "/" in text
                or "\\" in text
                or _PATH_LIKE_NAME.fullmatch(text) is not None
            ):
                paths.append(text)
    return tuple(dict.fromkeys(paths))


def sanitize_llm_text(
    value: Any,
    *,
    known_paths: Iterable[str | Path] = (),
) -> str:
    """Remove explicit and recognizable absolute local paths from one string."""

    text = str(value or "")
    variants: set[str] = set()
    for raw_path in known_paths:
        candidate = str(raw_path).strip()
        if not candidate:
            continue
        variants.update(
            {
                candidate,
                candidate.replace("\\", "/"),
                candidate.replace("/", "\\"),
            }
        )
    for candidate in sorted(variants, key=len, reverse=True):
        text = text.replace(candidate, "<local-path>")
    text = _WINDOWS_ABSOLUTE_PATH.sub("<local-path>", text)
    text = _POSIX_ABSOLUTE_PATH.sub("<local-path>", text)
    return text


def sanitize_llm_payload(
    value: Any,
    *,
    known_paths: Iterable[str | Path] = (),
) -> Any:
    """Recursively sanitize strings while preserving a JSON-like payload shape."""

    if isinstance(value, str):
        return sanitize_llm_text(value, known_paths=known_paths)
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_llm_payload(item, known_paths=known_paths)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_llm_payload(item, known_paths=known_paths) for item in value]
    return value


__all__ = ["issue_local_paths", "sanitize_llm_payload", "sanitize_llm_text"]
