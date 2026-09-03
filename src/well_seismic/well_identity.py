from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_API12_PREFIX = "API12:"
_PATH_API_TOKEN = re.compile(r"^(\d{14}|\d{12}|\d{10})(?:$|[^0-9])")


def canonical_api12(value: object) -> str | None:
    """Return one conservative US API-12 identity.

    Only the three documented source forms are accepted: API-10, API-12 and
    API-14.  API-10 is the base well identity and is equivalent to API-12 only
    when the latter has the explicit ``00`` suffix; API-14 contributes its
    first twelve digits.  Punctuation in ordinary API displays is ignored,
    while alphabetic or otherwise unexplained content is rejected.
    """

    text = str(value or "").strip()
    if not text:
        return None
    if text.upper().startswith(_API12_PREFIX):
        text = text[len(_API12_PREFIX) :].strip()
    # OOXML numeric cells can surface as ``490251120700.0``.  This is the only
    # decimal representation accepted for an identifier.
    text = re.sub(r"\.0+$", "", text)
    if re.search(r"[A-Za-z]", text):
        return None
    if re.search(r"[^0-9\s._-]", text):
        return None
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) == 10:
        digits += "00"
    elif len(digits) == 14:
        digits = digits[:12]
    elif len(digits) != 12:
        return None
    return _API12_PREFIX + digits


def canonical_api12_values(
    values: Iterable[object],
) -> tuple[list[str], list[str]]:
    """Canonicalise non-empty values and retain malformed evidence."""

    canonical: set[str] = set()
    invalid: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        identity = canonical_api12(text)
        if identity is None:
            invalid.add(text)
        else:
            canonical.add(identity)
    return sorted(canonical), sorted(invalid)


def api12_path_hints(path: str | Path) -> list[str]:
    """Extract only unambiguous leading API tokens from a LAS path.

    Public exports commonly use either an API-named parent directory or an
    API-prefixed LAS filename.  Arbitrary digit runs elsewhere in a path are
    deliberately ignored.
    """

    source = Path(path)
    raw_candidates: list[str] = []
    for value in (source.stem, source.parent.name):
        match = _PATH_API_TOKEN.match(str(value).strip())
        if match is not None:
            raw_candidates.append(match.group(1))
    canonical, _ = canonical_api12_values(raw_candidates)
    return canonical


__all__ = [
    "api12_path_hints",
    "canonical_api12",
    "canonical_api12_values",
]
