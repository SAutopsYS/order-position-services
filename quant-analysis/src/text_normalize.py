"""Deterministic subject/headline normalization for clustering."""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_SPACE_RE = re.compile(r"\s+")
_BOILERPLATE = (
    "announcement under regulation 30 lodr",
    "announcement under regulation 30 of the sebi listing obligations and disclosure requirements regulations 2015",
    "compliances",
)


def normalize_text(value: object) -> str:
    """Lowercase, strip punctuation, collapse spaces, drop light boilerplate."""
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    text = text.replace("''", "'").lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    for phrase in _BOILERPLATE:
        text = text.replace(phrase, " ")
    return _SPACE_RE.sub(" ", text).strip()
