"""Receiver-side text normalization shared by ASR wrappers and reports."""

from __future__ import annotations

import re
import unicodedata


_SPACE = re.compile(r"\s+")
_ENGLISH_PUNCTUATION = re.compile(r"[^a-z0-9' ]+")


def normalize_english(text: str) -> str:
    """Return the conservative comparison form used by Whisper reports.

    Candidate generation still receives the original transcription; this
    function is only for stable measurements and evidence summaries.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    value = unicodedata.normalize("NFKC", text).lower()
    value = _ENGLISH_PUNCTUATION.sub(" ", value)
    return _SPACE.sub(" ", value).strip()
