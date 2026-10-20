"""Small text-domain receiver metrics without model dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from .normalize import normalize_english


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, left in enumerate(reference, start=1):
        current = [index]
        for column, right in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class TranscriptMetrics:
    reference_words: int
    hypothesis_words: int
    word_errors: int
    word_error_rate: float
    normalized_exact_match: bool


def compare_transcripts(reference: str, hypothesis: str) -> TranscriptMetrics:
    left = normalize_english(reference).split()
    right = normalize_english(hypothesis).split()
    errors = _edit_distance(left, right)
    return TranscriptMetrics(
        reference_words=len(left),
        hypothesis_words=len(right),
        word_errors=errors,
        word_error_rate=(errors / len(left) if left else float(bool(right))),
        normalized_exact_match=left == right,
    )
