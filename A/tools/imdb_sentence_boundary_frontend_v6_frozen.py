#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_CANDIDATES = 4096

# Frozen after the IMDB row 1 result was revealed and before row 2 is read.
# These are generic sentence-start words. The list contains no sender text,
# payload information, token IDs, CRC value, or reference text.
SENTENCE_STARTERS = (
    "the",
    "this",
    "that",
    "it",
    "we",
    "they",
    "he",
    "she",
)

STARTER_PATTERN = re.compile(
    r"\s+("
    + "|".join(re.escape(word) for word in SENTENCE_STARTERS)
    + r")\b",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def candidate_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def resolve_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_complete_source_report(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_report = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(source_report, dict):
        raise TypeError("Source candidate report must be a JSON object.")

    if "candidate_count" not in source_report:
        raise KeyError("Source report has no candidate_count field.")

    declared_count = source_report["candidate_count"]

    if isinstance(declared_count, bool) or not isinstance(
        declared_count,
        int,
    ):
        raise TypeError("Source candidate_count must be an integer.")

    source_candidates = source_report.get("candidates")

    if not isinstance(source_candidates, list):
        raise TypeError("Source candidates must be a list.")

    if declared_count != len(source_candidates):
        raise RuntimeError(
            "Source candidate_count does not match the candidate list: "
            f"declared={declared_count}, actual={len(source_candidates)}."
        )

    if "truncated" not in source_report:
        raise KeyError("Source report has no truncated field.")

    source_truncated = source_report["truncated"]

    if not isinstance(source_truncated, bool):
        raise TypeError("Source truncated field must be Boolean.")

    if source_truncated:
        raise RuntimeError(
            "Refusing to expand an already-truncated source candidate space."
        )

    validated: list[dict[str, Any]] = []

    for index, candidate in enumerate(source_candidates):
        if not isinstance(candidate, dict):
            raise TypeError(f"Source candidate {index} is not an object.")

        if not isinstance(candidate.get("text"), str):
            raise TypeError(f"Source candidate {index} has no string text.")

        transforms = candidate.get("transforms", [])

        if not isinstance(transforms, list) or not all(
            isinstance(item, str) for item in transforms
        ):
            raise TypeError(
                f"Source candidate {index} has invalid transforms."
            )

        repair_count = candidate.get("repair_count", 0)

        if isinstance(repair_count, bool) or not isinstance(
            repair_count,
            int,
        ):
            raise TypeError(
                f"Source candidate {index} has invalid repair_count."
            )

        validated.append(candidate)

    return source_report, validated


def find_boundary_occurrences(
    text: str,
) -> list[tuple[int, int, str, str]]:
    occurrences: list[tuple[int, int, str, str]] = []

    for match in STARTER_PATTERN.finditer(text):
        observed_word = match.group(1)

        # Only restore a boundary when ASR emitted the starter in lowercase.
        # Existing title case is retained.
        if observed_word != observed_word.lower():
            continue

        prefix = text[: match.start()]

        # Do not insert a boundary before the first lexical word.
        if not prefix.strip():
            continue

        previous_character = prefix.rstrip()[-1]

        # A sentence boundary is already present.
        if previous_character in ".!?":
            continue

        replacement = ". " + observed_word[:1].upper() + observed_word[1:]

        occurrences.append(
            (
                match.start(),
                match.end(),
                replacement,
                (
                    "internal-sentence-boundary:"
                    f"{observed_word}->{replacement}"
                ),
            )
        )

    return occurrences


def apply_occurrence_subset(
    text: str,
    selected: tuple[tuple[int, int, str, str], ...],
) -> tuple[str, list[str]]:
    repaired = text

    for start, end, replacement, _ in reversed(selected):
        repaired = repaired[:start] + replacement + repaired[end:]

    transforms = [name for _, _, _, name in selected]
    return repaired, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Complete bounded candidate JSON emitted by frozen frontend v4.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")

    if args.max_candidates < 1:
        raise ValueError("--max-candidates must be at least 1.")

    source_report, source_candidates = load_complete_source_report(input_path)

    output_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    generated_variant_count = 0
    truncated = False

    def append_candidate(
        text: str,
        *,
        source: dict[str, Any],
        transforms: list[str],
        generation_stage: str,
        added_repairs: int,
    ) -> bool:
        nonlocal truncated

        if text in seen:
            return True

        if len(output_candidates) >= args.max_candidates:
            truncated = True
            return False

        seen.add(text)
        output_candidates.append(
            {
                "rank": len(output_candidates),
                "candidate_id": candidate_id(text),
                "generation_stage": generation_stage,
                "repair_count": int(source.get("repair_count", 0))
                + added_repairs,
                "transforms": transforms,
                "text": text,
            }
        )
        return True

    # Preserve every unique pre-v6 candidate before adding any v6 variant.
    for candidate in source_candidates:
        if not append_candidate(
            candidate["text"],
            source=candidate,
            transforms=list(candidate.get("transforms", [])),
            generation_stage=str(
                candidate.get("generation_stage", "pre-v6")
            ),
            added_repairs=0,
        ):
            break

    # Expand subsets in deterministic, increasing repair-count order.
    if not truncated:
        for candidate in source_candidates:
            text = candidate["text"]
            occurrences = find_boundary_occurrences(text)

            for subset_size in range(1, len(occurrences) + 1):
                for selected in itertools.combinations(
                    occurrences,
                    subset_size,
                ):
                    repaired, new_transforms = apply_occurrence_subset(
                        text,
                        selected,
                    )
                    generated_variant_count += 1
                    transforms = list(
                        candidate.get("transforms", [])
                    ) + new_transforms

                    if not append_candidate(
                        repaired,
                        source=candidate,
                        transforms=transforms,
                        generation_stage=(
                            "v6-frozen-internal-sentence-boundary"
                        ),
                        added_repairs=subset_size,
                    ):
                        break

                if truncated:
                    break

            if truncated:
                break

    report = {
        "status": "IMDB_SENTENCE_BOUNDARY_FRONTEND_V6_FROZEN_COMPLETE",
        "development_only": False,
        "frozen_for_prospective_blind_test": True,
        "official_blind_result": False,
        "introduced_after_imdb_row_1_reveal": True,
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "declared_candidate_count": source_report["candidate_count"],
            "candidate_count": len(source_candidates),
            "truncated": source_report["truncated"],
            "complete": True,
        },
        "max_candidates": args.max_candidates,
        "candidate_count": len(output_candidates),
        "generated_variant_attempts": generated_variant_count,
        "truncated": truncated,
        "rule": {
            "name": "generic-lowercase-sentence-starter-boundary-lattice",
            "sentence_starters": list(SENTENCE_STARTERS),
            "subset_order": "increasing-repair-count",
        },
        "constraints": {
            "loads_language_model": False,
            "uses_crc": False,
            "uses_payload": False,
            "uses_sender_stegotext": False,
            "uses_sender_token_ids": False,
            "uses_reference_text": False,
            "deterministic": True,
            "bounded_candidate_count": True,
            "requires_complete_source_candidate_space": True,
        },
        "candidates": output_candidates,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": report["status"],
                "source_candidates": len(source_candidates),
                "candidate_count": len(output_candidates),
                "generated_variant_attempts": generated_variant_count,
                "truncated": truncated,
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()