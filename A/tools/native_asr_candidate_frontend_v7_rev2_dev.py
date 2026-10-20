#!/usr/bin/env python
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

V3_PATH = (
    PROJECT_ROOT
    / "tools"
    / "native_asr_candidate_frontend_prospective_bounded_v3.py"
)


def load_v3():
    spec = importlib.util.spec_from_file_location(
        "native_candidate_frontend_v3_for_v4_dev",
        V3_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load v3 frontend: {V3_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sentence_boundary_variants(
    text: str,
) -> Iterable[tuple[str, str]]:
    """
    Recover sentence boundaries commonly normalized by ASR.

    Examples:
        ", but" -> ". But"
        ", however" -> ". However"

    This transformation uses only received ASR text. It does not use
    sender text, token IDs, payload, CRC, or reference transcription.
    """

    pattern = re.compile(
        r",\s+(but|however|yet|nevertheless)\b",
        flags=re.IGNORECASE,
    )

    occurrences = []

    for match in pattern.finditer(text):
        word = match.group(1)
        replacement = ". " + word[:1].upper() + word[1:].lower()

        occurrences.append(
            (
                match.start(),
                match.end(),
                replacement,
                (
                    "sentence-boundary:"
                    f"{match.group(0)}->{replacement}"
                ),
            )
        )

    # Generate individual and combined replacements.
    count = len(occurrences)

    for mask in range(1, 1 << count):
        selected = [
            occurrences[index]
            for index in range(count)
            if mask & (1 << index)
        ]

        repaired = text

        for start, end, replacement, _ in reversed(selected):
            repaired = (
                repaired[:start]
                + replacement
                + repaired[end:]
            )

        transform_name = "|".join(
            item[3] for item in selected
        )

        yield repaired, transform_name


def terminal_punctuation_variants(
    text: str,
) -> Iterable[tuple[str, str]]:
    """
    Enumerate bounded terminal punctuation alternatives because TTS
    generally does not preserve an exact terminal punctuation token.
    """

    body = text.rstrip()
    trailing_whitespace = text[len(body):]

    if not body:
        return

    punctuation = (".", ",", "!", "?", ";", ":", "")
    observed = body[-1] if body[-1] in ".,!?;:" else ""
    stem = body[:-1] if observed else body

    for replacement in punctuation:
        if replacement == observed:
            continue

        repaired = stem + replacement + trailing_whitespace

        yield (
            repaired,
            (
                "terminal-punctuation:"
                f"{observed or '<none>'}"
                f"->{replacement or '<none>'}"
            ),
        )


VOWELS = 'aeiou'
MAX_VOWEL_CLUSTER_VARIANTS_PER_TEXT = 8
MIN_VOWELS_FOR_CLUSTER_INSERTION = 3

WORD_PATTERN = re.compile(
    r'\b[A-Za-z]{4,16}\b',
)

CONTRACTION_PATTERN = re.compile(
    r'\b([A-Za-z]+)\s+(s|t|d|m|re|ve|ll)\b',
    flags=re.IGNORECASE,
)


def vowel_cluster_insertion_variants(
    text: str,
) -> Iterable[tuple[str, str]]:
    emitted = 0

    for match in WORD_PATTERN.finditer(text):
        word = match.group(0)
        lower = word.lower()
        vowel_count = sum(
            character in VOWELS
            for character in lower
        )

        if (
            vowel_count < MIN_VOWELS_FOR_CLUSTER_INSERTION
            or vowel_count * 2 < len(lower)
        ):
            continue
        positions: list[int] = []

        for index in range(1, len(lower)):
            if (
                lower[index - 1] in VOWELS
                and lower[index] in VOWELS
            ):
                for position in (
                    index - 1,
                    index,
                    index + 1,
                ):
                    if (
                        1 <= position < len(word)
                        and position not in positions
                    ):
                        positions.append(position)

        for position in positions:
            for vowel in VOWELS:
                if (
                    emitted
                    >= MAX_VOWEL_CLUSTER_VARIANTS_PER_TEXT
                ):
                    return

                if (
                    lower[position - 1] == vowel
                    or lower[position] == vowel
                ):
                    continue

                repaired_word = (
                    word[:position]
                    + vowel
                    + word[position:]
                )
                repaired = (
                    text[:match.start()]
                    + repaired_word
                    + text[match.end():]
                )
                emitted += 1

                yield (
                    repaired,
                    'vowel-cluster-insertion:'
                    + word
                    + '->'
                    + repaired_word,
                )


def contraction_apostrophe_variants(
    text: str,
) -> Iterable[tuple[str, str]]:
    for match in CONTRACTION_PATTERN.finditer(text):
        head = match.group(1)
        suffix = match.group(2)
        observed = match.group(0)
        replacements = (
            head + chr(39) + suffix,
            head + chr(39) + ' ' + suffix,
        )

        for replacement in replacements:
            repaired = (
                text[:match.start()]
                + replacement
                + text[match.end():]
            )

            yield (
                repaired,
                'contraction-apostrophe:'
                + observed
                + '->'
                + replacement,
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    v3 = load_v3()
    args = v3.parse_args()
    output_path = v3.resolve_project_path(args.output)

    original_build_stage_sequence = v3.build_stage_sequence

    def v4_stage_sequence(v422, rule):
        return (
            *original_build_stage_sequence(v422, rule),
            (
                'v7-dev-vowel-cluster-insertion',
                vowel_cluster_insertion_variants,
            ),
            (
                'v7-dev-contraction-apostrophe',
                contraction_apostrophe_variants,
            ),
            (
                "v4-dev-sentence-boundary",
                sentence_boundary_variants,
            ),
            (
                "v4-dev-terminal-punctuation",
                terminal_punctuation_variants,
            ),
        )

    # v3 main remains responsible for bounded deterministic expansion.
    v3.build_stage_sequence = v4_stage_sequence
    v3.main()

    report = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    report["status"] = (
        "PROSPECTIVE_BOUNDED_V4_DEV_"
        "NATIVE_ASR_CANDIDATE_FRONTEND_COMPLETE"
    )

    report["rule_source"] = (
        "frozen-v4.2.2-and-frozen-v3-rules-"
        "plus-v4-development-generic-punctuation-rules"
    )

    report["rule_names"].extend(
        [
            "v4-dev-sentence-boundary",
            "v4-dev-terminal-punctuation",
        ]
    )

    report["constraints"].update(
        {
            "new_rules_use_asr_text_only": True,
            "new_rules_use_crc": False,
            "new_rules_use_payload": False,
            "new_rules_use_sender_stegotext": False,
            "new_rules_use_sender_token_ids": False,
            "new_rules_use_reference_text": False,
        }
    )

    report["development_disclosure"] = {
        "official_blind_result": False,
        "introduced_after_imdb_row_0_reveal": True,
        "purpose": (
            "Development regression only. Freeze before testing "
            "an unseen IMDB row and unseen payload."
        ),
        "v3_source": str(V3_PATH),
        "v3_source_sha256": sha256_file(V3_PATH),
    }

    report['status'] = (
        'PROSPECTIVE_BOUNDED_V7_DEV_'
        'NATIVE_ASR_CANDIDATE_FRONTEND_COMPLETE'
    )
    report['rule_source'] = (
        'frozen-v3-v4-rules-plus-v7-generic-'
        'contraction-and-vowel-cluster-rules'
    )
    report['rule_names'] = list(dict.fromkeys(
        list(report.get('rule_names', []))
        + [
            'v7-dev-vowel-cluster-insertion',
            'v7-dev-contraction-apostrophe',
        ]
    ))
    report['constraints'].update(
        {
            'v7_uses_crc': False,
            'v7_uses_payload': False,
            'v7_uses_sender_stegotext': False,
            'v7_uses_sender_token_ids': False,
            'v7_uses_reference_text': False,
            'v7_min_vowels_for_cluster_insertion': (
                MIN_VOWELS_FOR_CLUSTER_INSERTION
            ),
            'v7_minimum_vowel_density': 0.5,
            'v7_max_vowel_variants_per_text': (
                MAX_VOWEL_CLUSTER_VARIANTS_PER_TEXT
            ),
        }
    )
    report['development_disclosure'] = {
        'official_blind_result': False,
        'introduced_after_imdb_row_2_reveal': True,
        'row_2_development_replay_only': True,
        'purpose': (
            'Freeze before testing an unseen IMDB row '
            'and unseen payload.'
        ),
        'v7_source': str(Path(__file__).resolve()),
        'v7_source_sha256': sha256_file(
            Path(__file__).resolve()
        ),
    }


    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate_count": report["candidate_count"],
                "truncated": report["truncated"],
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
