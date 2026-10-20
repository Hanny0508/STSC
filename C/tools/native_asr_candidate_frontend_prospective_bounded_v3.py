#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FROZEN_V422 = (
    PROJECT_ROOT
    / "tools"
    / "zts_crc_receiver_v4_2_2_frozen.py"
).resolve()

EXPECTED_V422_SHA256 = (
    "a86242b38821d177b1018126c83d56d867f59c56b9320031571fac6b2e7c9ca3"
)

EXPECTED_V3_RULE_SHA256 = (
    "5afdce9c4eeb7869d73615b610170c425327289b381bbbeceb5aee2b6174693c"
)

EXPECTED_V3_FAST2_SHA256 = (
    "fe66683902406cb2626c77956d5c581bde55a901754221db686646a05c39ab1f"
)

DEFAULT_MAX_CANDIDATES = 4096

Transform = Callable[[str], Iterable[tuple[str, str]]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def candidate_id(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


def extract_text(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("Input JSON must contain an object.")

    transcription = data.get("transcription")

    if isinstance(transcription, dict):
        text = transcription.get("text")

        if isinstance(text, str):
            return text

    for key in ("text", "asr_text", "transcript"):
        value = data.get(key)

        if isinstance(value, str):
            return value

    input_section = data.get("input")

    if isinstance(input_section, dict):
        for key in ("text", "asr_text", "transcript"):
            value = input_section.get(key)

            if isinstance(value, str):
                return value

    raise KeyError(
        "Unable to find ASR text in transcription.text, text, "
        "asr_text, transcript, or input.*."
    )


def load_frozen_rule_stack() -> tuple[
    Any,
    Any,
    Any,
    dict[str, Any],
]:
    require(
        FROZEN_V422.is_file(),
        f"Frozen v4.2.2 receiver missing: {FROZEN_V422}",
    )

    actual_v422_sha256 = sha256_file(FROZEN_V422)

    require(
        actual_v422_sha256 == EXPECTED_V422_SHA256,
        (
            "Frozen v4.2.2 SHA256 mismatch.\n"
            f"Expected: {EXPECTED_V422_SHA256}\n"
            f"Actual:   {actual_v422_sha256}"
        ),
    )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    v422 = load_module(
        "prospective_bounded_v3_receiver_v4_2_2",
        FROZEN_V422,
    )

    rule_path = resolve_project_path(
        v422.FROZEN_V3_RULE_SOURCE
    )
    fast2_path = resolve_project_path(
        v422.FROZEN_V3_FAST2
    )

    require(
        rule_path.is_file(),
        f"Frozen v3 rule source missing: {rule_path}",
    )

    require(
        fast2_path.is_file(),
        f"Frozen v3 fast2 source missing: {fast2_path}",
    )

    rule_sha256 = sha256_file(rule_path)
    fast2_sha256 = sha256_file(fast2_path)

    require(
        rule_sha256 == EXPECTED_V3_RULE_SHA256,
        (
            "Frozen v3 rule SHA256 mismatch.\n"
            f"Expected: {EXPECTED_V3_RULE_SHA256}\n"
            f"Actual:   {rule_sha256}"
        ),
    )

    require(
        fast2_sha256 == EXPECTED_V3_FAST2_SHA256,
        (
            "Frozen v3 fast2 SHA256 mismatch.\n"
            f"Expected: {EXPECTED_V3_FAST2_SHA256}\n"
            f"Actual:   {fast2_sha256}"
        ),
    )

    rule = v422.load_module(
        "prospective_bounded_v3_rule",
        rule_path,
    )

    fast2 = v422.load_module(
        "prospective_bounded_v3_fast2",
        fast2_path,
    )

    provenance = {
        "frozen_v4_2_2_receiver": str(FROZEN_V422),
        "frozen_v4_2_2_receiver_sha256": (
            actual_v422_sha256
        ),
        "frozen_v3_rule_source": str(rule_path),
        "frozen_v3_rule_source_sha256": rule_sha256,
        "frozen_v3_fast2_source": str(fast2_path),
        "frozen_v3_fast2_source_sha256": fast2_sha256,
    }

    return v422, rule, fast2, provenance


def build_stage_sequence(
    v422: Any,
    rule: Any,
) -> tuple[tuple[str, Transform], ...]:
    """
    Reuse the frozen historical repair stages.

    The broad title-case fallback is intentionally excluded because
    it caused candidate-space expansion beyond the fixed bound.
    """

    return (
        (
            "v4.2.2-transition-punctuation",
            v422.transition_punctuation_variants,
        ),
        (
            "v4.1-domain-term-case",
            v422.domain_term_case_variants,
        ),
        (
            "v4.1-combined-asr-confusion",
            v422.make_rule_bound_transform(
                v422.combined_confusion_variants,
                rule,
            ),
        ),
        (
            "v2-hyphenation",
            rule.hyphenation_variants,
        ),
        (
            "v3-singularization",
            rule.singularization_variants,
        ),
        (
            "v3-acronym-case",
            rule.acronym_case_variants,
        ),
        (
            "v4.1-technical-phrase-case",
            v422.technical_phrase_case_variants,
        ),
        (
            "v4.1-technical-phrase-comma",
            v422.technical_phrase_comma_variants,
        ),
        (
            "v3-apostrophe-cleanup",
            rule.apostrophe_cleanup_variants,
        ),
        (
            "v4.1-trailing-punctuation",
            v422.trailing_punctuation_variants,
        ),
        (
            "v2-domain-acronym",
            rule.domain_acronym_variants,
        ),
    )


def generate_candidates(
    raw_text: str,
    *,
    max_candidates: int,
) -> tuple[
    list[Any],
    bool,
    str | None,
    list[dict[str, Any]],
    dict[str, Any],
]:
    if max_candidates < 1:
        raise ValueError(
            "max_candidates must be at least 1."
        )

    v422, rule, fast2, provenance = (
        load_frozen_rule_stack()
    )

    base_candidates = [
        rule.LatticeCandidate(
            text=str(candidate.text),
            transforms=tuple(candidate.transforms),
        )
        for candidate in rule.build_candidates(raw_text)
    ]

    base_candidates = fast2.unique_candidates(
        rule,
        base_candidates,
    )

    first_stage_seen: dict[str, str] = {
        str(candidate.text): "base"
        for candidate in base_candidates
    }

    stage_summaries: list[dict[str, Any]] = []

    truncated = False
    stopped_after_stage: str | None = None

    if len(base_candidates) > max_candidates:
        base_candidates = base_candidates[:max_candidates]
        truncated = True
        stopped_after_stage = "base"

    candidates = base_candidates

    stage_summaries.append(
        {
            "stage": "base",
            "before_count": 0,
            "after_count": len(candidates),
            "added_count": len(candidates),
            "attempted_variants": len(candidates),
            "truncated": truncated,
        }
    )

    if not truncated:
        for stage_name, transform in build_stage_sequence(
            v422,
            rule,
        ):
            before_count = len(candidates)

            before_texts = {
                str(candidate.text)
                for candidate in candidates
            }

            (
                expanded,
                stage_truncated,
                attempted_variants,
            ) = v422.bounded_expand_candidates(
                rule,
                candidates,
                transform,
                max_candidates,
            )

            for candidate in expanded:
                candidate_text = str(candidate.text)

                if candidate_text not in before_texts:
                    first_stage_seen.setdefault(
                        candidate_text,
                        stage_name,
                    )

            after_count = len(expanded)

            stage_summaries.append(
                {
                    "stage": stage_name,
                    "before_count": before_count,
                    "after_count": after_count,
                    "added_count": (
                        after_count - before_count
                    ),
                    "attempted_variants": (
                        attempted_variants
                    ),
                    "truncated": bool(stage_truncated),
                }
            )

            candidates = expanded

            if stage_truncated:
                truncated = True
                stopped_after_stage = stage_name
                break

    metadata = {
        "provenance": provenance,
        "first_stage_seen": first_stage_seen,
    }

    return (
        candidates,
        truncated,
        stopped_after_stage,
        stage_summaries,
        metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a bounded prospective ASR candidate "
            "lattice by directly reusing frozen v4.2.2/v3 "
            "text-repair rules, excluding the broad title-case "
            "fallback. No language model, decoder, CRC, payload, "
            "sender secret, or reference sender text is used."
        )
    )

    source_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    source_group.add_argument("--text")
    source_group.add_argument("--text-json")

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = resolve_project_path(args.output)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output_path}"
        )

    if args.text is not None:
        raw_text = args.text

        source = {
            "type": "command-line-text",
        }

    else:
        input_path = resolve_project_path(
            args.text_json
        )

        require(
            input_path.is_file(),
            f"Input JSON missing: {input_path}",
        )

        input_data = json.loads(
            input_path.read_text(
                encoding="utf-8"
            )
        )

        raw_text = extract_text(input_data)

        source = {
            "type": "json",
            "path": str(input_path),
            "sha256": sha256_file(input_path),
        }

    (
        candidates,
        truncated,
        stopped_after_stage,
        stage_summaries,
        metadata,
    ) = generate_candidates(
        raw_text,
        max_candidates=args.max_candidates,
    )

    first_stage_seen = metadata["first_stage_seen"]

    report = {
        "status": (
            "PROSPECTIVE_BOUNDED_V3_NATIVE_ASR_"
            "CANDIDATE_FRONTEND_COMPLETE"
        ),
        "role": (
            "prospective-payload-independent-bounded-"
            "text-candidate-generation"
        ),
        "raw_text": raw_text,
        "source": source,
        "max_candidates": args.max_candidates,
        "candidate_count": len(candidates),
        "truncated": truncated,
        "stopped_after_stage": stopped_after_stage,
        "rule_source": (
            "frozen-v4.2.2-and-frozen-v3-rules-"
            "excluding-titlecase-fallback"
        ),
        "rule_names": [
            "base:frozen-v3-build-candidates",
            *[
                row["stage"]
                for row in stage_summaries
                if row["stage"] != "base"
            ],
        ],
        "stage_summaries": stage_summaries,
        "frozen_sources": metadata["provenance"],
        "constraints": {
            "loads_language_model": False,
            "calls_zero_shot_gls": False,
            "uses_crc": False,
            "uses_payload": False,
            "uses_sender_secret": False,
            "uses_secret_commitment": False,
            "uses_reference_sender_text": False,
            "contains_new_channel_spelling_rule": False,
            "uses_titlecase_fallback": False,
            "deterministic_stage_order": True,
            "bounded_candidate_count": True,
        },
        "candidates": [
            {
                "rank": rank,
                "candidate_id": candidate_id(
                    str(candidate.text)
                ),
                "generation_stage": (
                    first_stage_seen.get(
                        str(candidate.text),
                        "unknown",
                    )
                ),
                "repair_count": len(
                    tuple(candidate.transforms)
                ),
                "transforms": list(
                    tuple(candidate.transforms)
                ),
                "text": str(candidate.text),
            }
            for rank, candidate in enumerate(candidates)
        ],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "PROSPECTIVE BOUNDED V3 "
        "NATIVE ASR FRONTEND COMPLETE"
    )
    print("raw text:", repr(raw_text))
    print("candidates:", len(candidates))
    print("truncated:", truncated)
    print(
        "stopped after stage:",
        stopped_after_stage,
    )

    print("---")

    for stage in stage_summaries:
        print(
            stage["stage"],
            "before=",
            stage["before_count"],
            "added=",
            stage["added_count"],
            "after=",
            stage["after_count"],
            "attempted=",
            stage["attempted_variants"],
            "truncated=",
            stage["truncated"],
        )

    print("---")
    print("output:", output_path)
    print(
        "output sha256:",
        sha256_file(output_path),
    )
    print("language model loaded:", False)
    print("CRC used:", False)
    print("payload used:", False)
    print("titlecase fallback used:", False)

    if args.show:
        print("---")

        for row in report["candidates"]:
            print(
                row["rank"],
                row["generation_stage"],
                row["candidate_id"],
                row["transforms"],
                repr(row["text"]),
            )


if __name__ == "__main__":
    main()