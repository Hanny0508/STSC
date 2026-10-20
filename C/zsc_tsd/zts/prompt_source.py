"""Deterministic, sender-independent prompt resolution for ZTS.

The ZTS initial context (the "prompt") is **not** transmitted by the sender.
In this project it is a sentence taken from a public, SHA-256 pinned CSV of
Stanford ACL IMDb sentences:

    legacy/zero-shot-GLS/zgls_data/imdb.csv      (sentence_id,plaintext)

so the prompt is a pure function of

    (prompt dataset, zero-based data row index).

The sender publishes only a :class:`PromptReference`, which carries the
dataset fingerprint, the row index, the sentence identifier and the SHA-256
of the resolved prompt text -- never the prompt itself.  The receiver loads
its own copy of the dataset, re-derives the prompt locally, and verifies the
reference before decoding.

This removes the defect where the receiver obtained the exact prompt from a
sender-produced record: no sender-written file is read, and the reference
contains no plaintext prompt.

The row semantics deliberately match
``zts_secure_imdb_batch_runner_v1_4.load_imdb_rows``: the file is opened with
``utf-8-sig``, parsed with :class:`csv.DictReader`, and ``row_index`` is the
zero-based index over data rows (the header is not counted).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_COLUMNS = ("sentence_id", "plaintext")

_CHUNK = 8 * 1024 * 1024

# Dataset digest cache keyed by (resolved path, size, mtime_ns).
_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


class PromptSourceError(RuntimeError):
    """Raised when a prompt cannot be resolved or verified locally."""


def sha256_file(path: str | Path, *, chunk_size: int = _CHUNK) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dataset_digest(path: str | Path) -> str:
    """SHA-256 of the prompt dataset, cached per (path, size, mtime)."""

    source = Path(path).resolve()
    if not source.is_file():
        raise PromptSourceError(f"prompt dataset is missing: {source}")
    stat = source.stat()
    key = (str(source), stat.st_size, stat.st_mtime_ns)
    cached = _DIGEST_CACHE.get(key)
    if cached is None:
        cached = sha256_file(source)
        _DIGEST_CACHE[key] = cached
    return cached


def read_prompt_row(dataset: str | Path, row_index: int) -> tuple[str, str]:
    """Return ``(sentence_id, prompt_text)`` for a zero-based data row index."""

    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise PromptSourceError("row_index must be an integer")
    if row_index < 0:
        raise PromptSourceError("row_index must not be negative")

    source = Path(dataset)
    if not source.is_file():
        raise PromptSourceError(f"prompt dataset is missing: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in DATASET_COLUMNS if name not in fieldnames]
        if missing:
            raise PromptSourceError(
                f"prompt dataset is missing columns {missing}: {source}"
            )
        for index, row in enumerate(reader):
            if index != row_index:
                continue
            prompt = str(row["plaintext"])
            if not prompt.strip():
                raise PromptSourceError(
                    f"prompt dataset row {row_index} is empty"
                )
            return str(row.get("sentence_id", row_index)), prompt

    raise PromptSourceError(
        f"prompt dataset has no row {row_index}: {source}"
    )


@dataclass(frozen=True, slots=True)
class PromptReference:
    """A sender-publishable reference to the shared prompt (no plaintext)."""

    dataset_sha256: str
    row_index: int
    sentence_id: str
    prompt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_sha256, str) or len(self.dataset_sha256) != 64:
            raise PromptSourceError("dataset_sha256 must be a 64-char hex digest")
        if not isinstance(self.prompt_sha256, str) or len(self.prompt_sha256) != 64:
            raise PromptSourceError("prompt_sha256 must be a 64-char hex digest")
        if isinstance(self.row_index, bool) or not isinstance(self.row_index, int):
            raise PromptSourceError("row_index must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_sha256": self.dataset_sha256,
            "row_index": self.row_index,
            "sentence_id": self.sentence_id,
            "prompt_sha256": self.prompt_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "PromptReference":
        if not isinstance(value, dict):
            raise PromptSourceError("prompt reference must be an object")
        try:
            return cls(
                dataset_sha256=str(value["dataset_sha256"]),
                row_index=int(value["row_index"]),
                sentence_id=str(value["sentence_id"]),
                prompt_sha256=str(value["prompt_sha256"]),
            )
        except KeyError as exc:
            raise PromptSourceError(
                f"prompt reference is missing field {exc.args[0]!r}"
            ) from exc


def build_prompt_reference(
    dataset: str | Path, row_index: int
) -> PromptReference:
    """Build the publishable reference for a local dataset row."""

    sentence_id, prompt = read_prompt_row(dataset, row_index)
    return PromptReference(
        dataset_sha256=dataset_digest(dataset),
        row_index=int(row_index),
        sentence_id=sentence_id,
        prompt_sha256=sha256_text(prompt),
    )


def assert_reference_matches(
    local: PromptReference, remote: PromptReference
) -> None:
    differences: list[str] = []
    if local.dataset_sha256 != remote.dataset_sha256:
        differences.append("prompt dataset SHA-256 differs")
    if local.row_index != remote.row_index:
        differences.append(
            f"row index differs: local={local.row_index} remote={remote.row_index}"
        )
    if local.sentence_id != remote.sentence_id:
        differences.append("sentence id differs")
    if local.prompt_sha256 != remote.prompt_sha256:
        differences.append("resolved prompt SHA-256 differs")
    if differences:
        raise PromptSourceError(
            "prompt reference mismatch:\n- " + "\n- ".join(differences)
        )


def resolve_prompt(
    dataset: str | Path,
    reference: PromptReference,
    *,
    verify: bool = True,
) -> str:
    """Resolve the prompt locally and verify it against ``reference``."""

    if verify:
        actual_digest = dataset_digest(dataset)
        if actual_digest != reference.dataset_sha256:
            raise PromptSourceError(
                "local prompt dataset does not match the reference:\n"
                f"  local     = {actual_digest}\n"
                f"  reference = {reference.dataset_sha256}\n"
                f"  dataset   = {Path(dataset).resolve()}"
            )

    sentence_id, prompt = read_prompt_row(dataset, reference.row_index)

    if verify:
        if sentence_id != reference.sentence_id:
            raise PromptSourceError(
                "resolved sentence id does not match the reference: "
                f"local={sentence_id} reference={reference.sentence_id}"
            )
        actual_prompt_digest = sha256_text(prompt)
        if actual_prompt_digest != reference.prompt_sha256:
            raise PromptSourceError(
                "resolved prompt does not match the reference: "
                f"local={actual_prompt_digest} "
                f"reference={reference.prompt_sha256}"
            )

    return prompt


def resolve_prompt_reference(
    dataset: str | Path,
    reference: dict[str, Any] | PromptReference,
    *,
    verify: bool = True,
) -> str:
    """Convenience wrapper used by the receiver-side worker."""

    parsed = (
        reference
        if isinstance(reference, PromptReference)
        else PromptReference.from_dict(reference)
    )
    return resolve_prompt(dataset, parsed, verify=verify)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--dataset-sha256", action="store_true")
    parser.add_argument(
        "--reference",
        type=Path,
        help="JSON file holding a prompt reference; the prompt is resolved "
        "locally and verified against it.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.dataset_sha256:
        digest = dataset_digest(args.dataset)
        print(json.dumps({"dataset": str(args.dataset), "sha256": digest}))
        return 0

    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        prompt = resolve_prompt_reference(args.dataset, reference)
        result = {
            "status": "PROMPT_RESOLVED_LOCALLY",
            "row_index": int(reference["row_index"]),
            "sentence_id": str(reference["sentence_id"]),
            "prompt_sha256": sha256_text(prompt),
            "prompt_length": len(prompt),
        }
        if not args.json:
            result["prompt"] = prompt
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.row_index is None:
        parser.error("provide --row-index, --reference, or --dataset-sha256")

    reference = build_prompt_reference(args.dataset, args.row_index)
    payload: dict[str, Any] = {"reference": reference.to_dict()}
    if not args.json:
        payload["prompt"] = read_prompt_row(args.dataset, args.row_index)[1]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
