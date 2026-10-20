#!/usr/bin/env python
"""Build a deterministic prompt CSV from the original Stanford ACL IMDb data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


OFFICIAL_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
OFFICIAL_MD5 = "7c2ac02c03563afcf9b574c7e56c153a"
DATASET_CITATION = (
    "Maas et al., Learning Word Vectors for Sentiment Analysis, ACL 2011"
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
SPACE = re.compile(r"\s+")
WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
MEMBER = re.compile(r"^aclImdb/(train|test)/(pos|neg)/(\d+)_([0-9]+)\.txt$")


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "ZSC-TSD-reviewer/1.0"})
    with urllib.request.urlopen(request, timeout=60) as source, partial.open("wb") as sink:
        while block := source.read(1024 * 1024):
            sink.write(block)
    partial.replace(target)


def normalize(value: str) -> str:
    value = html.unescape(value.replace("<br />", " ").replace("<br/>", " "))
    return SPACE.sub(" ", value).strip()


def choose_sentence(review: str) -> tuple[int, str] | None:
    for index, sentence in enumerate(SENTENCE_SPLIT.split(normalize(review))):
        sentence = sentence.strip(' "')
        word_count = len(WORD.findall(sentence))
        if 12 <= word_count <= 45 and 70 <= len(sentence) <= 320:
            return index, sentence
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare real, provenance-recorded Stanford ACL IMDb prompts."
    )
    parser.add_argument(
        "--archive", type=Path, default=Path("artifacts/datasets/aclImdb_v1.tar.gz")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/datasets/imdb_acl_v1")
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    prompts = output_dir / "prompts.csv"
    manifest = output_dir / "dataset_manifest.json"
    if prompts.exists() or manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset output: {output_dir}")
    if not archive.is_file():
        if not args.download:
            raise FileNotFoundError(
                f"IMDb archive missing: {archive}. Re-run with --download to fetch the official Stanford file."
            )
        download(OFFICIAL_URL, archive)
    actual_md5 = digest_file(archive, "md5")
    if actual_md5 != OFFICIAL_MD5:
        raise RuntimeError(
            f"official ACL IMDb MD5 mismatch: expected={OFFICIAL_MD5} actual={actual_md5}"
        )

    rows: list[dict[str, object]] = []
    review_counts = {"train/pos": 0, "train/neg": 0, "test/pos": 0, "test/neg": 0}
    selected_counts = {key: 0 for key in review_counts}
    with tarfile.open(archive, "r:gz") as bundle:
        members = []
        for member in bundle.getmembers():
            match = MEMBER.fullmatch(member.name)
            if match and member.isfile():
                members.append((match.group(1), match.group(2), int(match.group(3)), member))
        members.sort(key=lambda item: (item[0], item[1], item[2]))
        for split, label, review_id, member in members:
            key = f"{split}/{label}"
            review_counts[key] += 1
            stream = bundle.extractfile(member)
            if stream is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            raw = stream.read()
            selected = choose_sentence(raw.decode("utf-8", errors="replace"))
            if selected is None:
                continue
            sentence_index, sentence = selected
            row_index = len(rows)
            rows.append(
                {
                    "sentence_id": f"aclImdb-{split}-{label}-{review_id:05d}-s{sentence_index:03d}",
                    "plaintext": sentence,
                    "row_index": row_index,
                    "source_split": split,
                    "source_label": label,
                    "source_review_id": review_id,
                    "source_member": member.name,
                    "source_review_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_sentence_index": sentence_index,
                }
            )
            selected_counts[key] += 1
            if len(rows) >= args.max_prompts:
                break
    if len(rows) < 1:
        raise RuntimeError("no usable IMDb prompt was generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with prompts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    value = {
        "status": "ACL_IMDB_PROMPTS_PREPARED",
        "schema": "zsc-tsd.imdb-prompt-dataset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Large Movie Review Dataset (aclImdb v1.0)",
            "publisher": "Stanford University",
            "official_url": OFFICIAL_URL,
            "citation": DATASET_CITATION,
            "archive_path": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_md5": actual_md5,
            "archive_sha256": digest_file(archive, "sha256"),
        },
        "derivation": {
            "method": "first deterministic sentence satisfying frozen length filters",
            "sentence_word_range": [12, 45],
            "sentence_character_range": [70, 320],
            "review_counts": review_counts,
            "selected_counts": selected_counts,
            "prompt_count": len(rows),
        },
        "prompts": {
            "path": str(prompts),
            "bytes": prompts.stat().st_size,
            "sha256": digest_file(prompts, "sha256"),
            "columns": fields,
        },
        "historical_compatibility_dataset_used": False,
    }
    manifest.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
