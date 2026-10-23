#!/usr/bin/env python
"""Run frozen v1.4 ZTS logic with real ACL IMDb prompts and DAQ-TTS audio.

The cryptographic, ZTS, ASR, and fragment-receiver decisions remain those of
v1.4.  This wrapper replaces only the prompt source and speech backend, while
injecting explicit provenance into the commitment and final report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real ACL IMDb + TransformerTTS/FACodec secure batch v1.5."
    )
    parser.add_argument("--bundle-public", required=True, type=Path)
    parser.add_argument("--bundle-private", required=True, type=Path)
    parser.add_argument("--imdb-csv", required=True, type=Path)
    parser.add_argument("--imdb-manifest", required=True, type=Path)
    parser.add_argument("--facodec-python", type=Path)
    parser.add_argument("--reference-wav", required=True, type=Path)
    parser.add_argument("--start-row", type=int, default=111)
    parser.add_argument("--max-attempts-per-shard", type=int, default=8)
    parser.add_argument("--max-v7-candidates", type=int, default=4096)
    parser.add_argument("--max-v6-candidates", type=int, default=32768)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parallel-worker", action="store_true")
    return parser.parse_args()


def validate_dataset(csv_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("ACL IMDb prompt CSV or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ACL_IMDB_PROMPTS_PREPARED":
        raise RuntimeError("IMDb manifest does not contain ACL_IMDB_PROMPTS_PREPARED")
    if manifest.get("dataset", {}).get("official_url") != (
        "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    ):
        raise RuntimeError("IMDb manifest is not tied to the official Stanford URL")
    expected = manifest.get("prompts", {}).get("sha256")
    actual = sha256_file(csv_path)
    if expected != actual:
        raise RuntimeError(f"IMDb prompt SHA-256 mismatch: expected={expected} actual={actual}")
    if manifest.get("dataset", {}).get("archive_md5") != "7c2ac02c03563afcf9b574c7e56c153a":
        raise RuntimeError("IMDb archive MD5 is not the frozen aclImdb v1 value")
    return manifest


def main() -> int:
    args = parse_args()
    csv_path = args.imdb_csv.resolve()
    manifest_path = args.imdb_manifest.resolve()
    dataset = validate_dataset(csv_path, manifest_path)
    if args.facodec_python is not None:
        os.environ["ZSC_TSD_FACODEC_PYTHON"] = str(args.facodec_python.resolve())
    reference_wav = args.reference_wav.resolve()
    if not reference_wav.is_file():
        raise FileNotFoundError(reference_wav)
    os.environ["ZSC_TSD_FACODEC_REFERENCE_WAV"] = str(reference_wav)

    sys.path.insert(0, str(ROOT / "tools"))
    import zts_secure_imdb_batch_runner_v1_4_parallel as legacy
    from zsc_tsd.speech.daqtts import DAQTTSClient

    old_dataset = legacy.IMDB_CSV
    legacy.IMDB_CSV = csv_path
    legacy.EXPECTED_HASHES.pop(old_dataset, None)
    legacy.EXPECTED_HASHES[csv_path] = sha256_file(csv_path)
    legacy.TransformerTTSClient = DAQTTSClient
    original_write = legacy.write_json_new
    provenance = {
        "prompt_dataset": {
            "name": dataset["dataset"]["name"],
            "publisher": dataset["dataset"]["publisher"],
            "official_url": dataset["dataset"]["official_url"],
            "archive_md5": dataset["dataset"]["archive_md5"],
            "archive_sha256": dataset["dataset"]["archive_sha256"],
            "prompt_csv_sha256": dataset["prompts"]["sha256"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "historical_compatibility_dataset_used": False,
        },
        "speech_backend": {
            "project_name": "DAQ-TTS",
            "composition": ["TransformerTTS", "FACodec"],
            "novel_algorithm_claimed": False,
            "facodec_mode": "V2 two-WAV zero-shot voice conversion",
            "reference_wav": str(reference_wav),
            "reference_wav_sha256": sha256_file(reference_wav),
            "source_residual_codes_used": False,
            "carrier_definition": (
                "The public carrier.wav is the output of the FACodec "
                "encoder/vector-quantizer/decoder path, not the raw TransformerTTS WAV."
            ),
        },
        "v1_5_wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }

    def write_with_v15_evidence(path: Path, value: object) -> None:
        if isinstance(value, dict) and path.name == "batch_commitment.json":
            value = dict(value)
            value["v1_5"] = provenance
        if isinstance(value, dict) and path.name == "final_result_after_reveal.json":
            value = dict(value)
            scope = dict(value.get("scope", {}))
            scope.update(
                {
                    "real_acl_imdb_prompts_with_provenance": True,
                    "real_transformertts": True,
                    "facodec_included": True,
                    "facodec_encode_quantize_decode_executed_per_carrier": True,
                    "daqtts_is_project_name_for_upstream_composition": True,
                    "daqtts_claimed_as_new_algorithm": False,
                }
            )
            value["scope"] = scope
            value["v1_5"] = provenance
            value["speech_backend"] = (
                "TransformerTTS + reference-conditioned FACodec V2 (DAQ-TTS)"
            )
        original_write(path, value)

    legacy.write_json_new = write_with_v15_evidence
    sys.argv = [
        str(Path(__file__).resolve()),
        "--bundle-public",
        str(args.bundle_public),
        "--bundle-private",
        str(args.bundle_private),
        "--start-row",
        str(args.start_row),
        "--max-attempts-per-shard",
        str(args.max_attempts_per_shard),
        "--max-v7-candidates",
        str(args.max_v7_candidates),
        "--max-v6-candidates",
        str(args.max_v6_candidates),
    ]
    sys.argv += ["--shard-start", str(args.shard_start), "--shard-end", str(args.shard_end)]
    if args.output_dir is not None:
        sys.argv += ["--output-dir", str(args.output_dir)]
    if args.parallel_worker:
        sys.argv += ["--parallel-worker"]
    legacy.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
