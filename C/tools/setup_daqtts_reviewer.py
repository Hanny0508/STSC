#!/usr/bin/env python
"""Create the isolated reviewer FACodec environment and fetch frozen weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = "zsc-tsd-facodec"
SOURCE_REVISION = "aac16b7268b1b96e70b9e3526b13e216984e80d0"
SOURCE_ARCHIVE = (
    "https://github.com/lifeiteng/naturalspeech3_facodec/archive/"
    + SOURCE_REVISION
    + ".zip"
)
REFERENCE_URL = (
    "https://raw.githubusercontent.com/lifeiteng/naturalspeech3_facodec/"
    + SOURCE_REVISION
    + "/audio/2.wav"
)
REFERENCE_SHA256 = "81faa543ca1ae17bb75acbfd7a883b7498db5fe1d0d779c2e8676727a4159e8a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_reference() -> Path:
    target = ROOT / "artifacts" / "facodec_reference" / "official_audio_2.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_suffix(".wav.part")
        request = urllib.request.Request(
            REFERENCE_URL, headers={"User-Agent": "ZSC-TSD-reviewer/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as sink:
            while block := source.read(1024 * 1024):
                sink.write(block)
        temporary.replace(target)
    actual = sha256_file(target)
    if actual != REFERENCE_SHA256:
        raise RuntimeError(
            f"official FACodec reference hash mismatch: expected={REFERENCE_SHA256} actual={actual}"
        )
    return target


def run(command: list[str]) -> None:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda", required=True)
    parser.add_argument("--hf-endpoint", default="")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu128")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def environment_names(conda: str) -> set[str]:
    process = subprocess.run(
        [conda, "env", "list", "--json"], check=True, capture_output=True, text=True
    )
    return {Path(item).name.lower() for item in json.loads(process.stdout)["envs"]}


def main() -> int:
    args = parse_args()
    conda = str(Path(args.conda).resolve())
    if not Path(conda).is_file() and not shutil.which(args.conda):
        raise FileNotFoundError(args.conda)
    exists = ENVIRONMENT.lower() in environment_names(conda)
    if not exists:
        run([conda, "create", "-y", "-n", ENVIRONMENT, "python=3.10", "pip"])
    if not exists or args.refresh:
        run(
            [
                conda,
                "run",
                "--no-capture-output",
                "-n",
                ENVIRONMENT,
                "python",
                "-m",
                "pip",
                "install",
                "torch==2.7.1",
                "torchaudio==2.7.1",
                "--index-url",
                args.torch_index_url,
            ]
        )
        run(
            [
                conda,
                "run",
                "--no-capture-output",
                "-n",
                ENVIRONMENT,
                "python",
                "-m",
                "pip",
                "install",
                "-r",
                str(ROOT / "requirements" / "reviewer-facodec.txt"),
            ]
        )
        run(
            [
                conda,
                "run",
                "--no-capture-output",
                "-n",
                ENVIRONMENT,
                "python",
                "-m",
                "pip",
                "install",
                SOURCE_ARCHIVE,
            ]
        )
        run(
            [
                conda,
                "run",
                "--no-capture-output",
                "-n",
                ENVIRONMENT,
                "python",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-e",
                str(ROOT),
            ]
        )
    else:
        print(f"= preserving existing environment: {ENVIRONMENT}", flush=True)
    reference = prepare_reference()
    report = ROOT / "artifacts" / "facodec_models" / "facodec_v2_checkpoint_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if args.hf_endpoint:
        environment["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    command = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        ENVIRONMENT,
        "python",
        str(ROOT / "tools" / "facodec_v2_voice_conversion_worker.py"),
        "--download-only",
        "--report",
        str(report),
    ]
    print("+ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    result = {
        "status": "ZSC_TSD_DAQTTS_SETUP_PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": ENVIRONMENT,
        "facodec_source_revision": SOURCE_REVISION,
        "checkpoint_report": str(report),
        "reference_wav": str(reference),
        "reference_wav_sha256": sha256_file(reference),
        "reference_source_url": REFERENCE_URL,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
