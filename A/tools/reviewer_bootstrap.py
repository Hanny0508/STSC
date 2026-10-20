#!/usr/bin/env python
"""One-command reviewer setup, model provisioning, and verification.

This script deliberately uses only the Python standard library until a model
download is requested.  It can therefore create the isolated Conda
environments before the project dependencies exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_LOCK = ROOT / "configs" / "reviewer_models_v1.json"
ENVIRONMENTS = {
    "full-a": (
        ("zsc-tsd-core", "3.10", "reviewer-core.txt", True),
        ("zsc-tsd-zts", "3.10", "reviewer-zts.txt", True),
        ("zsc-tsd-asr", "3.10", "reviewer-asr.txt", True),
        ("zsc-tsd-tts", "3.8", "reviewer-tts.txt", False),
    ),
    "relay-b": (
        ("zsc-tsd-relay", "3.12", "reviewer-relay.txt", True),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def run(command: list[str], *, dry_run: bool = False) -> None:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def conda_executable(value: str | None) -> str:
    candidate = value or os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not candidate:
        raise SystemExit(
            "Conda was not found. Install Miniconda/Anaconda or pass --conda <path>."
        )
    return candidate


def conda_environment_names(conda: str) -> set[str]:
    process = subprocess.run(
        [conda, "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = json.loads(process.stdout)
    return {Path(path).name.lower() for path in payload.get("envs", [])}


def setup_environments(args: argparse.Namespace) -> None:
    conda = conda_executable(args.conda)
    existing = set() if args.dry_run else conda_environment_names(conda)
    for name, python_version, requirements, install_project in ENVIRONMENTS[args.profile]:
        already_exists = name.lower() in existing
        if not already_exists:
            run(
                [conda, "create", "-y", "-n", name, f"python={python_version}", "pip"],
                dry_run=args.dry_run,
            )
        else:
            print(f"= environment already exists: {name}")
        if already_exists and not args.refresh_existing:
            print(f"= preserving existing environment without modification: {name}")
            continue
        if name in {"zsc-tsd-zts", "zsc-tsd-asr"}:
            torch_source = (
                [args.torch_wheel_url]
                if args.torch_wheel_url
                else [
                    "torch==2.7.1",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ]
            )
            run(
                [conda, "run", "--no-capture-output", "-n", name, "python", "-m", "pip", "install"]
                + torch_source,
                dry_run=args.dry_run,
            )
        run(
            [
                conda,
                "run",
                "--no-capture-output",
                "-n",
                name,
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(ROOT / "requirements" / requirements),
            ],
            dry_run=args.dry_run,
        )
        if install_project:
            run(
                [
                    conda,
                    "run",
                    "--no-capture-output",
                    "-n",
                    name,
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "-e",
                    str(ROOT),
                ],
                dry_run=args.dry_run,
            )
    if args.with_models:
        if args.profile != "full-a":
            raise SystemExit("--with-models is valid only for --profile full-a")
        command = [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            "zsc-tsd-core",
            "python",
            str(Path(__file__).resolve()),
            "models",
            "--accept-model-licenses",
        ]
        if args.hf_endpoint:
            command += ["--hf-endpoint", args.hf_endpoint]
        if args.dry_run:
            command.append("--dry-run")
        run(command, dry_run=args.dry_run)


def load_model_lock() -> dict[str, Any]:
    data = json.loads(MODEL_LOCK.read_text(encoding="utf-8"))
    if data.get("schema") != "zsc-tsd.reviewer-model-lock.v1":
        raise SystemExit("unsupported reviewer model lock")
    return data


def verify_model(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    target = ROOT / spec["target"]
    missing = [item for item in spec["required_files"] if not (target / item).is_file()]
    mismatches = []
    verified = {}
    for relative, expected in spec.get("verified_files", {}).items():
        path = target / relative
        if not path.is_file():
            continue
        actual_hash = sha256_file(path)
        actual_bytes = path.stat().st_size
        verified[relative] = {"bytes": actual_bytes, "sha256": actual_hash}
        if actual_hash != expected["sha256"]:
            mismatches.append(f"{relative}: sha256={actual_hash}")
        if "bytes" in expected and actual_bytes != expected["bytes"]:
            mismatches.append(f"{relative}: bytes={actual_bytes}")
    return {
        "name": name,
        "target": str(target),
        "pass": not missing and not mismatches,
        "missing": missing,
        "mismatches": mismatches,
        "verified_files": verified,
    }


def verify_models(args: argparse.Namespace) -> bool:
    models = load_model_lock()["models"]
    selected = [args.only] if args.only else list(models)
    rows = [verify_model(name, models[name]) for name in selected]
    passed = all(row["pass"] for row in rows)
    emit(
        {
            "status": "ZSC_TSD_MODEL_VERIFY_PASS" if passed else "ZSC_TSD_MODEL_VERIFY_FAIL",
            "lock_sha256": sha256_file(MODEL_LOCK),
            "models": rows,
        }
    )
    return passed


def download_huggingface(name: str, spec: dict[str, Any], endpoint: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is missing. Run setup --profile full-a first."
        ) from exc
    target = ROOT / spec["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=spec["revision"],
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def download_tts(spec: dict[str, Any]) -> None:
    target = ROOT / spec["target"]
    archive = ROOT / spec["archive_target"]
    if not archive.is_file():
        variable = spec["download_url_environment"]
        url = os.environ.get(variable)
        if not url:
            raise SystemExit(
                "TransformerTTS step-90000 archive is absent. Copy the authors' frozen "
                f"archive to {archive}, or set {variable} to its download URL. The "
                "extracted config and weights will still be SHA-256 verified."
            )
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(archive.suffix + ".part")
        if partial.exists():
            raise SystemExit(f"partial download already exists: {partial}")
        print(f"Downloading frozen TransformerTTS archive to {archive}")
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, archive)
    with tempfile.TemporaryDirectory(prefix="zsc-tsd-tts-") as temp:
        extracted = Path(temp)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                resolved = (extracted / member.filename).resolve()
                if extracted.resolve() not in resolved.parents and resolved != extracted.resolve():
                    raise SystemExit(f"unsafe path in TransformerTTS archive: {member.filename}")
            bundle.extractall(extracted)
        candidates = [
            path.parent
            for path in extracted.rglob("model_weights.hdf5")
            if (path.parent / "config.yaml").is_file()
        ]
        if len(candidates) != 1:
            raise SystemExit(
                f"expected exactly one TransformerTTS checkpoint in {archive}; found {len(candidates)}"
            )
        if target.exists() and any(target.iterdir()):
            raise SystemExit(f"refusing to replace nonempty invalid model directory: {target}")
        target.mkdir(parents=True, exist_ok=True)
        for name in spec["required_files"]:
            shutil.copy2(candidates[0] / name, target / name)


def provision_models(args: argparse.Namespace) -> None:
    if not args.accept_model_licenses:
        raise SystemExit(
            "Model provisioning requires --accept-model-licenses. This records only "
            "the reviewer's explicit acceptance; it does not bypass upstream access terms."
        )
    models = load_model_lock()["models"]
    selected = [args.only] if args.only else list(models)
    for name in selected:
        if name not in models:
            raise SystemExit(f"unknown model: {name}")
        spec = models[name]
        current = verify_model(name, spec)
        if current["pass"]:
            print(f"= model already verified: {name}")
            continue
        if current["mismatches"]:
            raise SystemExit(
                f"refusing to replace mismatched existing model {name}: "
                + "; ".join(current["mismatches"])
            )
        if args.dry_run:
            print(f"+ download {name} -> {ROOT / spec['target']}")
        elif spec["provider"] == "huggingface":
            download_huggingface(name, spec, args.hf_endpoint)
        else:
            download_tts(spec)
        if not args.dry_run:
            result = verify_model(name, spec)
            if not result["pass"]:
                raise SystemExit(f"downloaded model failed verification: {json.dumps(result)}")
            print(f"MODEL VERIFIED: {name}")
    if not args.dry_run and not verify_models(argparse.Namespace(only=args.only)):
        raise SystemExit(2)


def quick_test(args: argparse.Namespace) -> None:
    conda = conda_executable(args.conda)
    environment = "zsc-tsd-relay" if args.profile == "relay-b" else "zsc-tsd-core"
    if args.profile == "full-a":
        run([conda, "run", "--no-capture-output", "-n", environment, "python", str(Path(__file__).resolve()), "verify-models"])
        run([conda, "run", "--no-capture-output", "-n", environment, "python", "-m", "zsc_tsd", "preflight", "--profile", "paper_r1"])
        run([conda, "run", "--no-capture-output", "-n", environment, "python", "-m", "pytest", "-q", "tests"])
        if os.name == "nt":
            espeak = Path(r"C:\Program Files\eSpeak NG\espeak-ng.exe")
            if not espeak.is_file():
                raise SystemExit(
                    "eSpeak NG is required by TransformerTTS and was not found at "
                    f"{espeak}. Install it before the model-backed experiment."
                )
    else:
        run(
            [
                conda,
                "run",
                "--no-capture-output",
                "-n",
                environment,
                "python",
                "-c",
                "from zsc_tsd.control import verify_signed_event; "
                "from zsc_tsd.dual_channel.relay_b import RelayBConfig; "
                "print('ZSC_TSD_RELAY_IMPORT_PASS')",
            ]
        )


def write_plan(args: argparse.Namespace) -> None:
    emit(
        {
            "status": "ZSC_TSD_REVIEWER_PLAN",
            "profile": args.profile,
            "project_root": str(ROOT),
            "environments": [
                {"name": name, "python": version, "requirements": requirements}
                for name, version, requirements, _ in ENVIRONMENTS[args.profile]
            ],
            "models": list(load_model_lock()["models"]) if args.profile == "full-a" else [],
            "model_lock_sha256": sha256_file(MODEL_LOCK),
            "notes": [
                "No existing environment or model is deleted.",
                "Existing model files must pass the frozen SHA-256 checks.",
                "The two-host media experiment remains one command per physical node.",
            ],
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "setup", "quick-test"):
        item = commands.add_parser(command)
        item.add_argument("--profile", choices=tuple(ENVIRONMENTS), required=True)
        item.add_argument("--conda")
        if command == "setup":
            item.add_argument("--with-models", action="store_true")
            item.add_argument("--accept-model-licenses", action="store_true")
            item.add_argument("--hf-endpoint")
            item.add_argument(
                "--torch-wheel-url",
                help="optional exact CUDA wheel URL; default is official cu128 index",
            )
            item.add_argument("--dry-run", action="store_true")
            item.add_argument(
                "--refresh-existing",
                action="store_true",
                help="explicitly install into existing named environments",
            )
    models = commands.add_parser("models")
    models.add_argument("--only", choices=tuple(load_model_lock()["models"]))
    models.add_argument("--accept-model-licenses", action="store_true")
    models.add_argument("--hf-endpoint")
    models.add_argument("--dry-run", action="store_true")
    verify = commands.add_parser("verify-models")
    verify.add_argument("--only", choices=tuple(load_model_lock()["models"]))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    if args.command == "plan":
        write_plan(args)
    elif args.command == "setup":
        setup_environments(args)
    elif args.command == "models":
        provision_models(args)
    elif args.command == "verify-models":
        return 0 if verify_models(args) else 2
    elif args.command == "quick-test":
        quick_test(args)
    else:
        raise AssertionError(args.command)
    if args.command not in {"plan", "verify-models"}:
        emit({"status": "ZSC_TSD_REVIEWER_COMMAND_PASS", "command": args.command, "started_at_utc": started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
