"""Hash-addressed provenance manifests used by every public experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    relative = source
    if root is not None:
        try:
            relative = source.relative_to(Path(root).resolve())
        except ValueError:
            pass
    return {
        "path": str(relative),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def build_manifest(
    *,
    experiment: str,
    files: Iterable[str | Path],
    root: str | Path | None = None,
    command: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = [fingerprint(path, root=root) for path in files]
    records.sort(key=lambda row: row["path"])
    manifest: dict[str, Any] = {
        "schema": "zsc-tsd.provenance-manifest.v1",
        "experiment": experiment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "command": list(command or sys.argv),
        "files": records,
        "metadata": metadata or {},
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def write_json_new(path: str | Path, value: object) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target
