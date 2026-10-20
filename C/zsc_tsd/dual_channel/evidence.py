"""Evidence helpers for two-host experiment runs."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..manifest import build_manifest, sha256_file, write_json_new


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def wav_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with wave.open(str(source), "rb") as stream:
        frames = stream.getnframes()
        rate = stream.getframerate()
        result = {
            "channels": stream.getnchannels(),
            "sample_width_bytes": stream.getsampwidth(),
            "sample_rate_hz": rate,
            "frames": frames,
            "duration_microseconds": (frames * 1_000_000) // rate,
        }
    result.update({"bytes": source.stat().st_size, "sha256": sha256_file(source)})
    return result


def command_version(command: Iterable[str]) -> dict[str, Any]:
    args = list(command)
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
        return {"command": args, "returncode": completed.returncode, "first_line": first_line}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": args, "error": f"{type(exc).__name__}: {exc}"}


def runtime_document(*, node_id: str, ffmpeg: str | Path) -> dict[str, Any]:
    return {
        "schema": "zsc-tsd.runtime.v1",
        "node_id": node_id,
        "created_at_utc": utc_now(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "ffmpeg": command_version([str(ffmpeg), "-hide_banner", "-version"]),
    }


def write_evidence_manifest(
    output: str | Path,
    *,
    experiment: str,
    root: str | Path,
    files: Iterable[str | Path],
    metadata: dict[str, Any],
) -> Path:
    unique_files: list[Path] = []
    seen: set[Path] = set()
    for value in files:
        resolved = Path(value).resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(resolved)
    manifest = build_manifest(
        experiment=experiment,
        files=unique_files,
        root=root,
        metadata=metadata,
    )
    return write_json_new(output, manifest)


def write_log_record(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
