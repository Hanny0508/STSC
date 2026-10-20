"""Byte-exact local adapter used only for software integration tests."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ..manifest import sha256_file
from .base import MediaTransferResult, MediaTransport


class LocalFileTransport(MediaTransport):
    def transfer(self, source: str | Path, received: str | Path) -> MediaTransferResult:
        source_path = Path(source).resolve()
        target = Path(received).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        shutil.copyfile(source_path, target)
        elapsed = time.perf_counter() - started
        return MediaTransferResult(
            transport="local-file-copy-not-voip",
            source=source_path,
            received=target,
            source_sha256=sha256_file(source_path),
            received_sha256=sha256_file(target),
            source_bytes=source_path.stat().st_size,
            received_bytes=target.stat().st_size,
            elapsed_seconds=elapsed,
        )
