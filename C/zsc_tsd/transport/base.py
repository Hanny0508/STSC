"""Transport contracts shared by local tests and the RTP/Opus testbed."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MediaTransferResult:
    transport: str
    source: Path
    received: Path
    source_sha256: str
    received_sha256: str
    source_bytes: int
    received_bytes: int
    elapsed_seconds: float

    @property
    def byte_exact(self) -> bool:
        return self.source_sha256 == self.received_sha256


class MediaTransport(ABC):
    @abstractmethod
    def transfer(self, source: str | Path, received: str | Path) -> MediaTransferResult:
        """Move one audio carrier through the selected transport."""
