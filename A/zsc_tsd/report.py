"""Standard public/private directory layout for ZSC-TSD experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import write_json_new


@dataclass(frozen=True, slots=True)
class RunDirectory:
    root: Path
    public: Path
    sender_private: Path
    receiver: Path
    logs: Path

    @classmethod
    def create(cls, base: str | Path, experiment: str) -> "RunDirectory":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = Path(base).resolve() / f"{experiment}_{stamp}"
        root.mkdir(parents=True, exist_ok=False)
        paths = cls(
            root=root,
            public=root / "public",
            sender_private=root / "sender_private",
            receiver=root / "receiver",
            logs=root / "logs",
        )
        for path in (paths.public, paths.sender_private, paths.receiver, paths.logs):
            path.mkdir()
        return paths

    def write_public(self, name: str, value: object) -> Path:
        return write_json_new(self.public / name, value)

    def write_private(self, name: str, value: object) -> Path:
        return write_json_new(self.sender_private / name, value)

    def write_receiver(self, name: str, value: object) -> Path:
        return write_json_new(self.receiver / name, value)


def result_document(
    *,
    status: str,
    experiment: str,
    checks: dict[str, bool],
    measurements: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "zsc-tsd.experiment-result.v1",
        "status": status,
        "experiment": experiment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "measurements": measurements or {},
        "evidence": evidence or {},
    }
