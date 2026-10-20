"""Subprocess adapter for the frozen Whisper receiver environment."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ASRError(RuntimeError):
    """Raised when the independent Whisper worker fails."""


@dataclass(frozen=True, slots=True)
class WhisperASRConfig:
    python_executable: Path = Path(
        os.environ.get(
            "ZSC_TSD_ASR_PYTHON",
            "python",
        )
    )
    worker_script: Path = PROJECT_ROOT / "tools" / "whisper_asr_worker.py"
    model_dir: Path = PROJECT_ROOT / "artifacts" / "asr_models" / "whisper-small.en"
    timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    normalized: str
    output_path: Path
    report: dict[str, Any]
    stdout: str
    stderr: str


class WhisperASRClient:
    def __init__(self, config: WhisperASRConfig | None = None) -> None:
        self.config = config or WhisperASRConfig()

    def validate_environment(self) -> None:
        missing = [
            str(path)
            for path in (
                self.config.python_executable,
                self.config.worker_script,
                self.config.model_dir,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Whisper receiver files are missing:\n- " + "\n- ".join(missing))
        if self.config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def transcribe(self, audio_path: str | Path, output_path: str | Path) -> ASRResult:
        self.validate_environment()
        source = Path(audio_path).resolve()
        output = Path(output_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.config.python_executable),
            str(self.config.worker_script),
            "--model-dir",
            str(self.config.model_dir),
            "--audio",
            str(source),
            "--output",
            str(output),
        ]
        process = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.config.timeout_seconds,
            check=False,
        )
        if process.returncode != 0 or not output.is_file():
            raise ASRError(
                f"Whisper worker failed with exit code {process.returncode}\n"
                f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )
        report = json.loads(output.read_text(encoding="utf-8"))
        transcription = report.get("transcription") or {}
        return ASRResult(
            text=str(transcription.get("text", "")),
            normalized=str(transcription.get("normalized", "")),
            output_path=output,
            report=report,
            stdout=process.stdout,
            stderr=process.stderr,
        )
