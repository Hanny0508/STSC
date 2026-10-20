from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TTS_PYTHON = Path(
    os.environ.get(
        "ZSC_TSD_TTS_PYTHON",
        "python",
    )
)

DEFAULT_WORKER_SCRIPT = (
    PROJECT_ROOT
    / "tools"
    / "transformertts_smoke.py"
)

DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "tts_models"
    / "bdf06b9_ljspeech_step_90000"
    / "bdf06b9_ljspeech_step_90000"
)

DEFAULT_ESPEAK = Path(
    r"C:\Program Files\eSpeak NG\espeak-ng.exe"
)


class TransformerTTSError(RuntimeError):
    """Raised when the external TransformerTTS worker fails."""


@dataclass(frozen=True, slots=True)
class TransformerTTSConfig:
    python_executable: Path = DEFAULT_TTS_PYTHON
    worker_script: Path = DEFAULT_WORKER_SCRIPT
    model_dir: Path = DEFAULT_MODEL_DIR
    espeak_executable: Path = DEFAULT_ESPEAK
    griffin_lim_iterations: int = 32
    seed: int = 2023
    timeout_seconds: int = 600
    expected_step: str = "90000"
    expected_parameter_count: int = 52_396_772


@dataclass(frozen=True, slots=True)
class TransformerTTSResult:
    original_text: str
    spoken_text: str
    leading_space_count: int
    phonemes: str
    token_count: int
    mel_shape: tuple[int, int]
    model_step: str
    model_parameter_count: int
    output_path: Path
    report_path: Path
    audio_sha256: str
    sample_rate: int
    frame_count: int
    duration_seconds: float
    file_bytes: int
    stdout: str
    stderr: str
    report: dict[str, Any]


def _sha256_file(
    path: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def _leading_space_count(text: str) -> int:
    return len(text) - len(text.lstrip(" "))


class TransformerTTSClient:
    """
    Invoke the Python 3.8 TransformerTTS worker from the main project.

    TensorFlow remains isolated in the zsc-tsd-tts environment.
    """

    def __init__(
        self,
        config: TransformerTTSConfig | None = None,
    ) -> None:
        self.config = config or TransformerTTSConfig()

    def validate_environment(self) -> None:
        required_files = (
            self.config.python_executable,
            self.config.worker_script,
            self.config.model_dir / "config.yaml",
            self.config.model_dir / "model_weights.hdf5",
            self.config.espeak_executable,
        )

        missing = [
            str(path)
            for path in required_files
            if not path.is_file()
        ]

        if missing:
            raise FileNotFoundError(
                "TransformerTTS required files are missing:\n- "
                + "\n- ".join(missing)
            )

        if self.config.griffin_lim_iterations <= 0:
            raise ValueError(
                "griffin_lim_iterations must be positive"
            )

        if self.config.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive"
            )

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        *,
        report_path: str | Path | None = None,
    ) -> TransformerTTSResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError(
                "text contains no pronounceable content"
            )

        self.validate_environment()

        output = Path(output_path).resolve()
        report = (
            Path(report_path).resolve()
            if report_path is not None
            else output.with_suffix(".json")
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            str(self.config.python_executable),
            str(self.config.worker_script),
            "--model-dir",
            str(self.config.model_dir),
            "--output",
            str(output),
            "--report",
            str(report),
            "--espeak",
            str(self.config.espeak_executable),
            "--text",
            text,
            "--griffin-lim-iterations",
            str(self.config.griffin_lim_iterations),
            "--seed",
            str(self.config.seed),
        ]

        environment = os.environ.copy()
        environment.setdefault(
            "TF_CPP_MIN_LOG_LEVEL",
            "2",
        )

        try:
            process = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise TransformerTTSError(
                "TransformerTTS worker timed out after "
                f"{self.config.timeout_seconds} seconds"
            ) from error

        if process.returncode != 0:
            raise TransformerTTSError(
                "TransformerTTS worker failed with exit code "
                f"{process.returncode}.\n"
                f"STDOUT:\n{process.stdout}\n"
                f"STDERR:\n{process.stderr}"
            )

        if not output.is_file():
            raise TransformerTTSError(
                f"Worker did not create WAV file: {output}"
            )

        if not report.is_file():
            raise TransformerTTSError(
                f"Worker did not create JSON report: {report}"
            )

        try:
            report_value = json.loads(
                report.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise TransformerTTSError(
                f"Cannot read worker report: {report}"
            ) from error

        self._validate_report(
            report_value,
            expected_text=text,
            output_path=output,
        )

        model = report_value["model"]
        mel = report_value["mel"]
        audio = report_value["audio"]

        return TransformerTTSResult(
            original_text=str(
                report_value["original_text"]
            ),
            spoken_text=str(
                report_value["spoken_text"]
            ),
            leading_space_count=int(
                report_value["leading_space_count"]
            ),
            phonemes=str(
                report_value["phonemes"]
            ),
            token_count=int(
                report_value["token_count"]
            ),
            mel_shape=(
                int(mel["shape"][0]),
                int(mel["shape"][1]),
            ),
            model_step=str(model["step"]),
            model_parameter_count=int(
                model["parameter_count"]
            ),
            output_path=output,
            report_path=report,
            audio_sha256=str(audio["sha256"]),
            sample_rate=int(audio["sample_rate"]),
            frame_count=int(audio["frames"]),
            duration_seconds=float(
                audio["duration_seconds"]
            ),
            file_bytes=int(audio["file_bytes"]),
            stdout=process.stdout,
            stderr=process.stderr,
            report=report_value,
        )

    def _validate_report(
        self,
        report: dict[str, Any],
        *,
        expected_text: str,
        output_path: Path,
    ) -> None:
        if report.get("status") != "TEXT_TO_WAV_PASS":
            raise TransformerTTSError(
                "Worker report does not contain "
                "TEXT_TO_WAV_PASS"
            )

        if report.get("original_text") != expected_text:
            raise TransformerTTSError(
                "Worker report text differs from requested text"
            )

        expected_spaces = _leading_space_count(
            expected_text
        )

        if (
            int(report.get("leading_space_count", -1))
            != expected_spaces
        ):
            raise TransformerTTSError(
                "Leading-space metadata is inconsistent"
            )

        expected_spoken = expected_text[
            expected_spaces:
        ]

        if report.get("spoken_text") != expected_spoken:
            raise TransformerTTSError(
                "Worker spoken text is inconsistent"
            )

        model = report.get("model")

        if not isinstance(model, dict):
            raise TransformerTTSError(
                "Worker report has no model section"
            )

        if str(model.get("step")) != self.config.expected_step:
            raise TransformerTTSError(
                "Unexpected TransformerTTS checkpoint step: "
                f"{model.get('step')}"
            )

        if (
            int(model.get("parameter_count", -1))
            != self.config.expected_parameter_count
        ):
            raise TransformerTTSError(
                "Unexpected TransformerTTS parameter count: "
                f"{model.get('parameter_count')}"
            )

        mel = report.get("mel")

        if not isinstance(mel, dict):
            raise TransformerTTSError(
                "Worker report has no Mel section"
            )

        if mel.get("finite") is not True:
            raise TransformerTTSError(
                "Worker produced non-finite Mel values"
            )

        shape = mel.get("shape")

        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or int(shape[0]) != 80
            or int(shape[1]) <= 0
        ):
            raise TransformerTTSError(
                f"Unexpected Mel shape: {shape}"
            )

        audio = report.get("audio")

        if not isinstance(audio, dict):
            raise TransformerTTSError(
                "Worker report has no audio section"
            )

        if int(audio.get("sample_rate", 0)) != 22_050:
            raise TransformerTTSError(
                "Unexpected sample rate: "
                f"{audio.get('sample_rate')}"
            )

        if float(audio.get("duration_seconds", 0)) <= 0:
            raise TransformerTTSError(
                "Generated audio has invalid duration"
            )

        actual_sha256 = _sha256_file(output_path)

        if audio.get("sha256") != actual_sha256:
            raise TransformerTTSError(
                "WAV SHA-256 does not match worker report"
            )