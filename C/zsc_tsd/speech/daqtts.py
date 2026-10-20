"""DAQ-TTS engineering backend: TransformerTTS followed by FACodec.

DAQ-TTS is the project name for this composition.  The module deliberately
does not claim that either TransformerTTS or FACodec is a new contribution.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .transformertts import TransformerTTSClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACODEC_PYTHON = Path(
    os.environ.get(
        "ZSC_TSD_FACODEC_PYTHON",
        "python",
    )
)
DEFAULT_FACODEC_WORKER = PROJECT_ROOT / "tools" / "facodec_v2_voice_conversion_worker.py"
DEFAULT_REFERENCE_WAV = Path(
    os.environ.get(
        "ZSC_TSD_FACODEC_REFERENCE_WAV",
        str(PROJECT_ROOT / "artifacts" / "facodec_reference" / "official_audio_2.wav"),
    )
)


_DEFAULT_FACODEC_MODEL_DIR = (
    Path(os.environ["ZSC_TSD_FACODEC_MODEL_DIR"])
    if os.environ.get("ZSC_TSD_FACODEC_MODEL_DIR")
    else None
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DAQTTSError(RuntimeError):
    """Raised when either real stage of the DAQ-TTS backend fails."""


@dataclass(frozen=True, slots=True)
class DAQTTSConfig:
    facodec_python: Path = DEFAULT_FACODEC_PYTHON
    facodec_worker: Path = DEFAULT_FACODEC_WORKER
    reference_wav: Path = DEFAULT_REFERENCE_WAV
    facodec_model_dir: Path | None = _DEFAULT_FACODEC_MODEL_DIR
    facodec_device: str = "auto"
    timeout_seconds: int = 1800


@dataclass(frozen=True, slots=True)
class DAQTTSResult:
    output_path: Path
    report_path: Path
    audio_sha256: str
    transformertts_raw_path: Path
    transformertts_report_path: Path
    facodec_report_path: Path
    report: dict[str, Any]


class DAQTTSClient:
    """Run TransformerTTS followed by reference-conditioned FACodec V2."""

    def __init__(
        self,
        config: DAQTTSConfig | None = None,
        transformertts: TransformerTTSClient | None = None,
    ) -> None:
        self.config = config or DAQTTSConfig()
        self.transformertts = transformertts or TransformerTTSClient()

    def validate_environment(self) -> None:
        required = (
            self.config.facodec_python,
            self.config.facodec_worker,
            self.config.reference_wav,
        )
        missing = [str(path) for path in required if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError("DAQ-TTS FACodec files missing:\n- " + "\n- ".join(missing))
        if self.config.facodec_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("facodec_device must be auto, cuda, or cpu")
        if self.config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        *,
        report_path: str | Path | None = None,
    ) -> DAQTTSResult:
        self.validate_environment()
        output = Path(output_path).resolve()
        report = (
            Path(report_path).resolve()
            if report_path is not None
            else output.with_suffix(".daqtts.json")
        )
        if output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")
        if report.exists():
            raise FileExistsError(f"refusing to overwrite: {report}")
        report.parent.mkdir(parents=True, exist_ok=True)
        stem = report.stem
        raw = report.parent / f"{stem}.transformertts_raw.wav"
        raw_report = report.parent / f"{stem}.transformertts.json"
        facodec_report = report.parent / f"{stem}.facodec.json"
        for item in (raw, raw_report, facodec_report):
            if item.exists():
                raise FileExistsError(f"refusing to overwrite: {item}")

        tts_result = self.transformertts.synthesize(text, raw, report_path=raw_report)
        command = [
            str(self.config.facodec_python),
            str(self.config.facodec_worker),
            "--source",
            str(raw),
            "--reference",
            str(self.config.reference_wav.resolve()),
            "--output",
            str(output),
            "--report",
            str(facodec_report),
            "--device",
            self.config.facodec_device,
        ]
        if self.config.facodec_model_dir is not None:
            command += ["--model-dir", str(self.config.facodec_model_dir)]
        environment = os.environ.copy()
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
        if process.returncode != 0:
            raise DAQTTSError(
                f"FACodec worker exited with code {process.returncode}.\n"
                f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )
        if not output.is_file() or not facodec_report.is_file():
            raise DAQTTSError("FACodec worker did not create its output and report")
        facodec = json.loads(facodec_report.read_text(encoding="utf-8"))
        if facodec.get("status") != "FACODEC_V2_VOICE_CONVERSION_PASS":
            raise DAQTTSError("FACodec report does not contain FACODEC_V2_VOICE_CONVERSION_PASS")
        if facodec.get("source_audio", {}).get("sha256") != sha256_file(raw):
            raise DAQTTSError("FACodec input commitment differs from TransformerTTS WAV")
        if facodec.get("reference_audio", {}).get("sha256") != sha256_file(self.config.reference_wav):
            raise DAQTTSError("FACodec reference commitment differs from configured reference WAV")
        if facodec.get("output_audio", {}).get("sha256") != sha256_file(output):
            raise DAQTTSError("FACodec output commitment differs from carrier WAV")

        combined = {
            "status": "DAQ_TTS_REFERENCE_CONDITIONED_PASS",
            "schema": "zsc-tsd.daq-tts.v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "definition": {
                "project_name": "DAQ-TTS",
                "composition": ["TransformerTTS", "FACodec"],
                "novel_algorithm_claimed": False,
                "facodec_role": "V2 zero-shot voice conversion using a natural reference WAV",
                "source_residual_codes_used": False,
            },
            "path_executed": [
                "TransformerTTS text-to-Mel",
                "TransformerTTS Griffin-Lim waveform",
                "FACodecEncoderV2(source and reference)",
                "FACodecDecoderV2 vector quantizer",
                "FACodecDecoderV2.vq2emb(source,use_residual=False)",
                "FACodecDecoderV2.inference(source_codes,reference_speaker_embedding)",
            ],
            "transformertts": {
                "raw_wav": str(raw),
                "report": str(raw_report),
                "raw_wav_sha256": tts_result.audio_sha256,
                "sample_rate_hz": tts_result.sample_rate,
                "duration_seconds": tts_result.duration_seconds,
            },
            "facodec": facodec,
            "reference": {
                "path": str(self.config.reference_wav.resolve()),
                "sha256": sha256_file(self.config.reference_wav),
            },
            "carrier": {
                "path": str(output),
                "sha256": sha256_file(output),
                "sample_rate_hz": facodec["output_audio"]["sample_rate_hz"],
                "duration_seconds": facodec["output_audio"]["duration_seconds"],
            },
        }
        report.write_text(
            json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return DAQTTSResult(
            output_path=output,
            report_path=report,
            audio_sha256=sha256_file(output),
            transformertts_raw_path=raw,
            transformertts_report_path=raw_report,
            facodec_report_path=facodec_report,
            report=combined,
        )


__all__ = ["DAQTTSClient", "DAQTTSConfig", "DAQTTSError", "DAQTTSResult"]
