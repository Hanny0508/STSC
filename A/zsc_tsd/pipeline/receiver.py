"""Stable receiver facade over ASR, candidate generation, and fragment decode."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..speech import WhisperASRClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ReceiverPipelineError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FragmentRecovery:
    passed: bool
    recovered_data_hex: str | None
    asr_text: str
    candidate_v7_count: int
    candidate_v6_count: int
    receiver_status: str
    report_path: Path


class ReceiverPipeline:
    def __init__(
        self,
        *,
        asr: WhisperASRClient | None = None,
        python_executable: str | Path = sys.executable,
        max_v7_candidates: int = 4096,
        max_v6_candidates: int = 32768,
    ) -> None:
        self.asr = asr or WhisperASRClient()
        self.python_executable = Path(python_executable)
        self.max_v7_candidates = max_v7_candidates
        self.max_v6_candidates = max_v6_candidates

    @staticmethod
    def _load_report(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ReceiverPipelineError(f"report is not an object: {path}")
        return value

    def _run(self, command: list[str]) -> None:
        process = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if process.returncode != 0:
            raise ReceiverPipelineError(
                f"receiver command failed ({process.returncode}): {' '.join(command)}\n"
                f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )

    def recover_fragment(
        self,
        audio_path: str | Path,
        *,
        prompt: str,
        expected_shard_index: int,
        output_dir: str | Path,
    ) -> FragmentRecovery:
        target = Path(output_dir).resolve()
        target.mkdir(parents=True, exist_ok=False)
        asr_path = target / "asr.json"
        asr_result = self.asr.transcribe(audio_path, asr_path)

        v7_path = target / "candidates_v7_rev2.json"
        self._run(
            [
                str(self.python_executable),
                str(PROJECT_ROOT / "tools" / "native_asr_candidate_frontend_v7_rev2_dev.py"),
                "--text-json", str(asr_path), "--output", str(v7_path),
                "--max-candidates", str(self.max_v7_candidates),
            ]
        )
        v7 = self._load_report(v7_path)
        if v7.get("truncated") is not False:
            raise ReceiverPipelineError("v7 candidate space was truncated")

        v6_path = target / "candidates_v6.json"
        self._run(
            [
                str(self.python_executable),
                str(PROJECT_ROOT / "tools" / "imdb_sentence_boundary_frontend_v6_frozen.py"),
                "--input", str(v7_path), "--output", str(v6_path),
                "--max-candidates", str(self.max_v6_candidates),
            ]
        )
        v6 = self._load_report(v6_path)
        if v6.get("truncated") is not False:
            raise ReceiverPipelineError("v6 candidate space was truncated")

        receiver_path = target / "fragment_receiver.json"
        self._run(
            [
                str(self.python_executable),
                str(PROJECT_ROOT / "tools" / "zts_fragment_receiver_v1_dev.py"),
                "--candidates", str(v6_path), "--prompt", prompt,
                "--expected-shard-index", str(expected_shard_index),
                "--output", str(receiver_path),
            ]
        )
        receiver = self._load_report(receiver_path)
        distinct = receiver.get("distinct_crc16_valid_shard_data")
        passed = bool(
            receiver.get("status") == "ZSC_TSD_FRAGMENT_RECEIVER_V1_PASS"
            and receiver.get("unambiguous_shard_data") is True
            and isinstance(distinct, list)
            and len(distinct) == 1
            and isinstance(distinct[0], str)
        )
        return FragmentRecovery(
            passed=passed,
            recovered_data_hex=distinct[0] if passed else None,
            asr_text=asr_result.text,
            candidate_v7_count=int(v7.get("candidate_count", 0)),
            candidate_v6_count=int(v6.get("candidate_count", 0)),
            receiver_status=str(receiver.get("status", "UNKNOWN")),
            report_path=receiver_path,
        )
