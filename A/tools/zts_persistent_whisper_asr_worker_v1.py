from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_WORKER = ROOT / "tools" / "whisper_asr_worker.py"

READY = "ZSC_TSD_PERSISTENT_WHISPER_ASR_WORKER_V1_READY"
OK = "ZSC_TSD_PERSISTENT_WHISPER_ASR_WORKER_V1_OK"
STOPPED = "ZSC_TSD_PERSISTENT_WHISPER_ASR_WORKER_V1_STOPPED"
ERROR = "ZSC_TSD_PERSISTENT_WHISPER_ASR_WORKER_V1_ERROR"


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def load_original():
    spec = importlib.util.spec_from_file_location(
        "zsc_tsd_persistent_asr_original",
        ORIGINAL_WORKER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load original ASR worker: {ORIGINAL_WORKER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()

    original = load_original()

    required = (
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "preprocessor_config.json",
        model_dir / "tokenizer.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required ASR model files are missing:\n- " + "\n- ".join(missing)
        )
    if not original.torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this ASR worker")

    original_processor_class = original.AutoProcessor
    original_model_class = original.AutoModelForSpeechSeq2Seq

    load_started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        shared_processor = original_processor_class.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        shared_model = (
            original_model_class.from_pretrained(
                model_dir,
                torch_dtype=original.torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=True,
            )
            .to("cuda")
            .eval()
        )
    model_load_seconds = time.perf_counter() - load_started

    class ProcessorProxy:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any):
            return shared_processor

    class ModelProxy:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any):
            return shared_model

    original.AutoProcessor = ProcessorProxy
    original.AutoModelForSpeechSeq2Seq = ModelProxy

    emit(
        {
            "status": READY,
            "model_load_seconds": model_load_seconds,
            "pid": os.getpid(),
            "model_dir": str(model_dir),
        }
    )

    request_number = 0
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_number += 1
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("request must be a JSON object")

            op = request.get("op")
            if op == "shutdown":
                allowed = {"op"}
                extra = set(request) - allowed
                if extra:
                    raise ValueError(
                        "unexpected shutdown fields: " + ", ".join(sorted(extra))
                    )
                emit(
                    {
                        "status": STOPPED,
                        "request_number": request_number,
                    }
                )
                return

            if op != "transcribe":
                raise ValueError(f"unsupported op: {op!r}")

            allowed = {"op", "audio", "output"}
            extra = set(request) - allowed
            if extra:
                raise ValueError(
                    "unexpected transcribe fields: " + ", ".join(sorted(extra))
                )

            audio = request.get("audio")
            output = request.get("output")
            if not isinstance(audio, str) or not audio:
                raise TypeError("audio must be a nonempty string")
            if not isinstance(output, str) or not output:
                raise TypeError("output must be a nonempty string")

            audio_path = Path(audio).resolve()
            output_path = Path(output).resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            if output_path.exists():
                raise FileExistsError(output_path)

            old_argv = sys.argv[:]
            sys.argv = [
                str(ORIGINAL_WORKER),
                "--model-dir",
                str(model_dir),
                "--audio",
                str(audio_path),
                "--output",
                str(output_path),
            ]
            started = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    original.main()
            finally:
                sys.argv = old_argv
            elapsed_seconds = time.perf_counter() - started

            emit(
                {
                    "status": OK,
                    "request_number": request_number,
                    "elapsed_seconds": elapsed_seconds,
                    "output": str(output_path),
                }
            )
        except Exception as exc:
            emit(
                {
                    "status": ERROR,
                    "request_number": request_number,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise


if __name__ == "__main__":
    main()
