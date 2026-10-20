from __future__ import print_function

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_WORKER = ROOT / "tools" / "transformertts_smoke.py"

READY = "ZSC_TSD_PERSISTENT_TRANSFORMERTTS_WORKER_V1_READY"
OK = "ZSC_TSD_PERSISTENT_TRANSFORMERTTS_WORKER_V1_OK"
STOPPED = "ZSC_TSD_PERSISTENT_TRANSFORMERTTS_WORKER_V1_STOPPED"
ERROR = "ZSC_TSD_PERSISTENT_TRANSFORMERTTS_WORKER_V1_ERROR"


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def load_original():
    spec = importlib.util.spec_from_file_location(
        "zsc_tsd_persistent_tts_original",
        str(ORIGINAL_WORKER),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load original TTS worker: {}".format(ORIGINAL_WORKER))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--espeak", required=True, type=Path)
    parser.add_argument("--griffin-lim-iterations", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir.resolve()
    espeak = args.espeak.resolve()

    if args.griffin_lim_iterations <= 0:
        raise ValueError("griffin-lim-iterations must be positive")

    required = (
        ORIGINAL_WORKER,
        espeak,
        model_dir / "config.yaml",
        model_dir / "model_weights.hdf5",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required persistent TTS files are missing:\n- " + "\n- ".join(missing)
        )

    original = load_original()
    original_forward_transformer = original.ForwardTransformer

    # Reproduce the fresh-worker seed state before the one shared model load.
    original.np.random.seed(args.seed)
    original.tf.random.set_seed(args.seed)
    original.EspeakBackend.set_espeak_path(str(espeak))

    load_started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        shared_model = original_forward_transformer.load_model(model_dir.as_posix())
    model_load_seconds = time.perf_counter() - load_started

    class ForwardTransformerProxy(object):
        @staticmethod
        def load_model(*_args, **_kwargs):
            return shared_model

    original.ForwardTransformer = ForwardTransformerProxy

    emit(
        {
            "status": READY,
            "pid": os.getpid(),
            "model_dir": str(model_dir),
            "espeak": str(espeak),
            "griffin_lim_iterations": int(args.griffin_lim_iterations),
            "seed": int(args.seed),
            "model_load_seconds": model_load_seconds,
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
                extra = set(request) - {"op"}
                if extra:
                    raise ValueError(
                        "unexpected shutdown fields: " + ", ".join(sorted(extra))
                    )
                emit({"status": STOPPED, "request_number": request_number})
                return

            if op != "synthesize":
                raise ValueError("unsupported op: {!r}".format(op))

            allowed = {"op", "text", "output", "report"}
            extra = set(request) - allowed
            if extra:
                raise ValueError(
                    "unexpected synthesize fields: " + ", ".join(sorted(extra))
                )

            text = request.get("text")
            output = request.get("output")
            report = request.get("report")
            if not isinstance(text, str):
                raise TypeError("text must be a string")
            if not text.strip():
                raise ValueError("text contains no pronounceable content")
            if not isinstance(output, str) or not output:
                raise TypeError("output must be a nonempty string")
            if not isinstance(report, str) or not report:
                raise TypeError("report must be a nonempty string")

            output_path = Path(output).resolve()
            report_path = Path(report).resolve()
            if output_path.exists():
                raise FileExistsError(str(output_path))
            if report_path.exists():
                raise FileExistsError(str(report_path))

            old_argv = sys.argv[:]
            sys.argv = [
                str(ORIGINAL_WORKER),
                "--model-dir",
                str(model_dir),
                "--output",
                str(output_path),
                "--report",
                str(report_path),
                "--espeak",
                str(espeak),
                "--text",
                text,
                "--griffin-lim-iterations",
                str(args.griffin_lim_iterations),
                "--seed",
                str(args.seed),
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
                    "report": str(report_path),
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
