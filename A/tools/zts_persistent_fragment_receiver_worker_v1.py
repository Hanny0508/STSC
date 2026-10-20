from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
FRAGMENT_PATH = TOOLS / "zts_fragment_receiver_v1_dev.py"
BASE_PATH = TOOLS / "imdb_receiver_v6_prefix_dev.py"

for value in (str(ROOT), str(TOOLS)):
    if value not in sys.path:
        sys.path.insert(0, value)

from zsc_tsd.zts.adapter import ZTSModelAdapter


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SharedAdapterProxy:
    def __init__(self, shared: ZTSModelAdapter) -> None:
        object.__setattr__(self, "_shared", shared)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._shared, name)

    def load(self) -> "SharedAdapterProxy":
        self._shared.load()
        return self

    def configure_determinism(self) -> None:
        self._shared.configure_determinism()

    def release(self) -> None:
        # The worker owns the shared adapter lifetime.
        return None


def resolve_request_path(raw: object) -> Path:
    if not isinstance(raw, str) or raw == "":
        raise ValueError("request path must be a non-empty string")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def emit(value: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    sys.stdout.flush()


def main() -> None:
    base = load_module("persistent_worker_base_receiver_v6", BASE_PATH)
    fragment = load_module("persistent_worker_fragment_v1", FRAGMENT_PATH)

    shared = ZTSModelAdapter()
    load_started = time.perf_counter()
    shared.load()
    shared.configure_determinism()
    model_load_seconds = time.perf_counter() - load_started

    proxy = SharedAdapterProxy(shared)
    base.ZTSModelAdapter = lambda: proxy
    fragment.load_base_receiver = lambda: base

    memo = shared.model
    if not hasattr(memo, "clear_cache"):
        raise RuntimeError("loaded adapter model has no clear_cache()")

    emit(
        {
            "status": "ZSC_TSD_PERSISTENT_FRAGMENT_RECEIVER_WORKER_V1_READY",
            "model_load_seconds": model_load_seconds,
            "pid": __import__("os").getpid(),
        }
    )

    request_number = 0
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            request_number += 1
            started = time.perf_counter()
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                op = request.get("op")
                if op == "shutdown":
                    emit(
                        {
                            "status": "ZSC_TSD_PERSISTENT_FRAGMENT_RECEIVER_WORKER_V1_STOPPED",
                            "request_number": request_number,
                        }
                    )
                    return
                if op != "decode":
                    raise ValueError(f"unsupported op: {op!r}")

                allowed = {
                    "op",
                    "candidates",
                    "prompt",
                    "expected_shard_index",
                    "output",
                }
                extra = sorted(set(request) - allowed)
                if extra:
                    raise ValueError(f"unexpected request fields: {extra}")

                prompt = request.get("prompt")
                expected_index = request.get("expected_shard_index")
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")
                if not isinstance(expected_index, int):
                    raise ValueError("expected_shard_index must be an integer")

                candidate_path = resolve_request_path(request.get("candidates"))
                output_path = resolve_request_path(request.get("output"))
                if not candidate_path.is_file():
                    raise FileNotFoundError(f"candidate file missing: {candidate_path}")
                if output_path.exists():
                    raise FileExistsError(f"refusing to overwrite: {output_path}")

                memo.clear_cache()

                saved_argv = sys.argv[:]
                sys.argv = [
                    str(FRAGMENT_PATH),
                    "--candidates",
                    str(candidate_path),
                    "--prompt",
                    prompt,
                    "--expected-shard-index",
                    str(expected_index),
                    "--output",
                    str(output_path),
                ]
                captured = io.StringIO()
                try:
                    with contextlib.redirect_stdout(captured):
                        fragment.main()
                finally:
                    sys.argv = saved_argv

                elapsed = time.perf_counter() - started
                emit(
                    {
                        "status": "ZSC_TSD_PERSISTENT_FRAGMENT_RECEIVER_WORKER_V1_OK",
                        "request_number": request_number,
                        "elapsed_seconds": elapsed,
                        "output": str(output_path),
                    }
                )
            except BaseException as exc:
                emit(
                    {
                        "status": "ZSC_TSD_PERSISTENT_FRAGMENT_RECEIVER_WORKER_V1_ERROR",
                        "request_number": request_number,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
    finally:
        shared.release()


if __name__ == "__main__":
    main()
