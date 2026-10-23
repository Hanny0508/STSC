"""Parallel driver for the FACodec end-to-end batch (v1.4 runner).

The original runner is kept unchanged.  This driver launches several
zts_secure_imdb_batch_runner_v1_4_parallel.py workers, each covering a
disjoint range of shards, then merges the recovered shard data and performs
Reed-Solomon reconstruction and AES-GCM authentication on the merged set.

Wall-clock time of the parallel batch is the maximum of the worker elapsed
times, so the reported time is a real measurement, not an extrapolation.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from zts_secure_fragment_protocol_v1 import (  # noqa: E402
    FIXED_SECRET,
    SHARD_COUNT,
    decrypt_packet,
    reconstruct_packet,
    sha256_bytes,
    sha256_file,
)


def load_bundle(private_path: Path) -> dict:
    import json as _json

    data = _json.loads(private_path.read_text(encoding="utf-8"))
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel FACodec end-to-end batch driver.")
    parser.add_argument("--bundle-public", required=True, type=Path)
    parser.add_argument("--bundle-private", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--runner", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shard-total", type=int, default=SHARD_COUNT)
    parser.add_argument("--start-row", type=int, default=111)
    parser.add_argument("--max-attempts-per-shard", type=int, default=16)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--daqtts", action="store_true", help="wrap the v1.5 DAQ-TTS (FACodec) runner")
    parser.add_argument("--facodec-python", type=Path)
    parser.add_argument("--reference-wav", type=Path)
    parser.add_argument("--imdb-csv", type=Path)
    parser.add_argument("--imdb-manifest", type=Path)
    args = parser.parse_args()

    if args.runner is None:
        args.runner = TOOLS / (
            "zts_secure_imdb_daqtts_batch_runner_v1_5_parallel.py"
            if args.daqtts
            else "zts_secure_imdb_batch_runner_v1_4_parallel.py"
        )
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.workers > args.shard_total:
        raise SystemExit("--workers must not exceed --shard-total")
    if args.max_attempts_per_shard < 1:
        raise SystemExit("--max-attempts-per-shard must be positive")

    bundle = load_bundle(args.bundle_private.resolve())
    key = bytes.fromhex(str(bundle["aes_key_hex"]))

    chunk = math.ceil(args.shard_total / args.workers)
    ranges = []
    row_cursor = args.start_row
    for worker in range(args.workers):
        start = worker * chunk
        end = min(args.shard_total, start + chunk)
        if start >= end:
            break
        rows = (end - start) * args.max_attempts_per_shard
        ranges.append((worker, start, end, row_cursor))
        row_cursor += rows

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    processes = []
    for worker, start, end, start_row in ranges:
        out_dir = root / f"worker_{worker:02d}"
        if out_dir.exists():
            raise SystemExit(f"refusing to overwrite {out_dir}")
        command = [
            str(args.python.resolve()),
            str(args.runner.resolve()),
            "--bundle-public", str(args.bundle_public.resolve()),
            "--bundle-private", str(args.bundle_private.resolve()),
            "--start-row", str(start_row),
            "--max-attempts-per-shard", str(args.max_attempts_per_shard),
            "--shard-start", str(start),
            "--shard-end", str(end),
            "--output-dir", str(out_dir),
            "--parallel-worker",
        ]
        if args.daqtts:
            for name in ("imdb_csv", "imdb_manifest", "reference_wav"):
                value = getattr(args, name)
                if value is None:
                    raise SystemExit(f"--{name.replace('_', '-')} is required with --daqtts")
                command += ["--" + name.replace("_", "-"), str(value.resolve())]
            if args.facodec_python is not None:
                command += ["--facodec-python", str(args.facodec_python.resolve())]
        log_path = root / f"worker_{worker:02d}.log"
        log_stream = log_path.open("wb")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker, start, end, process, log_stream, log_path))
        print(f"WORKER {worker}: shards {start}..{end - 1} start_row={start_row}", flush=True)

    worker_reports = []
    for worker, start, end, process, log_stream, log_path in processes:
        code = process.wait()
        log_stream.close()
        print(f"WORKER {worker} exited with {code}", flush=True)
        report_path = root / f"worker_{worker:02d}" / "public" / "worker_result.json"
        if code != 0 or not report_path.is_file():
            for row in worker_reports:
                print("recovered so far:", row.get("shard_start"), "-", row.get("shard_end"))
            raise SystemExit(f"worker {worker} failed; see {log_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        worker_reports.append(report)

    elapsed = time.perf_counter() - started
    recovered: dict[int, bytes] = {}
    carrier_results = []
    for report in worker_reports:
        for index_text, hex_text in (report.get("recovered_shards_hex") or {}).items():
            recovered[int(index_text)] = bytes.fromhex(str(hex_text))
        carrier_results.extend(report.get("carrier_results") or [])

    packet = None
    plaintext = None
    rs_pass = False
    aes_pass = False
    error = None
    try:
        packet = reconstruct_packet(recovered)
        rs_pass = True
        plaintext = decrypt_packet(key=key, packet=packet)
        aes_pass = True
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    plaintext_exact = plaintext == FIXED_SECRET
    accepted = sum(1 for row in carrier_results if row.get("receiver_pass"))
    exact = sum(1 for row in carrier_results if row.get("exact_shard_data_after_reveal"))
    passed = (
        len(recovered) >= 22
        and rs_pass
        and aes_pass
        and plaintext_exact
    )
    result = {
        "schema": "zsc-tsd.parallel-facodec-batch-result.v1",
        "status": (
            "ZSC_TSD_SECURE_IMDB_BATCH_V1_4_PARALLEL_PASS"
            if passed
            else "ZSC_TSD_SECURE_IMDB_BATCH_V1_4_PARALLEL_FAIL"
        ),
        "runner_revision": "1.4-parallel",
        "workers": [{"worker": worker, "shard_start": start, "shard_end": end}
                    for worker, start, end, _row in ranges],
        "shard_total": args.shard_total,
        "unique_shards_recovered": len(recovered),
        "missing_shard_indices": [i for i in range(args.shard_total) if i not in recovered],
        "rs_reconstruction_pass": rs_pass,
        "aes_gcm_authentication_pass": aes_pass,
        "plaintext_exact_match": plaintext_exact,
        "fixed_secret_text_after_reveal": (
            plaintext.decode("utf-8") if plaintext is not None else None
        ),
        "recovery_error": error,
        "total_carriers": len(carrier_results),
        "accepted_carriers": accepted,
        "carrier_acceptance_rate": (accepted / len(carrier_results)) if carrier_results else 0.0,
        "carrier_exact_recovery_rate_after_reveal": (exact / len(carrier_results)) if carrier_results else 0.0,
        "parallel_wall_clock_seconds": elapsed,
        "worker_elapsed_seconds": {
            row.get("shard_start", -1): row.get("overall_elapsed_seconds") for row in worker_reports
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_results": [
            {"path": str(root / f"worker_{row.get('shard_start', i):02d}" / "public" / "worker_result.json")}
            for i, row in enumerate(worker_reports)
        ],
        "bundle_private": str(args.bundle_private.resolve()),
        "bundle_private_sha256": sha256_file(args.bundle_private.resolve()),
        "fixed_secret_sha256": sha256_bytes(FIXED_SECRET),
    }
    result_path = root / "final_result_parallel.json"
    if result_path.exists():
        raise SystemExit(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("PARALLEL BATCH:", result["status"])
    print("workers:", len(worker_reports))
    print("unique shards recovered:", len(recovered), "/", args.shard_total)
    print("total carriers:", len(carrier_results), "accepted:", accepted)
    print("RS reconstruction:", rs_pass, "AES-GCM:", aes_pass, "plaintext exact:", plaintext_exact)
    print("parallel wall clock seconds:", round(elapsed, 2), "=", round(elapsed / 60, 2), "min")
    print("result:", result_path)
    print("=" * 70)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
