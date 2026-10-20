#!/usr/bin/env python
"""Decode received RTP/Opus carriers and finalize the secure fragment batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from zts_secure_fragment_protocol_v1 import (
    FIXED_SECRET,
    MAX_MISSING_SHARDS,
    PROTOCOL_NAME,
    RS_CODEWORD_BYTES,
    RS_PARITY_BYTES,
    SHARD_COUNT,
    SHARD_DATA_BYTES,
    decrypt_packet,
    reconstruct_packet,
    sha256_bytes,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from reedsolo import RSCodec

# r3_bundle_geometry_applied


def bundle_geometry(private: dict[str, Any]) -> dict[str, Any]:
    """Shard geometry declared by the frozen bundle."""

    sharding = private.get("sharding", {})
    rs = private.get("reed_solomon", {})
    data_bytes = int(sharding.get("data_bytes_per_shard", SHARD_DATA_BYTES))
    shard_count = int(sharding.get("count", SHARD_COUNT))
    codeword_bytes = int(rs.get("codeword_bytes", RS_CODEWORD_BYTES))
    parity_bytes = int(rs.get("parity_bytes", RS_PARITY_BYTES))
    max_missing = int(rs.get("maximum_missing_shards", MAX_MISSING_SHARDS))
    frame_bits = int(sharding.get("frame_bits", 0))
    return {
        "data_bytes": data_bytes,
        "shard_count": shard_count,
        "codeword_bytes": codeword_bytes,
        "parity_bytes": parity_bytes,
        "max_missing": max_missing,
        "frame_bytes": (frame_bits // 8) if frame_bits else None,
        "required": (codeword_bytes - parity_bytes) // data_bytes,
    }


ROOT = Path(__file__).resolve().parents[1]
ASR_WORKER = ROOT / "tools" / "whisper_asr_worker.py"
CANDIDATE_FRONTEND = ROOT / "tools" / "native_asr_candidate_frontend_v7_rev2_dev.py"
SENTENCE_FRONTEND = ROOT / "tools" / "imdb_sentence_boundary_frontend_v6_frozen.py"
FRAGMENT_RECEIVER = ROOT / "tools" / "zts_fragment_receiver_v1_dev.py"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(command: list[str]) -> None:
    print("RUN: " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"subprocess exited with code {completed.returncode}: "
            + subprocess.list2cmdline(command)
        )


def require_candidates(path: Path) -> tuple[dict[str, Any], list[Any]]:
    report = read_json(path)
    rows = report.get("candidates")
    if not isinstance(rows, list) or report.get("candidate_count") != len(rows):
        raise RuntimeError(f"invalid candidate report: {path}")
    return report, rows


def carrier(args: argparse.Namespace) -> int:
    audio = args.audio.resolve()
    sender_path = args.sender_json.resolve()
    if not audio.is_file() or not sender_path.is_file():
        raise FileNotFoundError(audio if not audio.is_file() else sender_path)
    sender = read_json(sender_path)
    frame_bytes = None
    if getattr(args, "bundle_private", None) is not None:
        geometry = bundle_geometry(read_json(args.bundle_private.resolve()))
        frame_bytes = geometry["frame_bytes"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    asr_path = output / "asr.json"
    run_checked(
        [
            str(args.asr_python.resolve()),
            str(ASR_WORKER),
            "--model-dir",
            str(args.asr_model.resolve()),
            "--audio",
            str(audio),
            "--output",
            str(asr_path),
        ]
    )
    v7_path = output / "candidates_v7_rev2.json"
    run_checked(
        [
            sys.executable,
            str(CANDIDATE_FRONTEND),
            "--text-json",
            str(asr_path),
            "--output",
            str(v7_path),
            "--max-candidates",
            str(args.max_v7_candidates),
        ]
    )
    v7, v7_rows = require_candidates(v7_path)
    v6_rows: list[Any] = []
    receiver_result: dict[str, Any] = {}
    recovered_hex: str | None = None
    erasure_stage: str | None = None
    if v7.get("truncated") is not False:
        erasure_stage = "v7"
    else:
        v6_path = output / "candidates_v6.json"
        run_checked(
            [
                sys.executable,
                str(SENTENCE_FRONTEND),
                "--input",
                str(v7_path),
                "--output",
                str(v6_path),
                "--max-candidates",
                str(args.max_v6_candidates),
            ]
        )
        v6, v6_rows = require_candidates(v6_path)
        if v6.get("truncated") is not False:
            erasure_stage = "v6"
        else:
            receiver_path = output / "fragment_receiver.json"
            receiver_command = [
                sys.executable,
                str(FRAGMENT_RECEIVER),
                "--candidates",
                str(v6_path),
                "--prompt",
                str(sender["prompt"]),
                "--expected-shard-index",
                str(int(sender["shard_index"])),
                "--output",
                str(receiver_path),
            ]
            if frame_bytes:
                receiver_command += ["--expected-frame-bytes", str(frame_bytes)]
            run_checked(receiver_command)
            receiver_result = read_json(receiver_path)
            distinct = receiver_result.get("distinct_crc16_valid_shard_data")
            if (
                receiver_result.get("status") == "ZSC_TSD_FRAGMENT_RECEIVER_V1_PASS"
                and receiver_result.get("unambiguous_shard_data") is True
                and isinstance(distinct, list)
                and len(distinct) == 1
                and isinstance(distinct[0], str)
            ):
                recovered_hex = distinct[0]
    receiver_pass = recovered_hex is not None
    result = {
        "schema": "zsc-tsd.voip-carrier-postdecode.v1",
        "status": (
            "ZSC_TSD_VOIP_CARRIER_DECODE_PASS"
            if receiver_pass
            else "ZSC_TSD_VOIP_CARRIER_ERASURE"
        ),
        "carrier_id": str(sender["carrier_id"]),
        "shard_index": int(sender["shard_index"]),
        "attempt_number": int(sender["attempt_number"]),
        "received_wav": {
            "path": str(audio),
            "bytes": audio.stat().st_size,
            "sha256": sha256_file(audio),
        },
        "asr_text": read_json(asr_path).get("transcription", {}).get("text"),
        "candidate_v7_count": len(v7_rows),
        "candidate_v6_count": len(v6_rows) if erasure_stage != "v7" else None,
        "candidate_space_truncated_stage": erasure_stage,
        "receiver_status": receiver_result.get("status"),
        "receiver_pass": receiver_pass,
        "recovered_data_hex": recovered_hex,
        "exact_shard_data_after_reveal": (
            recovered_hex == str(sender["shard_data_hex"]) if receiver_pass else False
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = output / "carrier_result.json"
    write_json_new(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


def finalize(args: argparse.Namespace) -> int:
    result_paths = sorted(args.results_root.resolve().rglob("carrier_result.json"))
    if not result_paths:
        raise RuntimeError("no carrier_result.json files found")
    rows = [read_json(path) for path in result_paths]
    recovered: dict[int, bytes] = {}
    for row in rows:
        if row.get("receiver_pass") is True:
            recovered.setdefault(
                int(row["shard_index"]), bytes.fromhex(str(row["recovered_data_hex"]))
            )
    private = read_json(args.bundle_private.resolve())
    if private.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("private secure bundle protocol mismatch")
    key = bytes.fromhex(str(private["aes_key_hex"]))
    commitments = private.get("commitments", {})
    if sha256_bytes(key) != commitments.get("aes_key_sha256"):
        raise RuntimeError("private secure bundle key commitment mismatch")
    geometry = bundle_geometry(private)
    aad = bytes.fromhex(str(private.get("aad_hex", "")))
    expected_secret = str(private.get("fixed_secret_text", "")).encode("utf-8")
    if not expected_secret:
        expected_secret = FIXED_SECRET
    rs_pass = False
    aes_pass = False
    plaintext: bytes | None = None
    error: str | None = None
    try:
        data_bytes = geometry["data_bytes"]
        shard_count = geometry["shard_count"]
        if data_bytes == SHARD_DATA_BYTES and shard_count == SHARD_COUNT:
            packet = reconstruct_packet(recovered)
        else:
            codeword = bytearray(geometry["codeword_bytes"])
            erasures: list[int] = []
            for position in range(shard_count):
                if position in recovered:
                    start = data_bytes * position
                    codeword[start:start + data_bytes] = recovered[position]
                else:
                    erasures.extend(range(data_bytes * position, data_bytes * (position + 1)))
            if shard_count - len(recovered) > geometry["max_missing"]:
                raise ValueError(
                    f"too many missing shards: {shard_count - len(recovered)}; "
                    f"maximum is {geometry['max_missing']}"
                )
            decoded, _, _ = RSCodec(geometry["parity_bytes"]).decode(
                bytes(codeword), erase_pos=erasures
            )
            packet = bytes(decoded)
        rs_pass = True
        if aad:
            plaintext = AESGCM(key).decrypt(packet[:12], packet[12:], aad)
        else:
            plaintext = decrypt_packet(key=key, packet=packet)
        aes_pass = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    plaintext_exact = plaintext == expected_secret
    carrier_passes = sum(row.get("receiver_pass") is True for row in rows)
    exact_carriers = sum(row.get("exact_shard_data_after_reveal") is True for row in rows)
    required = geometry["required"]
    passed = (
        len(recovered) >= required and rs_pass and aes_pass and plaintext_exact
    )
    result = {
        "schema": "zsc-tsd.voip-secure-batch-result.v1",
        "status": (
            "ZSC_TSD_VOIP_SECURE_BATCH_END_TO_END_PASS"
            if passed
            else "ZSC_TSD_VOIP_SECURE_BATCH_END_TO_END_FAIL"
        ),
        "receiver_decode_pass": len(recovered) >= required,
        "rs_reconstruction_pass": rs_pass,
        "aes_gcm_authentication_pass": aes_pass,
        "plaintext_exact_match": plaintext_exact,
        "fixed_secret_sha256": sha256_bytes(expected_secret),
        "fixed_secret_text_after_reveal": (
            plaintext.decode("utf-8") if plaintext is not None else None
        ),
        "recovery_error": error,
        "shards_required": required,
        "shards_total": geometry["shard_count"],
        "unique_shards_recovered": len(recovered),
        "missing_shard_indices": [
            i for i in range(geometry["shard_count"]) if i not in recovered
        ],
        "shard_geometry": geometry,
        "total_voip_carriers": len(rows),
        "accepted_voip_carriers": carrier_passes,
        "exact_voip_carriers_after_reveal": exact_carriers,
        "voip_carrier_acceptance_rate": carrier_passes / len(rows),
        "scope": {
            "network_transport_included": True,
            "rtp_udp_opus_included": True,
            "sip_signaling_included": False,
            "asr_after_network_capture": True,
            "text_fragment_recovery_after_asr": True,
            "reed_solomon_after_network": True,
            "aes_gcm_after_network": True,
        },
        "carrier_result_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in result_paths
        ],
    }
    write_json_new(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("carrier")
    item.add_argument("--audio", required=True, type=Path)
    item.add_argument("--sender-json", required=True, type=Path)
    item.add_argument("--output-dir", required=True, type=Path)
    item.add_argument("--asr-python", required=True, type=Path)
    item.add_argument("--asr-model", required=True, type=Path)
    item.add_argument("--bundle-private", type=Path)
    item.add_argument("--max-v7-candidates", type=int, default=4096)
    item.add_argument("--max-v6-candidates", type=int, default=32768)
    end = commands.add_parser("finalize")
    end.add_argument("--results-root", required=True, type=Path)
    end.add_argument("--bundle-private", required=True, type=Path)
    end.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return carrier(args) if args.command == "carrier" else finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
