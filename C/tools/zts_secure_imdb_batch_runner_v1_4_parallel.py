#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from zsc_tsd.speech.transformertts import TransformerTTSClient
from zsc_tsd.zts.adapter import ZTSModelAdapter
from zsc_tsd.zts.decoder import ZTSDecoder
from zsc_tsd.zts.encoder import ZTSEncoder, ZTSEncodingError

from zts_secure_fragment_protocol_v1 import (
    FIXED_SECRET,
    PROTOCOL_NAME,
    SHARD_COUNT,
    SHARD_FRAME_BYTES,
    decode_shard_frame,
    decrypt_packet,
    reconstruct_packet,
    sha256_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_START_ROW = 111
DEFAULT_MAX_ATTEMPTS_PER_SHARD = 8
DEFAULT_MAX_V7_CANDIDATES = 4096
DEFAULT_MAX_V6_CANDIDATES = 32768

IMDB_CSV = ROOT / "legacy" / "zero-shot-GLS" / "zgls_data" / "imdb.csv"
ASR_PYTHON = Path(
    os.environ.get(
        "ZSC_TSD_ASR_PYTHON",
        "python",
    )
)
ASR_WORKER = ROOT / "tools" / "whisper_asr_worker.py"
ASR_MODEL = ROOT / "artifacts" / "asr_models" / "whisper-small.en"
CANDIDATE_FRONTEND = (
    ROOT / "tools" / "native_asr_candidate_frontend_v7_rev2_dev.py"
)
V3_DEPENDENCY = (
    ROOT / "tools" / "native_asr_candidate_frontend_prospective_bounded_v3.py"
)
SENTENCE_FRONTEND = (
    ROOT / "tools" / "imdb_sentence_boundary_frontend_v6_frozen.py"
)
BASE_RECEIVER = ROOT / "tools" / "imdb_receiver_v6_prefix_dev.py"
FRAGMENT_PROTOCOL = ROOT / "tools" / "zts_secure_fragment_protocol_v1.py"
FRAGMENT_RECEIVER = ROOT / "tools" / "zts_fragment_receiver_v1_dev.py"
ASR_WEIGHTS = ASR_MODEL / "model.safetensors"

EXPECTED_HASHES = {
    IMDB_CSV: "f8a3671e6d7275aa256c5676da6e85284e17665eccd333de8c4903d2db3b5704",
    ASR_WORKER: "275d08b36ec3e02bdd7cf1f8806a8bd7ede4ff9d388b85fca0f0a948de973c3f",
    ASR_WEIGHTS: "6014ac49b506df900f66f4aca6b0801eed7245594ace97bcaf73e0ae5b863066",
    CANDIDATE_FRONTEND: "2f24f904d5eab1a2e00503b37247ec817b737e3cb175923a668dfd3129ba6d47",
    V3_DEPENDENCY: "b7c080090058566e636846c6549b276d7c8d352603cbcac14a1da28929cbda24",
    SENTENCE_FRONTEND: "8f15b6254a34fa9f56673e66d873febc4a179822e9b3bf57ca2d9f931e404ca1",
    BASE_RECEIVER: "40ad96d50a7fbd261d25064db95704adce4558eddb6fc4f5a6d02728b61f088a",
    FRAGMENT_PROTOCOL: "b610dddb9daab44599083c799e68ad5bbb55b00f685772458fba29f660e4a56c",
    FRAGMENT_RECEIVER: "17ae6dcb9212a348b4184516dc1aa770e554c02433746968dba1327c4e2bade2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-public", required=True, type=Path)
    parser.add_argument("--bundle-private", required=True, type=Path)
    parser.add_argument("--start-row", type=int, default=DEFAULT_START_ROW)
    parser.add_argument(
        "--max-attempts-per-shard",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS_PER_SHARD,
    )
    parser.add_argument(
        "--max-v7-candidates", type=int, default=DEFAULT_MAX_V7_CANDIDATES
    )
    parser.add_argument(
        "--max-v6-candidates", type=int, default=DEFAULT_MAX_V6_CANDIDATES
    )
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=SHARD_COUNT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parallel-worker", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_hashes() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                "Frozen dependency hash mismatch:\n"
                f"path={path}\nexpected={expected}\nactual={actual}"
            )
        result[path.name] = {"path": str(path), "sha256": actual}
    return result


def require_runtime_receiver_hashes() -> None:
    paths = (
        ASR_WORKER,
        CANDIDATE_FRONTEND,
        V3_DEPENDENCY,
        SENTENCE_FRONTEND,
        BASE_RECEIVER,
        FRAGMENT_PROTOCOL,
        FRAGMENT_RECEIVER,
    )
    for path in paths:
        expected = EXPECTED_HASHES[path]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                "Runtime receiver dependency hash mismatch:\n"
                f"path={path}\nexpected={expected}\nactual={actual}"
            )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_checked(command: list[str]) -> None:
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    process = subprocess.run(command, cwd=str(ROOT), check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"subprocess exited with code {process.returncode}: "
            f"{subprocess.list2cmdline(command)}"
        )


def require_candidate_report(path: Path) -> tuple[dict[str, Any], list[Any]]:
    report = read_json(path)
    candidates = report.get("candidates")
    require(isinstance(candidates, list), f"candidate list missing: {path}")
    require(
        report.get("candidate_count") == len(candidates),
        f"candidate count mismatch: {path}",
    )
    require(
        report.get("truncated") in (True, False),
        f"candidate truncation flag missing: {path}",
    )
    return report, candidates


def candidate_capacity_erasure(
    *,
    carrier_id: str,
    shard_index: int,
    attempt_number: int,
    prompt_row: dict[str, Any],
    prompt: str,
    asr_path: Path,
    stage: str,
    candidate_v7_count: int,
    candidate_v6_count: int | None,
) -> dict[str, Any]:
    return {
        "carrier_id": carrier_id,
        "shard_index": shard_index,
        "attempt_number": attempt_number,
        "imdb_row_index": prompt_row["row_index"],
        "sentence_id": prompt_row["sentence_id"],
        "prompt": prompt,
        "asr_text": read_json(asr_path).get("transcription", {}).get("text"),
        "candidate_v7_count": candidate_v7_count,
        "candidate_v6_count": candidate_v6_count,
        "candidate_spaces_complete": False,
        "candidate_space_truncated_stage": stage,
        "receiver_status": (
            "ZSC_TSD_FRAGMENT_RECEIVER_V1_"
            "CANDIDATE_SPACE_TRUNCATED_ERASURE"
        ),
        "receiver_pass": False,
        "recovered_data_hex": None,
        "exact_shard_data_after_reveal": False,
        "receiver_elapsed_seconds": None,
        "receiver_maximum_recovered_bits": None,
    }


def load_imdb_rows(start: int, count: int) -> list[dict[str, str | int]]:
    if start < 0 or count < 1:
        raise ValueError("invalid IMDB row range")
    rows: list[dict[str, str | int]] = []
    stop = start + count
    with IMDB_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index < start:
                continue
            if index >= stop:
                break
            prompt = str(row["plaintext"])
            require(bool(prompt.strip()), f"IMDB row {index} is empty")
            rows.append(
                {
                    "row_index": index,
                    "sentence_id": str(row.get("sentence_id", index)),
                    "prompt": prompt,
                }
            )
    require(len(rows) == count, "IMDB dataset does not contain requested rows")
    return rows


def verify_bundle(
    public_path: Path, private_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[bytes], bytes]:
    public = read_json(public_path)
    private = read_json(private_path)
    require(public.get("protocol") == PROTOCOL_NAME, "public protocol mismatch")
    require(private.get("protocol") == PROTOCOL_NAME, "private protocol mismatch")
    require(
        public.get("commitments") == private.get("commitments"),
        "public/private commitments differ",
    )
    commitments = public["commitments"]
    key = bytes.fromhex(str(private["aes_key_hex"]))
    require(sha256_bytes(key) == commitments["aes_key_sha256"], "key commitment failed")
    shards_value = private.get("shards")
    require(isinstance(shards_value, list), "private shard list missing")
    require(len(shards_value) == SHARD_COUNT, "private shard count mismatch")
    frames: list[bytes] = []
    for expected_index, row in enumerate(shards_value):
        require(isinstance(row, dict), "invalid private shard row")
        frame = bytes.fromhex(str(row["frame_hex"]))
        decoded = decode_shard_frame(frame, expected_index=expected_index)
        require(decoded.data.hex() == row["data_hex"], "private shard data mismatch")
        frames.append(frame)
    require(
        sha256_bytes(b"".join(frames))
        == commitments["ordered_shard_frames_sha256"],
        "ordered shard commitment failed",
    )
    return public, private, frames, key


def main() -> None:
    args = parse_args()
    if args.start_row < 4:
        raise SystemExit("--start-row must be at least 4")
    if args.max_attempts_per_shard < 1:
        raise SystemExit("--max-attempts-per-shard must be positive")
    if args.max_v7_candidates < 1 or args.max_v7_candidates > 4096:
        raise SystemExit("--max-v7-candidates must be in [1, 4096]")
    if args.max_v6_candidates < 1 or args.max_v6_candidates > 65536:
        raise SystemExit("--max-v6-candidates must be in [1, 65536]")

    # The TransformerTTS client decodes captured worker output as UTF-8.
    # Force only the worker's standard streams to UTF-8 so arbitrary
    # generated Unicode cannot fail after successful WAV/report creation.
    os.environ["PYTHONIOENCODING"] = "utf-8"

    overall_started = time.perf_counter()
    frozen_sources = require_hashes()
    runner_path = Path(__file__).resolve()
    runner_sha256 = sha256_file(runner_path)
    public_bundle_path = args.bundle_public.resolve()
    private_bundle_path = args.bundle_private.resolve()
    public_bundle, private_bundle, frames, aes_key = verify_bundle(
        public_bundle_path, private_bundle_path
    )

    maximum_carriers = (args.shard_end - args.shard_start) * args.max_attempts_per_shard
    # Prompts are read only after all dependency and secure-bundle checks.
    prompt_rows = load_imdb_rows(args.start_row, maximum_carriers)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (
            ROOT / "artifacts" / "new_e2e" / f"secure_imdb_batch_v1_4_{timestamp}"
        )
    public_dir = output_dir / "public"
    private_dir = output_dir / "sender_private"
    receiver_dir = output_dir / "receiver"
    public_dir.mkdir(parents=True, exist_ok=False)
    private_dir.mkdir()
    receiver_dir.mkdir()

    tts_source = Path(inspect.getsourcefile(TransformerTTSClient) or "").resolve()
    batch_commitment = {
        "status": "ZSC_TSD_SECURE_IMDB_BATCH_V1_4_COMMITTED",
        "runner_revision": "1.4-parallel",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL_NAME,
        "fixed_secret_sha256": sha256_bytes(FIXED_SECRET),
        "secure_bundle_public_path": str(public_bundle_path),
        "secure_bundle_public_sha256": sha256_file(public_bundle_path),
        "secure_bundle_commitments": public_bundle["commitments"],
        "imdb_start_row": args.start_row,
        "maximum_carriers": maximum_carriers,
        "maximum_attempts_per_shard": args.max_attempts_per_shard,
        "adaptive_ack_policy": "retry-erased-or-ambiguous-shard-with-next-imdb-row",
        "sender_prompt_eligibility_policy": (
            "reject-strictly-non-retokenizable-prompt-before-tts-and-use-next-row"
        ),
        "shard_count": SHARD_COUNT,
        "shard_frame_bytes": SHARD_FRAME_BYTES,
        "candidate_capacity": {
            "v7": args.max_v7_candidates,
            "v6": args.max_v6_candidates,
            "rules_changed": False,
        },
        "candidate_space_overflow_policy": {
            "action": "carrier-erasure-and-next-attempt",
            "receiver_invoked_on_truncated_space": False,
            "counts_against_attempt_limit": True,
        },
        "tts_console_encoding": {
            "PYTHONIOENCODING": "utf-8",
            "affects_audio_or_receiver_rules": False,
        },
        "development_history": {
            "v1_2_aborted_after_successful_tts_before_console_print": True,
            "v1_2_batch_commitment_sha256": (
                "4ec2d0172d9ebd60f28fd77e1dad9557728fdb3eb30864025b334e9215e3b78e"
            ),
            "v1_3_aborted_candidate_space_capacity": True,
            "v1_3_batch_commitment_sha256": (
                "46e31daca5a357ddf0e1db4694aba3feb292dafdbcca9f82777879803da083e8"
            ),
            "v1_3_freeze_manifest_sha256": (
                "b17b53283b39d04912bc1105a589614004879a6b44a84dfa9c46ed5aad1a06dc"
            ),
            "rows_20_through_110_excluded_from_v1_4_result": True,
        },
        "frozen_sources": frozen_sources,
        "runner": {"path": str(runner_path), "sha256": runner_sha256},
        "transformertts_source": {
            "path": str(tts_source),
            "sha256": sha256_file(tts_source),
        },
        "receiver_isolation": {
            "fragment_receiver_runs_in_separate_process": True,
            "fragment_receiver_receives_expected_index": True,
            "fragment_receiver_receives_aes_key": False,
            "fragment_receiver_receives_rs_codeword": False,
            "fragment_receiver_receives_sender_text": False,
            "fragment_receiver_receives_sender_token_ids": False,
            "fragment_receiver_receives_reference_text": False,
        },
    }
    commitment_path = public_dir / "batch_commitment.json"
    write_json_new(commitment_path, batch_commitment)
    print("BATCH COMMITMENT:", commitment_path, flush=True)
    print("BATCH COMMITMENT SHA256:", sha256_file(commitment_path), flush=True)

    sender_adapter = ZTSModelAdapter()
    encoder = ZTSEncoder(sender_adapter)
    decoder = ZTSDecoder(sender_adapter)
    tts = TransformerTTSClient()
    prompt_cursor = 0
    carrier_results: list[dict[str, Any]] = []
    sender_prompt_rejections: list[dict[str, Any]] = []
    recovered_shards: dict[int, bytes] = {}

    try:
        for shard_index in range(args.shard_start, args.shard_end):
            frame = frames[shard_index]
            sent_data = decode_shard_frame(
                frame, expected_index=shard_index
            ).data
            for attempt_number in range(1, args.max_attempts_per_shard + 1):
                prompt_row = prompt_rows[prompt_cursor]
                prompt_cursor += 1
                prompt = str(prompt_row["prompt"])
                carrier_id = f"shard_{shard_index:02d}_attempt_{attempt_number}"
                carrier_public = public_dir / carrier_id
                carrier_private = private_dir / carrier_id
                carrier_receiver = receiver_dir / carrier_id
                carrier_public.mkdir()
                carrier_private.mkdir()
                carrier_receiver.mkdir()
                print("=" * 70, flush=True)
                print(
                    f"SHARD {shard_index}/{SHARD_COUNT - 1}, "
                    f"ATTEMPT {attempt_number}/{args.max_attempts_per_shard}, "
                    f"IMDB ROW {prompt_row['row_index']}",
                    flush=True,
                )

                try:
                    encoded = encoder.encode(frame, prompt, strict=True)
                except ZTSEncodingError as exc:
                    rejection = {
                        "shard_index": shard_index,
                        "attempt_slot": attempt_number,
                        "imdb_row_index": prompt_row["row_index"],
                        "sentence_id": prompt_row["sentence_id"],
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                        "tts_started": False,
                        "audio_transmitted": False,
                    }
                    sender_prompt_rejections.append(rejection)
                    print(
                        "SENDER PROMPT REJECTED BEFORE TTS:",
                        json.dumps(rejection, ensure_ascii=False),
                        flush=True,
                    )
                    continue
                sender_decoded = decoder.decode(
                    encoded.stegotext,
                    prompt,
                    expected_payload_size=SHARD_FRAME_BYTES,
                    strict=True,
                )
                sender_frame = sender_decoded.payload
                sender_shard = decode_shard_frame(
                    sender_frame, expected_index=shard_index
                )
                require(sender_frame == frame, "sender exact frame roundtrip failed")
                require(sender_shard.data == sent_data, "sender shard roundtrip failed")
                write_json_new(
                    carrier_private / "sender.json",
                    {
                        "carrier_id": carrier_id,
                        "shard_index": shard_index,
                        "attempt_number": attempt_number,
                        "imdb_row_index": prompt_row["row_index"],
                        "sentence_id": prompt_row["sentence_id"],
                        "prompt": prompt,
                        "frame_hex": frame.hex(),
                        "shard_data_hex": sent_data.hex(),
                        "stegotext": encoded.stegotext,
                        "source_bit_count": encoded.source_bit_count,
                        "used_bit_count": encoded.used_bit_count,
                        "generated_token_count": encoded.generated_token_count,
                        "retokenize_exact": encoded.retokenize_exact,
                        "perplexity": encoded.perplexity,
                        "sender_exact_roundtrip": True,
                    },
                )

                wav_path = carrier_public / "carrier.wav"
                tts.synthesize(
                    encoded.stegotext,
                    wav_path,
                    report_path=carrier_private / "tts.json",
                )
                asr_path = carrier_receiver / "asr.json"
                run_checked(
                    [
                        str(ASR_PYTHON),
                        str(ASR_WORKER),
                        "--model-dir",
                        str(ASR_MODEL),
                        "--audio",
                        str(wav_path),
                        "--output",
                        str(asr_path),
                    ]
                )
                candidate_v7_path = carrier_receiver / "candidates_v7_rev2.json"
                run_checked(
                    [
                        sys.executable,
                        str(CANDIDATE_FRONTEND),
                        "--text-json",
                        str(asr_path),
                        "--output",
                        str(candidate_v7_path),
                        "--max-candidates",
                        str(args.max_v7_candidates),
                    ]
                )
                candidate_v7, candidate_v7_rows = require_candidate_report(
                    candidate_v7_path
                )
                if candidate_v7.get("truncated") is not False:
                    carrier_result = candidate_capacity_erasure(
                        carrier_id=carrier_id,
                        shard_index=shard_index,
                        attempt_number=attempt_number,
                        prompt_row=prompt_row,
                        prompt=prompt,
                        asr_path=asr_path,
                        stage="v7",
                        candidate_v7_count=len(candidate_v7_rows),
                        candidate_v6_count=None,
                    )
                    carrier_results.append(carrier_result)
                    print(
                        "CARRIER RESULT:",
                        json.dumps(carrier_result, ensure_ascii=False),
                    )
                    print(
                        f"ERASURE shard {shard_index}: "
                        "v7 candidate space truncated",
                        flush=True,
                    )
                    continue
                candidate_v6_path = carrier_receiver / "candidates_v6.json"
                run_checked(
                    [
                        sys.executable,
                        str(SENTENCE_FRONTEND),
                        "--input",
                        str(candidate_v7_path),
                        "--output",
                        str(candidate_v6_path),
                        "--max-candidates",
                        str(args.max_v6_candidates),
                    ]
                )
                candidate_v6, candidate_v6_rows = require_candidate_report(
                    candidate_v6_path
                )
                if candidate_v6.get("truncated") is not False:
                    carrier_result = candidate_capacity_erasure(
                        carrier_id=carrier_id,
                        shard_index=shard_index,
                        attempt_number=attempt_number,
                        prompt_row=prompt_row,
                        prompt=prompt,
                        asr_path=asr_path,
                        stage="v6",
                        candidate_v7_count=len(candidate_v7_rows),
                        candidate_v6_count=len(candidate_v6_rows),
                    )
                    carrier_results.append(carrier_result)
                    print(
                        "CARRIER RESULT:",
                        json.dumps(carrier_result, ensure_ascii=False),
                    )
                    print(
                        f"ERASURE shard {shard_index}: "
                        "v6 candidate space truncated",
                        flush=True,
                    )
                    continue
                result_path = carrier_receiver / "fragment_receiver.json"
                require_runtime_receiver_hashes()
                run_checked(
                    [
                        sys.executable,
                        str(FRAGMENT_RECEIVER),
                        "--candidates",
                        str(candidate_v6_path),
                        "--prompt",
                        prompt,
                        "--expected-shard-index",
                        str(shard_index),
                        "--output",
                        str(result_path),
                    ]
                )
                require_runtime_receiver_hashes()
                receiver_result = read_json(result_path)
                distinct = receiver_result.get(
                    "distinct_crc16_valid_shard_data"
                )
                receiver_pass = (
                    receiver_result.get("status")
                    == "ZSC_TSD_FRAGMENT_RECEIVER_V1_PASS"
                    and receiver_result.get("unambiguous_shard_data") is True
                    and isinstance(distinct, list)
                    and len(distinct) == 1
                    and isinstance(distinct[0], str)
                )
                recovered_data = bytes.fromhex(distinct[0]) if receiver_pass else None
                # Comparison with sender data occurs only after receiver exit.
                exact_after_reveal = recovered_data == sent_data
                carrier_result = {
                    "carrier_id": carrier_id,
                    "shard_index": shard_index,
                    "attempt_number": attempt_number,
                    "imdb_row_index": prompt_row["row_index"],
                    "sentence_id": prompt_row["sentence_id"],
                    "prompt": prompt,
                    "asr_text": read_json(asr_path).get("transcription", {}).get("text"),
                    "candidate_v7_count": len(candidate_v7_rows),
                    "candidate_v6_count": len(candidate_v6_rows),
                    "candidate_spaces_complete": (
                        candidate_v7.get("truncated") is False
                        and candidate_v6.get("truncated") is False
                    ),
                    "receiver_status": receiver_result.get("status"),
                    "receiver_pass": receiver_pass,
                    "recovered_data_hex": (
                        recovered_data.hex() if recovered_data is not None else None
                    ),
                    "exact_shard_data_after_reveal": exact_after_reveal,
                    "receiver_elapsed_seconds": receiver_result.get("elapsed_seconds"),
                    "receiver_maximum_recovered_bits": receiver_result.get(
                        "maximum_recovered_bits"
                    ),
                }
                carrier_results.append(carrier_result)
                print("CARRIER RESULT:", json.dumps(carrier_result, ensure_ascii=False))
                if receiver_pass:
                    recovered_shards[shard_index] = recovered_data
                    print(f"ACK shard {shard_index}", flush=True)
                    break
                print(f"ERASURE shard {shard_index}; retrying if allowed", flush=True)
    finally:
        sender_adapter.release()

    if args.parallel_worker:
        worker_report = {
            "status": "ZSC_TSD_SECURE_IMDB_BATCH_V1_4_PARALLEL_WORKER_DONE",
            "shard_start": args.shard_start,
            "shard_end": args.shard_end,
            "recovered_shards_hex": {
                str(index): data.hex() for index, data in recovered_shards.items()
            },
            "carrier_results": carrier_results,
            "total_carriers": len(carrier_results),
            "overall_elapsed_seconds": time.perf_counter() - overall_started,
            "imdb_start_row": args.start_row,
        }
        worker_path = public_dir / "worker_result.json"
        write_json_new(worker_path, worker_report)
        print("WORKER RESULT:", worker_path)
        print("WORKER ELAPSED:", worker_report["overall_elapsed_seconds"])
        return

    require_hashes()
    packet: bytes | None = None
    plaintext: bytes | None = None
    rs_pass = False
    aes_gcm_pass = False
    recovery_error: str | None = None
    try:
        packet = reconstruct_packet(recovered_shards)
        rs_pass = True
        plaintext = decrypt_packet(key=aes_key, packet=packet)
        aes_gcm_pass = True
    except Exception as exc:
        recovery_error = f"{type(exc).__name__}: {exc}"

    candidate_space_erasure_count = sum(
        row.get("candidate_space_truncated_stage") is not None
        for row in carrier_results
    )
    accepted_carriers = sum(row["receiver_pass"] for row in carrier_results)
    exact_carriers = sum(
        row["exact_shard_data_after_reveal"] for row in carrier_results
    )
    recovered_unique = len(recovered_shards)
    total_carriers = len(carrier_results)
    plaintext_exact = plaintext == FIXED_SECRET
    passed = (
        recovered_unique >= 22
        and rs_pass
        and aes_gcm_pass
        and plaintext_exact
    )
    final_report = {
        "status": (
            "ZSC_TSD_SECURE_IMDB_BATCH_V1_4_PASS"
            if passed
            else "ZSC_TSD_SECURE_IMDB_BATCH_V1_4_FAIL"
        ),
        "runner_revision": "1.4",
        "protocol": PROTOCOL_NAME,
        "fixed_secret_text_after_reveal": (
            plaintext.decode("utf-8") if plaintext is not None else None
        ),
        "fixed_secret_sha256": sha256_bytes(FIXED_SECRET),
        "plaintext_exact_match": plaintext_exact,
        "rs_reconstruction_pass": rs_pass,
        "aes_gcm_authentication_pass": aes_gcm_pass,
        "recovery_error": recovery_error,
        "shards_required": 22,
        "shards_total": SHARD_COUNT,
        "unique_shards_recovered": recovered_unique,
        "missing_shard_indices": [
            index for index in range(SHARD_COUNT) if index not in recovered_shards
        ],
        "total_carriers": total_carriers,
        "sender_prompt_rejection_count": len(sender_prompt_rejections),
        "sender_prompt_rejections": sender_prompt_rejections,
        "candidate_space_erasure_count": candidate_space_erasure_count,
        "accepted_carriers": accepted_carriers,
        "exact_carriers_after_reveal": exact_carriers,
        "carrier_acceptance_rate": (
            accepted_carriers / total_carriers if total_carriers else 0.0
        ),
        "carrier_exact_recovery_rate_after_reveal": (
            exact_carriers / total_carriers if total_carriers else 0.0
        ),
        "unique_shard_recovery_rate": recovered_unique / SHARD_COUNT,
        "imdb_start_row": args.start_row,
        "imdb_rows_consumed": prompt_cursor,
        "maximum_attempts_per_shard": args.max_attempts_per_shard,
        "candidate_capacity": {
            "v7": args.max_v7_candidates,
            "v6": args.max_v6_candidates,
        },
        "tts_console_encoding": {
            "PYTHONIOENCODING": "utf-8",
            "affects_audio_or_receiver_rules": False,
        },
        "rows_20_through_110_excluded_from_v1_4_result": True,
        "carrier_results": carrier_results,
        "batch_commitment": str(commitment_path),
        "batch_commitment_sha256": sha256_file(commitment_path),
        "secure_bundle_public_sha256": sha256_file(public_bundle_path),
        "secure_bundle_commitments": public_bundle["commitments"],
        "frozen_sources": frozen_sources,
        "runner": {"path": str(runner_path), "sha256": runner_sha256},
        "overall_elapsed_seconds": time.perf_counter() - overall_started,
        "scope": {
            "fixed_semantic_secret": True,
            "real_aes_256_gcm": True,
            "real_reed_solomon": True,
            "real_binary_zts_fragments": True,
            "real_imdb_prompts": True,
            "real_transformertts": True,
            "real_whisper_asr": True,
            "payload_independent_candidate_generation": True,
            "independent_fragment_receiver_processes": True,
            "adaptive_retransmission": True,
            "sender_prompt_eligibility_filter": True,
            "tts_subprocess_stdout_utf8": True,
            "candidate_space_overflow_as_erasure": True,
            "network_transport_included": False,
            "facodec_included": False,
            "nisqa_included": False,
        },
    }
    final_path = public_dir / "final_result_after_reveal.json"
    write_json_new(final_path, final_report)
    print("=" * 70)
    print("FINAL RESULT:", final_report["status"])
    print("unique shards recovered:", recovered_unique, "/", SHARD_COUNT)
    print("total carriers:", total_carriers)
    print("carrier acceptance rate:", final_report["carrier_acceptance_rate"])
    print("RS reconstruction:", rs_pass)
    print("AES-GCM authentication:", aes_gcm_pass)
    print("plaintext exact match:", plaintext_exact)
    print("final report:", final_path)
    print("=" * 70)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
