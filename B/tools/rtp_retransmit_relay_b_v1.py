"""Reviewer harness: SQL-triggered same-carrier RTP/Opus retransmission on node B.

This harness does not generate or alter covert payloads. It only re-sends the
same cached carrier WAV after an authenticated RETRANSMIT_REQUESTED event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from zsc_tsd.control.postgres import PostgresControlStore
from zsc_tsd.transport import RTPProfile, sender_command
from zsc_tsd.dual_channel.control import ControlClient
from zsc_tsd.dual_channel.protocol import (
    PROTOCOL,
    DualChannelError,
    read_control_environment,
    require_uuid,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Node-B SQL-triggered same-carrier RTP/Opus retransmission harness."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--correlation-id", required=True)
    parser.add_argument("--carrier-wav", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--expected-a-ip", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--max-retransmissions", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def validate_request(
    event: dict[str, Any],
    *,
    expected_a_ip: str,
    carrier_sha256: str,
    maximum: int,
) -> tuple[int, RTPProfile]:
    if event.get("actor_id") != "zsc-voip-a":
        raise DualChannelError(
            f"RETRANSMIT_REQUESTED came from {event.get('actor_id')!r}, expected zsc-voip-a"
        )
    body = event.get("body")
    if not isinstance(body, dict):
        raise DualChannelError("RETRANSMIT_REQUESTED body is not an object")
    if body.get("protocol") != PROTOCOL:
        raise DualChannelError("RETRANSMIT_REQUESTED protocol mismatch")
    number = body.get("retransmission_no")
    if isinstance(number, bool) or not isinstance(number, int):
        raise DualChannelError("retransmission_no must be an integer")
    if number < 1 or number > maximum:
        raise DualChannelError(
            f"retransmission_no {number} outside allowed range 1..{maximum}"
        )
    if body.get("receiver_ip") != expected_a_ip:
        raise DualChannelError("retransmission receiver_ip does not match expected node A")
    if body.get("carrier_sha256") != carrier_sha256:
        raise DualChannelError("retransmission carrier commitment differs from cached carrier")
    port = body.get("rtp_port")
    payload_type = body.get("payload_type", 111)
    if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
        raise DualChannelError("retransmission rtp_port is invalid")
    if (
        isinstance(payload_type, bool)
        or not isinstance(payload_type, int)
        or not (0 <= payload_type <= 127)
    ):
        raise DualChannelError("retransmission payload_type is invalid")
    return number, RTPProfile(
        destination_ip=expected_a_ip,
        port=port,
        payload_type=payload_type,
    )


def main() -> int:
    args = parse_args()
    if args.max_retransmissions < 1:
        raise SystemExit("--max-retransmissions must be positive")
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        raise SystemExit("poll/timeout values must be positive")

    session_id = require_uuid(args.session_id, field="session_id")
    correlation_id = require_uuid(args.correlation_id, field="correlation_id")
    carrier = args.carrier_wav.resolve()
    ffmpeg = args.ffmpeg.resolve()
    evidence = args.evidence_dir.resolve()

    if not carrier.is_file():
        raise FileNotFoundError(carrier)
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    evidence.mkdir(parents=True, exist_ok=True)

    carrier_sha = sha256_file(carrier)
    carrier_bytes = carrier.stat().st_size

    dsn, master_key = read_control_environment()
    store = PostgresControlStore(dsn, master_key)
    client = ControlClient(
        store=store,
        master_key=master_key,
        session_id=session_id,
        actor_id="zsc-voip-b",
        correlation_id=correlation_id,
        evidence_directory=evidence,
        ttl_seconds=max(3600, int(args.timeout_seconds) + 600),
    )

    handled_ids: set[str] = set()
    handled_numbers: set[int] = set()
    deadline = time.monotonic() + args.timeout_seconds

    print(
        json.dumps(
            {
                "status": "ZSC_TSD_RETRANSMIT_RELAY_B_READY",
                "session_id": session_id,
                "correlation_id": correlation_id,
                "carrier_wav": str(carrier),
                "carrier_sha256": carrier_sha,
                "carrier_bytes": carrier_bytes,
                "max_retransmissions": args.max_retransmissions,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    while len(handled_numbers) < args.max_retransmissions:
        terminal = store.list_events(
            session_id=session_id,
            correlation_id=correlation_id,
        )
        if any(
            event["event_type"] in {"SESSION_COMPLETED", "SESSION_ABORTED"}
            for event in terminal
        ):
            print("ZSC_TSD_RETRANSMIT_RELAY_B_TERMINAL_STATE", flush=True)
            return 0

        requests = store.list_events(
            session_id=session_id,
            event_type="RETRANSMIT_REQUESTED",
            correlation_id=correlation_id,
        )
        pending = [event for event in requests if event["event_id"] not in handled_ids]
        pending.sort(key=lambda event: (event["issued_at_utc"], event["sequence_no"]))

        if pending:
            event = pending[0]
            event_id = str(event["event_id"])
            number, profile = validate_request(
                event,
                expected_a_ip=args.expected_a_ip,
                carrier_sha256=carrier_sha,
                maximum=args.max_retransmissions,
            )
            if number in handled_numbers:
                raise DualChannelError(
                    f"duplicate retransmission_no {number} with a new event_id"
                )
            expected_number = len(handled_numbers) + 1
            if number != expected_number:
                raise DualChannelError(
                    f"out-of-order retransmission request: got {number}, expected {expected_number}"
                )

            client.publish(
                "RETRANSMIT_STARTED",
                {
                    "protocol": PROTOCOL,
                    "phase": "RETRANSMIT_STARTED",
                    "retransmission_no": number,
                    "request_event_id": event_id,
                    "destination_ip": profile.destination_ip,
                    "destination_port": profile.port,
                    "payload_type": profile.payload_type,
                    "codec": "Opus",
                    "source_wav_sha256": carrier_sha,
                    "source_wav_bytes": carrier_bytes,
                },
            )

            log_path = evidence / f"retransmit_{number:02d}_ffmpeg.log"
            command = sender_command(ffmpeg, carrier, profile)
            started = time.perf_counter()
            with log_path.open("xb") as log_stream:
                completed = subprocess.run(
                    command,
                    cwd=str(evidence),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            client.publish(
                "RETRANSMIT_FINISHED",
                {
                    "protocol": PROTOCOL,
                    "phase": "RETRANSMIT_FINISHED",
                    "retransmission_no": number,
                    "request_event_id": event_id,
                    "ffmpeg_returncode": completed.returncode,
                    "elapsed_ms": elapsed_ms,
                    "source_wav_sha256": carrier_sha,
                    "source_wav_bytes": carrier_bytes,
                },
            )
            handled_ids.add(event_id)
            handled_numbers.add(number)

            print(
                json.dumps(
                    {
                        "status": (
                            "ZSC_TSD_RETRANSMIT_SEND_PASS"
                            if completed.returncode == 0
                            else "ZSC_TSD_RETRANSMIT_SEND_FAIL"
                        ),
                        "retransmission_no": number,
                        "request_event_id": event_id,
                        "ffmpeg_returncode": completed.returncode,
                        "elapsed_ms": elapsed_ms,
                        "source_wav_sha256": carrier_sha,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if completed.returncode != 0:
                return 2
            continue

        if time.monotonic() >= deadline:
            raise DualChannelError("timed out waiting for RETRANSMIT_REQUESTED")
        time.sleep(args.poll_seconds)

    print(
        json.dumps(
            {
                "status": "ZSC_TSD_RETRANSMIT_RELAY_B_MAX_REACHED",
                "retransmissions_sent": len(handled_numbers),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
