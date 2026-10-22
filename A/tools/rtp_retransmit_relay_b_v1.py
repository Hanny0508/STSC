"""Node-B responder for the SQL-triggered same-carrier RTP retransmission.

Waits for an authenticated RETRANSMIT_REQUESTED event published by node A,
re-sends the committed carrier with FFmpeg RTP/Opus (real-time, -re), and
publishes RETRANSMIT_STARTED / RETRANSMIT_FINISHED so that A can match the
retransmission attempt and verify the received carrier SHA-256.
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
from zsc_tsd.dual_channel.control import ControlClient
from zsc_tsd.dual_channel.protocol import (
    PROTOCOL,
    DualChannelError,
    read_control_environment,
    require_uuid,
)
from zsc_tsd.transport import RTPProfile, sender_command


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_request(
    store: PostgresControlStore,
    *,
    session_id: str,
    correlation_id: str,
    retransmission_no: int,
    expected_actor: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        events = store.list_events(
            session_id=session_id,
            event_type="RETRANSMIT_REQUESTED",
            correlation_id=correlation_id,
        )
        for event in events:
            body = event.get("body")
            if (
                event.get("actor_id") == expected_actor
                and isinstance(body, dict)
                and body.get("retransmission_no") == retransmission_no
            ):
                return event
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DualChannelError(
                f"timed out waiting for RETRANSMIT_REQUESTED retransmission_no={retransmission_no}"
            )
        time.sleep(min(poll_seconds, remaining))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Node-B same-carrier RTP/Opus retransmission responder."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--correlation-id", required=True)
    parser.add_argument("--carrier-wav", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--b-ip", required=True)
    parser.add_argument("--retransmission-no", type=int, required=True, choices=(1, 2))
    parser.add_argument("--output-base", type=Path, default=Path("E:/ZSC-TSD-VoIP/experiments"))
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()

    session_id = require_uuid(args.session_id, field="session_id")
    correlation_id = require_uuid(args.correlation_id, field="correlation_id")
    carrier = args.carrier_wav.resolve()
    ffmpeg = args.ffmpeg.resolve()
    if not carrier.is_file():
        raise FileNotFoundError(carrier)
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)

    dsn, master_key = read_control_environment()
    store = PostgresControlStore(dsn, master_key)
    evidence = args.output_base.resolve() / f"retransmit_b_{session_id}_{args.retransmission_no}"
    evidence.mkdir(parents=True, exist_ok=False)
    client = ControlClient(
        store=store,
        master_key=master_key,
        session_id=session_id,
        actor_id="zsc-voip-b",
        correlation_id=correlation_id,
        evidence_directory=evidence,
        ttl_seconds=max(600, int(args.timeout_seconds) + 120),
    )

    request = wait_request(
        store,
        session_id=session_id,
        correlation_id=correlation_id,
        retransmission_no=args.retransmission_no,
        expected_actor="zsc-voip-a",
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    body = request["body"]
    receiver_ip = str(body["receiver_ip"])
    rtp_port = int(body["rtp_port"])
    payload_type = int(body["payload_type"])
    expected_sha = str(body["carrier_sha256"]).lower()
    actual_sha = sha256_file(carrier)
    if actual_sha != expected_sha:
        raise DualChannelError(
            f"carrier SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )

    profile = RTPProfile(destination_ip=receiver_ip, port=rtp_port, payload_type=payload_type)
    client.publish(
        "RETRANSMIT_STARTED",
        {
            "protocol": PROTOCOL,
            "phase": "RETRANSMIT_STARTED",
            "retransmission_no": args.retransmission_no,
            "destination_ip": profile.destination_ip,
            "destination_port": profile.port,
            "payload_type": profile.payload_type,
            "carrier_sha256": actual_sha,
            "media": "RTP/UDP + Opus",
        },
    )
    command = sender_command(ffmpeg, carrier, profile)
    log_path = evidence / "ffmpeg_sender_retransmit.log"
    started = time.perf_counter()
    with log_path.open("xb") as log_stream:
        completed = subprocess.run(
            command, cwd=str(evidence), stdout=log_stream, stderr=subprocess.STDOUT, check=False
        )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if completed.returncode != 0:
        raise DualChannelError(f"B FFmpeg retransmission sender failed with exit code {completed.returncode}")

    result = {
        "schema": "zsc-tsd.dual-channel-retransmit-b-result.v1",
        "status": "ZSC_TSD_RETRANSMIT_B_PASS",
        "session_id": session_id,
        "correlation_id": correlation_id,
        "retransmission_no": args.retransmission_no,
        "carrier_sha256": actual_sha,
        "rtp": {
            "destination_ip": profile.destination_ip,
            "destination_port": profile.port,
            "payload_type": profile.payload_type,
            "elapsed_ms": elapsed_ms,
            "ffmpeg_returncode": completed.returncode,
        },
        "run_directory": str(evidence),
    }
    client.publish(
        "RETRANSMIT_FINISHED",
        {
            "protocol": PROTOCOL,
            "phase": "RETRANSMIT_FINISHED",
            "retransmission_no": args.retransmission_no,
            "carrier_sha256": actual_sha,
            "elapsed_ms": elapsed_ms,
            "ffmpeg_returncode": completed.returncode,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
