"""Node-A single retransmission receiver/request harness.

Starts a fresh RTP/Opus receiver, publishes one authenticated
RETRANSMIT_REQUESTED event, waits for the matching Node-B retransmission
events, and captures the retransmitted carrier WAV. This harness does not
invoke ASR/ZTS/RS/AES and does not modify frozen recovery semantics.
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
from zsc_tsd.transport import RTPProfile, receiver_command, receiver_sdp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_matching(
    store: PostgresControlStore,
    *,
    session_id: str,
    correlation_id: str,
    event_type: str,
    retransmission_no: int,
    expected_actor: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        events = store.list_events(
            session_id=session_id,
            event_type=event_type,
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
                f"timed out waiting for {event_type} retransmission_no={retransmission_no}"
            )
        time.sleep(min(poll_seconds, remaining))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Node-A single SQL-triggered same-carrier RTP retransmission receiver."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--correlation-id", required=True)
    parser.add_argument("--carrier-sha256", required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--a-ip", required=True)
    parser.add_argument("--rtp-port", type=int, default=5004)
    parser.add_argument("--payload-type", type=int, default=111)
    parser.add_argument("--retransmission-no", type=int, required=True, choices=(1, 2))
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()

    session_id = require_uuid(args.session_id, field="session_id")
    correlation_id = require_uuid(args.correlation_id, field="correlation_id")

    carrier_sha = args.carrier_sha256.strip().lower()
    if len(carrier_sha) != 64 or any(ch not in "0123456789abcdef" for ch in carrier_sha):
        raise DualChannelError("carrier-sha256 must be 64 lowercase/uppercase hexadecimal characters")

    ffmpeg = args.ffmpeg.resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    if not (1 <= args.rtp_port <= 65535):
        raise DualChannelError("rtp-port outside 1..65535")
    if not (0 <= args.payload_type <= 127):
        raise DualChannelError("payload-type outside 0..127")
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise DualChannelError("timeouts must be positive")

    output_wav = args.output_wav.resolve()
    evidence_dir = args.evidence_dir.resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    if output_wav.exists():
        raise FileExistsError(f"refusing to overwrite: {output_wav}")

    dsn, master_key = read_control_environment()
    store = PostgresControlStore(dsn, master_key)
    client = ControlClient(
        store=store,
        master_key=master_key,
        session_id=session_id,
        actor_id="zsc-voip-a",
        correlation_id=correlation_id,
        evidence_directory=evidence_dir,
        ttl_seconds=max(600, int(args.timeout_seconds) + 120),
    )

    profile = RTPProfile(
        destination_ip=args.a_ip,
        port=args.rtp_port,
        payload_type=args.payload_type,
    )
    sdp_path = evidence_dir / f"retransmit_{args.retransmission_no:02d}.sdp"
    sdp_path.write_text(
        receiver_sdp(profile, title=f"ZSC-TSD retransmission {args.retransmission_no}"),
        encoding="ascii",
        newline="\n",
    )

    log_path = evidence_dir / f"ffmpeg_receiver_retransmit_{args.retransmission_no:02d}.log"
    command = receiver_command(ffmpeg, sdp_path, output_wav)

    receiver_process: subprocess.Popen[bytes] | None = None
    started_at = time.perf_counter()
    try:
        with log_path.open("xb") as log_stream:
            receiver_process = subprocess.Popen(
                command,
                cwd=str(evidence_dir),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            time.sleep(1.0)
            if receiver_process.poll() is not None:
                raise DualChannelError("A FFmpeg retransmission receiver exited before request")

            request = client.publish(
                "RETRANSMIT_REQUESTED",
                {
                    "protocol": PROTOCOL,
                    "phase": "RETRANSMIT_REQUESTED",
                    "retransmission_no": args.retransmission_no,
                    "receiver_ip": args.a_ip,
                    "rtp_port": args.rtp_port,
                    "payload_type": args.payload_type,
                    "carrier_sha256": carrier_sha,
                },
            )

            started_event = wait_matching(
                store,
                session_id=session_id,
                correlation_id=correlation_id,
                event_type="RETRANSMIT_STARTED",
                retransmission_no=args.retransmission_no,
                expected_actor="zsc-voip-b",
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
            finished_event = wait_matching(
                store,
                session_id=session_id,
                correlation_id=correlation_id,
                event_type="RETRANSMIT_FINISHED",
                retransmission_no=args.retransmission_no,
                expected_actor="zsc-voip-b",
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )

            try:
                receiver_returncode = receiver_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                receiver_process.terminate()
                receiver_returncode = receiver_process.wait(timeout=10)
            receiver_process = None

        if not output_wav.is_file() or output_wav.stat().st_size <= 44:
            raise DualChannelError("A did not capture a nonempty retransmission WAV")

        finished_body = finished_event.get("body")
        sender_returncode = (
            finished_body.get("ffmpeg_returncode")
            if isinstance(finished_body, dict)
            else None
        )
        if sender_returncode != 0:
            raise DualChannelError(
                f"B retransmission sender failed with exit code {sender_returncode!r}"
            )

        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        received_sha = sha256_file(output_wav)
        result = {
            "schema": "zsc-tsd.retransmission-receiver-a.v1",
            "status": "ZSC_TSD_RETRANSMISSION_CAPTURE_PASS",
            "protocol": PROTOCOL,
            "session_id": session_id,
            "correlation_id": correlation_id,
            "retransmission_no": args.retransmission_no,
            "source_carrier_sha256": carrier_sha,
            "received_wav_sha256": received_sha,
            "received_wav_bytes": output_wav.stat().st_size,
            "receiver_ffmpeg_returncode": receiver_returncode,
            "sender_ffmpeg_returncode": sender_returncode,
            "elapsed_ms": elapsed_ms,
            "request_event_id": request["event_id"],
            "started_event_id": started_event["event_id"],
            "finished_event_id": finished_event["event_id"],
        }
        result_path = evidence_dir / f"retransmit_{args.retransmission_no:02d}_result.json"
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite: {result_path}")
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        client.publish(
            "RECEIVER_MEDIA_CAPTURED",
            {
                "protocol": PROTOCOL,
                "phase": "RECEIVER_MEDIA_CAPTURED",
                "retransmission_no": args.retransmission_no,
                "received_wav_sha256": received_sha,
                "received_wav_bytes": output_wav.stat().st_size,
                "receiver_ffmpeg_returncode": receiver_returncode,
                "relay_sender_pass": True,
            },
        )

        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        if receiver_process is not None and receiver_process.poll() is None:
            receiver_process.terminate()
            try:
                receiver_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                receiver_process.kill()
                receiver_process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
