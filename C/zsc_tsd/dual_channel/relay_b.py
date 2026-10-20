"""Node B relay for the authenticated two-host RTP/Opus experiment."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..control.postgres import PostgresControlStore
from ..manifest import write_json_new
from ..transport import RTPProfile, sender_command
from .control import ControlClient
from .evidence import runtime_document, utc_now, wav_metadata, write_evidence_manifest
from .media import download_carrier
from .protocol import PROTOCOL, DualChannelError, read_control_environment, require_uuid


@dataclass(frozen=True, slots=True)
class RelayBConfig:
    session_id: str
    correlation_id: str
    output_base: Path
    ffmpeg: Path
    expected_a_ip: str
    b_ip: str
    event_ttl_seconds: int = 3600
    wait_timeout_seconds: int = 300


def run_relay(config: RelayBConfig) -> dict[str, Any]:
    session_id = require_uuid(config.session_id, field="session_id")
    correlation_id = require_uuid(config.correlation_id, field="correlation_id")
    ffmpeg = config.ffmpeg.resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    root = config.output_base.resolve() / f"dual_channel_{session_id}_relay_b"
    root.mkdir(parents=True, exist_ok=False)
    evidence = root / "evidence"
    media = root / "media"
    logs = root / "logs"
    for path in (evidence, media, logs):
        path.mkdir()
    dsn, master_key = read_control_environment()
    store = PostgresControlStore(dsn, master_key)
    client = ControlClient(
        store=store,
        master_key=master_key,
        session_id=session_id,
        actor_id="zsc-voip-b",
        correlation_id=correlation_id,
        evidence_directory=evidence,
        ttl_seconds=config.event_ttl_seconds,
    )
    runtime_path = write_json_new(
        evidence / "runtime_b.json",
        runtime_document(node_id="zsc-voip-b", ffmpeg=ffmpeg),
    )
    try:
        carrier_event = client.wait(
            "CARRIER_PREPARED",
            timeout_seconds=config.wait_timeout_seconds,
            expected_actor="zsc-voip-a",
        )
        body = carrier_event["body"]
        if body.get("protocol") != PROTOCOL:
            raise DualChannelError("unsupported dual-channel protocol")
        commitment = body.get("carrier_commitment")
        provisioning = body.get("provisioning")
        receiver = body.get("rtp_receiver")
        if not all(isinstance(value, dict) for value in (commitment, provisioning, receiver)):
            raise DualChannelError("CARRIER_PREPARED metadata is incomplete")
        received = download_carrier(
            str(provisioning["url"]),
            media / "carrier_from_a.wav",
            expected_server_ip=config.expected_a_ip,
            expected_sha256=str(commitment["sha256"]),
            expected_bytes=int(commitment["bytes"]),
            timeout_seconds=config.wait_timeout_seconds,
        )
        relay_meta = wav_metadata(received.path)
        client.publish(
            "RELAY_MEDIA_VERIFIED",
            {
                "protocol": PROTOCOL,
                "phase": "RELAY_MEDIA_VERIFIED",
                "carrier_id": str(body["carrier_id"]),
                "carrier_sha256": relay_meta["sha256"],
                "carrier_bytes": relay_meta["bytes"],
                "verification": {
                    "sha256_pass": relay_meta["sha256"] == commitment["sha256"],
                    "byte_count_pass": relay_meta["bytes"] == commitment["bytes"],
                    "wav_parse_pass": True,
                },
                "provisioning_counted_as_voip_media": False,
            },
        )
        ready = client.wait(
            "RECEIVER_RTP_READY",
            timeout_seconds=config.wait_timeout_seconds,
            expected_actor="zsc-voip-a",
        )
        ready_body = ready["body"]
        if ready_body.get("receiver_ip") != config.expected_a_ip:
            raise DualChannelError("A readiness event contains an unexpected receiver IP")
        profile = RTPProfile(
            destination_ip=config.expected_a_ip,
            port=int(ready_body["rtp_port"]),
            payload_type=int(ready_body["payload_type"]),
        )
        command = sender_command(ffmpeg, received.path, profile)
        client.publish(
            "RELAY_RTP_STARTED",
            {
                "protocol": PROTOCOL,
                "phase": "RELAY_RTP_STARTED",
                "destination_ip": profile.destination_ip,
                "destination_port": profile.port,
                "codec": "Opus",
                "bitrate_bps": profile.bitrate_bps,
                "frame_duration_ms": profile.frame_duration_ms,
            },
        )
        log_path = logs / "ffmpeg_sender_b.log"
        started = time.perf_counter()
        with log_path.open("xb") as log_stream:
            completed = subprocess.run(
                command,
                cwd=str(root),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        client.publish(
            "RELAY_RTP_FINISHED",
            {
                "protocol": PROTOCOL,
                "phase": "RELAY_RTP_FINISHED",
                "ffmpeg_returncode": completed.returncode,
                "elapsed_ms": elapsed_ms,
                "source_wav_sha256": relay_meta["sha256"],
            },
        )
        status = (
            "ZSC_TSD_DUAL_CHANNEL_RELAY_PASS"
            if completed.returncode == 0
            else "ZSC_TSD_DUAL_CHANNEL_RELAY_FAIL"
        )
        result = {
            "schema": "zsc-tsd.dual-channel-relay-result.v2",
            "status": status,
            "protocol": PROTOCOL,
            "session_id": session_id,
            "correlation_id": correlation_id,
            "carrier": relay_meta,
            "rtp": {
                "destination_ip": profile.destination_ip,
                "destination_port": profile.port,
                "ffmpeg_returncode": completed.returncode,
                "elapsed_ms": elapsed_ms,
            },
            "created_at_utc": utc_now(),
        }
        result_path = write_json_new(evidence / "relay_result.json", result)
        write_evidence_manifest(
            evidence / "evidence_manifest_b.json",
            experiment="dual_channel_v2_relay_b",
            root=root,
            files=[runtime_path, result_path, received.path, log_path, *evidence.glob("*.json")],
            metadata={"session_id": session_id, "status": status},
        )
        if completed.returncode != 0:
            raise DualChannelError("B FFmpeg RTP sender failed")
        return {"status": status, "run_directory": str(root), **result}
    except Exception as exc:
        try:
            client.publish(
                "SESSION_ABORTED",
                {
                    "protocol": PROTOCOL,
                    "phase": "SESSION_ABORTED",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        except Exception:
            pass
        raise
