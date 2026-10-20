"""Node A coordinator for the authenticated two-host experiment."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid

# r3_sync_gate_applied
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..control.postgres import PostgresControlStore
from ..manifest import sha256_file, write_json_new
from ..transport import RTPProfile, receiver_command, receiver_sdp
from .control import ControlClient
from .evidence import runtime_document, utc_now, wav_metadata, write_evidence_manifest
from .media import CarrierProvisionServer
from .protocol import (
    PROTOCOL,
    DualChannelError,
    decode_checks,
    load_json_object,
    read_control_environment,
)


@dataclass(frozen=True, slots=True)
class CoordinatorAConfig:
    source_wav: Path
    output_base: Path
    ffmpeg: Path
    a_ip: str
    b_ip: str
    provisioning_port: int = 5602
    rtp_port: int = 5004
    event_ttl_seconds: int = 3600
    wait_timeout_seconds: int = 300
    decoder_command_json: Path | None = None
    batch_id: str | None = None
    carrier_id: str | None = None
    sync_manifest: Path | None = None
    sync_gate_python: Path | None = None
    skip_sync_gate: bool = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _discover_sync_manifest(root: Path, explicit: Path | None) -> Path | None:
    """Locate the frozen synchronization manifest that this tree commits to."""

    if explicit is not None:
        candidate = explicit if explicit.is_absolute() else (root / explicit)
        return candidate
    candidates = sorted((root / "artifacts" / "freezes").glob("*/zts_manifest*.json"))
    return candidates[-1] if candidates else None


def _run_sync_gate(manifest_path: Path, interpreter: Path | None) -> dict[str, Any]:
    """Compare this node's own profile with the frozen one, before decoding.

    The receiver reads only its *own* installation; the frozen manifest is a
    commitment to what a conforming installation looks like.  Hashing the
    checkpoint needs the model environment, so the check runs in the same
    interpreter as the decoder.
    """

    tool = PROJECT_ROOT / "tools" / "verify_sync.py"
    if not tool.is_file():
        return {
            "status": "SYNC_GATE_TOOL_MISSING",
            "passed": False,
            "differences": [f"verification tool not found: {tool}"],
        }
    python_exe = Path(interpreter) if interpreter else Path(sys.executable)
    completed = subprocess.run(
        [
            str(python_exe),
            str(tool),
            "--root",
            str(PROJECT_ROOT),
            "--manifest",
            str(manifest_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "SYNC_GATE_UNPARSABLE",
            "differences": [
                (completed.stdout or completed.stderr or "").strip()[-500:]
            ],
        }
    payload["interpreter"] = str(python_exe)
    payload["returncode"] = completed.returncode
    payload["passed"] = bool(
        completed.returncode == 0 and payload.get("status") == "SYNC_GATE_PASS"
    )
    return payload


def _new_run(base: Path, session_id: str) -> dict[str, Path]:
    root = base.resolve() / f"dual_channel_{session_id}"
    root.mkdir(parents=True, exist_ok=False)
    result = {
        "root": root,
        "public": root / "public",
        "sender_private": root / "sender_private",
        "receiver_a": root / "receiver_a",
        "relay_b": root / "relay_b",
        "events": root / "events",
        "logs": root / "logs",
    }
    for key, path in result.items():
        if key != "root":
            path.mkdir()
    return result


def _decoder_command(
    template_path: Path,
    *,
    received_wav: Path,
    output_json: Path,
    run_root: Path,
) -> list[str]:
    value = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise DualChannelError("decoder command JSON must be a nonempty string array")
    substitutions = {
        "received_wav": str(received_wav),
        "output_json": str(output_json),
        "run_root": str(run_root),
    }
    try:
        return [item.format(**substitutions) for item in value]
    except KeyError as exc:
        raise DualChannelError(f"unknown decoder command placeholder: {exc}") from exc


def run_coordinator(config: CoordinatorAConfig) -> dict[str, Any]:
    source = config.source_wav.resolve()
    ffmpeg = config.ffmpeg.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    source_meta = wav_metadata(source)
    dsn, master_key = read_control_environment()
    session_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())
    paths = _new_run(config.output_base, session_id)
    store = PostgresControlStore(dsn, master_key)
    client = ControlClient(
        store=store,
        master_key=master_key,
        session_id=session_id,
        actor_id="zsc-voip-a",
        correlation_id=correlation_id,
        evidence_directory=paths["events"],
        ttl_seconds=config.event_ttl_seconds,
    )
    runtime_path = write_json_new(
        paths["public"] / "runtime_a.json",
        runtime_document(node_id="zsc-voip-a", ffmpeg=ffmpeg),
    )
    context = {
        "schema": "zsc-tsd.dual-channel-context.v2",
        "protocol": PROTOCOL,
        "session_id": session_id,
        "correlation_id": correlation_id,
        "created_at_utc": utc_now(),
        "nodes": {"a": config.a_ip, "b": config.b_ip},
        "ports": {"provisioning_http": config.provisioning_port, "rtp_udp": config.rtp_port},
        "claims": {
            "sip_call": False,
            "rtp_opus_media": True,
            "postgresql_hmac_control": True,
            "provisioning_is_voip": False,
        },
    }
    if config.batch_id is not None:
        context["batch"] = {
            "batch_id": config.batch_id,
            "carrier_id": config.carrier_id or source_meta["sha256"][:16],
        }
    context_path = write_json_new(paths["public"] / "session_context.json", context)
    session_body = {
        "protocol": PROTOCOL,
        "phase": "SESSION_OPENED",
        "node_a": config.a_ip,
        "node_b": config.b_ip,
        "media_direction": "B_TO_A",
        "media_transport": "RTP/UDP Opus",
        "control_transport": "PostgreSQL/TCP over Tailscale",
        "secret_payload_in_control_channel": False,
    }
    if config.batch_id is not None:
        session_body.update(
            {
                "batch_id": config.batch_id,
                "carrier_id": config.carrier_id or source_meta["sha256"][:16],
            }
        )
    client.publish(
        "SESSION_OPENED",
        session_body,
    )
    receiver_process: subprocess.Popen[bytes] | None = None
    server: CarrierProvisionServer | None = None
    receiver_log = paths["logs"] / "ffmpeg_receiver_a.log"
    received_wav = paths["receiver_a"] / "received_from_b_16k.wav"
    checks: dict[str, bool] = {
        "control_hmac_roundtrip_pass": False,
        "relay_carrier_commitment_pass": False,
        "rtp_opus_media_pass": False,
        "received_wav_valid": False,
    }
    decoder_report: dict[str, Any] | None = None
    try:
        server = CarrierProvisionServer(
            bind_ip=config.a_ip,
            port=config.provisioning_port,
            expected_client_ip=config.b_ip,
            carrier_path=source,
            carrier_sha256=str(source_meta["sha256"]),
        )
        server.start()
        carrier_body = {
            "protocol": PROTOCOL,
            "phase": "CARRIER_PREPARED",
            "carrier_id": config.carrier_id or source_meta["sha256"][:16],
            "carrier_commitment": {
                "sha256": source_meta["sha256"],
                "bytes": source_meta["bytes"],
                "duration_microseconds": source_meta["duration_microseconds"],
            },
            "provisioning": {
                "url": server.url,
                "transport": "HTTP over private Tailscale",
                "experimental_setup_only": True,
                "counted_as_voip_media": False,
            },
            "rtp_receiver": {
                "ip": config.a_ip,
                "port": config.rtp_port,
                "payload_type": 111,
                "codec": "Opus",
                "clock_rate_hz": 48000,
            },
        }
        if config.batch_id is not None:
            carrier_body["batch_id"] = config.batch_id
        client.publish(
            "CARRIER_PREPARED",
            carrier_body,
        )
        print(
            json.dumps(
                {
                    "status": "ZSC_TSD_NODE_A_WAITING_FOR_RELAY_B",
                    "session_id": session_id,
                    "correlation_id": correlation_id,
                    "run_directory": str(paths["root"]),
                    "next_action": "start dual-channel-relay on node B",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        relay_verified = client.wait(
            "RELAY_MEDIA_VERIFIED",
            timeout_seconds=config.wait_timeout_seconds,
            expected_actor="zsc-voip-b",
        )
        verification = relay_verified["body"].get("verification", {})
        checks["relay_carrier_commitment_pass"] = bool(
            verification.get("sha256_pass") and verification.get("byte_count_pass")
        )
        if not checks["relay_carrier_commitment_pass"]:
            raise DualChannelError("relay B did not verify the carrier commitment")
        server.close()
        server = None

        profile = RTPProfile(destination_ip=config.a_ip, port=config.rtp_port)
        sdp_path = paths["receiver_a"] / "receiver.sdp"
        sdp_path.write_text(
            receiver_sdp(profile, title="ZSC-TSD authenticated B-to-A RTP Opus"),
            encoding="ascii",
            newline="\n",
        )
        command = receiver_command(ffmpeg, sdp_path, received_wav)
        with receiver_log.open("xb") as log_stream:
            receiver_process = subprocess.Popen(
                command,
                cwd=str(paths["root"]),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            time.sleep(1.0)
            if receiver_process.poll() is not None:
                raise DualChannelError("A FFmpeg receiver exited before RTP readiness")
            client.publish(
                "RECEIVER_RTP_READY",
                {
                    "protocol": PROTOCOL,
                    "phase": "RECEIVER_RTP_READY",
                    "receiver_ip": config.a_ip,
                    "rtp_port": config.rtp_port,
                    "payload_type": 111,
                    "codec": "Opus",
                },
            )
            relay_finished = client.wait(
                "RELAY_RTP_FINISHED",
                timeout_seconds=config.wait_timeout_seconds,
                expected_actor="zsc-voip-b",
            )
            try:
                receiver_returncode = receiver_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                receiver_process.terminate()
                receiver_returncode = receiver_process.wait(timeout=10)
        receiver_process = None
        sender_ok = bool(relay_finished["body"].get("ffmpeg_returncode") == 0)
        if not received_wav.is_file() or received_wav.stat().st_size <= 44:
            raise DualChannelError("A did not capture a nonempty WAV from RTP")
        received_meta = wav_metadata(received_wav)
        checks["received_wav_valid"] = True
        checks["rtp_opus_media_pass"] = sender_ok
        client.publish(
            "RECEIVER_MEDIA_CAPTURED",
            {
                "protocol": PROTOCOL,
                "phase": "RECEIVER_MEDIA_CAPTURED",
                "received_wav": received_meta,
                "ffmpeg_returncode": receiver_returncode,
                "relay_sender_pass": sender_ok,
            },
        )

        sync_result: dict[str, Any] | None = None
        if not config.skip_sync_gate:
            manifest_path = _discover_sync_manifest(PROJECT_ROOT, config.sync_manifest)
            if manifest_path is None:
                sync_result = {
                    "status": "SYNC_GATE_MANIFEST_MISSING",
                    "passed": False,
                    "differences": [
                        "no frozen synchronization manifest below artifacts/freezes"
                    ],
                }
            else:
                try:
                    sync_result = _run_sync_gate(
                        manifest_path, config.sync_gate_python
                    )
                except Exception as exc:  # noqa: BLE001
                    sync_result = {
                        "status": "SYNC_GATE_ERROR",
                        "manifest": str(manifest_path),
                        "passed": False,
                        "differences": [f"{type(exc).__name__}: {exc}"],
                    }
            checks["sync_manifest_pass"] = bool(sync_result.get("passed"))
            write_json_new(paths["public"] / "sync_gate.json", sync_result)
            client.publish(
                "RECEIVER_SYNC_VERIFIED",
                {
                    "protocol": PROTOCOL,
                    "phase": "RECEIVER_SYNC_VERIFIED",
                    "status": sync_result.get("status"),
                    "manifest": sync_result.get("manifest"),
                    "local_protocol_sha256": sync_result.get("local_protocol_sha256"),
                    "frozen_protocol_sha256": sync_result.get("frozen_protocol_sha256"),
                    "differences": sync_result.get("differences", []),
                },
            )

        if config.decoder_command_json is not None:
            decode_output = paths["receiver_a"] / "decode_result.json"
            command = _decoder_command(
                config.decoder_command_json,
                received_wav=received_wav,
                output_json=decode_output,
                run_root=paths["root"],
            )
            decode_log = paths["logs"] / "decoder_a.log"
            started = time.perf_counter()
            with decode_log.open("xb") as log_stream:
                completed = subprocess.run(
                    command,
                    cwd=str(paths["root"]),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            decode_elapsed_ms = int((time.perf_counter() - started) * 1000)
            if completed.returncode != 0:
                raise DualChannelError(
                    f"decoder command failed with exit code {completed.returncode}"
                )
            if not decode_output.is_file():
                raise DualChannelError("decoder command did not create decode_result.json")
            decoder_report = load_json_object(decode_output)
            checks.update(decode_checks(decoder_report))
            client.publish(
                "RECEIVER_DECODED",
                {
                    "protocol": PROTOCOL,
                    "phase": "RECEIVER_DECODED",
                    "decode_result_sha256": sha256_file(decode_output),
                    "decode_elapsed_ms": decode_elapsed_ms,
                    "checks": decode_checks(decoder_report),
                },
            )

        checks["control_hmac_roundtrip_pass"] = True
        media_pass = all(
            checks[name]
            for name in (
                "control_hmac_roundtrip_pass",
                "relay_carrier_commitment_pass",
                "rtp_opus_media_pass",
                "received_wav_valid",
            )
        )
        end_to_end_pass = bool(
            decoder_report is not None
            and media_pass
            and all(decode_checks(decoder_report).values())
            and checks.get("sync_manifest_pass", True)
        )
        status = (
            "ZSC_TSD_DUAL_CHANNEL_END_TO_END_PASS"
            if end_to_end_pass
            else "ZSC_TSD_DUAL_CHANNEL_MEDIA_PASS"
            if media_pass
            else "ZSC_TSD_DUAL_CHANNEL_FAIL"
        )
        client.publish(
            "SESSION_COMPLETED",
            {
                "protocol": PROTOCOL,
                "phase": "SESSION_COMPLETED",
                "status": status,
                "media_pass": media_pass,
                "decoder_executed": decoder_report is not None,
                "end_to_end_pass": end_to_end_pass,
            },
        )
        timeline = store.list_events(session_id=session_id)
        timeline_path = write_json_new(paths["public"] / "event_timeline.json", timeline)
        final = {
            "schema": "zsc-tsd.dual-channel-result.v2",
            "status": status,
            "protocol": PROTOCOL,
            "session_id": session_id,
            "correlation_id": correlation_id,
            "claims": {
                "sip_call_implemented": False,
                "rtp_opus_media_transmitted": media_pass,
                "end_to_end_secret_recovery": end_to_end_pass,
                "sync_profile_verified": bool(checks.get("sync_manifest_pass")),
            },
            "checks": checks,
            "source_carrier": source_meta,
            "received_carrier": received_meta,
            "decoder_executed": decoder_report is not None,
            "created_at_utc": utc_now(),
        }
        final_path = write_json_new(paths["public"] / "final_result.json", final)
        evidence_files = [
            runtime_path,
            context_path,
            timeline_path,
            final_path,
            receiver_log,
            received_wav,
            *paths["events"].glob("*.json"),
        ]
        if decoder_report is not None:
            evidence_files.extend(
                [paths["receiver_a"] / "decode_result.json", paths["logs"] / "decoder_a.log"]
            )
        write_evidence_manifest(
            paths["public"] / "evidence_manifest.json",
            experiment="dual_channel_v2_node_a",
            root=paths["root"],
            files=evidence_files,
            metadata={"session_id": session_id, "status": status},
        )
        return {"status": status, "run_directory": str(paths["root"]), **final}
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
        failure = {
            "schema": "zsc-tsd.dual-channel-result.v2",
            "status": "ZSC_TSD_DUAL_CHANNEL_ABORTED",
            "session_id": session_id,
            "correlation_id": correlation_id,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "checks": checks,
            "created_at_utc": utc_now(),
        }
        if not (paths["public"] / "final_result.json").exists():
            write_json_new(paths["public"] / "final_result.json", failure)
        raise
    finally:
        if server is not None:
            server.close()
        if receiver_process is not None and receiver_process.poll() is None:
            receiver_process.terminate()
