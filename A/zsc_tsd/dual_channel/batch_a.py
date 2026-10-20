"""Node-A coordinator for a frozen multi-carrier RTP/Opus post-decode batch."""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..manifest import sha256_file, write_json_new

# r3_batch_a_bundle_applied
from .coordinator_a import CoordinatorAConfig, run_coordinator
from .evidence import utc_now
from .protocol import DualChannelError, load_json_object


@dataclass(frozen=True, slots=True)
class BatchAConfig:
    secure_batch_root: Path
    bundle_private: Path
    output_base: Path
    ffmpeg: Path
    asr_python: Path
    zts_python: Path
    asr_model: Path
    a_ip: str
    b_ip: str
    provisioning_port: int = 5602
    rtp_port: int = 5004
    event_ttl_seconds: int = 14_400
    wait_timeout_seconds: int = 900
    sync_manifest: Path | None = None
    sync_gate_python: Path | None = None
    skip_sync_gate: bool = False


def _discover(root: Path) -> list[dict[str, Any]]:
    public = root / "public"
    private = root / "sender_private"
    rows: list[dict[str, Any]] = []
    for wav in public.glob("*/carrier.wav"):
        carrier_id = wav.parent.name
        sender_path = private / carrier_id / "sender.json"
        if not sender_path.is_file():
            raise DualChannelError(f"sender metadata missing for {carrier_id}")
        sender = load_json_object(sender_path)
        if sender.get("carrier_id") != carrier_id:
            raise DualChannelError(f"carrier_id mismatch in {sender_path}")
        rows.append(
            {
                "carrier_id": carrier_id,
                "shard_index": int(sender["shard_index"]),
                "attempt_number": int(sender["attempt_number"]),
                "wav": wav.resolve(),
                "sender_json": sender_path.resolve(),
            }
        )
    rows.sort(key=lambda row: (row["shard_index"], row["attempt_number"], row["carrier_id"]))
    if not rows:
        raise DualChannelError(f"no frozen carriers found below {public}")
    if len({row["carrier_id"] for row in rows}) != len(rows):
        raise DualChannelError("duplicate carrier_id in secure batch")
    return rows


def run_batch_a(config: BatchAConfig) -> dict[str, Any]:
    secure_root = config.secure_batch_root.resolve()
    bundle_private = config.bundle_private.resolve()
    for path in (secure_root, bundle_private, config.ffmpeg, config.asr_python, config.zts_python, config.asr_model):
        if not path.exists():
            raise FileNotFoundError(path)
    carriers = _discover(secure_root)
    batch_id = str(uuid.uuid4())
    root = config.output_base.resolve() / f"dual_channel_batch_{batch_id}"
    sessions = root / "sessions"
    decoded = root / "postdecode"
    public = root / "public"
    for path in (sessions, decoded, public):
        path.mkdir(parents=True, exist_ok=False)
    selection = {
        "schema": "zsc-tsd.voip-batch-selection.v1",
        "batch_id": batch_id,
        "created_at_utc": utc_now(),
        "selection_policy": "all carrier.wav files frozen before network start",
        "adaptive_post_network_selection": False,
        "secure_batch_root": str(secure_root),
        "secure_batch_root_name": secure_root.name,
        "carrier_count": len(carriers),
        "carriers": [
            {
                "carrier_id": row["carrier_id"],
                "shard_index": row["shard_index"],
                "attempt_number": row["attempt_number"],
                "wav": str(row["wav"]),
                "wav_bytes": row["wav"].stat().st_size,
                "wav_sha256": sha256_file(row["wav"]),
                "sender_json": str(row["sender_json"]),
                "sender_json_sha256": sha256_file(row["sender_json"]),
            }
            for row in carriers
        ],
    }
    selection_path = write_json_new(public / "batch_selection.json", selection)
    print(
        json.dumps(
            {
                "status": "ZSC_TSD_BATCH_A_READY",
                "batch_id": batch_id,
                "carrier_count": len(carriers),
                "next_action": (
                    "start dual-channel-batch-relay on B with this batch_id "
                    "and expected_sessions"
                ),
                "selection_sha256": sha256_file(selection_path),
                "run_directory": str(root),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    session_rows: list[dict[str, Any]] = []
    postdecode_tool = Path(__file__).resolve().parents[2] / "tools" / "zts_voip_postdecode_v1.py"
    for number, row in enumerate(carriers, 1):
        print(
            f"A BATCH CARRIER {number}/{len(carriers)}: {row['carrier_id']}",
            flush=True,
        )
        media = run_coordinator(
            CoordinatorAConfig(
                source_wav=row["wav"],
                output_base=sessions,
                ffmpeg=config.ffmpeg,
                a_ip=config.a_ip,
                b_ip=config.b_ip,
                provisioning_port=config.provisioning_port,
                rtp_port=config.rtp_port,
                event_ttl_seconds=config.event_ttl_seconds,
                wait_timeout_seconds=config.wait_timeout_seconds,
                batch_id=batch_id,
                carrier_id=row["carrier_id"],
                sync_manifest=config.sync_manifest,
                sync_gate_python=(config.sync_gate_python or config.zts_python),
                skip_sync_gate=config.skip_sync_gate,
            )
        )
        session_root = Path(media["run_directory"])
        received = session_root / "receiver_a" / "received_from_b_16k.wav"
        carrier_output = decoded / row["carrier_id"]
        completed = subprocess.run(
            [
                str(config.zts_python.resolve()),
                str(postdecode_tool),
                "carrier",
                "--audio",
                str(received),
                "--sender-json",
                str(row["sender_json"]),
                "--output-dir",
                str(carrier_output),
                "--asr-python",
                str(config.asr_python.resolve()),
                "--asr-model",
                str(config.asr_model.resolve()),
                "--bundle-private",
                str(bundle_private.resolve()),
            ],
            cwd=postdecode_tool.parents[1],
            check=False,
        )
        if completed.returncode != 0:
            raise DualChannelError(
                f"post-VoIP carrier decoder failed for {row['carrier_id']} "
                f"with exit code {completed.returncode}"
            )
        carrier_result = load_json_object(carrier_output / "carrier_result.json")
        session_rows.append(
            {
                "carrier_id": row["carrier_id"],
                "session_id": media["session_id"],
                "correlation_id": media["correlation_id"],
                "media_status": media["status"],
                "postdecode_status": carrier_result["status"],
                "session_directory": str(session_root),
            }
        )
    final_path = public / "final_result_after_voip.json"
    completed = subprocess.run(
        [
            str(config.zts_python.resolve()),
            str(postdecode_tool),
            "finalize",
            "--results-root",
            str(decoded),
            "--bundle-private",
            str(bundle_private),
            "--output",
            str(final_path),
        ],
        cwd=postdecode_tool.parents[1],
        check=False,
    )
    final = load_json_object(final_path)
    summary = {
        "schema": "zsc-tsd.dual-channel-batch-a-result.v1",
        "status": final["status"],
        "batch_id": batch_id,
        "carrier_count": len(carriers),
        "completed_media_sessions": len(session_rows),
        "selection_sha256": sha256_file(selection_path),
        "final_result_sha256": sha256_file(final_path),
        "final_result": final,
        "sessions": session_rows,
        "created_at_utc": utc_now(),
    }
    summary_path = write_json_new(public / "batch_summary.json", summary)
    result = {"run_directory": str(root), "summary_path": str(summary_path), **summary}
    if completed.returncode not in (0, 2):
        raise DualChannelError(
            f"batch finalizer exited unexpectedly with {completed.returncode}"
        )
    return result
