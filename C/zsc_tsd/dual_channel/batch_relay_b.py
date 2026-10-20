"""Persistent node-B relay for a precommitted multi-carrier batch."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..control.postgres import PostgresControlStore
from .protocol import DualChannelError, read_control_environment, require_uuid
from .relay_b import RelayBConfig, run_relay


@dataclass(frozen=True, slots=True)
class BatchRelayBConfig:
    batch_id: str
    expected_sessions: int
    output_base: Path
    ffmpeg: Path
    expected_a_ip: str
    b_ip: str
    event_ttl_seconds: int = 14_400
    wait_timeout_seconds: int = 900
    poll_seconds: float = 0.5


def run_batch_relay(config: BatchRelayBConfig) -> dict[str, Any]:
    batch_id = require_uuid(config.batch_id, field="batch_id")
    if config.expected_sessions < 1:
        raise DualChannelError("expected_sessions must be positive")
    dsn, master_key = read_control_environment()
    store = PostgresControlStore(dsn, master_key)
    handled: set[str] = set()
    results: list[dict[str, Any]] = []
    last_progress = time.monotonic()
    print(
        json.dumps(
            {
                "status": "ZSC_TSD_BATCH_RELAY_B_WAITING",
                "batch_id": batch_id,
                "expected_sessions": config.expected_sessions,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    while len(handled) < config.expected_sessions:
        events = store.list_batch_carriers(batch_id=batch_id)
        pending = [event for event in events if event["session_id"] not in handled]
        if not pending:
            if time.monotonic() - last_progress >= config.wait_timeout_seconds:
                raise DualChannelError(
                    "batch relay timed out waiting for the next CARRIER_PREPARED event"
                )
            time.sleep(config.poll_seconds)
            continue
        event = pending[0]
        body = event["body"]
        carrier_id = str(body.get("carrier_id", ""))
        if not carrier_id:
            raise DualChannelError("batch carrier event has no carrier_id")
        already_finished = store.list_events(
            session_id=event["session_id"],
            event_type="RELAY_RTP_FINISHED",
            correlation_id=event["correlation_id"],
        )
        if already_finished:
            handled.add(event["session_id"])
            results.append(
                {
                    "carrier_id": carrier_id,
                    "session_id": event["session_id"],
                    "status": "ZSC_TSD_DUAL_CHANNEL_RELAY_PASS",
                    "run_directory": None,
                    "resumed_from_authenticated_database_event": True,
                }
            )
            last_progress = time.monotonic()
            continue
        print(
            f"BATCH RELAY {len(handled) + 1}/{config.expected_sessions}: {carrier_id}",
            flush=True,
        )
        result = run_relay(
            RelayBConfig(
                session_id=event["session_id"],
                correlation_id=event["correlation_id"],
                output_base=config.output_base,
                ffmpeg=config.ffmpeg,
                expected_a_ip=config.expected_a_ip,
                b_ip=config.b_ip,
                event_ttl_seconds=config.event_ttl_seconds,
                wait_timeout_seconds=config.wait_timeout_seconds,
            )
        )
        handled.add(event["session_id"])
        results.append(
            {
                "carrier_id": carrier_id,
                "session_id": event["session_id"],
                "status": result["status"],
                "run_directory": result["run_directory"],
            }
        )
        last_progress = time.monotonic()
    passed = all(row["status"] == "ZSC_TSD_DUAL_CHANNEL_RELAY_PASS" for row in results)
    return {
        "schema": "zsc-tsd.dual-channel-batch-relay-result.v1",
        "status": (
            "ZSC_TSD_DUAL_CHANNEL_BATCH_RELAY_PASS"
            if passed
            else "ZSC_TSD_DUAL_CHANNEL_BATCH_RELAY_FAIL"
        ),
        "batch_id": batch_id,
        "expected_sessions": config.expected_sessions,
        "completed_sessions": len(results),
        "results": results,
    }
