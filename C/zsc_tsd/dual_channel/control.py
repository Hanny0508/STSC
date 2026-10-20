"""Small authenticated event client used by both experiment nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..control.events import create_signed_event
from ..control.postgres import PostgresControlStore
from ..manifest import write_json_new
from .protocol import DualChannelError, assert_metadata_only


class ControlClient:
    def __init__(
        self,
        *,
        store: PostgresControlStore,
        master_key: bytes,
        session_id: str,
        actor_id: str,
        correlation_id: str,
        evidence_directory: str | Path,
        ttl_seconds: int,
    ) -> None:
        self.store = store
        self.master_key = master_key
        self.session_id = session_id
        self.actor_id = actor_id
        self.correlation_id = correlation_id
        self.evidence_directory = Path(evidence_directory)
        self.evidence_directory.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        existing = store.list_events(session_id=session_id)
        self.sequence_no = 1 + max(
            (
                int(event["sequence_no"])
                for event in existing
                if event["actor_id"] == actor_id
            ),
            default=0,
        )

    def publish(self, event_type: str, body: Mapping[str, Any]) -> dict[str, Any]:
        assert_metadata_only(body)
        event = create_signed_event(
            self.master_key,
            session_id=self.session_id,
            actor_id=self.actor_id,
            sequence_no=self.sequence_no,
            event_type=event_type,
            correlation_id=self.correlation_id,
            body=body,
            ttl_seconds=self.ttl_seconds,
        )
        stored = self.store.append(event)
        name = f"{self.sequence_no:03d}_{event_type.lower()}_{self.actor_id}.json"
        write_json_new(self.evidence_directory / name, stored)
        self.sequence_no += 1
        return stored

    def wait(
        self,
        event_type: str,
        *,
        timeout_seconds: float,
        expected_actor: str,
    ) -> dict[str, Any]:
        event = self.store.wait_for_event(
            session_id=self.session_id,
            event_type=event_type,
            correlation_id=self.correlation_id,
            timeout_seconds=timeout_seconds,
        )
        if event is None:
            raise DualChannelError(f"timed out waiting for {event_type}")
        if event["actor_id"] != expected_actor:
            raise DualChannelError(
                f"{event_type} came from {event['actor_id']}, expected {expected_actor}"
            )
        return event
