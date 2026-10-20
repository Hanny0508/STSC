from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from .events import verify_signed_event


class ControlStoreError(RuntimeError):
    """Raised when the authenticated control event store cannot proceed."""


def _load_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ControlStoreError(
            "psycopg is required; install requirements/control.txt"
        ) from exc
    return psycopg, dict_row


class PostgresControlStore:
    def __init__(self, dsn: str, master_key: bytes) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("PostgreSQL DSN must be nonempty")
        self.dsn = dsn
        self.master_key = master_key

    def initialize(self, schema_path: str | Path | None = None) -> None:
        psycopg, _ = _load_psycopg()
        path = Path(schema_path) if schema_path else Path(__file__).with_name("schema.sql")
        sql = path.read_text(encoding="utf-8")
        with psycopg.connect(self.dsn) as connection:
            connection.execute(sql)

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        verified = verify_signed_event(self.master_key, event)
        psycopg, dict_row = _load_psycopg()
        sql = """
            INSERT INTO zsc_control_events (
                event_id, session_id, actor_id, sequence_no, event_type,
                correlation_id, issued_at, expires_at, event_nonce, body,
                hmac_sha256
            ) VALUES (
                %(event_id)s, %(session_id)s, %(actor_id)s, %(sequence_no)s,
                %(event_type)s, %(correlation_id)s, %(issued_at_utc)s,
                %(expires_at_utc)s, %(event_nonce)s, %(body)s::jsonb,
                %(hmac_sha256)s
            )
            RETURNING *
        """
        parameters = dict(verified)
        parameters["body"] = json.dumps(
            verified["body"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        try:
            with psycopg.connect(self.dsn, row_factory=dict_row) as connection:
                row = connection.execute(sql, parameters).fetchone()
        except psycopg.errors.UniqueViolation as exc:
            raise ControlStoreError(
                "replayed event, nonce, or actor sequence was rejected"
            ) from exc
        if row is None:
            raise ControlStoreError("PostgreSQL did not return inserted event")
        return self._row_to_verified_event(row)

    def get(self, event_id: str) -> dict[str, Any] | None:
        psycopg, dict_row = _load_psycopg()
        with psycopg.connect(self.dsn, row_factory=dict_row) as connection:
            row = connection.execute(
                "SELECT * FROM zsc_control_events WHERE event_id = %s", (event_id,)
            ).fetchone()
        return None if row is None else self._row_to_verified_event(row)

    def list_events(
        self,
        *,
        session_id: str,
        event_type: str | None = None,
        correlation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        psycopg, dict_row = _load_psycopg()
        clauses = ["session_id = %s"]
        parameters: list[Any] = [session_id]
        if event_type is not None:
            clauses.append("event_type = %s")
            parameters.append(event_type)
        if correlation_id is not None:
            clauses.append("correlation_id = %s")
            parameters.append(correlation_id)
        sql = (
            "SELECT * FROM zsc_control_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY received_at, event_id"
        )
        with psycopg.connect(self.dsn, row_factory=dict_row) as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._row_to_verified_event(row) for row in rows]

    def wait_for_event(
        self,
        *,
        session_id: str,
        event_type: str,
        correlation_id: str,
        timeout_seconds: float,
        poll_seconds: float = 0.5,
    ) -> dict[str, Any] | None:
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("timeout and poll intervals must be positive")
        deadline = time.monotonic() + timeout_seconds
        while True:
            rows = self.list_events(
                session_id=session_id,
                event_type=event_type,
                correlation_id=correlation_id,
            )
            if rows:
                return rows[-1]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(poll_seconds, remaining))

    def list_batch_carriers(self, *, batch_id: str) -> list[dict[str, Any]]:
        """Return HMAC-verified CARRIER_PREPARED events for one batch.

        This narrow query lets relay B discover fresh per-carrier sessions
        without accepting an unauthenticated side channel or scanning unrelated
        historical experiments.
        """

        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError("batch_id must be nonempty")
        psycopg, dict_row = _load_psycopg()
        sql = """
            SELECT * FROM zsc_control_events
            WHERE event_type = 'CARRIER_PREPARED'
              AND body ->> 'batch_id' = %s
            ORDER BY received_at, event_id
        """
        with psycopg.connect(self.dsn, row_factory=dict_row) as connection:
            rows = connection.execute(sql, (batch_id,)).fetchall()
        return [self._row_to_verified_event(row) for row in rows]

    def _row_to_verified_event(self, row: Mapping[str, Any]) -> dict[str, Any]:
        def utc_text(value: Any) -> str:
            return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

        event = {
            "schema": "zsc-tsd.control-event.v1",
            "event_id": str(row["event_id"]),
            "session_id": str(row["session_id"]),
            "actor_id": str(row["actor_id"]),
            "sequence_no": int(row["sequence_no"]),
            "event_type": str(row["event_type"]),
            "correlation_id": str(row["correlation_id"]),
            "issued_at_utc": utc_text(row["issued_at"]),
            "expires_at_utc": utc_text(row["expires_at"]),
            "event_nonce": str(row["event_nonce"]),
            "body": dict(row["body"]),
            "hmac_sha256": str(row["hmac_sha256"]),
        }
        return verify_signed_event(self.master_key, event)
