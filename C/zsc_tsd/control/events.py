from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


CONTROL_SCHEMA = "zsc-tsd.control-event.v1"
CONTROL_HMAC_DOMAIN = b"ZSC-TSD-CONTROL-EVENT-HMAC-V1\x00"
CONTROL_KDF_INFO = b"ZSC-TSD-CONTROL-ACTOR-KEY-V1\x00"
MINIMUM_MASTER_KEY_BYTES = 32
DEFAULT_MAX_CLOCK_SKEW_SECONDS = 120

EVENT_TYPES = frozenset(
    {
        "SESSION_OPENED",
        "CARRIER_PREPARED",
        "RELAY_MEDIA_VERIFIED",
        "RELAY_RTP_STARTED",
        "RELAY_RTP_FINISHED",
        "RECEIVER_RTP_READY",
        "RECEIVER_MEDIA_CAPTURED",
        "RECEIVER_DECODED",
        "RECEIVER_SYNC_VERIFIED",
        "RETRANSMIT_REQUESTED",
        "RETRANSMIT_STARTED",
        "RETRANSMIT_FINISHED",
        "SESSION_COMPLETED",
        "SESSION_ABORTED",
    }
)

SIGNED_FIELDS = (
    "schema",
    "event_id",
    "session_id",
    "actor_id",
    "sequence_no",
    "event_type",
    "correlation_id",
    "issued_at_utc",
    "expires_at_utc",
    "event_nonce",
    "body",
)


class ControlEventError(ValueError):
    """Base exception for malformed or unauthenticated control events."""


class ControlEventSignatureError(ControlEventError):
    """Raised when an event HMAC does not verify."""


class ControlEventExpired(ControlEventError):
    """Raised when an event is expired or issued too far in the future."""


def _require_master_key(master_key: bytes) -> bytes:
    if not isinstance(master_key, bytes):
        raise TypeError("control master key must be bytes")
    if len(master_key) < MINIMUM_MASTER_KEY_BYTES:
        raise ControlEventError(
            f"control master key must contain at least {MINIMUM_MASTER_KEY_BYTES} bytes"
        )
    return master_key


def _normalize_uuid(value: str, *, field: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ControlEventError(f"{field} must be a valid UUID") from exc


def _normalize_timestamp(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise ControlEventError(f"{field} must be a string")
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ControlEventError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ControlEventError(f"{field} must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    normalized = _normalize_timestamp(value, field="timestamp")
    return datetime.fromisoformat(normalized[:-1] + "+00:00")


def _validate_json_value(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise ControlEventError(
            f"floating-point values are forbidden in signed JSON: {path}"
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ControlEventError(f"signed JSON key is not a string: {path}")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ControlEventError(
        f"unsupported signed JSON value at {path}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return protocol-specific deterministic UTF-8 JSON bytes.

    Floats are deliberately forbidden so different runtimes cannot disagree
    about their textual representation.
    """

    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def derive_actor_hmac_key(
    master_key: bytes,
    *,
    session_id: str,
    actor_id: str,
) -> bytes:
    master_key = _require_master_key(master_key)
    normalized_session = _normalize_uuid(session_id, field="session_id")
    if not isinstance(actor_id, str) or not actor_id.strip():
        raise ControlEventError("actor_id must be a nonempty string")
    salt = hashlib.sha256(
        b"ZSC-TSD-CONTROL-SALT-V1\x00" + uuid.UUID(normalized_session).bytes
    ).digest()
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=CONTROL_KDF_INFO + actor_id.encode("utf-8"),
    ).derive(master_key)


def _unsigned_event(event: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise ControlEventError("control event must be an object")
    missing = [field for field in SIGNED_FIELDS if field not in event]
    if missing:
        raise ControlEventError("control event is missing: " + ", ".join(missing))
    unexpected = set(event) - set(SIGNED_FIELDS) - {"hmac_sha256"}
    if unexpected:
        raise ControlEventError(
            "control event has unsigned fields: " + ", ".join(sorted(unexpected))
        )

    body = event["body"]
    if not isinstance(body, dict):
        raise ControlEventError("body must be an object")
    event_type = str(event["event_type"])
    if event_type not in EVENT_TYPES:
        raise ControlEventError(f"unsupported event_type: {event_type}")
    sequence_no = event["sequence_no"]
    if isinstance(sequence_no, bool) or not isinstance(sequence_no, int):
        raise ControlEventError("sequence_no must be an integer")
    if sequence_no < 1:
        raise ControlEventError("sequence_no must be positive")

    actor_id = event["actor_id"]
    correlation_id = event["correlation_id"]
    event_nonce = event["event_nonce"]
    for field, value in (
        ("actor_id", actor_id),
        ("correlation_id", correlation_id),
        ("event_nonce", event_nonce),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ControlEventError(f"{field} must be a nonempty string")

    result = {
        "schema": str(event["schema"]),
        "event_id": _normalize_uuid(str(event["event_id"]), field="event_id"),
        "session_id": _normalize_uuid(str(event["session_id"]), field="session_id"),
        "actor_id": actor_id,
        "sequence_no": sequence_no,
        "event_type": event_type,
        "correlation_id": correlation_id,
        "issued_at_utc": _normalize_timestamp(
            event["issued_at_utc"], field="issued_at_utc"
        ),
        "expires_at_utc": _normalize_timestamp(
            event["expires_at_utc"], field="expires_at_utc"
        ),
        "event_nonce": event_nonce,
        "body": body,
    }
    if result["schema"] != CONTROL_SCHEMA:
        raise ControlEventError(f"unsupported control schema: {result['schema']}")
    canonical_json_bytes(result)
    return result


def _signature(master_key: bytes, unsigned: Mapping[str, Any]) -> str:
    actor_key = derive_actor_hmac_key(
        master_key,
        session_id=str(unsigned["session_id"]),
        actor_id=str(unsigned["actor_id"]),
    )
    return hmac.new(
        actor_key,
        CONTROL_HMAC_DOMAIN + canonical_json_bytes(dict(unsigned)),
        hashlib.sha256,
    ).hexdigest()


def create_signed_event(
    master_key: bytes,
    *,
    session_id: str,
    actor_id: str,
    sequence_no: int,
    event_type: str,
    correlation_id: str,
    body: Mapping[str, Any],
    ttl_seconds: int = 600,
    now: datetime | None = None,
    event_id: str | None = None,
    event_nonce: str | None = None,
) -> dict[str, Any]:
    if ttl_seconds < 1:
        raise ControlEventError("ttl_seconds must be positive")
    issued = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = issued + timedelta(seconds=ttl_seconds)
    event: dict[str, Any] = {
        "schema": CONTROL_SCHEMA,
        "event_id": event_id or str(uuid.uuid4()),
        "session_id": session_id,
        "actor_id": actor_id,
        "sequence_no": sequence_no,
        "event_type": event_type,
        "correlation_id": correlation_id,
        "issued_at_utc": issued.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "expires_at_utc": expires.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "event_nonce": event_nonce or secrets.token_hex(16),
        "body": dict(body),
    }
    unsigned = _unsigned_event(event)
    event = dict(unsigned)
    event["hmac_sha256"] = _signature(master_key, unsigned)
    return event


def verify_signed_event(
    master_key: bytes,
    event: Mapping[str, Any],
    *,
    now: datetime | None = None,
    max_clock_skew_seconds: int = DEFAULT_MAX_CLOCK_SKEW_SECONDS,
    check_time: bool = True,
) -> dict[str, Any]:
    unsigned = _unsigned_event(event)
    received = event.get("hmac_sha256")
    if not isinstance(received, str) or len(received) != 64:
        raise ControlEventSignatureError("control event has no valid HMAC field")
    expected = _signature(master_key, unsigned)
    if not hmac.compare_digest(received.lower(), expected):
        raise ControlEventSignatureError("control event HMAC verification failed")

    if check_time:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        issued = _parse_timestamp(unsigned["issued_at_utc"])
        expires = _parse_timestamp(unsigned["expires_at_utc"])
        if expires <= issued:
            raise ControlEventExpired("control event expiry is not after issue time")
        if issued > current + timedelta(seconds=max_clock_skew_seconds):
            raise ControlEventExpired("control event was issued too far in the future")
        if expires < current:
            raise ControlEventExpired("control event has expired")

    verified = dict(unsigned)
    verified["hmac_sha256"] = received.lower()
    return verified
