"""Authenticated PostgreSQL control-channel primitives for ZSC-TSD."""

from .events import (
    ControlEventError,
    ControlEventExpired,
    ControlEventSignatureError,
    canonical_json_bytes,
    create_signed_event,
    derive_actor_hmac_key,
    verify_signed_event,
)

__all__ = [
    "ControlEventError",
    "ControlEventExpired",
    "ControlEventSignatureError",
    "canonical_json_bytes",
    "create_signed_event",
    "derive_actor_hmac_key",
    "verify_signed_event",
]
