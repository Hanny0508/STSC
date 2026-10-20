"""Shared invariants for the ZSC-TSD dual-channel experiment.

The media and control paths have deliberately different responsibilities:

* RTP/UDP + Opus carries speech from relay B to receiver A.
* PostgreSQL carries authenticated metadata and acknowledgements only.
* HTTP over the private Tailscale network is an experiment provisioning step
  used to place A's already-generated carrier on relay B.  It is recorded but
  is not described as part of the VoIP media path.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


PROTOCOL = "ZSC-TSD-DUAL-CHANNEL-v2"
DSN_ENV = "ZSC_TSD_CONTROL_DSN"
MASTER_KEY_ENV = "ZSC_TSD_CONTROL_MASTER_KEY_B64"


class DualChannelError(RuntimeError):
    """Raised when an authenticated two-host run cannot safely continue."""


def read_control_environment() -> tuple[str, bytes]:
    """Load control credentials without printing or persisting their values."""

    dsn = os.environ.get(DSN_ENV, "").strip()
    raw_key = os.environ.get(MASTER_KEY_ENV, "").strip()
    if not dsn:
        raise DualChannelError(f"missing environment variable: {DSN_ENV}")
    if not raw_key:
        raise DualChannelError(f"missing environment variable: {MASTER_KEY_ENV}")
    try:
        key = base64.b64decode(raw_key, validate=True)
    except (ValueError, TypeError) as exc:
        raise DualChannelError(f"{MASTER_KEY_ENV} is not valid Base64") from exc
    if len(key) < 32:
        raise DualChannelError(f"{MASTER_KEY_ENV} must decode to at least 32 bytes")
    return dsn, key


def require_uuid(value: str, *, field: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise DualChannelError(f"{field} must be a UUID") from exc


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DualChannelError(f"JSON root must be an object: {source}")
    return value


def decode_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    """Extract the minimum checks required for an end-to-end PASS claim.

    The accepted keys match the frozen secure IMDB batch report.  Missing
    checks are false.  This prevents a successful RTP transfer from being
    mislabeled as successful secret-message recovery.
    """

    receiver_pass = bool(
        report.get("receiver_decode_pass")
        or report.get("receiver_pass")
        or str(report.get("status", "")).endswith("_PASS")
    )
    return {
        "receiver_decode_pass": receiver_pass,
        "rs_reconstruction_pass": bool(report.get("rs_reconstruction_pass")),
        "aes_gcm_authentication_pass": bool(
            report.get("aes_gcm_authentication_pass")
        ),
        "plaintext_exact_match": bool(report.get("plaintext_exact_match")),
    }


def assert_metadata_only(body: Mapping[str, Any]) -> None:
    """Reject obvious attempts to put secret content in the control channel."""

    forbidden_fragments = {
        "plaintext",
        "payload_hex",
        "ciphertext",
        "stegotext",
        "token_ids",
        "reference_transcript",
        "audio_bytes",
        "wav_base64",
    }
    allowed_boolean_declarations = {
        "secret_payload_allowed",
        "secret_payload_in_control_channel",
        "secret_payload_received_from_control_channel",
    }
    allowed_boolean_results = {"plaintext_exact_match"}

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).lower()
                if normalized in allowed_boolean_declarations:
                    if item is not False:
                        raise DualChannelError(
                            f"control metadata must declare {path}.{key}=false"
                        )
                elif normalized in allowed_boolean_results:
                    if not isinstance(item, bool):
                        raise DualChannelError(
                            f"control result flag must be boolean: {path}.{key}"
                        )
                elif any(fragment in normalized for fragment in forbidden_fragments):
                    raise DualChannelError(
                        f"secret-bearing field is forbidden in control metadata: {path}.{key}"
                    )
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(dict(body), "body")
