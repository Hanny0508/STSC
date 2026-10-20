"""Key creation and loading helpers that never print secret material."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from pathlib import Path


def generate_key(length: int = 32) -> bytes:
    if isinstance(length, bool) or not isinstance(length, int) or length < 32:
        raise ValueError("key length must be an integer of at least 32 bytes")
    return secrets.token_bytes(length)


def key_commitment(key: bytes) -> str:
    if not isinstance(key, bytes) or not key:
        raise ValueError("key must be nonempty bytes")
    return hashlib.sha256(key).hexdigest()


def decode_base64_key(value: str, *, minimum_bytes: int = 32) -> bytes:
    try:
        key = base64.b64decode(value.strip(), validate=True)
    except ValueError as exc:
        raise ValueError("key is not valid Base64") from exc
    if len(key) < minimum_bytes:
        raise ValueError(f"key must contain at least {minimum_bytes} bytes")
    return key


def read_key_file(path: str | Path, *, minimum_bytes: int = 32) -> bytes:
    return decode_base64_key(
        Path(path).read_text(encoding="ascii"), minimum_bytes=minimum_bytes
    )


def read_key_environment(name: str, *, minimum_bytes: int = 32) -> bytes:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing environment variable: {name}")
    return decode_base64_key(value, minimum_bytes=minimum_bytes)
