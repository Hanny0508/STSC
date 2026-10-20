"""Small, explicit AES-256-GCM envelope used before ZTS fragmentation."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Mapping

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


KEY_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16


@dataclass(frozen=True, slots=True)
class AESGCMEnvelope:
    nonce: bytes
    ciphertext_and_tag: bytes

    def to_json(self) -> dict[str, str]:
        return {
            "algorithm": "AES-256-GCM",
            "nonce_b64": base64.b64encode(self.nonce).decode("ascii"),
            "ciphertext_and_tag_b64": base64.b64encode(
                self.ciphertext_and_tag
            ).decode("ascii"),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "AESGCMEnvelope":
        if value.get("algorithm") != "AES-256-GCM":
            raise ValueError("unsupported encrypted-envelope algorithm")
        try:
            nonce = base64.b64decode(str(value["nonce_b64"]), validate=True)
            packet = base64.b64decode(
                str(value["ciphertext_and_tag_b64"]), validate=True
            )
        except (KeyError, ValueError) as exc:
            raise ValueError("malformed AES-GCM envelope") from exc
        return cls(nonce=nonce, ciphertext_and_tag=packet)


def _require_key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) != KEY_BYTES:
        raise ValueError("AES-256-GCM key must contain exactly 32 bytes")
    return key


def encrypt(
    plaintext: bytes,
    *,
    key: bytes,
    aad: bytes = b"ZSC-TSD|AES-256-GCM|v1",
    nonce: bytes | None = None,
) -> AESGCMEnvelope:
    if not isinstance(plaintext, bytes) or not plaintext:
        raise ValueError("plaintext must be nonempty bytes")
    nonce_value = os.urandom(NONCE_BYTES) if nonce is None else nonce
    if not isinstance(nonce_value, bytes) or len(nonce_value) != NONCE_BYTES:
        raise ValueError("AES-GCM nonce must contain exactly 12 bytes")
    return AESGCMEnvelope(
        nonce=nonce_value,
        ciphertext_and_tag=AESGCM(_require_key(key)).encrypt(
            nonce_value, plaintext, aad
        ),
    )


def decrypt(
    envelope: AESGCMEnvelope,
    *,
    key: bytes,
    aad: bytes = b"ZSC-TSD|AES-256-GCM|v1",
) -> bytes:
    if len(envelope.nonce) != NONCE_BYTES:
        raise ValueError("AES-GCM nonce must contain exactly 12 bytes")
    if len(envelope.ciphertext_and_tag) <= TAG_BYTES:
        raise ValueError("AES-GCM packet contains no ciphertext")
    return AESGCM(_require_key(key)).decrypt(
        envelope.nonce, envelope.ciphertext_and_tag, aad
    )
