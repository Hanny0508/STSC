from __future__ import annotations

import pytest
from cryptography.exceptions import InvalidTag

from zsc_tsd.crypto import decrypt, encrypt


KEY = bytes(range(32))
NONCE = bytes(range(12))
AAD = b"zsc-tsd-test"


def test_aes_gcm_roundtrip_and_serialization() -> None:
    envelope = encrypt(b"fixed secret", key=KEY, aad=AAD, nonce=NONCE)
    restored = type(envelope).from_json(envelope.to_json())
    assert restored == envelope
    assert decrypt(restored, key=KEY, aad=AAD) == b"fixed secret"


def test_aes_gcm_rejects_tampering() -> None:
    envelope = encrypt(b"fixed secret", key=KEY, aad=AAD, nonce=NONCE)
    damaged = type(envelope)(
        nonce=envelope.nonce,
        ciphertext_and_tag=envelope.ciphertext_and_tag[:-1]
        + bytes([envelope.ciphertext_and_tag[-1] ^ 1]),
    )
    with pytest.raises(InvalidTag):
        decrypt(damaged, key=KEY, aad=AAD)


def test_aes_gcm_rejects_wrong_key_size() -> None:
    with pytest.raises(ValueError):
        encrypt(b"fixed secret", key=b"short", nonce=NONCE)
