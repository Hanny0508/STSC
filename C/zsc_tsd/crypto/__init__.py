"""Cryptographic and integrity primitives for ZSC-TSD."""

from .aes_gcm import AESGCMEnvelope, decrypt, encrypt
from .framing import (
    CRC32PayloadFrame,
    PayloadFrameError,
    PayloadFrameIntegrityError,
    decode_crc32_frame,
    encode_crc32_frame,
    inspect_crc32_frame,
)
from .keys import generate_key, key_commitment, read_key_environment, read_key_file

__all__ = [
    "AESGCMEnvelope",
    "CRC32PayloadFrame",
    "PayloadFrameError",
    "PayloadFrameIntegrityError",
    "decode_crc32_frame",
    "decrypt",
    "encode_crc32_frame",
    "encrypt",
    "generate_key",
    "inspect_crc32_frame",
    "key_commitment",
    "read_key_environment",
    "read_key_file",
]
