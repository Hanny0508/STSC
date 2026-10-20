#!/usr/bin/env python
from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from reedsolo import RSCodec, ReedSolomonError


PROTOCOL_NAME = "ZSC-TSD-SECURE-FRAGMENT-v1"
FIXED_SECRET_TEXT = "Meet at Gate 07."
FIXED_SECRET = FIXED_SECRET_TEXT.encode("utf-8")
AAD = b"ZSC-TSD|IMDB|AES-256-GCM|v1"

AES_KEY_BYTES = 32
AES_NONCE_BYTES = 12
AES_TAG_BYTES = 16
PLAINTEXT_BYTES = 16
ENCRYPTED_PACKET_BYTES = AES_NONCE_BYTES + PLAINTEXT_BYTES + AES_TAG_BYTES

RS_PARITY_BYTES = 16
RS_CODEWORD_BYTES = ENCRYPTED_PACKET_BYTES + RS_PARITY_BYTES
SHARD_DATA_BYTES = 2
SHARD_COUNT = RS_CODEWORD_BYTES // SHARD_DATA_BYTES
MAX_MISSING_SHARDS = RS_PARITY_BYTES // SHARD_DATA_BYTES

SHARD_INDEX_BYTES = 1
SHARD_CRC_BYTES = 2
SHARD_FRAME_BYTES = SHARD_INDEX_BYTES + SHARD_DATA_BYTES + SHARD_CRC_BYTES
SHARD_FRAME_BITS = SHARD_FRAME_BYTES * 8
CRC_DOMAIN = b"ZSC-TSD-SHARD-v1|"


class SecureFragmentError(ValueError):
    """Base exception for malformed protocol material."""


class ShardIntegrityError(SecureFragmentError):
    """Raised when a shard index or CRC16 is invalid."""


class SecurePacketError(SecureFragmentError):
    """Raised when RS or AES-GCM reconstruction fails."""


@dataclass(frozen=True, slots=True)
class DecodedShard:
    frame: bytes
    index: int
    data: bytes
    stored_crc16: int
    computed_crc16: int

    @property
    def valid(self) -> bool:
        return self.stored_crc16 == self.computed_crc16


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_exact(value: bytes, *, name: str, size: int) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != size:
        raise SecureFragmentError(
            f"{name} must be exactly {size} bytes; got {len(value)}"
        )
    return value


def crc16_for_header(
    header: bytes,
    *,
    data_bytes: int = SHARD_DATA_BYTES,
) -> int:
    _bytes_exact(
        header,
        name="shard header",
        size=SHARD_INDEX_BYTES + data_bytes,
    )
    return binascii.crc_hqx(CRC_DOMAIN + header, 0xFFFF)


def encode_shard_frame(index: int, data: bytes) -> bytes:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("shard index must be an integer")
    if not 0 <= index < SHARD_COUNT:
        raise SecureFragmentError(
            f"shard index must be in [0, {SHARD_COUNT - 1}]"
        )
    _bytes_exact(data, name="shard data", size=SHARD_DATA_BYTES)
    header = bytes((index,)) + data
    crc = crc16_for_header(header)
    return header + crc.to_bytes(2, "big")


def decode_shard_frame(
    frame: bytes,
    *,
    expected_index: int | None = None,
    data_bytes: int = SHARD_DATA_BYTES,
    shard_count: int = SHARD_COUNT,
) -> DecodedShard:
    # r3_frame_geometry_applied
    header_bytes = SHARD_INDEX_BYTES + data_bytes
    _bytes_exact(
        frame, name="shard frame", size=header_bytes + SHARD_CRC_BYTES
    )
    index = frame[0]
    data = frame[SHARD_INDEX_BYTES:header_bytes]
    stored = int.from_bytes(frame[header_bytes:header_bytes + SHARD_CRC_BYTES], "big")
    computed = crc16_for_header(frame[:header_bytes], data_bytes=data_bytes)
    if not 0 <= index < shard_count:
        raise ShardIntegrityError(f"invalid shard index: {index}")
    if expected_index is not None and index != expected_index:
        raise ShardIntegrityError(
            f"unexpected shard index: actual={index}, expected={expected_index}"
        )
    if stored != computed:
        raise ShardIntegrityError(
            f"CRC16 failed: stored={stored:04x}, computed={computed:04x}"
        )
    return DecodedShard(
        frame=frame,
        index=index,
        data=data,
        stored_crc16=stored,
        computed_crc16=computed,
    )


def encrypt_packet(*, key: bytes, nonce: bytes) -> bytes:
    _bytes_exact(key, name="AES-256 key", size=AES_KEY_BYTES)
    _bytes_exact(nonce, name="AES-GCM nonce", size=AES_NONCE_BYTES)
    encrypted = AESGCM(key).encrypt(nonce, FIXED_SECRET, AAD)
    packet = nonce + encrypted
    return _bytes_exact(
        packet,
        name="encrypted packet",
        size=ENCRYPTED_PACKET_BYTES,
    )


def decrypt_packet(*, key: bytes, packet: bytes) -> bytes:
    _bytes_exact(key, name="AES-256 key", size=AES_KEY_BYTES)
    _bytes_exact(
        packet,
        name="encrypted packet",
        size=ENCRYPTED_PACKET_BYTES,
    )
    nonce = packet[:AES_NONCE_BYTES]
    ciphertext_and_tag = packet[AES_NONCE_BYTES:]
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext_and_tag, AAD)
    except InvalidTag as exc:
        raise SecurePacketError("AES-GCM authentication failed") from exc
    if plaintext != FIXED_SECRET:
        raise SecurePacketError(
            "authenticated plaintext differs from the fixed secret"
        )
    return plaintext


def rs_encode(packet: bytes) -> bytes:
    _bytes_exact(
        packet,
        name="encrypted packet",
        size=ENCRYPTED_PACKET_BYTES,
    )
    codeword = bytes(RSCodec(RS_PARITY_BYTES).encode(packet))
    return _bytes_exact(
        codeword,
        name="RS codeword",
        size=RS_CODEWORD_BYTES,
    )


def make_shard_frames(codeword: bytes) -> tuple[bytes, ...]:
    _bytes_exact(codeword, name="RS codeword", size=RS_CODEWORD_BYTES)
    return tuple(
        encode_shard_frame(
            index,
            codeword[
                index * SHARD_DATA_BYTES : (index + 1) * SHARD_DATA_BYTES
            ],
        )
        for index in range(SHARD_COUNT)
    )


def reconstruct_packet(shards: Mapping[int, bytes]) -> bytes:
    if not isinstance(shards, Mapping):
        raise TypeError("shards must be a mapping from index to two bytes")
    normalized: dict[int, bytes] = {}
    for index, data in shards.items():
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("all shard indices must be integers")
        if not 0 <= index < SHARD_COUNT:
            raise SecureFragmentError(f"invalid shard index: {index}")
        normalized[index] = _bytes_exact(
            data,
            name=f"shard {index} data",
            size=SHARD_DATA_BYTES,
        )
    missing = [index for index in range(SHARD_COUNT) if index not in normalized]
    if len(missing) > MAX_MISSING_SHARDS:
        raise SecurePacketError(
            f"too many missing shards: {len(missing)}; maximum is "
            f"{MAX_MISSING_SHARDS}"
        )
    codeword = bytearray(RS_CODEWORD_BYTES)
    erasures: list[int] = []
    for index in range(SHARD_COUNT):
        start = index * SHARD_DATA_BYTES
        if index in normalized:
            codeword[start : start + SHARD_DATA_BYTES] = normalized[index]
        else:
            erasures.extend((start, start + 1))
    try:
        decoded = RSCodec(RS_PARITY_BYTES).decode(
            bytes(codeword), erase_pos=erasures
        )
    except ReedSolomonError as exc:
        raise SecurePacketError("Reed-Solomon reconstruction failed") from exc
    packet = bytes(decoded[0])
    return _bytes_exact(
        packet,
        name="reconstructed encrypted packet",
        size=ENCRYPTED_PACKET_BYTES,
    )


def protocol_metadata() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "fixed_secret_utf8_bytes": len(FIXED_SECRET),
        "aad_hex": AAD.hex(),
        "aes": {
            "algorithm": "AES-256-GCM",
            "key_bytes": AES_KEY_BYTES,
            "nonce_bytes": AES_NONCE_BYTES,
            "tag_bytes": AES_TAG_BYTES,
        },
        "reed_solomon": {
            "data_bytes": ENCRYPTED_PACKET_BYTES,
            "parity_bytes": RS_PARITY_BYTES,
            "codeword_bytes": RS_CODEWORD_BYTES,
            "maximum_missing_shards": MAX_MISSING_SHARDS,
        },
        "sharding": {
            "count": SHARD_COUNT,
            "data_bytes_per_shard": SHARD_DATA_BYTES,
            "frame_bytes": SHARD_FRAME_BYTES,
            "frame_bits": SHARD_FRAME_BITS,
            "frame_layout": "index:u8 || data:2B || crc16_ccitt:u16be",
            "crc_initial_value": "ffff",
            "expected_index_is_authenticated_control_metadata": True,
        },
    }


def write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_bundle(key: bytes, nonce: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = encrypt_packet(key=key, nonce=nonce)
    codeword = rs_encode(packet)
    frames = make_shard_frames(codeword)
    decoded_shards = {
        index: decode_shard_frame(frame, expected_index=index).data
        for index, frame in enumerate(frames)
    }
    recovered_packet = reconstruct_packet(
        {
            index: data
            for index, data in decoded_shards.items()
            if index >= MAX_MISSING_SHARDS
        }
    )
    recovered_plaintext = decrypt_packet(key=key, packet=recovered_packet)
    if recovered_plaintext != FIXED_SECRET:
        raise AssertionError("protocol self-test did not recover fixed secret")
    metadata = protocol_metadata()
    public = {
        "status": "ZSC_TSD_SECURE_FRAGMENT_V1_BUNDLE_COMMITTED",
        **metadata,
        "commitments": {
            "fixed_secret_sha256": sha256_bytes(FIXED_SECRET),
            "aes_key_sha256": sha256_bytes(key),
            "nonce_sha256": sha256_bytes(nonce),
            "encrypted_packet_sha256": sha256_bytes(packet),
            "rs_codeword_sha256": sha256_bytes(codeword),
            "ordered_shard_frames_sha256": sha256_bytes(b"".join(frames)),
        },
        "self_test": {
            "all_shard_crc16_checks_pass": True,
            "rs_maximum_declared_erasure_roundtrip_pass": True,
            "self_test_missing_shards": list(range(MAX_MISSING_SHARDS)),
            "aes_gcm_authentication_pass": True,
            "fixed_plaintext_exact_match": True,
        },
    }
    private = {
        "status": "ZSC_TSD_SECURE_FRAGMENT_V1_SENDER_PRIVATE",
        **metadata,
        "fixed_secret_text": FIXED_SECRET_TEXT,
        "fixed_secret_hex": FIXED_SECRET.hex(),
        "aes_key_hex": key.hex(),
        "nonce_hex": nonce.hex(),
        "encrypted_packet_hex": packet.hex(),
        "rs_codeword_hex": codeword.hex(),
        "shards": [
            {
                "index": index,
                "data_hex": frame[1:3].hex(),
                "frame_hex": frame.hex(),
                "frame_bitstring": "".join(f"{byte:08b}" for byte in frame),
            }
            for index, frame in enumerate(frames)
        ],
        "commitments": public["commitments"],
    }
    return public, private


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-output", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--key-hex")
    parser.add_argument("--nonce-hex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.key_hex is None) != (args.nonce_hex is None):
        raise SystemExit("--key-hex and --nonce-hex must be supplied together")
    if args.key_hex is None:
        key = secrets.token_bytes(AES_KEY_BYTES)
        nonce = secrets.token_bytes(AES_NONCE_BYTES)
        entropy_source = "secrets.token_bytes"
    else:
        try:
            key = bytes.fromhex(args.key_hex)
            nonce = bytes.fromhex(args.nonce_hex)
        except ValueError as exc:
            raise SystemExit("key or nonce is not valid hexadecimal") from exc
        entropy_source = "explicit-development-input"
    _bytes_exact(key, name="AES-256 key", size=AES_KEY_BYTES)
    _bytes_exact(nonce, name="AES-GCM nonce", size=AES_NONCE_BYTES)
    public, private = create_bundle(key, nonce)
    public["entropy_source"] = entropy_source
    private["entropy_source"] = entropy_source
    public_path = args.public_output.resolve()
    private_path = args.private_output.resolve()
    if public_path == private_path:
        raise SystemExit("public and private output paths must differ")
    if public_path.exists() or private_path.exists():
        raise SystemExit("STOP: refusing to overwrite an output file")
    write_json_new(private_path, private)
    write_json_new(public_path, public)
    print("ZSC-TSD SECURE FRAGMENT V1 BUNDLE PASS")
    print("public:", public_path)
    print("private:", private_path)
    print("shards:", SHARD_COUNT)
    print("frame bits:", SHARD_FRAME_BITS)
    print("maximum missing shards:", MAX_MISSING_SHARDS)
    print("fixed secret commitment:", public["commitments"]["fixed_secret_sha256"])


if __name__ == "__main__":
    main()
