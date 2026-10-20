from __future__ import annotations

import zlib
from dataclasses import dataclass


class PayloadFrameError(ValueError):
    """Base exception for malformed payload frames."""


class PayloadFrameIntegrityError(PayloadFrameError):
    """Raised when the stored CRC32 does not match the payload."""


@dataclass(frozen=True)
class CRC32PayloadFrame:
    raw: bytes
    payload: bytes
    stored_crc32: int
    computed_crc32: int

    @property
    def valid(self) -> bool:
        return self.stored_crc32 == self.computed_crc32

    @property
    def checksum_hex(self) -> str:
        return f"{self.stored_crc32:08x}"


def _as_bytes(
    value: bytes | bytearray | memoryview,
    *,
    name: str,
) -> bytes:
    if not isinstance(
        value,
        (bytes, bytearray, memoryview),
    ):
        raise TypeError(
            f"{name} must be bytes, bytearray, or memoryview"
        )

    return bytes(value)


def crc32_value(
    payload: bytes | bytearray | memoryview,
) -> int:
    payload_bytes = _as_bytes(
        payload,
        name="payload",
    )

    return zlib.crc32(payload_bytes) & 0xFFFFFFFF


def encode_crc32_frame(
    payload: bytes | bytearray | memoryview,
) -> bytes:
    payload_bytes = _as_bytes(
        payload,
        name="payload",
    )

    if not payload_bytes:
        raise PayloadFrameError(
            "payload must not be empty"
        )

    checksum = crc32_value(payload_bytes)

    return (
        payload_bytes
        + checksum.to_bytes(
            4,
            byteorder="big",
            signed=False,
        )
    )


def inspect_crc32_frame(
    frame: bytes | bytearray | memoryview,
    *,
    expected_payload_size: int | None = None,
) -> CRC32PayloadFrame:
    frame_bytes = _as_bytes(
        frame,
        name="frame",
    )

    if len(frame_bytes) < 5:
        raise PayloadFrameError(
            "CRC32 frame must contain at least "
            "one payload byte and four checksum bytes"
        )

    payload = frame_bytes[:-4]

    if (
        expected_payload_size is not None
        and len(payload) != expected_payload_size
    ):
        raise PayloadFrameError(
            "Unexpected payload size: "
            f"actual={len(payload)}, "
            f"expected={expected_payload_size}"
        )

    stored_crc32 = int.from_bytes(
        frame_bytes[-4:],
        byteorder="big",
        signed=False,
    )

    computed_crc32 = crc32_value(payload)

    return CRC32PayloadFrame(
        raw=frame_bytes,
        payload=payload,
        stored_crc32=stored_crc32,
        computed_crc32=computed_crc32,
    )


def decode_crc32_frame(
    frame: bytes | bytearray | memoryview,
    *,
    expected_payload_size: int | None = None,
) -> bytes:
    result = inspect_crc32_frame(
        frame,
        expected_payload_size=(
            expected_payload_size
        ),
    )

    if not result.valid:
        raise PayloadFrameIntegrityError(
            "CRC32 validation failed: "
            f"stored={result.stored_crc32:08x}, "
            f"computed={result.computed_crc32:08x}"
        )

    return result.payload