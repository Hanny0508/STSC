import pytest

from zsc_tsd.crypto.framing import (
    PayloadFrameError,
    PayloadFrameIntegrityError,
    decode_crc32_frame,
    encode_crc32_frame,
    inspect_crc32_frame,
)


def test_crc32_frame_roundtrip() -> None:
    payload = b"ZTS1"

    frame = encode_crc32_frame(payload)
    inspected = inspect_crc32_frame(
        frame,
        expected_payload_size=4,
    )

    assert frame.hex() == "5a5453312e09b81b"
    assert inspected.payload == payload
    assert inspected.valid is True
    assert inspected.checksum_hex == "2e09b81b"

    recovered = decode_crc32_frame(
        frame,
        expected_payload_size=4,
    )

    assert recovered == payload


def test_crc32_frame_rejects_modified_payload() -> None:
    frame = bytearray(
        encode_crc32_frame(b"ZTS1")
    )

    frame[0] ^= 0x01

    with pytest.raises(
        PayloadFrameIntegrityError
    ):
        decode_crc32_frame(
            frame,
            expected_payload_size=4,
        )


def test_crc32_frame_rejects_wrong_size() -> None:
    frame = encode_crc32_frame(b"ZTS1")

    with pytest.raises(PayloadFrameError):
        decode_crc32_frame(
            frame,
            expected_payload_size=5,
        )


def test_crc32_frame_rejects_empty_payload() -> None:
    with pytest.raises(PayloadFrameError):
        encode_crc32_frame(b"")