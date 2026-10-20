from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import pytest

from zsc_tsd.control.events import create_signed_event, verify_signed_event
from zsc_tsd.dual_channel.media import CarrierProvisionServer, download_carrier
from zsc_tsd.dual_channel.protocol import (
    DualChannelError,
    assert_metadata_only,
    decode_checks,
)


def test_end_to_end_claim_requires_all_decode_checks() -> None:
    report = {
        "status": "ZSC_TSD_SECURE_IMDB_BATCH_V1_4_PASS",
        "rs_reconstruction_pass": True,
        "aes_gcm_authentication_pass": True,
        "plaintext_exact_match": True,
    }
    assert all(decode_checks(report).values())
    report["plaintext_exact_match"] = False
    assert not all(decode_checks(report).values())


def test_secret_bearing_control_metadata_is_rejected() -> None:
    assert_metadata_only({"carrier_sha256": "ab", "secret_payload_allowed": False})
    with pytest.raises(DualChannelError, match="forbidden"):
        assert_metadata_only({"nested": {"payload_hex": "00"}})
    with pytest.raises(DualChannelError, match="must declare"):
        assert_metadata_only({"secret_payload_in_control_channel": True})


def test_receiver_rtp_ready_is_authenticated_event() -> None:
    key = bytes(range(32))
    event = create_signed_event(
        key,
        session_id="d716350d-683c-42d0-82a5-a12d1aa8699f",
        actor_id="zsc-voip-a",
        sequence_no=3,
        event_type="RECEIVER_RTP_READY",
        correlation_id="9ef80b7b-339c-41a2-a4eb-42cc70bff131",
        body={"phase": "RECEIVER_RTP_READY", "rtp_port": 5004},
        ttl_seconds=60,
    )
    assert verify_signed_event(key, event)["event_type"] == "RECEIVER_RTP_READY"


def test_restricted_carrier_provision_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 160)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "received.wav"
    with CarrierProvisionServer(
        bind_ip="127.0.0.1",
        port=0,
        expected_client_ip="127.0.0.1",
        carrier_path=source,
        carrier_sha256=digest,
    ) as server:
        result = download_carrier(
            server.url,
            output,
            expected_server_ip="127.0.0.1",
            expected_sha256=digest,
            expected_bytes=source.stat().st_size,
        )
        assert server.wait_for_transfer(2)
    assert result.sha256 == digest
    assert output.read_bytes() == source.read_bytes()
