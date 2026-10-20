from __future__ import annotations

from pathlib import Path

from zsc_tsd.config import load_profile


def test_local_profile_inherits_paper_profile() -> None:
    profile = load_profile("local_e2e")
    assert profile["zts"]["embedding"]["method"] == "huffman"
    assert profile["transport"]["counts_as_voip"] is False


def test_voip_profile_is_rtp_not_sip() -> None:
    profile = load_profile("voip")
    assert profile["transport"]["backend"] == "rtp_udp_opus_over_tailscale"
    assert profile["transport"]["sip_signaling_implemented"] is False
    assert profile["control_channel"]["server_port"] == 55432
