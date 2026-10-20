from __future__ import annotations

from zsc_tsd.transport import RTPProfile, receiver_sdp, sender_command


def test_rtp_opus_plan_matches_frozen_media_profile() -> None:
    profile = RTPProfile(destination_ip="192.0.2.10")
    command = sender_command("ffmpeg.exe", "carrier.wav", profile)
    sdp = receiver_sdp(profile)
    assert "libopus" in command
    assert "voip" in command
    assert "rtp://192.0.2.10:5004?pkt_size=1200" in command
    assert "m=audio 5004 RTP/AVP 111" in sdp
    assert "a=rtpmap:111 opus/48000/2" in sdp
