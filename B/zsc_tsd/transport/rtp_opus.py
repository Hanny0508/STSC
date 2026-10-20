"""FFmpeg command construction for the two-host RTP/UDP Opus experiment."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RTPProfile:
    destination_ip: str
    port: int = 5004
    payload_type: int = 111
    sample_rate_hz: int = 48_000
    bitrate_bps: int = 24_000
    frame_duration_ms: int = 20
    packet_size: int = 1200

    def validate(self) -> None:
        ipaddress.ip_address(self.destination_ip)
        if not 1 <= self.port <= 65535:
            raise ValueError("RTP port must be in [1, 65535]")
        if not 96 <= self.payload_type <= 127:
            raise ValueError("dynamic RTP payload type must be in [96, 127]")
        if self.sample_rate_hz != 48_000:
            raise ValueError("Opus RTP clock rate must be 48000 Hz")
        if self.frame_duration_ms not in {2, 5, 10, 20, 40, 60}:
            raise ValueError("unsupported Opus frame duration")


def sender_command(ffmpeg: str | Path, source_wav: str | Path, profile: RTPProfile) -> list[str]:
    profile.validate()
    return [
        str(ffmpeg), "-hide_banner", "-loglevel", "info", "-re", "-i",
        str(Path(source_wav)), "-map", "0:a:0", "-vn", "-ac", "1", "-ar",
        str(profile.sample_rate_hz), "-c:a", "libopus", "-application", "voip",
        "-b:a", str(profile.bitrate_bps), "-vbr", "off", "-frame_duration",
        str(profile.frame_duration_ms), "-payload_type", str(profile.payload_type),
        "-f", "rtp",
        f"rtp://{profile.destination_ip}:{profile.port}?pkt_size={profile.packet_size}",
    ]


def receiver_sdp(profile: RTPProfile, *, title: str = "ZSC-TSD RTP Opus") -> str:
    profile.validate()
    return (
        "v=0\n"
        "o=- 0 0 IN IP4 127.0.0.1\n"
        f"s={title}\n"
        f"c=IN IP4 {profile.destination_ip}\n"
        "t=0 0\n"
        f"m=audio {profile.port} RTP/AVP {profile.payload_type}\n"
        f"b=AS:{profile.bitrate_bps // 1000}\n"
        f"a=rtpmap:{profile.payload_type} opus/48000/2\n"
    )


def receiver_command(
    ffmpeg: str | Path,
    sdp_path: str | Path,
    output_wav: str | Path,
    *,
    output_sample_rate_hz: int = 16_000,
) -> list[str]:
    return [
        str(ffmpeg), "-hide_banner", "-loglevel", "info", "-protocol_whitelist",
        "file,crypto,data,udp,rtp", "-fflags", "+genpts", "-analyzeduration",
        "10000000", "-probesize", "1000000", "-i", str(Path(sdp_path)),
        "-map", "0:a:0", "-c:a", "pcm_s16le", "-ar",
        str(output_sample_rate_hz), "-ac", "1", "-n", str(Path(output_wav)),
    ]
