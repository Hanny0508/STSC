"""Create the reproducible FFmpeg plan for the two-host RTP/Opus testbed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zsc_tsd.manifest import write_json_new
from zsc_tsd.transport import RTPProfile, receiver_command, receiver_sdp, sender_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--source-wav", required=True)
    parser.add_argument("--receiver-ip", required=True)
    parser.add_argument("--port", type=int, default=5004)
    parser.add_argument("--receiver-sdp", required=True, type=Path)
    parser.add_argument("--received-wav", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profile = RTPProfile(destination_ip=args.receiver_ip, port=args.port)
    if args.receiver_sdp.exists():
        raise FileExistsError(f"refusing to overwrite: {args.receiver_sdp}")
    args.receiver_sdp.parent.mkdir(parents=True, exist_ok=True)
    args.receiver_sdp.write_text(receiver_sdp(profile), encoding="ascii", newline="\n")
    write_json_new(
        args.output,
        {
            "schema": "zsc-tsd.rtp-opus-plan.v1",
            "scope": "RTP/UDP Opus media transport; no SIP signaling claim",
            "sender_command": sender_command(args.ffmpeg, args.source_wav, profile),
            "receiver_command": receiver_command(
                args.ffmpeg, args.receiver_sdp, args.received_wav
            ),
            "receiver_sdp": str(args.receiver_sdp.resolve()),
        },
    )
    print(json.dumps({"status": "RTP_OPUS_PLAN_READY", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
