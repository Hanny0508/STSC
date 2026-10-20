from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from zsc_tsd.transport import RTPProfile, sender_command


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--source-wav", required=True)
    ap.add_argument("--dest-ip", required=True)
    ap.add_argument("--dest-port", required=True, type=int)
    ap.add_argument("--payload-type", required=True, type=int)
    ap.add_argument("--sample-rate", required=True, type=int)
    ap.add_argument("--bitrate", required=True, type=int)
    ap.add_argument("--frame-duration-ms", required=True, type=int)
    ap.add_argument("--packet-size", required=True, type=int)

    ap.add_argument("--loss", required=True, type=float)
    ap.add_argument("--jitter-ms", required=True, type=float)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--impairment-log", required=True, type=Path)

    ap.add_argument("--local-ip", default="127.0.0.1")
    ap.add_argument("--local-port", default=5006, type=int)
    ap.add_argument("--idle-timeout", default=0.5, type=float)

    args = ap.parse_args()

    relay_script = Path(__file__).with_name("rtp_impairment_relay.py")
    if not relay_script.is_file():
        raise FileNotFoundError(relay_script)

    args.impairment_log.parent.mkdir(parents=True, exist_ok=True)

    relay_cmd = [
        sys.executable,
        str(relay_script),
        "--listen-ip", args.local_ip,
        "--listen-port", str(args.local_port),
        "--dest-ip", args.dest_ip,
        "--dest-port", str(args.dest_port),
        "--loss", str(args.loss),
        "--jitter-ms", str(args.jitter_ms),
        "--seed", str(args.seed),
        "--log", str(args.impairment_log),
        "--idle-timeout", str(args.idle_timeout),
    ]

    relay = subprocess.Popen(
        relay_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    ready = False
    assert relay.stdout is not None

    try:
        while True:
            line = relay.stdout.readline()
            if not line:
                break
            print("[IMPAIRMENT] " + line.rstrip(), flush=True)
            if "RTP_IMPAIRMENT_READY" in line:
                ready = True
                break

        if not ready:
            raise RuntimeError("RTP impairment relay failed before READY")

        local_profile = RTPProfile(
            destination_ip=args.local_ip,
            port=args.local_port,
            payload_type=args.payload_type,
            sample_rate_hz=args.sample_rate,
            bitrate_bps=args.bitrate,
            frame_duration_ms=args.frame_duration_ms,
            packet_size=args.packet_size,
        )

        ffmpeg_cmd = sender_command(
            args.ffmpeg,
            args.source_wav,
            local_profile,
        )

        print(
            "[IMPAIRMENT] media_path="
            f"FFmpeg->{args.local_ip}:{args.local_port}"
            f"->{args.dest_ip}:{args.dest_port}",
            flush=True,
        )

        completed = subprocess.run(
            ffmpeg_cmd,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )

        remaining, _ = relay.communicate(timeout=10)
        if remaining:
            for line in remaining.splitlines():
                print("[IMPAIRMENT] " + line, flush=True)

        if completed.returncode != 0:
            return completed.returncode

        if relay.returncode != 0:
            raise RuntimeError(
                f"RTP impairment relay exited with code {relay.returncode}"
            )

        print("[IMPAIRMENT] RTP_IMPAIRMENT_SENDER_PASS", flush=True)
        return 0

    finally:
        if relay.poll() is None:
            relay.kill()
            relay.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
