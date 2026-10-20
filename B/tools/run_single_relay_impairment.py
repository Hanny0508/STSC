from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import zsc_tsd.dual_channel.relay_b as relay_b
from zsc_tsd.dual_channel.relay_b import RelayBConfig, run_relay


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Single-carrier B relay with RTP packet loss/jitter."
    )
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--correlation-id", required=True)
    ap.add_argument("--ffmpeg", required=True, type=Path)
    ap.add_argument("--expected-a-ip", required=True)
    ap.add_argument("--b-ip", required=True)
    ap.add_argument("--output-base", required=True, type=Path)

    ap.add_argument("--loss-pct", required=True, type=float)
    ap.add_argument("--jitter-ms", required=True, type=float)
    ap.add_argument("--seed", required=True, type=int)

    ap.add_argument("--local-port", type=int, default=5006)
    ap.add_argument("--wait-timeout-seconds", type=int, default=600)
    ap.add_argument("--event-ttl-seconds", type=int, default=3600)

    args = ap.parse_args()

    if args.loss_pct not in {5.0, 10.0, 15.0}:
        raise ValueError("--loss-pct must be 5, 10, or 15")
    if args.jitter_ms not in {0.0, 30.0, 60.0}:
        raise ValueError("--jitter-ms must be 0, 30, or 60")

    impairment_sender = (
        Path(__file__).resolve().parent / "rtp_impairment_sender.py"
    )
    if not impairment_sender.is_file():
        raise FileNotFoundError(impairment_sender)

    args.output_base.mkdir(parents=True, exist_ok=True)

    original_sender_command = relay_b.sender_command

    def impaired_sender_command(ffmpeg, source_wav, profile):
        source = Path(source_wav).resolve()

        run_root = source.parent.parent
        log_dir = run_root / "logs"

        condition = (
            f"loss{int(args.loss_pct):02d}"
            f"_jitter{int(args.jitter_ms):03d}"
            f"_seed{args.seed}"
        )

        impairment_log = (
            log_dir / f"rtp_impairment_{condition}.jsonl"
        )

        return [
            sys.executable,
            str(impairment_sender),

            "--ffmpeg", str(ffmpeg),
            "--source-wav", str(source),

            "--dest-ip", profile.destination_ip,
            "--dest-port", str(profile.port),

            "--payload-type", str(profile.payload_type),
            "--sample-rate", str(profile.sample_rate_hz),
            "--bitrate", str(profile.bitrate_bps),
            "--frame-duration-ms", str(profile.frame_duration_ms),
            "--packet-size", str(profile.packet_size),

            "--loss", str(args.loss_pct / 100.0),
            "--jitter-ms", str(args.jitter_ms),
            "--seed", str(args.seed),

            "--impairment-log", str(impairment_log),

            "--local-ip", "127.0.0.1",
            "--local-port", str(args.local_port),
            "--idle-timeout", "0.5",
        ]

    relay_b.sender_command = impaired_sender_command

    print(json.dumps({
        "status": "ZSC_TSD_SINGLE_IMPAIRMENT_RELAY_START",
        "session_id": args.session_id,
        "correlation_id": args.correlation_id,
        "loss_pct": args.loss_pct,
        "jitter_ms": args.jitter_ms,
        "seed": args.seed,
        "media_path": (
            f"FFmpeg->127.0.0.1:{args.local_port}"
            f"->{args.expected_a_ip}:5004"
        ),
    }, indent=2, sort_keys=True), flush=True)

    try:
        result = run_relay(
            RelayBConfig(
                session_id=args.session_id,
                correlation_id=args.correlation_id,
                output_base=args.output_base,
                ffmpeg=args.ffmpeg,
                expected_a_ip=args.expected_a_ip,
                b_ip=args.b_ip,
                event_ttl_seconds=args.event_ttl_seconds,
                wait_timeout_seconds=args.wait_timeout_seconds,
            )
        )
    finally:
        relay_b.sender_command = original_sender_command

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    if result["status"] != "ZSC_TSD_DUAL_CHANNEL_RELAY_PASS":
        return 1

    print("ZSC_TSD_SINGLE_IMPAIRMENT_RELAY_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
