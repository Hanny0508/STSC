from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import zsc_tsd.dual_channel.relay_b as relay_b
from zsc_tsd.dual_channel.batch_relay_b import (
    BatchRelayBConfig,
    run_batch_relay,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Experimental node-B batch relay with real RTP packet loss/jitter."
    )
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--expected-sessions", required=True, type=int)
    ap.add_argument("--ffmpeg", required=True, type=Path)
    ap.add_argument("--expected-a-ip", required=True)
    ap.add_argument("--b-ip", required=True)

    ap.add_argument(
        "--output-base",
        type=Path,
        default=Path(
            r"E:\ZSC-TSD-VoIP\control_receiver_v1\artifacts\network_impairment_runs"
        ),
    )

    ap.add_argument(
        "--loss-pct",
        required=True,
        type=float,
        choices=[5.0, 10.0, 15.0],
    )
    ap.add_argument(
        "--jitter-ms",
        required=True,
        type=float,
        choices=[0.0, 30.0, 60.0],
    )
    ap.add_argument("--base-seed", required=True, type=int)
    ap.add_argument("--local-port", type=int, default=5006)
    ap.add_argument("--wait-timeout-seconds", type=int, default=900)
    ap.add_argument("--event-ttl-seconds", type=int, default=14400)

    args = ap.parse_args()

    impairment_sender = (
        Path(__file__).resolve().parent / "rtp_impairment_sender.py"
    )
    if not impairment_sender.is_file():
        raise FileNotFoundError(impairment_sender)

    args.output_base.mkdir(parents=True, exist_ok=True)

    original_sender_command = relay_b.sender_command
    counter = 0

    def impaired_sender_command(ffmpeg, source_wav, profile):
        nonlocal counter

        carrier_index = counter
        counter += 1
        seed = args.base_seed + carrier_index

        source = Path(source_wav).resolve()
        run_root = source.parent.parent
        log_dir = run_root / "logs"

        condition = (
            f"loss{int(args.loss_pct):02d}"
            f"_jitter{int(args.jitter_ms):03d}"
        )

        impairment_log = (
            log_dir
            / f"rtp_impairment_{condition}_seed{seed}.jsonl"
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
            "--seed", str(seed),

            "--impairment-log", str(impairment_log),

            "--local-ip", "127.0.0.1",
            "--local-port", str(args.local_port),
            "--idle-timeout", "0.5",
        ]

    relay_b.sender_command = impaired_sender_command

    experiment = {
        "schema": "zsc-tsd.network-impairment-batch.v1",
        "network_impairment": {
            "layer": "RTP/UDP packet",
            "codec": "Opus",
            "loss_model": "independent Bernoulli",
            "configured_loss_pct": args.loss_pct,
            "jitter_model": f"Uniform(0,{args.jitter_ms} ms)",
            "jitter_max_ms": args.jitter_ms,
            "base_seed": args.base_seed,
            "per_carrier_seed": "base_seed + zero_based_carrier_index",
            "local_impairment_port": args.local_port,
            "final_destination": f"{args.expected_a_ip}:5004",
        },
    }

    print(
        json.dumps(
            {
                "status": "ZSC_TSD_NETWORK_IMPAIRMENT_BATCH_START",
                **experiment,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        result = run_batch_relay(
            BatchRelayBConfig(
                batch_id=args.batch_id,
                expected_sessions=args.expected_sessions,
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

    final = {
        **experiment,
        "batch_result": result,
    }

    print(
        json.dumps(final, indent=2, sort_keys=True),
        flush=True,
    )

    if result["status"] != "ZSC_TSD_DUAL_CHANNEL_BATCH_RELAY_PASS":
        return 1

    print(
        "ZSC_TSD_NETWORK_IMPAIRMENT_BATCH_PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
