"""Reviewer-facing command line interface for the ZSC-TSD artifact."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .transport import RTPProfile, receiver_command, receiver_sdp, sender_command


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zsc-tsd",
        description="Reproducible entry points for the ZSC-TSD research artifact.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="check lightweight project structure")
    preflight.add_argument("--profile", default="paper_r1")

    show = commands.add_parser("show-config", help="print a resolved YAML profile")
    show.add_argument("profile")

    batch = commands.add_parser("secure-imdb-batch", help="run the frozen v1.4 secure batch")
    batch.add_argument("--bundle-public", required=True)
    batch.add_argument("--bundle-private", required=True)
    batch.add_argument("--start-row", type=int)
    batch.add_argument("--max-attempts-per-shard", type=int)

    real_imdb = commands.add_parser(
        "prepare-acl-imdb",
        help="download and deterministically derive prompts from Stanford ACL IMDb",
    )
    real_imdb.add_argument("--archive", type=Path)
    real_imdb.add_argument("--output-dir", type=Path)
    real_imdb.add_argument("--download", action="store_true")
    real_imdb.add_argument("--max-prompts", type=int, default=50_000)

    facodec = commands.add_parser(
        "facodec-roundtrip", help="execute official FACodec encode/quantize/decode"
    )
    facodec.add_argument("--input", required=True, type=Path)
    facodec.add_argument("--output", required=True, type=Path)
    facodec.add_argument("--report", required=True, type=Path)
    facodec.add_argument("--model-dir", type=Path)
    facodec.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    facodec_v2 = commands.add_parser(
        "facodec-v2-convert",
        help="convert a source WAV to a reference speaker with FACodec V2",
    )
    facodec_v2.add_argument("--source", required=True, type=Path)
    facodec_v2.add_argument("--reference", required=True, type=Path)
    facodec_v2.add_argument("--output", required=True, type=Path)
    facodec_v2.add_argument("--report", required=True, type=Path)
    facodec_v2.add_argument("--model-dir", type=Path)
    facodec_v2.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    daq_batch = commands.add_parser(
        "secure-imdb-daqtts-batch",
        help="run v1.5 with real ACL IMDb prompts and TransformerTTS plus FACodec",
    )
    daq_batch.add_argument("--bundle-public", required=True)
    daq_batch.add_argument("--bundle-private", required=True)
    daq_batch.add_argument("--imdb-csv", required=True)
    daq_batch.add_argument("--imdb-manifest", required=True)
    daq_batch.add_argument("--facodec-python", type=Path)
    daq_batch.add_argument("--reference-wav", required=True, type=Path)
    daq_batch.add_argument("--start-row", type=int, default=111)
    daq_batch.add_argument("--max-attempts-per-shard", type=int, default=8)

    rtp = commands.add_parser("rtp-plan", help="print truthful FFmpeg RTP/Opus commands")
    rtp.add_argument("--ffmpeg", required=True)
    rtp.add_argument("--source-wav", required=True)
    rtp.add_argument("--receiver-ip", required=True)
    rtp.add_argument("--port", type=int, default=5004)
    rtp.add_argument("--receiver-sdp", required=True)
    rtp.add_argument("--received-wav", required=True)

    coordinator = commands.add_parser(
        "dual-channel-run",
        help="run node A coordinator for authenticated B-to-A RTP/Opus",
    )
    coordinator.add_argument("--source-wav", required=True, type=Path)
    coordinator.add_argument("--ffmpeg", required=True, type=Path)
    coordinator.add_argument("--a-ip", required=True)
    coordinator.add_argument("--b-ip", required=True)
    coordinator.add_argument(
        "--output-base", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments"
    )
    coordinator.add_argument("--provisioning-port", type=int, default=5602)
    coordinator.add_argument("--rtp-port", type=int, default=5004)
    coordinator.add_argument("--event-ttl-seconds", type=int, default=3600)
    coordinator.add_argument("--wait-timeout-seconds", type=int, default=300)
    coordinator.add_argument("--decoder-command-json", type=Path)

    coordinator.add_argument(
        "--sync-manifest",
        type=Path,
        help="frozen synchronization manifest verified before decoding "
        "(default: newest artifacts/freezes/*/zts_manifest*.json)",
    )
    coordinator.add_argument(
        "--sync-gate-python",
        type=Path,
        help="interpreter that hashes the model files (default: this interpreter)",
    )
    coordinator.add_argument(
        "--no-sync-gate",
        action="store_true",
        help="skip the synchronization-profile gate",
    )


    relay = commands.add_parser(
        "dual-channel-relay",
        help="run node B authenticated carrier relay and RTP/Opus sender",
    )
    relay.add_argument("--session-id", required=True)
    relay.add_argument("--correlation-id", required=True)
    relay.add_argument("--ffmpeg", required=True, type=Path)
    relay.add_argument("--expected-a-ip", required=True)
    relay.add_argument("--b-ip", required=True)
    relay.add_argument(
        "--output-base", type=Path, default=Path("E:/ZSC-TSD-VoIP/experiments")
    )
    relay.add_argument("--event-ttl-seconds", type=int, default=3600)
    relay.add_argument("--wait-timeout-seconds", type=int, default=300)

    batch_a = commands.add_parser(
        "dual-channel-batch-a",
        help="transmit and post-decode every precommitted secure-batch carrier",
    )
    batch_a.add_argument("--secure-batch-root", required=True, type=Path)
    batch_a.add_argument("--bundle-private", required=True, type=Path)
    batch_a.add_argument("--ffmpeg", required=True, type=Path)
    batch_a.add_argument("--asr-python", required=True, type=Path)
    batch_a.add_argument("--zts-python", required=True, type=Path)
    batch_a.add_argument("--asr-model", required=True, type=Path)
    batch_a.add_argument("--a-ip", required=True)
    batch_a.add_argument("--b-ip", required=True)
    batch_a.add_argument(
        "--output-base", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments"
    )
    batch_a.add_argument("--provisioning-port", type=int, default=5602)
    batch_a.add_argument("--rtp-port", type=int, default=5004)
    batch_a.add_argument("--event-ttl-seconds", type=int, default=14_400)
    batch_a.add_argument("--wait-timeout-seconds", type=int, default=900)

    batch_a.add_argument("--sync-manifest", type=Path)
    batch_a.add_argument("--sync-gate-python", type=Path)
    batch_a.add_argument("--no-sync-gate", action="store_true")


    batch_b = commands.add_parser(
        "dual-channel-batch-relay",
        help="relay every authenticated carrier session in one frozen batch",
    )
    batch_b.add_argument("--batch-id", required=True)
    batch_b.add_argument("--expected-sessions", required=True, type=int)
    batch_b.add_argument("--ffmpeg", required=True, type=Path)
    batch_b.add_argument("--expected-a-ip", required=True)
    batch_b.add_argument("--b-ip", required=True)
    batch_b.add_argument(
        "--output-base", type=Path, default=Path("E:/ZSC-TSD-VoIP/experiments")
    )
    batch_b.add_argument("--event-ttl-seconds", type=int, default=14_400)
    batch_b.add_argument("--wait-timeout-seconds", type=int, default=900)
    return parser


def _preflight(profile: str) -> int:
    from .config import load_profile

    required = [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "zsc_tsd" / "zts" / "encoder.py",
        PROJECT_ROOT / "zsc_tsd" / "speech" / "transformertts.py",
        PROJECT_ROOT / "zsc_tsd" / "control" / "events.py",
        PROJECT_ROOT / "zsc_tsd" / "dual_channel" / "coordinator_a.py",
        PROJECT_ROOT / "zsc_tsd" / "dual_channel" / "relay_b.py",
        PROJECT_ROOT / "zsc_tsd" / "dual_channel" / "batch_a.py",
        PROJECT_ROOT / "zsc_tsd" / "dual_channel" / "batch_relay_b.py",
        PROJECT_ROOT / "tools" / "zts_secure_imdb_batch_runner_v1_4.py",
        PROJECT_ROOT / "tools" / "zts_voip_postdecode_v1.py",
        PROJECT_ROOT / "tools" / "whisper_asr_worker.py",
        PROJECT_ROOT / "tools" / "facodec_v2_voice_conversion_worker.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    resolved = load_profile(profile)
    result = {
        "status": "ZSC_TSD_PREFLIGHT_PASS" if not missing else "ZSC_TSD_PREFLIGHT_FAIL",
        "version": __version__,
        "project_root": str(PROJECT_ROOT),
        "profile": resolved.get("_meta", {}).get("profile_path"),
        "required_files": len(required),
        "missing_files": missing,
        "control_environment_present": {
            "dsn": bool(os.environ.get("ZSC_TSD_CONTROL_DSN")),
            "master_key": bool(os.environ.get("ZSC_TSD_CONTROL_MASTER_KEY_B64")),
        },
    }
    _json(result)
    return 0 if not missing else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        return _preflight(args.profile)
    if args.command == "show-config":
        from .config import load_profile

        _json(load_profile(args.profile))
        return 0
    if args.command == "secure-imdb-batch":
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "zts_secure_imdb_batch_runner_v1_4.py"),
            "--bundle-public", args.bundle_public,
            "--bundle-private", args.bundle_private,
        ]
        if args.start_row is not None:
            command += ["--start-row", str(args.start_row)]
        if args.max_attempts_per_shard is not None:
            command += ["--max-attempts-per-shard", str(args.max_attempts_per_shard)]
        return subprocess.call(command, cwd=str(PROJECT_ROOT))
    if args.command == "prepare-acl-imdb":
        command = [sys.executable, str(PROJECT_ROOT / "tools" / "prepare_acl_imdb_prompts.py")]
        if args.archive is not None:
            command += ["--archive", str(args.archive)]
        if args.output_dir is not None:
            command += ["--output-dir", str(args.output_dir)]
        if args.download:
            command.append("--download")
        command += ["--max-prompts", str(args.max_prompts)]
        return subprocess.call(command, cwd=str(PROJECT_ROOT))
    if args.command == "facodec-roundtrip":
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "facodec_roundtrip_worker.py"),
            "--input", str(args.input),
            "--output", str(args.output),
            "--report", str(args.report),
            "--device", args.device,
        ]
        if args.model_dir is not None:
            command += ["--model-dir", str(args.model_dir)]
        return subprocess.call(command, cwd=str(PROJECT_ROOT))
    if args.command == "facodec-v2-convert":
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "facodec_v2_voice_conversion_worker.py"),
            "--source", str(args.source),
            "--reference", str(args.reference),
            "--output", str(args.output),
            "--report", str(args.report),
            "--device", args.device,
        ]
        if args.model_dir is not None:
            command += ["--model-dir", str(args.model_dir)]
        return subprocess.call(command, cwd=str(PROJECT_ROOT))
    if args.command == "secure-imdb-daqtts-batch":
        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "zts_secure_imdb_daqtts_batch_runner_v1_5.py"),
            "--bundle-public", args.bundle_public,
            "--bundle-private", args.bundle_private,
            "--imdb-csv", args.imdb_csv,
            "--imdb-manifest", args.imdb_manifest,
            "--reference-wav", str(args.reference_wav),
            "--start-row", str(args.start_row),
            "--max-attempts-per-shard", str(args.max_attempts_per_shard),
        ]
        if args.facodec_python is not None:
            command += ["--facodec-python", str(args.facodec_python)]
        return subprocess.call(command, cwd=str(PROJECT_ROOT))
    if args.command == "rtp-plan":
        profile = RTPProfile(destination_ip=args.receiver_ip, port=args.port)
        _json(
            {
                "scope": "RTP/UDP Opus media transport; SIP signaling not implemented",
                "sender_command": sender_command(args.ffmpeg, args.source_wav, profile),
                "receiver_sdp_path": args.receiver_sdp,
                "receiver_sdp": receiver_sdp(profile),
                "receiver_command": receiver_command(
                    args.ffmpeg, args.receiver_sdp, args.received_wav
                ),
            }
        )
        return 0
    if args.command == "dual-channel-run":
        from .dual_channel.coordinator_a import CoordinatorAConfig, run_coordinator

        result = run_coordinator(
            CoordinatorAConfig(
                source_wav=args.source_wav,
                output_base=args.output_base,
                ffmpeg=args.ffmpeg,
                a_ip=args.a_ip,
                b_ip=args.b_ip,
                provisioning_port=args.provisioning_port,
                rtp_port=args.rtp_port,
                event_ttl_seconds=args.event_ttl_seconds,
                wait_timeout_seconds=args.wait_timeout_seconds,
                decoder_command_json=args.decoder_command_json,
                sync_manifest=args.sync_manifest,
                sync_gate_python=args.sync_gate_python,
                skip_sync_gate=args.no_sync_gate,
            )
        )
        _json(result)
        return 0
    if args.command == "dual-channel-relay":
        from .dual_channel.relay_b import RelayBConfig, run_relay

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
        _json(result)
        return 0
    if args.command == "dual-channel-batch-a":
        from .dual_channel.batch_a import BatchAConfig, run_batch_a

        result = run_batch_a(
            BatchAConfig(
                secure_batch_root=args.secure_batch_root,
                bundle_private=args.bundle_private,
                output_base=args.output_base,
                ffmpeg=args.ffmpeg,
                asr_python=args.asr_python,
                zts_python=args.zts_python,
                asr_model=args.asr_model,
                a_ip=args.a_ip,
                b_ip=args.b_ip,
                provisioning_port=args.provisioning_port,
                rtp_port=args.rtp_port,
                event_ttl_seconds=args.event_ttl_seconds,
                wait_timeout_seconds=args.wait_timeout_seconds,
                sync_manifest=args.sync_manifest,
                sync_gate_python=args.sync_gate_python,
                skip_sync_gate=args.no_sync_gate,
            )
        )
        _json(result)
        return 0 if str(result["status"]).endswith("_PASS") else 2
    if args.command == "dual-channel-batch-relay":
        from .dual_channel.batch_relay_b import BatchRelayBConfig, run_batch_relay

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
        _json(result)
        return 0 if str(result["status"]).endswith("_PASS") else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
