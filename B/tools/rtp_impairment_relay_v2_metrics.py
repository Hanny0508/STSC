from __future__ import annotations

import argparse
import heapq
import json
import random
import socket
import statistics
import time
from pathlib import Path
from typing import Any


RTP_CLOCK_RATE_HZ = 48_000
EXPECTED_FRAME_SAMPLES = 960
EXPECTED_FRAME_MS = 20.0
FRAME_MAPPING_MIN_CONSISTENCY_PCT = 95.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UDP/RTP impairment relay with reviewer-facing packet metrics."
    )
    p.add_argument("--listen-ip", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, default=5006)
    p.add_argument("--dest-ip", required=True)
    p.add_argument("--dest-port", type=int, default=5004)
    p.add_argument("--loss-pct", type=float, default=0.0)
    p.add_argument(
        "--jitter-ms",
        type=float,
        default=0.0,
        help="Maximum additional one-way random delay, sampled uniformly in [0, jitter_ms].",
    )
    p.add_argument("--seed", type=int, default=20260829)
    p.add_argument(
        "--idle-timeout",
        type=float,
        default=3.0,
        help="Stop after this many seconds without a new packet once traffic has started.",
    )
    p.add_argument(
        "--startup-timeout",
        type=float,
        default=120.0,
        help="Stop if no packet arrives within this many seconds.",
    )
    p.add_argument("--log-json", required=True)
    p.add_argument(
        "--packet-log-jsonl",
        help="Optional per-packet JSONL path. Defaults beside --log-json.",
    )
    return p.parse_args()


def parse_rtp_header(payload: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rtp_valid": False,
        "rtp_version": None,
        "rtp_marker": None,
        "rtp_payload_type": None,
        "rtp_sequence": None,
        "rtp_timestamp": None,
        "rtp_ssrc": None,
    }
    if len(payload) < 12:
        return result
    first = payload[0]
    version = first >> 6
    if version != 2:
        return result
    result.update(
        {
            "rtp_valid": True,
            "rtp_version": version,
            "rtp_marker": bool(payload[1] & 0x80),
            "rtp_payload_type": payload[1] & 0x7F,
            "rtp_sequence": int.from_bytes(payload[2:4], "big"),
            "rtp_timestamp": int.from_bytes(payload[4:8], "big"),
            "rtp_ssrc": int.from_bytes(payload[8:12], "big"),
        }
    )
    return result


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo
    return float(ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": float(max(values)),
    }


def main() -> int:
    a = parse_args()
    if not (0.0 <= a.loss_pct < 100.0):
        raise SystemExit("--loss-pct must be in [0, 100)")
    if a.jitter_ms < 0:
        raise SystemExit("--jitter-ms must be >= 0")

    log_json = Path(a.log_json)
    packet_log = (
        Path(a.packet_log_jsonl)
        if a.packet_log_jsonl
        else log_json.with_suffix(".packets.jsonl")
    )

    rng = random.Random(a.seed)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((a.listen_ip, a.listen_port))
    rx.settimeout(0.01)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Preserve v1 impairment semantics:
    # 1) exactly one rng.random() per received packet for Bernoulli loss;
    # 2) only surviving packets call rng.uniform() when jitter_ms > 0.
    pending: list[tuple[float, int, bytes, int]] = []
    seqno = 0
    received = dropped = forwarded = reordered = 0
    first_packet_at: float | None = None
    last_packet_at: float | None = None
    last_forwarded_input_seq = -1
    start = time.monotonic()

    packet_records: list[dict[str, Any]] = []
    sampled_delays_ms: list[float] = []
    actual_forward_delays_ms: list[float] = []

    current_drop_run = 0
    max_drop_run = 0

    rtp_valid_packets = 0
    rtp_invalid_packets = 0
    input_rtp_gap_packets = 0
    input_rtp_duplicates = 0
    input_rtp_backward_or_reordered = 0
    rtp_timestamp_steps: list[int] = []
    previous_rtp_seq: int | None = None
    previous_rtp_timestamp: int | None = None
    previous_rtp_ssrc: int | None = None

    def mark_forwarded(record_index: int, send_monotonic: float) -> None:
        record = packet_records[record_index]
        record["forwarded"] = True
        record["forward_monotonic_s"] = send_monotonic - start
        actual_ms = 1000.0 * (send_monotonic - record["_receive_monotonic_abs"])
        record["actual_forward_delay_ms"] = actual_ms
        actual_forward_delays_ms.append(actual_ms)

    print(
        f"RTP_IMPAIRMENT_RELAY_V2_READY listen={a.listen_ip}:{a.listen_port} "
        f"dest={a.dest_ip}:{a.dest_port} loss={a.loss_pct:.2f}% "
        f"jitter=[0,{a.jitter_ms:.2f}]ms seed={a.seed}",
        flush=True,
    )

    while True:
        now = time.monotonic()

        while pending and pending[0][0] <= now:
            _, input_seq, payload, record_index = heapq.heappop(pending)
            tx.sendto(payload, (a.dest_ip, a.dest_port))
            send_time = time.monotonic()
            forwarded += 1
            if input_seq < last_forwarded_input_seq:
                reordered += 1
            last_forwarded_input_seq = max(last_forwarded_input_seq, input_seq)
            mark_forwarded(record_index, send_time)

        if first_packet_at is None:
            if now - start >= a.startup_timeout:
                break
        elif not pending and last_packet_at is not None and now - last_packet_at >= a.idle_timeout:
            break

        try:
            payload, source_address = rx.recvfrom(65535)
        except socket.timeout:
            continue

        now = time.monotonic()
        if first_packet_at is None:
            first_packet_at = now
        last_packet_at = now
        received += 1
        input_seq = seqno
        seqno += 1

        rtp = parse_rtp_header(payload)
        if rtp["rtp_valid"]:
            rtp_valid_packets += 1
            current_seq = int(rtp["rtp_sequence"])
            current_ts = int(rtp["rtp_timestamp"])
            current_ssrc = int(rtp["rtp_ssrc"])
            if (
                previous_rtp_seq is not None
                and previous_rtp_timestamp is not None
                and previous_rtp_ssrc == current_ssrc
            ):
                seq_delta = (current_seq - previous_rtp_seq) & 0xFFFF
                if seq_delta == 0:
                    input_rtp_duplicates += 1
                elif 1 < seq_delta < 0x8000:
                    input_rtp_gap_packets += seq_delta - 1
                elif seq_delta >= 0x8000:
                    input_rtp_backward_or_reordered += 1

                if seq_delta == 1:
                    ts_delta = (current_ts - previous_rtp_timestamp) & 0xFFFFFFFF
                    if 0 < ts_delta < 0x80000000:
                        rtp_timestamp_steps.append(ts_delta)
            previous_rtp_seq = current_seq
            previous_rtp_timestamp = current_ts
            previous_rtp_ssrc = current_ssrc
        else:
            rtp_invalid_packets += 1

        record: dict[str, Any] = {
            "input_index": input_seq,
            "receive_monotonic_s": now - start,
            "_receive_monotonic_abs": now,
            "source_ip": source_address[0],
            "source_port": source_address[1],
            "packet_bytes": len(payload),
            **rtp,
            "dropped": False,
            "sampled_delay_ms": None,
            "scheduled_send_monotonic_s": None,
            "forwarded": False,
            "forward_monotonic_s": None,
            "actual_forward_delay_ms": None,
        }
        packet_records.append(record)
        record_index = len(packet_records) - 1

        # IMPORTANT: identical loss RNG call placement to v1.
        if rng.random() < (a.loss_pct / 100.0):
            dropped += 1
            record["dropped"] = True
            current_drop_run += 1
            max_drop_run = max(max_drop_run, current_drop_run)
            continue

        current_drop_run = 0

        # IMPORTANT: identical jitter RNG call placement to v1.
        delay_s = rng.uniform(0.0, a.jitter_ms / 1000.0) if a.jitter_ms else 0.0
        sampled_ms = 1000.0 * delay_s
        sampled_delays_ms.append(sampled_ms)
        record["sampled_delay_ms"] = sampled_ms
        record["scheduled_send_monotonic_s"] = now + delay_s - start
        heapq.heappush(pending, (now + delay_s, input_seq, payload, record_index))

    while pending:
        send_at, input_seq, payload, record_index = heapq.heappop(pending)
        delay = send_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        tx.sendto(payload, (a.dest_ip, a.dest_port))
        send_time = time.monotonic()
        forwarded += 1
        if input_seq < last_forwarded_input_seq:
            reordered += 1
        last_forwarded_input_seq = max(last_forwarded_input_seq, input_seq)
        mark_forwarded(record_index, send_time)

    duration = 0.0
    if first_packet_at is not None and last_packet_at is not None:
        duration = max(0.0, last_packet_at - first_packet_at)

    step_counts: dict[int, int] = {}
    for step in rtp_timestamp_steps:
        step_counts[step] = step_counts.get(step, 0) + 1
    dominant_step = None
    dominant_count = 0
    if step_counts:
        dominant_step, dominant_count = max(
            step_counts.items(), key=lambda item: (item[1], -item[0])
        )
    step_observations = len(rtp_timestamp_steps)
    step_consistency_pct = (
        100.0 * dominant_count / step_observations if step_observations else 0.0
    )
    frame_mapping_verified = bool(
        dominant_step == EXPECTED_FRAME_SAMPLES
        and step_consistency_pct >= FRAME_MAPPING_MIN_CONSISTENCY_PCT
    )
    verified_frame_duration_ms = EXPECTED_FRAME_MS if frame_mapping_verified else None
    max_consecutive_speech_frame_loss_ms = (
        max_drop_run * EXPECTED_FRAME_MS if frame_mapping_verified else None
    )

    result = {
        "schema": "zsc-tsd.rtp-impairment-relay-metrics.v2",
        "listen_ip": a.listen_ip,
        "listen_port": a.listen_port,
        "dest_ip": a.dest_ip,
        "dest_port": a.dest_port,
        "configured_loss_pct": a.loss_pct,
        "configured_jitter_ms_max": a.jitter_ms,
        "jitter_distribution": "uniform_[0,J]",
        "seed": a.seed,
        "impairment_semantics": {
            "loss": "independent Bernoulli drop per received UDP/RTP packet",
            "jitter": "uniform additional one-way delay in [0,J] for non-dropped packets",
            "rng_call_order_matches_v1": True,
        },
        "packets_received": received,
        "packets_dropped": dropped,
        "packets_forwarded": forwarded,
        "observed_drop_pct": (100.0 * dropped / received) if received else 0.0,
        "forward_reordering_events": reordered,
        "max_consecutive_dropped_packets": max_drop_run,
        "traffic_duration_s": duration,
        "sampled_additional_delay_ms": stats(sampled_delays_ms),
        "actual_forward_delay_ms": stats(actual_forward_delays_ms),
        "rtp_input_audit": {
            "rtp_valid_packets": rtp_valid_packets,
            "rtp_invalid_packets": rtp_invalid_packets,
            "input_rtp_sequence_gap_packets": input_rtp_gap_packets,
            "input_rtp_duplicates": input_rtp_duplicates,
            "input_rtp_backward_or_reordered_events": input_rtp_backward_or_reordered,
            "rtp_clock_rate_hz": RTP_CLOCK_RATE_HZ,
            "timestamp_step_observations": step_observations,
            "dominant_timestamp_step_samples": dominant_step,
            "dominant_timestamp_step_count": dominant_count,
            "dominant_timestamp_step_consistency_pct": step_consistency_pct,
            "expected_timestamp_step_samples": EXPECTED_FRAME_SAMPLES,
            "packet_frame_mapping_verified": frame_mapping_verified,
            "verified_frame_duration_ms": verified_frame_duration_ms,
            "max_consecutive_speech_frame_loss_ms": max_consecutive_speech_frame_loss_ms,
        },
        "packet_log_jsonl": str(packet_log),
    }

    log_json.parent.mkdir(parents=True, exist_ok=True)
    packet_log.parent.mkdir(parents=True, exist_ok=True)

    with packet_log.open("w", encoding="utf-8", newline="\n") as stream:
        for record in packet_records:
            clean = {k: v for k, v in record.items() if not k.startswith("_")}
            stream.write(json.dumps(clean, sort_keys=True) + "\n")

    log_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("RTP_IMPAIRMENT_RELAY_V2_DONE")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
