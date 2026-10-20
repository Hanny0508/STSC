from __future__ import annotations

import argparse
import heapq
import json
import random
import socket
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UDP/RTP impairment relay for real packet-loss/jitter experiments.")
    p.add_argument("--listen-ip", default="127.0.0.1")
    p.add_argument("--listen-port", type=int, default=5006)
    p.add_argument("--dest-ip", required=True)
    p.add_argument("--dest-port", type=int, default=5004)
    p.add_argument("--loss-pct", type=float, default=0.0)
    p.add_argument("--jitter-ms", type=float, default=0.0,
                   help="Maximum additional one-way random delay, sampled uniformly in [0, jitter_ms].")
    p.add_argument("--seed", type=int, default=20260829)
    p.add_argument("--idle-timeout", type=float, default=3.0,
                   help="Stop after this many seconds without a new packet once traffic has started.")
    p.add_argument("--startup-timeout", type=float, default=120.0,
                   help="Stop if no packet arrives within this many seconds.")
    p.add_argument("--log-json", required=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if not (0.0 <= a.loss_pct < 100.0):
        raise SystemExit("--loss-pct must be in [0, 100)")
    if a.jitter_ms < 0:
        raise SystemExit("--jitter-ms must be >= 0")

    rng = random.Random(a.seed)
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((a.listen_ip, a.listen_port))
    rx.settimeout(0.01)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pending = []
    seqno = 0
    received = dropped = forwarded = reordered = 0
    first_packet_at = None
    last_packet_at = None
    last_forwarded_input_seq = -1
    start = time.monotonic()

    print(
        f"RTP_IMPAIRMENT_RELAY_READY listen={a.listen_ip}:{a.listen_port} "
        f"dest={a.dest_ip}:{a.dest_port} loss={a.loss_pct:.2f}% "
        f"jitter=[0,{a.jitter_ms:.2f}]ms seed={a.seed}",
        flush=True,
    )

    while True:
        now = time.monotonic()

        while pending and pending[0][0] <= now:
            _, input_seq, payload = heapq.heappop(pending)
            tx.sendto(payload, (a.dest_ip, a.dest_port))
            forwarded += 1
            if input_seq < last_forwarded_input_seq:
                reordered += 1
            last_forwarded_input_seq = max(last_forwarded_input_seq, input_seq)

        if first_packet_at is None:
            if now - start >= a.startup_timeout:
                break
        elif not pending and last_packet_at is not None and now - last_packet_at >= a.idle_timeout:
            break

        try:
            payload, _ = rx.recvfrom(65535)
        except socket.timeout:
            continue

        now = time.monotonic()
        if first_packet_at is None:
            first_packet_at = now
        last_packet_at = now
        received += 1
        input_seq = seqno
        seqno += 1

        if rng.random() < (a.loss_pct / 100.0):
            dropped += 1
            continue

        delay_s = rng.uniform(0.0, a.jitter_ms / 1000.0) if a.jitter_ms else 0.0
        heapq.heappush(pending, (now + delay_s, input_seq, payload))

    while pending:
        send_at, input_seq, payload = heapq.heappop(pending)
        delay = send_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        tx.sendto(payload, (a.dest_ip, a.dest_port))
        forwarded += 1
        if input_seq < last_forwarded_input_seq:
            reordered += 1
        last_forwarded_input_seq = max(last_forwarded_input_seq, input_seq)

    duration = 0.0
    if first_packet_at is not None and last_packet_at is not None:
        duration = max(0.0, last_packet_at - first_packet_at)

    result = {
        "listen_ip": a.listen_ip,
        "listen_port": a.listen_port,
        "dest_ip": a.dest_ip,
        "dest_port": a.dest_port,
        "configured_loss_pct": a.loss_pct,
        "configured_jitter_ms_max": a.jitter_ms,
        "jitter_distribution": "uniform_[0,J]",
        "seed": a.seed,
        "packets_received": received,
        "packets_dropped": dropped,
        "packets_forwarded": forwarded,
        "observed_drop_pct": (100.0 * dropped / received) if received else 0.0,
        "forward_reordering_events": reordered,
        "traffic_duration_s": duration,
    }

    out = Path(a.log_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RTP_IMPAIRMENT_RELAY_DONE")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
