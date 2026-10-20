from __future__ import annotations

import argparse
import heapq
import json
import random
import select
import socket
import time
from pathlib import Path


def rtp_fields(packet: bytes) -> tuple[int | None, int | None]:
    if len(packet) < 12:
        return None, None
    version = packet[0] >> 6
    if version != 2:
        return None, None
    seq = int.from_bytes(packet[2:4], "big")
    timestamp = int.from_bytes(packet[4:8], "big")
    return seq, timestamp


def write_jsonl(stream, obj: dict) -> None:
    stream.write(json.dumps(obj, sort_keys=True) + "\n")
    stream.flush()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="RTP/UDP packet impairment relay for ZSC-TSD network experiments."
    )
    ap.add_argument("--listen-ip", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=5006)
    ap.add_argument("--dest-ip", required=True)
    ap.add_argument("--dest-port", type=int, default=5004)
    ap.add_argument("--loss", type=float, required=True,
                    help="Independent packet loss probability, e.g. 0.05.")
    ap.add_argument("--jitter-ms", type=float, required=True,
                    help="Maximum added delay. Each forwarded packet gets U(0,J) ms.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--idle-timeout", type=float, default=10.0)
    args = ap.parse_args()

    if not 0.0 <= args.loss < 1.0:
        raise ValueError("--loss must be in [0,1)")
    if args.jitter_ms < 0:
        raise ValueError("--jitter-ms must be >= 0")
    if args.idle_timeout <= 0:
        raise ValueError("--idle-timeout must be > 0")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.log.exists():
        raise FileExistsError(args.log)

    summary_path = args.log.with_suffix(".summary.json")
    if summary_path.exists():
        raise FileExistsError(summary_path)

    rng = random.Random(args.seed)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.listen_ip, args.listen_port))
    sock.setblocking(False)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    destination = (args.dest_ip, args.dest_port)
    queue: list[tuple[float, int, bytes, dict]] = []
    queue_counter = 0
    received = 0
    dropped = 0
    forwarded = 0
    started = time.monotonic()
    last_receive = started

    with args.log.open("x", encoding="utf-8", newline="\n") as log:
        write_jsonl(log, {
            "event": "START",
            "listen_ip": args.listen_ip,
            "listen_port": args.listen_port,
            "dest_ip": args.dest_ip,
            "dest_port": args.dest_port,
            "loss_probability": args.loss,
            "jitter_distribution": f"Uniform(0,{args.jitter_ms} ms)",
            "jitter_max_ms": args.jitter_ms,
            "seed": args.seed,
            "idle_timeout_s": args.idle_timeout,
            "wall_time": time.time(),
        })

        print(json.dumps({
            "status": "RTP_IMPAIRMENT_READY",
            "listen": f"{args.listen_ip}:{args.listen_port}",
            "destination": f"{args.dest_ip}:{args.dest_port}",
            "loss": args.loss,
            "jitter_ms": args.jitter_ms,
            "seed": args.seed,
            "log": str(args.log),
        }, sort_keys=True), flush=True)

        while True:
            now = time.monotonic()

            while queue and queue[0][0] <= now:
                due, _, packet, meta = heapq.heappop(queue)
                sent_at = time.monotonic()
                send_sock.sendto(packet, destination)
                forwarded += 1
                meta.update({
                    "event": "FORWARD",
                    "actual_added_delay_ms": (sent_at - meta["recv_monotonic"]) * 1000.0,
                    "forward_monotonic": sent_at,
                })
                write_jsonl(log, meta)

            if (
                received > 0
                and not queue
                and now - last_receive >= args.idle_timeout
            ):
                break

            timeout = 0.05
            if queue:
                timeout = max(0.0, min(timeout, queue[0][0] - now))

            readable, _, _ = select.select([sock], [], [], timeout)
            if not readable:
                continue

            while True:
                try:
                    packet, source = sock.recvfrom(65535)
                except BlockingIOError:
                    break

                recv_at = time.monotonic()
                last_receive = recv_at
                received += 1
                seq, rtp_timestamp = rtp_fields(packet)

                base = {
                    "packet_index": received,
                    "rtp_seq": seq,
                    "rtp_timestamp": rtp_timestamp,
                    "packet_bytes": len(packet),
                    "source_ip": source[0],
                    "source_port": source[1],
                    "recv_monotonic": recv_at,
                }

                if rng.random() < args.loss:
                    dropped += 1
                    base.update({
                        "event": "DROP",
                        "scheduled_delay_ms": None,
                    })
                    write_jsonl(log, base)
                    continue

                delay_ms = rng.uniform(0.0, args.jitter_ms) if args.jitter_ms else 0.0
                due = recv_at + delay_ms / 1000.0
                base["scheduled_delay_ms"] = delay_ms
                queue_counter += 1
                heapq.heappush(queue, (due, queue_counter, packet, base))

        elapsed = time.monotonic() - started
        actual_loss = dropped / received if received else 0.0
        summary = {
            "status": "RTP_IMPAIRMENT_PASS",
            "received_packets": received,
            "dropped_packets": dropped,
            "forwarded_packets": forwarded,
            "configured_loss_probability": args.loss,
            "actual_loss_fraction": actual_loss,
            "jitter_max_ms": args.jitter_ms,
            "jitter_distribution": f"Uniform(0,{args.jitter_ms} ms)",
            "seed": args.seed,
            "elapsed_s": elapsed,
            "log": str(args.log),
        }
        write_jsonl(log, {"event": "SUMMARY", **summary})
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    sock.close()
    send_sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


