#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zsc_tsd.control import create_signed_event, verify_signed_event
from zsc_tsd.control.postgres import PostgresControlStore


MASTER_KEY_ENV = "ZSC_TSD_CONTROL_MASTER_KEY_B64"
DSN_ENV = "ZSC_TSD_CONTROL_DSN"


def read_master_key() -> bytes:
    raw = os.environ.get(MASTER_KEY_ENV)
    if not raw:
        raise SystemExit(f"missing environment variable: {MASTER_KEY_ENV}")
    try:
        key = base64.b64decode(raw, validate=True)
    except ValueError as exc:
        raise SystemExit(f"{MASTER_KEY_ENV} is not valid Base64") from exc
    if len(key) < 32:
        raise SystemExit(f"{MASTER_KEY_ENV} must decode to at least 32 bytes")
    return key


def read_dsn() -> str:
    value = os.environ.get(DSN_ENV)
    if not value:
        raise SystemExit(f"missing environment variable: {DSN_ENV}")
    return value


def read_json_object(value: str) -> dict[str, Any]:
    if value.startswith("@"):
        source = Path(value[1:])
        parsed = json.loads(source.read_text(encoding="utf-8"))
    else:
        parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise SystemExit("event body must be a JSON object")
    return parsed


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HMAC-authenticated PostgreSQL control channel for ZSC-TSD."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("new-session")
    init_db = subparsers.add_parser("init-db")
    init_db.add_argument("--schema", type=Path)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--session-id", required=True)
    publish.add_argument("--actor-id", required=True)
    publish.add_argument("--sequence-no", required=True, type=int)
    publish.add_argument("--event-type", required=True)
    publish.add_argument("--correlation-id", required=True)
    publish.add_argument("--body", required=True, help="JSON object or @path.json")
    publish.add_argument("--ttl-seconds", type=int, default=600)
    publish.add_argument("--output", type=Path)

    verify = subparsers.add_parser("verify-file")
    verify.add_argument("path", type=Path)
    verify.add_argument("--ignore-time", action="store_true")

    get_event = subparsers.add_parser("get")
    get_event.add_argument("--event-id", required=True)

    list_events = subparsers.add_parser("list")
    list_events.add_argument("--session-id", required=True)
    list_events.add_argument("--event-type")
    list_events.add_argument("--correlation-id")

    wait = subparsers.add_parser("wait")
    wait.add_argument("--session-id", required=True)
    wait.add_argument("--event-type", required=True)
    wait.add_argument("--correlation-id", required=True)
    wait.add_argument("--timeout-seconds", type=float, default=600)
    wait.add_argument("--poll-seconds", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "new-session":
        print(str(uuid.uuid4()))
        return

    master_key = read_master_key()
    if args.command == "verify-file":
        event = json.loads(args.path.read_text(encoding="utf-8"))
        print_json(
            verify_signed_event(master_key, event, check_time=not args.ignore_time)
        )
        return

    store = PostgresControlStore(read_dsn(), master_key)
    if args.command == "init-db":
        store.initialize(args.schema)
        print("ZSC-TSD CONTROL DATABASE INITIALIZED")
        return
    if args.command == "publish":
        event = create_signed_event(
            master_key,
            session_id=args.session_id,
            actor_id=args.actor_id,
            sequence_no=args.sequence_no,
            event_type=args.event_type,
            correlation_id=args.correlation_id,
            body=read_json_object(args.body),
            ttl_seconds=args.ttl_seconds,
        )
        stored = store.append(event)
        if args.output is not None:
            if args.output.exists():
                raise SystemExit(f"refusing to overwrite: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(stored, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print_json(stored)
        return
    if args.command == "get":
        event = store.get(args.event_id)
        if event is None:
            raise SystemExit(3)
        print_json(event)
        return
    if args.command == "list":
        print_json(
            store.list_events(
                session_id=args.session_id,
                event_type=args.event_type,
                correlation_id=args.correlation_id,
            )
        )
        return
    if args.command == "wait":
        event = store.wait_for_event(
            session_id=args.session_id,
            event_type=args.event_type,
            correlation_id=args.correlation_id,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        if event is None:
            print("CONTROL EVENT WAIT TIMEOUT", file=sys.stderr)
            raise SystemExit(4)
        print_json(event)
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
