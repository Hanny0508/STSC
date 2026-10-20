#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from zts_secure_fragment_protocol_v1 import (
    SHARD_CRC_BYTES,
    SHARD_FRAME_BYTES,
    SHARD_INDEX_BYTES,
    SHARD_COUNT,
    decode_shard_frame,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_RECEIVER = ROOT / "tools" / "imdb_receiver_v6_prefix_dev.py"


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_base_receiver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "zts_fragment_receiver_v1_base", BASE_RECEIVER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load base receiver: {BASE_RECEIVER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Payload-independent 40-bit fragment receiver. It reuses the "
            "verified Receiver v6 decision-prefix cache, then validates only "
            "the authenticated expected shard index and CRC16."
        )
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--expected-shard-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--expected-frame-bytes",
        type=int,
        default=SHARD_FRAME_BYTES,
        help="shard frame width in bytes (default: the packaged constant)",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# r3_frame_geometry_applied
def main() -> None:
    args = parse_args()
    if not 0 <= args.expected_shard_index < SHARD_COUNT:
        raise SystemExit(
            f"--expected-shard-index must be in [0, {SHARD_COUNT - 1}]"
        )
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"STOP: refusing to overwrite: {output_path}")

    base = load_base_receiver()

    frame_bytes = getattr(args, "expected_frame_bytes", None) or SHARD_FRAME_BYTES
    data_bytes = frame_bytes - SHARD_INDEX_BYTES - SHARD_CRC_BYTES
    if data_bytes < 1:
        raise SystemExit(f"--expected-frame-bytes too small: {frame_bytes}")

    def validate_fragment(frame: bytes, **_: object) -> bytes:
        return decode_shard_frame(
            frame,
            expected_index=args.expected_shard_index,
            data_bytes=data_bytes,
        ).data

    base.decode_crc32_frame = validate_fragment
    base.parse_args = lambda: argparse.Namespace(
        candidates=args.candidates,
        prompt=args.prompt,
        expected_frame_bytes=frame_bytes,
        output=args.output,
        continue_after_accept=True,
        overwrite=args.overwrite,
    )
    base.main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    accepted = result.get("accepted", [])
    for row in accepted:
        row.pop("crc32_pass", None)
        row["expected_shard_index"] = args.expected_shard_index
        row["decoded_shard_data_hex"] = row.pop("recovered_payload_hex")
        row["shard_crc16_pass"] = True

    distinct = sorted(
        {row["decoded_shard_data_hex"] for row in accepted}
    )
    unambiguous = len(distinct) == 1
    result.update(
        {
            "status": (
                "ZSC_TSD_FRAGMENT_RECEIVER_V1_PASS"
                if unambiguous
                else "ZSC_TSD_FRAGMENT_RECEIVER_V1_FAIL"
            ),
            "protocol": "ZSC-TSD-SECURE-FRAGMENT-v1",
            "development_only": True,
            "full_candidate_search": True,
            "expected_shard_index": args.expected_shard_index,
            "expected_frame_bytes": frame_bytes,
            "expected_frame_bits": frame_bytes * 8,
            "shard_data_bytes": data_bytes,
            "distinct_crc16_valid_shard_data": distinct,
            "unambiguous_shard_data": unambiguous,
            "accepted_candidate_count": len(accepted),
            "base_receiver": {
                "path": str(BASE_RECEIVER),
                "sha256": sha256_file(BASE_RECEIVER),
                "decision_prefix_semantics_reused": True,
            },
            "selection_constraints": {
                "uses_expected_shard_index": True,
                "uses_crc16_after_zts_decode": True,
                "uses_aes_key": False,
                "uses_aes_gcm_tag": False,
                "uses_rs_codeword": False,
                "uses_sender_stegotext": False,
                "uses_sender_token_ids": False,
                "uses_reference_text": False,
            },
        }
    )
    result.pop("distinct_crc32_valid_payloads", None)
    result.pop("unambiguous_payload", None)
    metadata = result.get("receiver_control_metadata", {})
    metadata.pop("expected_payload_bytes", None)
    metadata["expected_shard_index"] = args.expected_shard_index
    isolation = result.get("receiver_isolation", {})
    isolation.pop("crc32_used_only_after_zts_decode", None)
    isolation["crc16_used_only_after_zts_decode"] = True
    isolation["aes_gcm_not_used_for_candidate_selection"] = True
    write_json(output_path, result)

    print("=" * 70)
    print("FRAGMENT RESULT:", result["status"])
    print("expected shard index:", args.expected_shard_index)
    print("distinct valid data:", distinct)
    print("accepted candidates:", len(accepted))
    print("output:", output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
