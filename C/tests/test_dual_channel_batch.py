from __future__ import annotations

import json
import wave
from pathlib import Path

from zsc_tsd.dual_channel.batch_a import _discover


def _carrier(root: Path, name: str, shard: int, attempt: int) -> None:
    public = root / "public" / name
    private = root / "sender_private" / name
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    with wave.open(str(public / "carrier.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 16)
    (private / "sender.json").write_text(
        json.dumps(
            {
                "carrier_id": name,
                "shard_index": shard,
                "attempt_number": attempt,
                "prompt": "test prompt",
                "shard_data_hex": "0000",
            }
        ),
        encoding="utf-8",
    )


def test_batch_discovery_is_precommitted_and_deterministic(tmp_path: Path) -> None:
    _carrier(tmp_path, "shard_01_attempt_2", 1, 2)
    _carrier(tmp_path, "shard_00_attempt_1", 0, 1)
    _carrier(tmp_path, "shard_01_attempt_1", 1, 1)
    rows = _discover(tmp_path)
    assert [row["carrier_id"] for row in rows] == [
        "shard_00_attempt_1",
        "shard_01_attempt_1",
        "shard_01_attempt_2",
    ]
