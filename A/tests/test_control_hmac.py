from __future__ import annotations

import copy
import unittest
import uuid
from datetime import datetime, timedelta, timezone

from zsc_tsd.control import (
    ControlEventError,
    ControlEventExpired,
    ControlEventSignatureError,
    canonical_json_bytes,
    create_signed_event,
    verify_signed_event,
)


MASTER_KEY = bytes(range(32))
OTHER_KEY = bytes(reversed(range(32)))
SESSION_ID = "d3b07384-d9a0-4f9c-9c7b-64a3dd942b41"
NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


class ControlHMACTests(unittest.TestCase):
    def make_event(self, **changes):
        arguments = {
            "session_id": SESSION_ID,
            "actor_id": "zsc-voip-a",
            "sequence_no": 1,
            "event_type": "CARRIER_PREPARED",
            "correlation_id": "shard_00_attempt_1",
            "body": {
                "shard_index": 0,
                "attempt_number": 1,
                "prompt_sha256": "a" * 64,
                "rtp": {"payload_type": 111, "bitrate_bps": 24000},
            },
            "ttl_seconds": 600,
            "now": NOW,
            "event_id": "5409d8f8-e04b-43cb-a05a-9b8ab875ced3",
            "event_nonce": "00" * 16,
        }
        arguments.update(changes)
        return create_signed_event(MASTER_KEY, **arguments)

    def test_canonical_json_is_stable(self):
        left = {"b": 2, "a": [True, None, "文本"]}
        right = {"a": [True, None, "文本"], "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))

    def test_roundtrip(self):
        event = self.make_event()
        verified = verify_signed_event(MASTER_KEY, event, now=NOW)
        self.assertEqual(verified, event)
        self.assertEqual(len(event["hmac_sha256"]), 64)

    def test_body_tamper_is_rejected(self):
        event = copy.deepcopy(self.make_event())
        event["body"]["shard_index"] = 1
        with self.assertRaises(ControlEventSignatureError):
            verify_signed_event(MASTER_KEY, event, now=NOW)

    def test_wrong_key_is_rejected(self):
        with self.assertRaises(ControlEventSignatureError):
            verify_signed_event(OTHER_KEY, self.make_event(), now=NOW)

    def test_actor_tamper_is_rejected(self):
        event = self.make_event()
        event["actor_id"] = "zsc-voip-b"
        with self.assertRaises(ControlEventSignatureError):
            verify_signed_event(MASTER_KEY, event, now=NOW)

    def test_expired_event_is_rejected(self):
        with self.assertRaises(ControlEventExpired):
            verify_signed_event(
                MASTER_KEY, self.make_event(ttl_seconds=1), now=NOW + timedelta(seconds=2)
            )

    def test_future_event_is_rejected(self):
        with self.assertRaises(ControlEventExpired):
            verify_signed_event(
                MASTER_KEY,
                self.make_event(now=NOW + timedelta(minutes=10)),
                now=NOW,
            )

    def test_float_in_signed_body_is_rejected(self):
        with self.assertRaises(ControlEventError):
            self.make_event(body={"latency_seconds": 0.25})

    def test_unsigned_extra_field_is_rejected(self):
        event = self.make_event()
        event["not_signed"] = "unsafe"
        with self.assertRaises(ControlEventError):
            verify_signed_event(MASTER_KEY, event, now=NOW)


if __name__ == "__main__":
    unittest.main()
