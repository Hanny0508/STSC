"""Assemble the text-domain detectability (D3) dataset from archived runs.

Every archived carrier already provides:
  * sender_private/<carrier>/sender.json  -> the stego text and the row used
  * receiver/<carrier>/asr.json           -> the Whisper transcript of the audio
  * public/<carrier>/carrier.wav          -> the audio actually transmitted

so only the cover side needs new synthesis.  This script inventories what is
available and reports the effective bits-per-word distribution, which is the
natural "embedding rate" axis for the table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WORD = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", default="artifacts/new_e2e/secure_imdb_batch_v1_8_*")
    parser.add_argument("--out", type=Path, default=Path("artifacts/d3/dataset_inventory.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    run_dirs = sorted(p for p in root.glob(args.pattern) if p.is_dir())
    print(f"run directories: {len(run_dirs)}")

    records: list[dict] = []
    skipped: dict[str, int] = {}

    for run in run_dirs:
        commitment = read_json(run / "public" / "batch_commitment.json") or {}
        frame_bytes = commitment.get("shard_frame_bytes")
        payload_bits = int(frame_bytes) * 8 if frame_bytes else None

        sender_root = run / "sender_private"
        if not sender_root.is_dir():
            skipped["no sender_private"] = skipped.get("no sender_private", 0) + 1
            continue

        for carrier in sorted(p for p in sender_root.iterdir() if p.is_dir()):
            sender = read_json(carrier / "sender.json")
            if not sender:
                skipped["no sender.json"] = skipped.get("no sender.json", 0) + 1
                continue
            stegotext = sender.get("stegotext")
            if not isinstance(stegotext, str) or not stegotext.strip():
                skipped["no stegotext"] = skipped.get("no stegotext", 0) + 1
                continue

            asr = read_json(run / "receiver" / carrier.name / "asr.json")
            transcript = None
            if isinstance(asr, dict):
                transcript = (asr.get("transcription") or {}).get("text")

            words = len(WORD.findall(stegotext))
            bpw = (payload_bits / words) if (payload_bits and words) else None

            records.append(
                {
                    "run": run.name,
                    "carrier": carrier.name,
                    "imdb_row_index": sender.get("imdb_row_index"),
                    "payload_bits": payload_bits,
                    "stegotext_words": words,
                    "bpw": None if bpw is None else round(bpw, 3),
                    "stegotext": stegotext,
                    "transcript": transcript,
                    "wav": str(run / "public" / carrier.name / "carrier.wav"),
                    "shard_index": sender.get("shard_index"),
                }
            )

    with_transcript = sum(1 for r in records if r["transcript"])
    print(f"carriers with stegotext      : {len(records)}")
    print(f"carriers with ASR transcript : {with_transcript}")
    if skipped:
        print(f"skipped                      : {skipped}")

    if records:
        bpws = [r["bpw"] for r in records if r["bpw"]]
        bpws_sorted = sorted(bpws)
        n = len(bpws_sorted)
        print("\nBPW distribution of the archived carriers:")
        print(f"  min {bpws_sorted[0]:.2f}   p25 {bpws_sorted[n//4]:.2f}   "
              f"median {bpws_sorted[n//2]:.2f}   p75 {bpws_sorted[3*n//4]:.2f}   max {bpws_sorted[-1]:.2f}")

        by_payload: dict[int, list[float]] = {}
        for r in records:
            if r["bpw"]:
                by_payload.setdefault(r["payload_bits"], []).append(r["bpw"])
        print("\n  by fragment payload size:")
        for bits in sorted(by_payload):
            vals = sorted(by_payload[bits])
            print(f"    {bits:>2} bit  n={len(vals):4d}  median BPW = {vals[len(vals)//2]:.2f}")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
