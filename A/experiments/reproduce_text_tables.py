"""Aggregate immutable final-result JSON files into a reviewer-readable CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "source",
    "status",
    "runner_revision",
    "total_carriers",
    "accepted_carriers",
    "carrier_acceptance_rate",
    "unique_shards_recovered",
    "rs_reconstruction_pass",
    "aes_gcm_authentication_pass",
    "plaintext_exact_match",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    rows = []
    for path in sorted(args.artifacts.rglob("final_result_after_reveal.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        row = {field: value.get(field) for field in FIELDS}
        row["source"] = str(path)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "RESULT_TABLE_CREATED", "rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
