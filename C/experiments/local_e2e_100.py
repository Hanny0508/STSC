"""Repeat the formal one-carrier local integration runner over a prompt file.

This experiment uses local file transport and must not be reported as VoIP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zsc_tsd.pipeline import run_local_fragment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--frame-hex", required=True)
    parser.add_argument("--expected-shard-index", required=True, type=int)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-base", default="artifacts/new_e2e", type=Path)
    args = parser.parse_args()
    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    outputs = []
    for prompt in prompts[: args.limit]:
        outputs.append(
            str(
                run_local_fragment(
                    frame=bytes.fromhex(args.frame_hex),
                    prompt=prompt,
                    expected_shard_index=args.expected_shard_index,
                    output_base=args.output_base,
                )
            )
        )
    print(json.dumps({"status": "LOCAL_E2E_BATCH_COMPLETE", "runs": len(outputs), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
