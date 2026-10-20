"""Bits-per-word of the archived carriers under two tokenisation conventions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median

WORD_REGEX = re.compile(r"[A-Za-z]+(?:['\u2019-][A-Za-z]+)*")
# Closer to nltk.word_tokenize: contractions split, punctuation kept as tokens.
NLTK_LIKE = re.compile(r"[A-Za-z]+(?:n['\u2019]t|['\u2019](?:s|re|ve|ll|d|m))?|[^\w\s]")


def stats(values: list[float]) -> dict:
    values = sorted(values)
    n = len(values)
    return {
        "n": n,
        "min": round(values[0], 3),
        "p25": round(values[n // 4], 3),
        "median": round(median(values), 3),
        "p75": round(values[3 * n // 4], 3),
        "max": round(values[-1], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path,
                        default=Path("artifacts/d3/dataset_inventory.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    records = json.loads(
        (root / args.inventory).read_text(encoding="utf-8")
    )["records"]

    plain: list[float] = []
    nltkish: list[float] = []
    for record in records:
        bits = record.get("payload_bits")
        text = record.get("stegotext")
        if not bits or not isinstance(text, str):
            continue
        w1 = len(WORD_REGEX.findall(text))
        w2 = len(NLTK_LIKE.findall(text))
        if w1:
            plain.append(bits / w1)
        if w2:
            nltkish.append(bits / w2)

    print("current operating point: threshold=0.002  max_bpw=5  egs_mode=huffman")
    print()
    print("bits-per-word, alphanumeric-word tokenisation :", json.dumps(stats(plain)))
    print("bits-per-word, nltk-like tokenisation         :", json.dumps(stats(nltkish)))
    print()
    print("for reference, the manuscript reports ZTS BPW = 1.89 / 2.37 / 2.84 (Table II)")
    print("and ZTS-LLaMA               BPW = 1.906 / 2.420 / 3.376 (Table IX)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
