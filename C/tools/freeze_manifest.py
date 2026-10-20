"""Create a SHA-256 provenance manifest for explicit input files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zsc_tsd.manifest import build_manifest, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    value = build_manifest(
        experiment=args.experiment,
        files=args.files,
        root=PROJECT_ROOT,
    )
    write_json_new(args.output, value)
    print(json.dumps({"status": "MANIFEST_CREATED", "output": str(args.output), "manifest_sha256": value["manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
