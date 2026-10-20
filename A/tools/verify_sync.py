#!/usr/bin/env python
"""Synchronization gate: compare the local ZTS manifest with a frozen one.

This is the check that the reviewer asked for and that was previously never
invoked in the main recovery path: the sender and receiver must agree on the
model checkpoint, the tokenizer, the implementation sources, the generation
parameters and the runtime versions before a session is attempted.

Usage
-----
Create a frozen manifest for the current tree:

    python tools/verify_sync.py --root . --create --out artifacts/freezes/r3/zts_manifest_r3.json

Verify the current tree against a frozen manifest:

    python tools/verify_sync.py --root . --manifest artifacts/freezes/r3/zts_manifest_r3.json

Exit status is 0 when the manifests match and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DEFAULT = Path(__file__).resolve().parents[1]


def build_local_manifest(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from zsc_tsd.zts.validation import create_zts_manifest
    from zsc_tsd.zts.adapter import ZTSModelAdapter

    return create_zts_manifest(ZTSModelAdapter())


def load_manifest(path: Path):
    if str(ROOT_DEFAULT) not in sys.path:
        sys.path.insert(0, str(ROOT_DEFAULT))
    from zsc_tsd.zts.validation import ZTSManifest

    return ZTSManifest.read_json(path)


def compare(local, remote):
    from zsc_tsd.zts.validation import compare_zts_manifests

    return compare_zts_manifests(local, remote)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()

    local = build_local_manifest(root)

    if args.create:
        out = args.out
        if out is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            out = Path(f"artifacts/freezes/r3_{stamp}/zts_manifest_r3.json")
        out = out if out.is_absolute() else (root / out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            print(f"refusing to overwrite existing manifest: {out}")
            return 1
        local.write_json(out)
        print(
            json.dumps(
                {
                    "status": "ZTS_MANIFEST_CREATED",
                    "path": str(out),
                    "manifest_version": local.manifest_version,
                    "protocol_sha256": local.protocol_sha256,
                    "files": len(local.model_files),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.manifest is None:
        parser.error("provide --manifest to verify, or --create to freeze")

    manifest_path = (
        args.manifest if args.manifest.is_absolute() else (root / args.manifest)
    )
    remote = load_manifest(manifest_path)
    differences = compare(local, remote)

    result = {
        "status": (
            "SYNC_GATE_PASS" if not differences else "SYNC_GATE_FAIL"
        ),
        "manifest": str(manifest_path),
        "local_protocol_sha256": local.protocol_sha256,
        "frozen_protocol_sha256": remote.protocol_sha256,
        "differences": differences,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not differences else 1


if __name__ == "__main__":
    raise SystemExit(main())
