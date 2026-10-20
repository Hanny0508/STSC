#!/usr/bin/env python
"""Make the batch runner authoritative for its own import tree.

The environment carries an editable install of ``zsc-tsd`` whose PEP 660
module map was generated when it was installed.  Modules added to the
checkout afterwards are invisible through that map, so a run can silently
depend on a different (installed) copy of the package than the one it was
launched from -- a reproducibility hazard in its own right.

This edit inserts the checkout's root and its ``tools`` directory at the
front of ``sys.path`` before any local import, so the process always uses the
tree it was started from.

Exact-text replacement with a one-occurrence assertion; aborts without
writing if the expectation is not met.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

RUNNER = Path("tools") / "zts_payload_capacity_runner.py"

OLD = (
    "from typing import Any\n"
    "\n"
    "from zsc_tsd.zts.adapter import ZTSModelAdapter\n"
)

NEW = (
    "from typing import Any\n"
    "\n"
    "# Use this checkout as the authoritative source tree.  An editable\n"
    "# install of zsc_tsd caches a fixed module map, so without this the\n"
    "# process could import a different copy of the package than the one it\n"
    "# was launched from.\n"
    "for _search_path in (\n"
    "    str(Path(__file__).resolve().parents[1]),\n"
    "    str(Path(__file__).resolve().parent),\n"
    "):\n"
    "    if _search_path not in sys.path:\n"
    "        sys.path.insert(0, _search_path)\n"
    "\n"
    "from zsc_tsd.zts.adapter import ZTSModelAdapter\n"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    path = root / RUNNER
    if not path.is_file():
        print(f"missing target: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    count = text.count(OLD)
    if count != 1:
        if NEW in text:
            print("already applied")
            return 0
        print(
            f"ABORTED - no files were modified: expected 1 import prologue, "
            f"found {count}"
        )
        return 1

    updated = text.replace(OLD, NEW, 1)

    if args.dry_run:
        print("dry run OK: sys.path bootstrap would be inserted")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = root.parent / f"_r3_path_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backup = backup_dir / RUNNER.name
    shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8", newline="")

    report = {
        "status": "R3_PATH_BOOTSTRAP_APPLIED",
        "applied_utc": stamp,
        "file": str(RUNNER),
        "backup": str(backup),
    }
    (backup_dir / "path_bootstrap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
