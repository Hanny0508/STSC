"""Backward-compatible alias for the reviewer-facing package preflight."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from zsc_tsd.cli import main


raise SystemExit(main(["preflight", *sys.argv[1:]]))
