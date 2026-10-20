#!/usr/bin/env python
"""Compatibility entry point for hash-locked reviewer model provisioning."""

from __future__ import annotations

import sys

from reviewer_bootstrap import main


if __name__ == "__main__":
    raise SystemExit(main(["models", *sys.argv[1:]]))
