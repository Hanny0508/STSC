#!/usr/bin/env python
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from native_zero_shot_gls_bootstrap import (
    PROJECT_ROOT,
    bootstrap_native_core,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run an existing sender or receiver unchanged while selecting "
            "the frozen original zero-shot-GLS C++ Huffman backend."
        )
    )
    parser.add_argument("--script", required=True)
    parser.add_argument(
        "--native-manifest",
        default=(
            "artifacts/new_e2e/"
            "native_zero_shot_gls_core_manifest.json"
        ),
    )

    return parser.parse_known_args()


def resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main() -> None:
    args, child_arguments = parse_args()

    native_info = bootstrap_native_core(args.native_manifest)

    script_path = resolve_project_path(args.script)
    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    print("NATIVE ZERO-SHOT GLS RUNNER")
    print("child script:", script_path)
    print("native package:", native_info["package"])
    print(
        "native extension sha256:",
        native_info["huffman_extension_sha256"],
    )
    print("child arguments:", child_arguments)
    print("---")

    sys.argv = [str(script_path), *child_arguments]
    runpy.run_path(str(script_path), run_name="__main__")

    import zgls_utils

    package_after = Path(zgls_utils.__file__).resolve()

    if str(package_after) != native_info["package"]:
        raise RuntimeError(
            "Native zgls_utils package was replaced during execution:\n"
            f"Expected: {native_info['package']}\n"
            f"Actual:   {package_after}"
        )

    print("---")
    print("NATIVE ZERO-SHOT GLS RUNNER COMPLETE")
    print("native package after execution:", package_after)


if __name__ == "__main__":
    main()