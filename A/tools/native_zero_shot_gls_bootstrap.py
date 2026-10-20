#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "new_e2e"
    / "native_zero_shot_gls_core_manifest.json"
)

EXPECTED_MANIFEST_SHA256 = (
    "4cce7d16a936c74cab1d5cc2f53d8473dd31363f84e1dba7f0911c2aef7a25ea"
)

EXPECTED_STATUS = "NATIVE_ZERO_SHOT_GLS_CORE_FROZEN"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def normalized_relative_name(raw: str) -> str:
    return raw.replace("\\", "/").lower()


def bootstrap_native_core(
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """
    Select and verify the original zero-shot-GLS C++ Huffman backend.

    This function must run before codec.py or hide_extract.py is imported.
    It does not load the Llama model.
    """

    manifest_path = resolve_project_path(manifest_path)

    require(
        manifest_path.is_file(),
        f"Native core manifest is missing: {manifest_path}",
    )

    manifest_sha256 = sha256_file(manifest_path)
    require(
        manifest_sha256 == EXPECTED_MANIFEST_SHA256,
        (
            "Native core manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Actual:   {manifest_sha256}"
        ),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    require(
        manifest.get("status") == EXPECTED_STATUS,
        (
            "Native core manifest has invalid status: "
            f"{manifest.get('status')!r}"
        ),
    )

    manifest_files = manifest.get("files")
    require(
        isinstance(manifest_files, dict) and manifest_files,
        "Native core manifest has no files section.",
    )

    resolved_files: dict[str, Path] = {}

    for relative_name, metadata in manifest_files.items():
        require(
            isinstance(relative_name, str),
            "Manifest contains an invalid file name.",
        )
        require(
            isinstance(metadata, dict),
            f"Invalid manifest metadata: {relative_name}",
        )

        path = resolve_project_path(relative_name)
        require(path.is_file(), f"Native core file missing: {path}")

        actual_bytes = path.stat().st_size
        expected_bytes = int(metadata["bytes"])

        require(
            actual_bytes == expected_bytes,
            (
                f"Native core file size mismatch: {path}\n"
                f"Expected: {expected_bytes}\n"
                f"Actual:   {actual_bytes}"
            ),
        )

        actual_sha256 = sha256_file(path)
        expected_sha256 = str(metadata["sha256"])

        require(
            actual_sha256 == expected_sha256,
            (
                f"Native core SHA256 mismatch: {path}\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual_sha256}"
            ),
        )

        resolved_files[relative_name] = path

    huffman_wrapper: Path | None = None
    native_extensions: list[Path] = []

    for relative_name, path in resolved_files.items():
        normalized = normalized_relative_name(relative_name)

        if normalized.endswith("/zgls_utils/huffman.py"):
            huffman_wrapper = path

        if normalized.endswith(".pyd"):
            native_extensions.append(path)

    require(
        huffman_wrapper is not None,
        "Manifest does not contain the original Huffman wrapper.",
    )
    require(
        len(native_extensions) == 1,
        (
            "Expected exactly one native Huffman extension; found "
            f"{len(native_extensions)}."
        ),
    )

    native_extension = native_extensions[0]
    package_directory = huffman_wrapper.parent
    build_root = package_directory.parent

    require(
        native_extension.parent == package_directory,
        "Huffman wrapper and native extension are in different packages.",
    )

    # codec.py and hide_extract.py keep references to the zgls_utils module
    # imported at their own import time. Therefore bootstrap must happen first.
    already_loaded_core = [
        name
        for name in ("codec", "hide_extract")
        if name in sys.modules
    ]
    require(
        not already_loaded_core,
        (
            "Native bootstrap was called too late. Already loaded: "
            f"{already_loaded_core}"
        ),
    )

    for module_name in list(sys.modules):
        if (
            module_name == "zgls_utils"
            or module_name.startswith("zgls_utils.")
        ):
            del sys.modules[module_name]

    build_root_string = str(build_root)

    sys.path[:] = [
        item
        for item in sys.path
        if str(Path(item or ".").resolve()) != str(build_root)
    ]
    sys.path.insert(0, build_root_string)

    importlib.invalidate_caches()

    package = importlib.import_module("zgls_utils")
    wrapper = importlib.import_module("zgls_utils.huffman")
    extension = importlib.import_module("zgls_utils._huffman")

    package_path = Path(package.__file__).resolve()
    wrapper_path = Path(wrapper.__file__).resolve()
    extension_path = Path(extension.__file__).resolve()

    require(
        package_path.parent == package_directory,
        f"Wrong zgls_utils package loaded: {package_path}",
    )
    require(
        wrapper_path == huffman_wrapper,
        f"Wrong Huffman wrapper loaded: {wrapper_path}",
    )
    require(
        extension_path == native_extension,
        f"Wrong native Huffman extension loaded: {extension_path}",
    )

    result = {
        "status": "NATIVE_ZERO_SHOT_GLS_BOOTSTRAP_PASS",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "build_root": str(build_root),
        "package": str(package_path),
        "huffman_wrapper": str(wrapper_path),
        "huffman_wrapper_sha256": sha256_file(wrapper_path),
        "huffman_extension": str(extension_path),
        "huffman_extension_sha256": sha256_file(extension_path),
        "verified_file_count": len(resolved_files),
    }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and activate the frozen native zero-shot-GLS core."
        )
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = bootstrap_native_core(args.manifest)

    print(result["status"])
    print("manifest:", result["manifest"])
    print("manifest sha256:", result["manifest_sha256"])
    print("package:", result["package"])
    print("Huffman wrapper:", result["huffman_wrapper"])
    print("Huffman wrapper sha256:", result["huffman_wrapper_sha256"])
    print("Huffman extension:", result["huffman_extension"])
    print("Huffman extension sha256:", result["huffman_extension_sha256"])
    print("verified files:", result["verified_file_count"])
    print("model loaded:", False)


if __name__ == "__main__":
    main()