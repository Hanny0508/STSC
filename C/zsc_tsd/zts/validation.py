from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .adapter import (
    COMPAT_ROOT,
    LEGACY_SRC,
    PROJECT_ROOT,
    ZTSConfig,
    ZTSModelAdapter,
)

MANIFEST_VERSION = 2

# These files define the model weights, quantization scheme, and tokenizer.
MODEL_MANIFEST_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "quantize_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
IMPLEMENTATION_MANIFEST_FILES = (
    (
        "implementation/legacy/codec.py",
        LEGACY_SRC / "codec.py",
    ),
    (
        "implementation/legacy/hide_extract.py",
        LEGACY_SRC / "hide_extract.py",
    ),
    (
        "implementation/legacy/prompt_gen.py",
        LEGACY_SRC / "prompt_gen.py",
    ),
    (
        "implementation/legacy/search.py",
        LEGACY_SRC / "search.py",
    ),
    (
        "implementation/legacy/strategy.py",
        LEGACY_SRC / "strategy.py",
    ),
    (
        "implementation/legacy/p_utils.py",
        LEGACY_SRC / "p_utils.py",
    ),
    (
        "implementation/compat/zgls_utils/huffman.py",
        COMPAT_ROOT / "zgls_utils" / "huffman.py",
    ),
    (
        "implementation/zsc_tsd/zts/adapter.py",
        PROJECT_ROOT / "zsc_tsd" / "zts" / "adapter.py",
    ),
    (
        "implementation/zsc_tsd/zts/encoder.py",
        PROJECT_ROOT / "zsc_tsd" / "zts" / "encoder.py",
    ),
    (
        "implementation/zsc_tsd/zts/decoder.py",
        PROJECT_ROOT / "zsc_tsd" / "zts" / "decoder.py",
    ),
)


class ZTSConsistencyError(RuntimeError):
    """Raised when sender and receiver ZTS configurations differ."""


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ZTSManifest:
    """
    Reproducibility and sender/receiver consistency manifest.

    The protocol fingerprint excludes local absolute paths, so the same
    checkpoint copied to another machine produces the same fingerprint.
    """

    manifest_version: int
    model_files: tuple[FileFingerprint, ...]
    zts_config: dict[str, Any]
    software_versions: dict[str, str]
    protocol_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "model_files": [
                {
                    "name": item.name,
                    "size": item.size,
                    "sha256": item.sha256,
                }
                for item in self.model_files
            ],
            "zts_config": dict(self.zts_config),
            "software_versions": dict(self.software_versions),
            "protocol_sha256": self.protocol_sha256,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            self.to_json() + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ZTSManifest":
        files = tuple(
            FileFingerprint(
                name=str(item["name"]),
                size=int(item["size"]),
                sha256=str(item["sha256"]),
            )
            for item in value["model_files"]
        )

        return cls(
            manifest_version=int(value["manifest_version"]),
            model_files=files,
            zts_config=dict(value["zts_config"]),
            software_versions=dict(value["software_versions"]),
            protocol_sha256=str(value["protocol_sha256"]),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "ZTSManifest":
        value = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        if not isinstance(value, dict):
            raise ValueError("ZTS manifest root must be a JSON object")

        return cls.from_dict(value)


def _sha256_file(
    path: Path,
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _protocol_payload(manifest: "ZTSManifest") -> dict[str, Any]:
    """The exact value that ``protocol_sha256`` is derived from."""

    return {
        "manifest_version": manifest.manifest_version,
        "model_files": [
            {
                "name": item.name,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in manifest.model_files
        ],
        "zts_config": manifest.zts_config,
        "software_versions": manifest.software_versions,
    }


def _recompute_protocol_sha256(manifest: "ZTSManifest") -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_protocol_payload(manifest))
    ).hexdigest()

# r3_manifest_compare_applied


def create_zts_manifest(
    adapter: ZTSModelAdapter,
) -> ZTSManifest:
    """
    Hash the local checkpoint, tokenizer, and shared ZTS parameters.

    model.safetensors is approximately 4.27 GB, so this function may take
    several seconds. It does not load the model into GPU memory.
    """

    adapter.validate_files()

    file_fingerprints: list[FileFingerprint] = []

    for filename in MODEL_MANIFEST_FILES:
        path = adapter.model_dir / filename

        # generation_config.json and tokenizer.json are useful but some
        # compatible checkpoints may not contain them.
        if not path.is_file():
            continue

        file_fingerprints.append(
            FileFingerprint(
                name=filename,
                size=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )

    for logical_name, path in IMPLEMENTATION_MANIFEST_FILES:
        if not path.is_file():
            raise FileNotFoundError(
                "Required ZTS implementation file is missing: "
                f"{path}"
            )

        file_fingerprints.append(
            FileFingerprint(
                name=logical_name,
                size=path.stat().st_size,
                sha256=_sha256_file(path),
            )
        )

    zts_config = asdict(adapter.config)

    software_versions = {
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "gptqmodel": _package_version("GPTQModel"),
        "accelerate": _package_version("accelerate"),
        "safetensors": _package_version("safetensors"),
        "bitstring": _package_version("bitstring"),
        "tokenicer": _package_version("tokenicer"),
        "numpy": _package_version("numpy"),
    }

    # The fingerprint includes the checkpoint, tokenizer, implementation
    # sources, shared parameters, and critical software versions.
    # This allows equivalent sender/receiver installations in different
    # local directories while still enforcing identical token decisions.
    protocol_value = {
        "manifest_version": MANIFEST_VERSION,
        "model_files": [
            {
                "name": item.name,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in file_fingerprints
        ],
        "zts_config": zts_config,
        "software_versions": software_versions,
    }

    protocol_sha256 = hashlib.sha256(
        _canonical_json_bytes(protocol_value)
    ).hexdigest()

    return ZTSManifest(
        manifest_version=MANIFEST_VERSION,
        model_files=tuple(file_fingerprints),
        zts_config=zts_config,
        software_versions=software_versions,
        protocol_sha256=protocol_sha256,
    )


def compare_zts_manifests(
    local: ZTSManifest,
    remote: ZTSManifest,
) -> list[str]:
    """Return human-readable sender/receiver consistency differences."""

    differences: list[str] = []

    if local.manifest_version != remote.manifest_version:
        differences.append(
            "manifest version differs: "
            f"local={local.manifest_version}, "
            f"remote={remote.manifest_version}"
        )

    if local.zts_config != remote.zts_config:
        differences.append(
            "ZTS generation parameters differ"
        )

    local_files = {
        item.name: item
        for item in local.model_files
    }
    remote_files = {
        item.name: item
        for item in remote.model_files
    }

    for filename in sorted(local_files.keys() | remote_files.keys()):
        local_file = local_files.get(filename)
        remote_file = remote_files.get(filename)

        if local_file is None:
            differences.append(
                f"model file exists only remotely: {filename}"
            )
            continue

        if remote_file is None:
            differences.append(
                f"model file exists only locally: {filename}"
            )
            continue

        if local_file.size != remote_file.size:
            differences.append(
                f"model file size differs for {filename}: "
                f"local={local_file.size}, "
                f"remote={remote_file.size}"
            )

        if local_file.sha256 != remote_file.sha256:
            differences.append(
                f"model file SHA-256 differs for {filename}"
            )

    if local.software_versions != remote.software_versions:
        differing = sorted(
            key
            for key in set(local.software_versions) | set(remote.software_versions)
            if local.software_versions.get(key) != remote.software_versions.get(key)
        )
        detail = ", ".join(
            f"{key} (local={local.software_versions.get(key)}, "
            f"remote={remote.software_versions.get(key)})"
            for key in differing
        )
        differences.append(f"software versions differ: {detail}")

    # Recompute each fingerprint from the manifest contents instead of trusting
    # the stored value, so an edited manifest cannot keep an old fingerprint.
    for label, manifest in (("local", local), ("remote", remote)):
        recomputed = _recompute_protocol_sha256(manifest)
        if recomputed != manifest.protocol_sha256:
            differences.append(
                f"{label} manifest fingerprint does not match its own contents: "
                f"stored={manifest.protocol_sha256}, recomputed={recomputed}"
            )

    if local.protocol_sha256 != remote.protocol_sha256:
        differences.append(
            "protocol fingerprint differs: "
            f"local={local.protocol_sha256}, "
            f"remote={remote.protocol_sha256}"
        )

    return differences


def assert_zts_manifests_match(
    local: ZTSManifest,
    remote: ZTSManifest,
) -> None:
    differences = compare_zts_manifests(local, remote)

    if differences:
        raise ZTSConsistencyError(
            "Sender and receiver ZTS manifests do not match:\n- "
            + "\n- ".join(differences)
        )


def create_default_manifest() -> ZTSManifest:
    """Create a manifest using the default paper-compatible setup."""

    return create_zts_manifest(
        ZTSModelAdapter(config=ZTSConfig())
    )