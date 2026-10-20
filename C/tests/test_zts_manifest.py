from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from zsc_tsd.zts import (
    ZTSConsistencyError,
    ZTSManifest,
    assert_zts_manifests_match,
    compare_zts_manifests,
)


MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "new_e2e"
    / "zts_manifest_rtx5090.json"
)


def load_manifest() -> ZTSManifest:
    if not MANIFEST_PATH.is_file():
        pytest.skip(f"frozen manifest not installed: {MANIFEST_PATH}")
    return ZTSManifest.read_json(MANIFEST_PATH)


def test_manifest_matches_itself() -> None:
    manifest = load_manifest()

    assert_zts_manifests_match(manifest, manifest)
    assert compare_zts_manifests(manifest, manifest) == []


def test_manifest_rejects_decoder_parameter_mismatch() -> None:
    local = load_manifest()
    remote_value = deepcopy(local.to_dict())

    # Reproduce the historical encoder/decoder mismatch:
    # sender alpha=1.25, receiver alpha=1.0.
    remote_value["zts_config"]["temperature_alpha"] = 1.0

    # A different configuration must also have a different advertised
    # protocol fingerprint.
    remote_value["protocol_sha256"] = "0" * 64

    remote = ZTSManifest.from_dict(remote_value)
    differences = compare_zts_manifests(local, remote)

    assert any(
        "generation parameters differ" in item
        for item in differences
    )
    assert any(
        "protocol fingerprint differs" in item
        for item in differences
    )

    with pytest.raises(
        ZTSConsistencyError,
        match="manifests do not match",
    ):
        assert_zts_manifests_match(local, remote)
