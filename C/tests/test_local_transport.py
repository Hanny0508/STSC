from __future__ import annotations

from pathlib import Path

import pytest

from zsc_tsd.transport import LocalFileTransport


def test_local_transport_is_byte_exact_and_truthfully_named(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    received = tmp_path / "received.wav"
    source.write_bytes(b"RIFF-test-audio")
    result = LocalFileTransport().transfer(source, received)
    assert result.byte_exact is True
    assert result.transport == "local-file-copy-not-voip"
    assert received.read_bytes() == source.read_bytes()


def test_local_transport_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    received = tmp_path / "received.wav"
    source.write_bytes(b"one")
    received.write_bytes(b"two")
    with pytest.raises(FileExistsError):
        LocalFileTransport().transfer(source, received)
