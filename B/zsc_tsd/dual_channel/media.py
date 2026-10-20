"""Restricted single-carrier provisioning over the private A/B network."""

from __future__ import annotations

import hashlib
import http.server
import ipaddress
import shutil
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import DualChannelError


@dataclass(frozen=True, slots=True)
class DownloadResult:
    path: Path
    bytes: int
    sha256: str
    source_url: str


class _CarrierHandler(http.server.BaseHTTPRequestHandler):
    server: Any

    def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
        if self.client_address[0] != self.server.expected_client_ip:
            self.send_error(403, "unexpected client")
            return
        if urllib.parse.urlsplit(self.path).path != "/carrier.wav":
            self.send_error(404, "not found")
            return
        source: Path = self.server.carrier_path
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(source.stat().st_size))
        self.send_header("X-Content-SHA256", self.server.carrier_sha256)
        self.end_headers()
        with source.open("rb") as stream:
            shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
        self.server.transfer_completed.set()

    def log_message(self, _format: str, *_args: object) -> None:
        return


class CarrierProvisionServer:
    """Serve exactly one WAV to the expected B Tailscale address."""

    def __init__(
        self,
        *,
        bind_ip: str,
        port: int,
        expected_client_ip: str,
        carrier_path: str | Path,
        carrier_sha256: str,
    ) -> None:
        ipaddress.ip_address(bind_ip)
        ipaddress.ip_address(expected_client_ip)
        if not 0 <= port <= 65535:
            raise ValueError("provisioning port must be in [0, 65535]")
        source = Path(carrier_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        server = http.server.ThreadingHTTPServer((bind_ip, port), _CarrierHandler)
        server.daemon_threads = True
        server.expected_client_ip = expected_client_ip
        server.carrier_path = source
        server.carrier_sha256 = carrier_sha256
        server.transfer_completed = threading.Event()
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="zsc-tsd-carrier-provision",
            daemon=True,
        )
        actual_port = int(server.server_address[1])
        self.url = f"http://{bind_ip}:{actual_port}/carrier.wav"

    def start(self) -> None:
        self._thread.start()

    def wait_for_transfer(self, timeout_seconds: float) -> bool:
        return bool(self._server.transfer_completed.wait(timeout_seconds))

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "CarrierProvisionServer":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def download_carrier(
    url: str,
    output: str | Path,
    *,
    expected_server_ip: str,
    expected_sha256: str,
    expected_bytes: int,
    timeout_seconds: float = 60,
    maximum_bytes: int = 256 * 1024 * 1024,
) -> DownloadResult:
    """Download one carrier and enforce address, size, and hash commitments."""

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != expected_server_ip:
        raise DualChannelError("carrier URL does not match the expected A node")
    ipaddress.ip_address(parsed.hostname)
    if expected_bytes < 44 or expected_bytes > maximum_bytes:
        raise DualChannelError("carrier byte commitment is outside the safe range")
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite: {temporary}")
    digest = hashlib.sha256()
    count = 0
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            header_hash = response.headers.get("X-Content-SHA256", "").lower()
            if header_hash and header_hash != expected_sha256.lower():
                raise DualChannelError("carrier HTTP hash header differs from commitment")
            with temporary.open("xb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > maximum_bytes or count > expected_bytes:
                        raise DualChannelError("carrier exceeded its committed size")
                    digest.update(block)
                    stream.write(block)
        actual_hash = digest.hexdigest()
        if count != expected_bytes:
            raise DualChannelError(
                f"carrier size mismatch: expected {expected_bytes}, received {count}"
            )
        if actual_hash != expected_sha256.lower():
            raise DualChannelError("carrier SHA256 does not match signed commitment")
        temporary.replace(target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return DownloadResult(target.resolve(), count, actual_hash, url)
