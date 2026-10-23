"""Checksummed downloads from the two allowed hosts, with timeouts, retries and a size cap.

Policy (docs/DESIGN.md, section 3.3): only netlib.org and miplib.zib.de, HTTPS, connect and
read timeouts, 3 attempts with backoff, streaming to ``<file>.part`` then an atomic rename,
a running byte total checked against the configured cap before each download, and SHA-256
checksums that must match the committed manifest once recorded.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_HOSTS = frozenset({"netlib.org", "www.netlib.org", "miplib.zib.de"})
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 60.0
ATTEMPTS = 3
USER_AGENT = "lpbench (https://github.com/h-a-forster/lpbench)"


class DownloadError(RuntimeError):
    pass


class ChecksumMismatch(DownloadError):
    pass


class CapExceeded(DownloadError):
    pass


def check_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise DownloadError(f"refusing to download from {url!r} (host or scheme not allowed)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Budget:
    """Running byte total for one fetch, checked against the cap before each download."""

    cap_bytes: int
    used_bytes: int = 0

    def reserve(self, nbytes: int, label: str) -> None:
        if self.used_bytes + nbytes > self.cap_bytes:
            raise CapExceeded(
                f"downloading {label} ({nbytes} bytes) would exceed the cap of "
                f"{self.cap_bytes} bytes ({self.used_bytes} already used)"
            )
        self.used_bytes += nbytes


class _HostCheckingRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        check_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HostCheckingRedirect)


def _open(url: str, method: str = "GET"):  # type: ignore[no-untyped-def]
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    return _OPENER.open(req, timeout=CONNECT_TIMEOUT_S)


def content_length(url: str) -> int | None:
    """Size announced by a HEAD request, or None if the server does not say."""
    check_host(url)
    for attempt in range(ATTEMPTS):
        try:
            with _open(url, "HEAD") as resp:
                value = resp.headers.get("Content-Length")
                return int(value) if value is not None else None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return None  # pragma: no cover


def download(
    url: str,
    dest: Path,
    budget: Budget,
    expected_sha256: str | None = None,
    size_hint: int | None = None,
) -> tuple[str, int, bool]:
    """Download ``url`` to ``dest`` unless an identical file is already there.

    Returns ``(sha256, bytes, downloaded)``. An existing file whose checksum matches
    ``expected_sha256`` is reused without network access. A mismatch is a hard error.
    """
    check_host(url)
    dest = Path(dest)
    if dest.exists():
        digest = sha256_file(dest)
        if expected_sha256 is None or digest == expected_sha256:
            return digest, dest.stat().st_size, False
        dest.unlink()  # stale or corrupted cache entry: fetch again and re-check
    size = content_length(url)
    budget.reserve(size if size is not None else (size_hint or 0), dest.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_exc: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            h = hashlib.sha256()
            n = 0
            with _open(url) as resp, open(part, "wb") as out:
                with contextlib.suppress(AttributeError):  # read timeout on the socket
                    resp.fp.raw._sock.settimeout(READ_TIMEOUT_S)  # type: ignore[union-attr]
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
                    if size is None and budget.used_bytes + n > budget.cap_bytes:
                        raise CapExceeded(f"{dest.name}: download cap exceeded mid-stream")
            if size is not None and n != size:
                raise DownloadError(f"{dest.name}: got {n} bytes, expected {size}")
            digest = h.hexdigest()
            if expected_sha256 is not None and digest != expected_sha256:
                part.unlink(missing_ok=True)
                raise ChecksumMismatch(
                    f"{dest.name}: SHA-256 {digest} does not match the manifest "
                    f"({expected_sha256}); use --update-checksums to accept a changed file"
                )
            part.replace(dest)
            if size is None:
                budget.used_bytes += n
            return digest, n, True
        except (ChecksumMismatch, CapExceeded):
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, DownloadError) as exc:
            last_exc = exc
            part.unlink(missing_ok=True)
            if attempt < ATTEMPTS - 1:
                time.sleep(2.0 * (attempt + 1))
    raise DownloadError(f"{url}: failed after {ATTEMPTS} attempts: {last_exc}")
