"""Explicit, snapshot-recorded downloads from official benchmark hosts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass(frozen=True, slots=True)
class DownloadedFile:
    name: str
    url: str
    sha256: str
    size: int


_SNAPSHOTS = {
    "miplib": {
        "version": "MIPLIB-2017-benchmark-v2/solu-v36",
        "files": (
            ("benchmark.zip", "https://miplib.zib.de/downloads/benchmark.zip"),
            ("benchmark-v2.test", "https://miplib.zib.de/downloads/benchmark-v2.test"),
            ("miplib2017-v36.solu", "https://miplib.zib.de/downloads/miplib2017-v36.solu"),
        ),
    },
    "qplib": {
        "version": "QPLIB-public-snapshot",
        "files": (
            ("qplib.zip", "https://qplib.zib.de/qplib.zip"),
            ("qplib.solu", "https://qplib.zib.de/qplib.solu"),
        ),
    },
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _https_origin(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Benchmark URL contains an invalid port.") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Benchmark downloads require a credential-free HTTPS origin.")
    return parsed.hostname.lower(), 443 if port is None else port


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects that could leave the audited benchmark origin."""

    def __init__(self, origin: tuple[str, int]) -> None:
        super().__init__()
        self._origin = origin

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _https_origin(newurl) != self._origin:
            raise urllib.error.HTTPError(
                newurl,
                code,
                "Cross-origin benchmark redirect refused.",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url: str, destination: Path, *, overwrite: bool) -> DownloadedFile:
    origin = _https_origin(url)
    if destination.exists() and not overwrite:
        return DownloadedFile(destination.name, url, _hash(destination), destination.stat().st_size)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    request = urllib.request.Request(url, headers={"User-Agent": "qqa-benchmark-fetch/1"})
    opener = urllib.request.build_opener(_SameOriginRedirectHandler(origin))
    try:
        with (
            os.fdopen(descriptor, "wb") as output,
            opener.open(request, timeout=60) as response,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    except OSError as exc:
        public_host = urllib.parse.urlsplit(url).hostname or "public benchmark host"
        raise RuntimeError(
            f"Could not download public benchmark file {destination.name!r} from {public_host}."
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return DownloadedFile(destination.name, url, _hash(destination), destination.stat().st_size)


def _safe_extract(archive: Path, destination: Path) -> list[str]:
    extracted: list[str] = []
    root = destination.resolve()
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            posix_name = PurePosixPath(member.filename)
            windows_name = PureWindowsPath(member.filename)
            if (
                not member.filename
                or member.filename in seen
                or "\\" in member.filename
                or posix_name.is_absolute()
                or windows_name.is_absolute()
                or bool(windows_name.drive)
                or ".." in posix_name.parts
                or ".." in windows_name.parts
            ):
                raise ValueError(f"Unsafe or duplicate archive member {member.filename!r}.")
            seen.add(member.filename)
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member {member.filename!r}.")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(member.filename)
    return extracted


def fetch_benchmark(
    library: str,
    output: str | Path,
    *,
    extract: bool = True,
    overwrite: bool = False,
) -> dict:
    """Download an official MIPLIB/QPLIB snapshot and write provenance JSON."""
    key = library.lower()
    if key not in _SNAPSHOTS:
        raise ValueError("library must be 'miplib' or 'qplib'.")
    destination = Path(output).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = _SNAPSHOTS[key]
    files = [
        _download(url, destination / name, overwrite=overwrite) for name, url in snapshot["files"]
    ]
    extracted: list[str] = []
    if extract:
        for downloaded in files:
            path = destination / downloaded.name
            if path.suffix.lower() == ".zip":
                extracted.extend(_safe_extract(path, destination / "instances"))
    metadata = {
        "library": key,
        "snapshot": snapshot["version"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "files": [asdict(file) for file in files],
        "extracted_files": extracted,
    }
    (destination / "snapshot.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def fetch_instance(
    library: str,
    instance: str,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Download one named public instance without fetching a full archive."""
    key = library.lower()
    destination = Path(output).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    if key == "miplib":
        safe_name = Path(instance).name.removesuffix(".mps.gz").removesuffix(".mps")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", safe_name) is None:
            raise ValueError("MIPLIB instance must be a portable instance identifier.")
        filename = f"{safe_name}.mps.gz"
        url = f"https://miplib.zib.de/WebData/instances/{urllib.parse.quote(filename)}"
    elif key == "qplib":
        safe_name = Path(instance).name.removeprefix("QPLIB_").removesuffix(".qplib")
        if not safe_name.isdigit():
            raise ValueError("QPLIB instance must be a numeric identifier.")
        safe_name = str(int(safe_name)).zfill(4)
        filename = f"QPLIB_{safe_name}.qplib"
        url = f"https://qplib.zib.de/qplib/{urllib.parse.quote(filename)}"
    else:
        raise ValueError("library must be 'miplib' or 'qplib'.")
    downloaded = _download(url, destination / filename, overwrite=overwrite)
    metadata = {
        "library": key,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "file": asdict(downloaded),
    }
    (destination / f"{filename}.snapshot.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


__all__ = ["DownloadedFile", "fetch_benchmark", "fetch_instance"]
