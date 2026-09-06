"""Audited benchmark sources and local snapshot verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class BenchmarkRegistryEntry:
    key: str
    title: str
    snapshot: str
    official_hosts: tuple[str, ...]
    accepted_formats: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


_REGISTRY = {
    "miplib": BenchmarkRegistryEntry(
        "miplib",
        "MIPLIB 2017 benchmark set",
        "MIPLIB-2017-benchmark-v2/solu-v36",
        ("miplib.zib.de",),
        ("mps", "mps.gz"),
    ),
    "qplib": BenchmarkRegistryEntry(
        "qplib",
        "QPLIB public collection",
        "QPLIB-public-snapshot",
        ("qplib.zib.de",),
        ("qplib",),
    ),
}


def benchmark_registry() -> tuple[BenchmarkRegistryEntry, ...]:
    return tuple(_REGISTRY[key] for key in sorted(_REGISTRY))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_benchmark_snapshot(path: str | Path) -> dict:
    """Verify registry identity, official origins, sizes, and file hashes."""
    root = Path(path)
    manifest_path = root / "snapshot.json" if root.is_dir() else root
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = str(payload.get("library", "")).lower()
    if key not in _REGISTRY:
        raise ValueError("Snapshot library is absent from the audited registry.")
    entry = _REGISTRY[key]
    if payload.get("snapshot") != entry.snapshot:
        raise ValueError("Snapshot version does not match the audited registry.")
    records = payload.get("files")
    if not isinstance(records, list):
        raise ValueError("Snapshot files must be a list.")
    verified = []
    seen_names: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not {"name", "url", "sha256", "size"} <= record.keys():
            raise ValueError("Snapshot file record is incomplete.")
        name = Path(str(record["name"]))
        if name.name != str(record["name"]):
            raise ValueError("Snapshot filenames must be portable basenames.")
        if name.name in seen_names:
            raise ValueError("Snapshot filenames must be unique.")
        seen_names.add(name.name)
        parsed_url = urlsplit(str(record["url"]))
        host = (parsed_url.hostname or "").lower()
        try:
            port = parsed_url.port
        except ValueError as exc:
            raise ValueError("Snapshot URL contains an invalid port.") from exc
        if (
            parsed_url.scheme != "https"
            or host not in entry.official_hosts
            or port not in {None, 443}
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("Snapshot URL is not an audited official origin.")
        source = manifest_path.parent / name
        size = record["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"Snapshot file size is invalid: {name.name}.")
        expected_digest = record["sha256"]
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError(f"Snapshot checksum is invalid: {name.name}.")
        if not source.is_file() or source.stat().st_size != size:
            raise ValueError(f"Snapshot file size mismatch: {name.name}.")
        digest = _sha256(source)
        if digest != expected_digest:
            raise ValueError(f"Snapshot checksum mismatch: {name.name}.")
        verified.append({"name": name.name, "sha256": digest, "size": source.stat().st_size})
    if not verified:
        raise ValueError("Snapshot contains no auditable files.")
    return {"library": key, "snapshot": entry.snapshot, "verified_files": verified}


__all__ = ["BenchmarkRegistryEntry", "audit_benchmark_snapshot", "benchmark_registry"]
