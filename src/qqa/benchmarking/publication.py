"""Create portable, deterministic public artifacts from benchmark campaigns."""

from __future__ import annotations

import gzip
import hashlib
import ipaddress
import json
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_string(value: str) -> bool:
    parsed = urlsplit(value)
    if parsed.hostname is not None:
        host = parsed.hostname.lower().rstrip(".")
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            return True
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return False
        return bool(
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_unspecified
        )
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def validate_portable_payload(payload: Any, *, location: str = "root") -> None:
    """Reject absolute paths, private hosts, and environment-specific keys."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} contains a non-string key.")
            if key.lower() in {
                "absolute_path",
                "cwd",
                "home",
                "hostname",
                "internal_url",
                "server",
                "username",
            }:
                raise ValueError(f"Private environment key at {location}.{key}.")
            validate_portable_payload(value, location=f"{location}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            validate_portable_payload(value, location=f"{location}[{index}]")
        return
    if isinstance(payload, str) and _private_string(payload):
        raise ValueError(f"Private environment value at {location}.")


def _canonical_json(payload: Any, *, pretty: bool) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_gzip(path: Path, payload: bytes) -> None:
    with (
        path.open("wb") as stream,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=stream,
            mtime=0,
        ) as compressed,
    ):
        compressed.write(payload)


def publish_benchmark_campaigns(
    campaigns: Mapping[str, str | Path],
    snapshots: Mapping[str, str | Path],
    output: str | Path,
    *,
    implementation_revision: str | None = None,
) -> dict[str, Any]:
    """Publish full compressed and compact JSON results plus one manifest.

    Input paths are used only for reading and are never serialised. Library
    names must align between ``campaigns`` and ``snapshots``.
    """
    if not campaigns or set(campaigns) != set(snapshots):
        raise ValueError("campaigns and snapshots must have the same non-empty keys.")
    if implementation_revision is not None and (
        not 7 <= len(implementation_revision) <= 64
        or any(character not in "0123456789abcdef" for character in implementation_revision)
    ):
        raise ValueError("implementation_revision must be a 7-64 character lowercase hex hash.")
    destination = Path(output).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema_version": 1, "libraries": {}}
    if implementation_revision is not None:
        manifest["implementation_revision"] = implementation_revision

    for library in sorted(campaigns):
        campaign_path = Path(campaigns[library]).expanduser()
        snapshot_path = Path(snapshots[library]).expanduser()
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        validate_portable_payload(campaign)
        validate_portable_payload(snapshot)

        results = campaign.get("results", [])
        failures = campaign.get("failures", [])
        compact_results = []
        for row in results:
            compact = dict(row)
            trajectory = compact.pop("trajectory", [])
            solution_values = compact.pop("solution_values", None)
            compact["trajectory_points"] = len(trajectory)
            compact["solution_value_count"] = (
                None if solution_values is None else len(solution_values)
            )
            compact_results.append(compact)
        compact_payload = {
            "schema_version": 1,
            "library": library,
            "comparison_config": campaign["comparison_config"],
            "summary": campaign["summary"],
            "results": compact_results,
            "failures": failures,
        }
        validate_portable_payload(compact_payload)

        compact_name = f"{library}-results.json"
        full_name = f"{library}-campaign.json.gz"
        compact_bytes = _canonical_json(compact_payload, pretty=True)
        full_bytes = _canonical_json(campaign, pretty=False)
        (destination / compact_name).write_bytes(compact_bytes)
        _write_gzip(destination / full_name, full_bytes)

        extracted = snapshot.get("extracted_files", [])
        suffix = ".qplib" if library.lower() == "qplib" else ".mps.gz"
        snapshot_summary = {
            "library": snapshot.get("library", library),
            "snapshot": snapshot.get("snapshot"),
            "retrieved_at": snapshot.get("retrieved_at"),
            "files": snapshot.get("files", []),
            "instance_count": sum(str(name).endswith(suffix) for name in extracted),
        }
        validate_portable_payload(snapshot_summary)
        manifest["libraries"][library] = {
            "snapshot": snapshot_summary,
            "comparison_config": campaign["comparison_config"],
            "summary": campaign["summary"],
            "completed_runs": len(results),
            "failed_runs": len(failures),
            "compact_results": {
                "name": compact_name,
                "sha256": _sha256_bytes(compact_bytes),
            },
            "full_campaign": {
                "name": full_name,
                "sha256": _sha256_file(destination / full_name),
            },
        }

    validate_portable_payload(manifest)
    manifest_bytes = _canonical_json(manifest, pretty=True)
    (destination / "manifest.json").write_bytes(manifest_bytes)
    return manifest


__all__ = ["publish_benchmark_campaigns", "validate_portable_payload"]
