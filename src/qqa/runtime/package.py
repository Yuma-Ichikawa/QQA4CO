"""Auditable model/result exchange packages with content checksums."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qqa.result import SolveResult
from qqa.runtime.security import validate_portable_payload

_PAYLOAD_MEMBERS = frozenset({"model-summary.json", "result.json", "events.json"})
_PACKAGE_MEMBERS = _PAYLOAD_MEMBERS | {"manifest.json"}


@dataclass(frozen=True, slots=True)
class PackageManifest:
    schema_version: int
    files: dict[str, str]
    model_fingerprint: str
    result_status: str

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError("Result package schema version must be an integer.")
        if not isinstance(self.files, dict):
            raise TypeError("Result package files must be a dictionary.")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            for name, checksum in self.files.items()
        ):
            raise ValueError("Result package files contain an invalid name or checksum.")
        if not isinstance(self.model_fingerprint, str) or not self.model_fingerprint.strip():
            raise ValueError("Result package model fingerprint must be non-empty.")
        if not isinstance(self.result_status, str) or not self.result_status.strip():
            raise ValueError("Result package result status must be non-empty.")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def export_result_package(
    result: SolveResult,
    path: str | Path,
    *,
    model_summary: dict[str, Any],
    model_fingerprint: str,
) -> Path:
    """Write a self-verifying result package containing no machine metadata."""
    if not isinstance(result, SolveResult):
        raise TypeError("result must be a SolveResult.")
    if not isinstance(model_summary, dict):
        raise TypeError("model_summary must be a dictionary.")
    if not isinstance(model_fingerprint, str) or not model_fingerprint.strip():
        raise ValueError("model_fingerprint must be a non-empty string.")
    result_payload = result.to_dict(include_solutions=True)
    event_payload = [
        event.to_dict() if callable(getattr(event, "to_dict", None)) else event
        for event in result.events
    ]
    validate_portable_payload(model_summary)
    validate_portable_payload(result_payload)
    validate_portable_payload(event_payload)
    validate_portable_payload({"model_fingerprint": model_fingerprint})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        "model-summary.json": _json_bytes(model_summary),
        "result.json": _json_bytes(result_payload),
        "events.json": _json_bytes(event_payload),
    }
    manifest = PackageManifest(
        1,
        {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
        model_fingerprint,
        result.status.value,
    )
    payloads["manifest.json"] = _json_bytes(asdict(manifest))
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, payload in payloads.items():
                bundle.writestr(name, payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def verify_result_package(path: str | Path) -> PackageManifest:
    """Validate member names, schema, portable payloads, and every checksum."""
    with zipfile.ZipFile(path) as bundle:
        listed = bundle.namelist()
        names = set(listed)
        if names != _PACKAGE_MEMBERS or len(listed) != len(names):
            raise ValueError("Result package contains an unexpected member set.")
        try:
            raw = json.loads(bundle.read("manifest.json"))
            manifest = PackageManifest(**raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("Result package manifest is invalid.") from exc
        if manifest.schema_version != 1:
            raise ValueError("Unsupported result package schema.")
        validate_portable_payload(raw)
        if not isinstance(manifest.files, dict) or set(manifest.files) != _PAYLOAD_MEMBERS:
            raise ValueError("Result package manifest does not cover every payload member.")
        payloads: dict[str, Any] = {}
        for name in sorted(_PAYLOAD_MEMBERS):
            expected = manifest.files[name]
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
            ):
                raise ValueError(f"Result package checksum is invalid: {name}.")
            raw_payload = bundle.read(name)
            if hashlib.sha256(raw_payload).hexdigest() != expected:
                raise ValueError(f"Result package checksum mismatch: {name}.")
            try:
                payload = json.loads(raw_payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Result package payload is not valid JSON: {name}.") from exc
            validate_portable_payload(payload)
            payloads[name] = payload
        result_payload = payloads["result.json"]
        if (
            not isinstance(result_payload, dict)
            or result_payload.get("status") != manifest.result_status
        ):
            raise ValueError("Result package status does not match its manifest.")
    return manifest


__all__ = ["PackageManifest", "export_result_package", "verify_result_package"]
