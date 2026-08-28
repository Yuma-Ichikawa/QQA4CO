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


@dataclass(frozen=True, slots=True)
class PackageManifest:
    schema_version: int
    files: dict[str, str]
    model_fingerprint: str
    result_status: str


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
    """Validate member names, schema, and every content checksum."""
    with zipfile.ZipFile(path) as bundle:
        listed = bundle.namelist()
        names = set(listed)
        expected_names = {"manifest.json", "model-summary.json", "result.json", "events.json"}
        if names != expected_names or len(listed) != len(names):
            raise ValueError("Result package contains an unexpected member set.")
        raw = json.loads(bundle.read("manifest.json"))
        manifest = PackageManifest(**raw)
        if manifest.schema_version != 1:
            raise ValueError("Unsupported result package schema.")
        validate_portable_payload(raw)
        for name, expected in manifest.files.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise ValueError(f"Result package checksum mismatch: {name}.")
    return manifest


__all__ = ["PackageManifest", "export_result_package", "verify_result_package"]
