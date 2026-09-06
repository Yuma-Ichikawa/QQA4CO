"""Portable, atomic solver checkpoints without executable pickle payloads."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import tempfile
import zipfile
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from qqa.runtime.security import validate_portable_payload

_MAX_ARCHIVE_BYTES = 2 * 1024**3


def _update_fingerprint(digest, value: Any) -> None:
    """Hash model semantics recursively without serialising environment data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        digest.update(json.dumps(value, sort_keys=True, allow_nan=True).encode())
        return
    if isinstance(value, Enum):
        _update_fingerprint(digest, value.value)
        return
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
        return
    if isinstance(value, dict) or hasattr(value, "items"):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            _update_fingerprint(digest, str(key))
            _update_fingerprint(digest, item)
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"sequence:{len(value)}".encode())
        for item in value:
            _update_fingerprint(digest, item)
        return
    if is_dataclass(value):
        digest.update(f"{type(value).__module__}.{type(value).__qualname__}".encode())
        for item in fields(value):
            _update_fingerprint(digest, item.name)
            _update_fingerprint(digest, getattr(value, item.name))
        return
    if callable(value):
        digest.update(f"callable:{value.__module__}.{value.__qualname__}".encode())
        with suppress(OSError, TypeError):
            digest.update(inspect.getsource(value).encode())
        closure = getattr(value, "__closure__", None)
        if closure:
            for cell in closure:
                with suppress(ValueError):
                    _update_fingerprint(digest, cell.cell_contents)
        return
    digest.update(f"opaque:{type(value).__module__}.{type(value).__qualname__}".encode())


@dataclass(frozen=True, slots=True)
class Checkpoint:
    model_fingerprint: str
    config: dict[str, Any]
    epoch: int
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.model_fingerprint, str) or not self.model_fingerprint.strip():
            raise ValueError("Checkpoint model_fingerprint must be a non-empty string.")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
            or self.schema_version != 1
        ):
            raise ValueError("Unsupported checkpoint schema.")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, Integral) or self.epoch < 0:
            raise ValueError("Checkpoint epoch must be a non-negative integer.")
        if not isinstance(self.config, dict) or not isinstance(self.metadata, dict):
            raise TypeError("Checkpoint config and metadata must be dictionaries.")
        if not isinstance(self.tensors, dict) or any(
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or not torch.is_tensor(value)
            for name, value in self.tensors.items()
        ):
            raise TypeError("Checkpoint tensors must map simple names to tensors.")
        validate_portable_payload(self.config)
        validate_portable_payload(self.metadata)
        validate_portable_payload({"model_fingerprint": self.model_fingerprint})


def fingerprint_problem(problem: Any) -> str:
    """Hash model structure/source and numerical tensors without recording paths."""
    digest = hashlib.sha256()
    model_type = type(problem)
    digest.update(f"{model_type.__module__}.{model_type.__qualname__}".encode())
    for name in ("name", "num_vars", "num_nodes", "num_node", "num_category"):
        value = getattr(problem, name, None)
        if isinstance(value, (str, int, float)):
            digest.update(f"{name}={value}".encode())
    with suppress(OSError, TypeError):
        digest.update(inspect.getsource(model_type).encode())
    tensors: list[torch.Tensor] = []
    for value in getattr(problem, "__dict__", {}).values():
        if isinstance(value, torch.Tensor):
            tensors.append(value)
    qubo = getattr(problem, "sparse_qubo", None)
    if qubo is not None:
        for name in ("linear", "edge_index", "edge_weight"):
            value = getattr(qubo, name, None)
            if isinstance(value, torch.Tensor):
                tensors.append(value)
    model_ir = getattr(problem, "model_ir", None)
    if model_ir is not None:
        _update_fingerprint(digest, model_ir)
        expressions = [model_ir.objective, *(row.expression for row in model_ir.constraints)]
        for expression in expressions:
            for factor in expression.factors:
                for name in (
                    "indices",
                    "edge_index",
                    "weights",
                    "signs",
                    "durations",
                    "demands",
                ):
                    value = getattr(factor, name, None)
                    if isinstance(value, torch.Tensor):
                        tensors.append(value)
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, tensor.detach().cpu().numpy(), allow_pickle=False)
    return buffer.getvalue()


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> Path:
    """Atomically write a checksum-protected checkpoint archive."""
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError("checkpoint must be a Checkpoint.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensor_payloads = {name: _tensor_bytes(value) for name, value in checkpoint.tensors.items()}
    checksums = {name: hashlib.sha256(value).hexdigest() for name, value in tensor_payloads.items()}
    manifest = {
        "schema_version": 1,
        "model_fingerprint": checkpoint.model_fingerprint,
        "config": checkpoint.config,
        "epoch": checkpoint.epoch,
        "metadata": checkpoint.metadata,
        "tensor_checksums": checksums,
    }
    validate_portable_payload(manifest)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(
                "manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            for name, payload in tensor_payloads.items():
                bundle.writestr(f"tensors/{name}.npy", payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> Checkpoint:
    """Load and verify a checkpoint; no Python objects are deserialised."""
    source = Path(path)
    with zipfile.ZipFile(source) as bundle:
        listed = bundle.namelist()
        names = set(listed)
        if len(listed) != len(names):
            raise ValueError("Checkpoint archive contains duplicate members.")
        if (
            "manifest.json" not in names
            or any(
                PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                for name in names
            )
            or sum(item.file_size for item in bundle.infolist()) > _MAX_ARCHIVE_BYTES
        ):
            raise ValueError("Invalid checkpoint archive layout.")
        try:
            manifest = json.loads(bundle.read("manifest.json"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Checkpoint manifest is not valid JSON.") from exc
        required_keys = {
            "schema_version",
            "model_fingerprint",
            "config",
            "epoch",
            "metadata",
            "tensor_checksums",
        }
        if not isinstance(manifest, dict) or set(manifest) != required_keys:
            raise ValueError("Checkpoint manifest has an invalid field set.")
        schema_version = manifest["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            raise ValueError("Unsupported checkpoint schema.")
        if (
            not isinstance(manifest["model_fingerprint"], str)
            or not manifest["model_fingerprint"].strip()
            or isinstance(manifest["epoch"], bool)
            or not isinstance(manifest["epoch"], int)
            or manifest["epoch"] < 0
            or not isinstance(manifest["config"], dict)
            or not isinstance(manifest["metadata"], dict)
            or not isinstance(manifest["tensor_checksums"], dict)
        ):
            raise ValueError("Checkpoint manifest contains invalid field values.")
        validate_portable_payload(manifest)
        checksums = manifest["tensor_checksums"]
        if any(
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            for name, expected in checksums.items()
        ):
            raise ValueError("Checkpoint manifest contains an invalid tensor checksum.")
        expected_members = {
            "manifest.json",
            *(f"tensors/{name}.npy" for name in checksums),
        }
        if names != expected_members:
            raise ValueError("Checkpoint archive contains unexpected members.")
        tensors = {}
        for name, expected in checksums.items():
            member = f"tensors/{name}.npy"
            payload = bundle.read(member)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Checkpoint tensor checksum mismatch: {name}.")
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            tensors[name] = torch.from_numpy(array.copy()).to(device)
    return Checkpoint(
        manifest["model_fingerprint"],
        manifest["config"],
        manifest["epoch"],
        tensors,
        manifest["metadata"],
    )


__all__ = ["Checkpoint", "fingerprint_problem", "load_checkpoint", "save_checkpoint"]
