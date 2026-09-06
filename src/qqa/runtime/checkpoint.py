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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.schema_version != 1 or checkpoint.epoch < 0:
        raise ValueError("Unsupported checkpoint schema or negative epoch.")
    validate_portable_payload(checkpoint.config)
    validate_portable_payload(checkpoint.metadata)
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
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(
                "manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            for name, payload in tensor_payloads.items():
                if not name or "/" in name or "\\" in name:
                    raise ValueError("Checkpoint tensor names must be simple relative names.")
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
        manifest = json.loads(bundle.read("manifest.json"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema.")
        validate_portable_payload(manifest.get("config", {}))
        validate_portable_payload(manifest.get("metadata", {}))
        expected_members = {
            "manifest.json",
            *(f"tensors/{name}.npy" for name in manifest.get("tensor_checksums", {})),
        }
        if names != expected_members:
            raise ValueError("Checkpoint archive contains unexpected members.")
        tensors = {}
        for name, expected in manifest.get("tensor_checksums", {}).items():
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ValueError("Invalid checkpoint tensor name.")
            member = f"tensors/{name}.npy"
            payload = bundle.read(member)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Checkpoint tensor checksum mismatch: {name}.")
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            tensors[name] = torch.from_numpy(array.copy()).to(device)
    return Checkpoint(
        str(manifest["model_fingerprint"]),
        dict(manifest["config"]),
        int(manifest["epoch"]),
        tensors,
        dict(manifest.get("metadata", {})),
    )


__all__ = ["Checkpoint", "fingerprint_problem", "load_checkpoint", "save_checkpoint"]
