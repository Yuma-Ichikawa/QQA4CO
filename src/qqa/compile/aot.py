"""Persistent ahead-of-time graph and AOTInductor cache for sparse QUBOs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from qqa.compile.sparse_qubo import SparseQUBO

_CACHE_FORMAT = 1


class _SparseQUBOModule(torch.nn.Module):
    linear: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor

    def __init__(self, qubo: SparseQUBO) -> None:
        super().__init__()
        self.register_buffer("linear", qubo.linear.detach().clone())
        self.register_buffer("edge_index", qubo.edge_index.detach().clone())
        self.register_buffer("edge_weight", qubo.edge_weight.detach().clone())
        self.constant = float(qubo.constant)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        linear = torch.matmul(values, self.linear)
        endpoints = values[:, self.edge_index]
        pairwise = torch.matmul(endpoints[:, 0] * endpoints[:, 1], self.edge_weight)
        return linear + pairwise + self.constant


@dataclass(frozen=True, slots=True)
class AOTCompiledSparseQUBO:
    """Callable cached artifact plus reproducibility metadata."""

    module: Any
    artifact: Path
    key: str
    cache_hit: bool
    backend: Literal["export", "inductor"]

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return self.module(values)


def _default_cache_dir() -> Path:
    configured = os.environ.get("QQA_AOT_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "qqa" / "aot"


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def _cache_key(
    qubo: SparseQUBO,
    example: torch.Tensor,
    backend: Literal["export", "inductor"],
    dynamic_batch: bool,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "format": _CACHE_FORMAT,
        "backend": backend,
        "dynamic_batch": dynamic_batch,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "device": example.device.type,
        "dtype": str(example.dtype),
        "shape": [None if dynamic_batch else example.shape[0], example.shape[1]],
        "constant": qubo.constant,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for tensor in (qubo.linear, qubo.edge_index, qubo.edge_weight):
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _load_artifact(
    artifact: Path,
    backend: Literal["export", "inductor"],
    device: torch.device,
):
    if backend == "export":
        return torch.export.load(artifact).module().to(device)
    loader = getattr(getattr(torch, "_inductor", None), "aoti_load_package", None)
    if not callable(loader):
        raise RuntimeError("This PyTorch build does not provide AOTInductor package loading.")
    device_index = device.index if device.type == "cuda" and device.index is not None else -1
    return loader(artifact, device_index=device_index)


def compile_sparse_qubo_aot(
    qubo: SparseQUBO,
    example_values: torch.Tensor,
    *,
    cache_dir: str | Path | None = None,
    backend: Literal["export", "inductor"] = "export",
    dynamic_batch: bool = True,
    force: bool = False,
) -> AOTCompiledSparseQUBO:
    """Capture, persist, and reuse a sparse-QUBO tensor graph.

    ``backend="export"`` stores a device-movable, autograd-capable
    :class:`torch.export.ExportedProgram`. ``backend="inductor"`` additionally
    packages native AOTInductor code for deployment/inference and is therefore
    device/toolchain specific. The cache key includes model tensors, Torch and
    Python versions, device type, dtype, shape policy, and cache format.
    """
    if not isinstance(qubo, SparseQUBO):
        raise TypeError("qubo must be a SparseQUBO.")
    if backend not in {"export", "inductor"}:
        raise ValueError("backend must be 'export' or 'inductor'.")
    if not torch.is_tensor(example_values) or example_values.ndim != 2:
        raise TypeError("example_values must be a rank-2 torch tensor.")
    if example_values.shape[0] < 1 or example_values.shape[1] != qubo.num_variables:
        raise ValueError("example_values must have shape (batch >= 1, qubo.num_variables).")
    if not example_values.is_floating_point() or not torch.isfinite(example_values).all():
        raise ValueError("example_values must be a finite floating-point tensor.")
    if backend == "inductor" and dynamic_batch:
        raise ValueError("AOTInductor cache currently requires dynamic_batch=False.")

    key = _cache_key(qubo, example_values, backend, dynamic_batch)
    root = _default_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / f"{key}.pt2"
    manifest = root / f"{key}.json"
    if not force and artifact.is_file() and manifest.is_file():
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if metadata == {"format": _CACHE_FORMAT, "key": key, "backend": backend}:
                module = _load_artifact(artifact, backend, example_values.device)
                return AOTCompiledSparseQUBO(module, artifact, key, True, backend)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            # Rebuild stale/incompatible artifacts atomically below.
            pass

    module = _SparseQUBOModule(qubo).to(
        device=example_values.device,
        dtype=example_values.dtype,
    )
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = ({0: torch.export.Dim("batch", min=1)},)
    export_values = (
        example_values.expand(2, -1).clone()
        if dynamic_batch and example_values.shape[0] == 1
        else example_values
    )
    exported = torch.export.export(
        module,
        (export_values,),
        dynamic_shapes=dynamic_shapes,
        strict=True,
    )
    descriptor = {"format": _CACHE_FORMAT, "key": key, "backend": backend}
    with tempfile.TemporaryDirectory(prefix="qqa-aot-", dir=root) as temporary:
        temporary_root = Path(temporary)
        temporary_artifact = temporary_root / artifact.name
        if backend == "export":
            torch.export.save(exported, temporary_artifact)
        else:
            compiler = getattr(
                getattr(torch, "_inductor", None),
                "aoti_compile_and_package",
                None,
            )
            if not callable(compiler):
                raise RuntimeError("This PyTorch build does not provide AOTInductor packaging.")
            # Torch 2.13 compares the returned package filename with the
            # supplied value using strict equality; pass ``str`` so its
            # internal string result does not spuriously differ from a Path.
            compiler(exported, package_path=str(temporary_artifact))
        temporary_manifest = temporary_root / manifest.name
        temporary_manifest.write_text(
            json.dumps(descriptor, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_artifact, artifact)
        os.replace(temporary_manifest, manifest)

    loaded = _load_artifact(artifact, backend, example_values.device)
    return AOTCompiledSparseQUBO(loaded, artifact, key, False, backend)


__all__ = ["AOTCompiledSparseQUBO", "compile_sparse_qubo_aot"]
