"""Persistent ahead-of-time graph and AOTInductor cache for sparse QUBOs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import torch

from qqa.compile.sparse_qubo import SparseQUBO

_CACHE_FORMAT = 2


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


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment_metadata(example: torch.Tensor) -> dict[str, Any]:
    cuda_capability = None
    driver_version = None
    if example.device.type == "cuda" and torch.cuda.is_available():
        with suppress(RuntimeError, ValueError):
            cuda_capability = list(torch.cuda.get_device_capability(example.device))
        probe = getattr(torch._C, "_cuda_getDriverVersion", None)
        if callable(probe):
            with suppress(RuntimeError):
                driver_version = int(probe())
    return {
        "torch": torch.__version__,
        "torch_revision": getattr(torch.version, "git_version", None),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "device": example.device.type,
        "cuda_capability": cuda_capability,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver_version,
        "hip_runtime": torch.version.hip,
        "triton": _package_version("triton"),
        "qqa": _package_version("qqa"),
        "aot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _artifact_lock(path: Path, *, timeout: float = 60.0):
    """Cross-process lock using one atomic, cache-local lock file."""
    started = time.monotonic()
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                stale = time.time() - path.stat().st_mtime > max(300.0, timeout * 2)
            except FileNotFoundError:
                continue
            if stale:
                with suppress(FileNotFoundError):
                    path.unlink()
                continue
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Timed out waiting for an AOT cache artifact lock.") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        with suppress(FileNotFoundError):
            path.unlink()


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
        "environment": _environment_metadata(example),
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

    def load_cached():
        if force or not artifact.is_file() or not manifest.is_file():
            return None
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                metadata.get("format") == _CACHE_FORMAT
                and metadata.get("key") == key
                and metadata.get("backend") == backend
                and metadata.get("artifact_sha256") == _file_sha256(artifact)
            ):
                module = _load_artifact(artifact, backend, example_values.device)
                return AOTCompiledSparseQUBO(module, artifact, key, True, backend)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            pass  # Rebuild stale, corrupt, or incompatible artifacts below.
        return None

    cached = load_cached()
    if cached is not None:
        return cached

    lock = root / f"{key}.lock"
    with _artifact_lock(lock):
        cached = load_cached()
        if cached is not None:
            return cached
        module = _SparseQUBOModule(qubo).to(
            device=example_values.device,
            dtype=example_values.dtype,
        )
        dynamic_shapes = ({0: torch.export.Dim("batch", min=1)},) if dynamic_batch else None
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
                compiler(exported, package_path=str(temporary_artifact))
            descriptor = {
                "format": _CACHE_FORMAT,
                "key": key,
                "backend": backend,
                "artifact_sha256": _file_sha256(temporary_artifact),
                "environment": _environment_metadata(example_values),
            }
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
