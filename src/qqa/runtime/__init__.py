"""Lazy event-driven runtime contracts shared by solvers, services, and UIs."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Checkpoint": ("qqa.runtime.checkpoint", "Checkpoint"),
    "EventKind": ("qqa.runtime.events", "EventKind"),
    "EventRecorder": ("qqa.runtime.events", "EventRecorder"),
    "PackageManifest": ("qqa.runtime.package", "PackageManifest"),
    "ReplicaPortfolio": ("qqa.runtime.population", "ReplicaPortfolio"),
    "ReplicaRole": ("qqa.runtime.population", "ReplicaRole"),
    "SolveContext": ("qqa.runtime.context", "SolveContext"),
    "SolveEvent": ("qqa.runtime.events", "SolveEvent"),
    "WarmStateBundle": ("qqa.runtime.population", "WarmStateBundle"),
    "export_result_package": ("qqa.runtime.package", "export_result_package"),
    "fingerprint_problem": ("qqa.runtime.checkpoint", "fingerprint_problem"),
    "load_checkpoint": ("qqa.runtime.checkpoint", "load_checkpoint"),
    "save_checkpoint": ("qqa.runtime.checkpoint", "save_checkpoint"),
    "validate_portable_payload": ("qqa.runtime.security", "validate_portable_payload"),
    "verify_result_package": ("qqa.runtime.package", "verify_result_package"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
