"""Fail-closed validation for serialised metadata and deployment boundaries."""

from __future__ import annotations

import ipaddress
import math
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret|credential|"
    r"hostname|server|cluster|worker|absolute[_-]?path|local[_-]?path)",
    re.IGNORECASE,
)


def _private_string(value: str) -> bool:
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return True
    parsed = urlsplit(value)
    host = parsed.hostname
    if host is None:
        return False
    lowered = host.lower()
    if lowered in {"localhost", "localhost.localdomain"} or lowered.endswith(
        (".local", ".internal")
    ):
        return True
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return False
    return not address.is_global


def validate_portable_payload(value: Any, *, maximum_depth: int = 32) -> None:
    """Reject secrets, machine topology, paths, private URLs, and exotic objects."""

    def visit(item: Any, depth: int) -> None:
        if depth > maximum_depth:
            raise ValueError("Portable metadata exceeds the maximum nesting depth.")
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Portable metadata numbers must be finite.")
            return
        if isinstance(item, str):
            if _private_string(item):
                raise ValueError("Portable metadata contains a local path or private endpoint.")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError("Portable metadata keys must be strings.")
                if _SENSITIVE_KEY.search(key):
                    raise ValueError(f"Portable metadata key {key!r} is sensitive.")
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
            return
        raise TypeError(f"Unsupported portable metadata type: {type(item).__name__}.")

    visit(value, 0)


__all__ = ["validate_portable_payload"]
