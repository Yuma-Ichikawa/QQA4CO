"""Wheel-shipped benchmark runner and report renderer.

The implementation lives in the package so :mod:`qqa.bench` and the
``qqa bench-*`` commands behave identically from a source checkout, wheel,
or sdist install.  Files under ``scripts/`` are deliberately thin command
wrappers around these modules.
"""

from __future__ import annotations

__all__: list[str] = []
