#!/usr/bin/env python3
"""Compatibility CLI for :mod:`qqa.benchmarking.runner`."""

from __future__ import annotations

from qqa.benchmarking.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
