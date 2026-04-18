"""Smoke tests for the ``qqa`` CLI."""

from __future__ import annotations

import subprocess
import sys


def _run(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "qqa.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_version_prints_version():
    import qqa

    out = _run("version")
    assert out.returncode == 0
    assert qqa.__version__ in out.stdout


def test_cli_help_describes_subcommands():
    out = _run("--help")
    assert out.returncode == 0
    for cmd in ("solve", "bench", "gui", "version"):
        assert cmd in out.stdout


def test_cli_solve_small_mis():
    out = _run(
        "solve",
        "--problem",
        "mis",
        "--size",
        "16",
        "--sol-size",
        "16",
        "--epochs",
        "120",
        "--quiet",
    )
    assert out.returncode == 0, out.stderr
    assert "best_obj" in out.stdout


def test_cli_solve_sk_small():
    out = _run(
        "solve",
        "--problem",
        "sk",
        "--size",
        "20",
        "--sol-size",
        "16",
        "--epochs",
        "80",
        "--quiet",
    )
    assert out.returncode == 0, out.stderr
