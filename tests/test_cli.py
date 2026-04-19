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


def test_cli_solve_min_dominating_set():
    """Phase-B: --problem min_dominating_set is wired end-to-end."""
    out = _run(
        "solve",
        "--problem",
        "min_dominating_set",
        "--size",
        "16",
        "--sol-size",
        "16",
        "--epochs",
        "80",
        "--quiet",
    )
    assert out.returncode == 0, out.stderr
    assert "best_obj" in out.stdout
    assert "dominating set size" in out.stdout


def test_cli_solve_balanced_graph_partition():
    """Phase-B: --problem bgp + --num-category exposes BalancedGraphPartition."""
    out = _run(
        "solve",
        "--problem",
        "bgp",
        "--size",
        "12",
        "--num-category",
        "3",
        "--sol-size",
        "16",
        "--epochs",
        "80",
        "--quiet",
    )
    assert out.returncode == 0, out.stderr
    assert "edge cut" in out.stdout


def test_cli_solve_no_polish_flag_accepted():
    """--no-polish must be accepted and produce a finite result."""
    out = _run(
        "solve",
        "--problem",
        "mis",
        "--size",
        "16",
        "--sol-size",
        "16",
        "--epochs",
        "80",
        "--no-polish",
        "--quiet",
    )
    assert out.returncode == 0, out.stderr
    assert "best_obj" in out.stdout
