"""Smoke tests for the ``qqa`` CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest


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
    for cmd in ("solve", "tex", "example", "doctor", "bench", "gui", "version"):
        assert cmd in out.stdout


def test_cli_score_summary_is_compact_and_auditable(capsys):
    from qqa.cli import _print_score

    _print_score(
        {
            "label": "cost",
            "value": 12.5,
            "unit": "USD",
            "feasible": True,
            "extra": {
                "variables": {"units": 3.0},
                "constraints": {
                    "capacity": {
                        "violation": 0.0,
                        "feasible": True,
                    }
                },
                "large_internal_payload": list(range(100)),
            },
        }
    )
    output = capsys.readouterr().out
    assert "cost=12.5 USD; feasible=true" in output
    assert 'solution   : {"units": 3.0}' in output
    assert "constraints: 0/1 failed" in output
    assert "large_internal_payload" not in output


def test_cli_lists_realistic_examples_and_doctor_json():
    listed = _run("example", "list")
    assert listed.returncode == 0, listed.stderr
    assert "microgrid-pareto" in listed.stdout
    doctor = _run("doctor", "--json")
    assert doctor.returncode == 0, doctor.stderr
    assert '"recommended_device"' in doctor.stdout


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


def test_cli_solve_pspin_glass():
    """Physics catalog: --problem pspin --p-order routes to PSpinGlass."""
    out = _run(
        "solve",
        "--problem",
        "pspin",
        "--size",
        "10",
        "--p-order",
        "3",
        "--sol-size",
        "16",
        "--epochs",
        "60",
        "--quiet",
    )
    assert out.returncode == 0, out.stderr
    assert "best_obj" in out.stdout
    assert "energy / spin" in out.stdout


def test_cli_solve_random_field_ising():
    """Physics catalog: --problem rfim --h-std --coupling-J routes to RFIM."""
    out = _run(
        "solve",
        "--problem",
        "rfim",
        "--size",
        "4",
        "--dim",
        "2",
        "--h-std",
        "1.0",
        "--coupling-J",
        "1.0",
        "--sol-size",
        "16",
        "--epochs",
        "60",
        "--quiet",
    )
    assert out.returncode == 0, out.stderr
    assert "best_obj" in out.stdout
    assert "energy / spin" in out.stdout


def test_cli_mixed_problem_file_writes_html_report(tmp_path):
    model = tmp_path / "mixed_model.py"
    model.write_text(
        """
import qqa

problem = qqa.MixedProblem(
    [qqa.Integer("quantity", 0, 8), qqa.Real("slack", 0.0, 2.0)],
    lambda v: (v["quantity"] - 4).square() + v["slack"].square(),
    name="cli-mixed",
)
""",
        encoding="utf-8",
    )
    report = tmp_path / "mixed-report.html"
    out = _run(
        "solve",
        "--problem-file",
        str(model),
        "--sol-size",
        "16",
        "--epochs",
        "80",
        "--quiet",
        "--report",
        str(report),
    )
    assert out.returncode == 0, out.stderr
    assert report.is_file()
    assert "Machine-readable result JSON" in report.read_text(encoding="utf-8")


def test_cli_scip_backend_reports_proof():
    pytest.importorskip("pyscipopt")
    out = _run(
        "solve",
        "--problem",
        "maxcut",
        "--size",
        "8",
        "--backend",
        "scip",
        "--sol-size",
        "32",
        "--epochs",
        "10",
        "--scip-time-limit",
        "10",
        "--quiet",
    )
    assert out.returncode == 0, out.stderr
    assert "scip_status: optimal" in out.stdout
    assert "gap        : 0.0" in out.stdout
