"""Smoke tests for the ``qqa`` CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


def _run(*args: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    if timeout is None:
        timeout = int(os.environ.get("QQA_TEST_SUBPROCESS_TIMEOUT_SECONDS", "120"))
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
    for cmd in ("ask", "solve", "tex", "example", "doctor", "bench", "gui", "version"):
        assert cmd in out.stdout


def test_positional_model_forwards_only_explicit_profile_overrides(tmp_path, monkeypatch):
    import qqa
    from qqa.cli import _cmd_solve, build_parser
    from qqa.portfolio import ModelInspection, SolverPlan
    from qqa.result import SolveStatus

    model = tmp_path / "small.qubo"
    model.write_text("2\n0 0 1\n1 1 -1\n", encoding="utf-8")
    captured = {}
    inspection = ModelInspection(
        "small",
        2,
        2,
        0,
        {"binary": 2},
        {},
        2,
        0.5,
        1.0,
        2,
        "minimize",
        ("sparse-qubo",),
    )
    plan = SolverPlan(inspection, "fast", "sparse-factor-qqa", (), None, 8, 64, ())

    def fake_solve(path, **kwargs):
        captured.update(path=path, **kwargs)
        return SimpleNamespace(
            status=SolveStatus.FEASIBLE,
            plan=plan,
            best_obj=0.0,
            feasible=True,
            runtime=0.0,
            best_bound=None,
            relative_gap=None,
        )

    monkeypatch.setattr(qqa, "solve", fake_solve)
    args = build_parser().parse_args(
        [
            "solve",
            str(model),
            "--profile",
            "fast",
            "--sol-size",
            "8",
            "--epochs",
            "5",
            "--schedule",
            "adaptive",
            "--restart-patience",
            "0",
            "--gradient-clip",
            "0",
            "--no-polish",
        ]
    )
    assert _cmd_solve(args) == 0
    assert captured["replicas"] == 8
    assert captured["epochs"] == 5
    assert captured["schedule"] == "adaptive"
    assert captured["restart_patience"] == 0
    assert captured["gradient_clip_norm"] is None
    assert captured["polish"] is False
    assert "temperature" not in captured


def test_legacy_catalogue_defaults_are_preserved(monkeypatch):
    import qqa
    from qqa.cli import _cmd_solve, build_parser

    captured = {}

    def fake_anneal(problem, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(best_obj=0.0, runtime=0.0, score={}, diagnostics={})

    monkeypatch.setattr(qqa, "anneal", fake_anneal)
    args = build_parser().parse_args(
        ["solve", "--problem", "sk", "--size", "4", "--schedule", "linear", "--quiet"]
    )
    assert _cmd_solve(args) == 0
    assert captured["sol_size"] == 100
    assert captured["num_epochs"] == 1000
    assert captured["schedule"](0, 2) == pytest.approx(-2.0)
    assert captured["schedule"](1, 2) == pytest.approx(0.1)


def test_benchmark_compare_defaults_to_conservative_balanced_qqa_profile():
    from qqa.cli import build_parser

    args = build_parser().parse_args(
        ["benchmark", "compare", "example.mps", "--output", "result.json"]
    )
    assert args.solvers == ("scip-aggressive", "sg-cqqa")
    assert args.baseline_solver == "scip-aggressive"
    assert args.execution_order == "balanced"
    assert args.maximum_problem_variables == 32
    assert args.minimum_core_size == 16
    assert args.maximum_core_saturation == pytest.approx(0.9)
    assert args.maximum_call_time == pytest.approx(0.15)
    assert args.qqa_fix_fraction == pytest.approx(0.25)
    assert args.minimum_relative_improvement == pytest.approx(0.001)
    assert args.qplib_problem_types is None
    assert not args.no_subscip_repair

    qplib = build_parser().parse_args(
        [
            "benchmark",
            "compare",
            "example.qplib",
            "--qplib-problem-types",
            "QML",
            "LIQ",
            "--output",
            "result.json",
        ]
    )
    assert qplib.qplib_problem_types == ["QML", "LIQ"]

    run = build_parser().parse_args(["benchmark", "run", "example.mps"])
    for option in (
        "core_size",
        "maximum_problem_variables",
        "minimum_core_size",
        "maximum_core_saturation",
        "completion_time",
        "minimum_relative_improvement",
        "maximum_overhead_fraction",
    ):
        assert getattr(run, option) == getattr(args, option)

    merged = build_parser().parse_args(
        ["benchmark", "merge", "shard.json.gz", "--output", "all.json", "--quiet"]
    )
    assert merged.quiet

    published = build_parser().parse_args(
        [
            "benchmark",
            "publish",
            "--campaign",
            "miplib=campaign.json",
            "--snapshot",
            "miplib=snapshot.json",
            "--output",
            "public-results",
        ]
    )
    assert published.campaign == ["miplib=campaign.json"]
    assert published.snapshot == ["miplib=snapshot.json"]


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


def test_doctor_returns_portable_fallback_when_runtime_probe_times_out(monkeypatch):
    from qqa.commands import system

    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired([sys.executable], timeout=0.01)

    monkeypatch.setattr(system.subprocess, "run", time_out)
    payload = system._probe_runtime(0.01)
    assert payload["probe_status"] == "timed_out"
    assert payload["recommended_device"] == "cpu"
    assert payload["cuda_available"] is None


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
        "--allow-unsafe-python",
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
        os.environ.get("QQA_TEST_SCIP_TIME_LIMIT_SECONDS", "10"),
        "--quiet",
    )
    assert out.returncode == 0, out.stderr
    assert "scip_status: optimal" in out.stdout
    assert "gap        : 0.0" in out.stdout
