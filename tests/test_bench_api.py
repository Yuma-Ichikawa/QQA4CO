"""Public API contracts for :mod:`qqa.bench` and the ``qqa bench-*`` CLI.

These tests exist because the bench module is the *documented* entry
point for third parties — a silent rename or signature change would
break every external benchmark notebook. The aim is to pin down:

* ``DEFAULT_RESULTS_DIR`` stays as the ``bench_results`` relative path,
* ``list_suites`` / ``resolve_suite`` delegate correctly,
* ``plot`` wraps ``plot_benchmarks.main`` end-to-end and respects
  ``DEFAULT_RESULTS_DIR`` when a relative path is passed,
* ``qqa bench-list`` and ``qqa bench-plot`` are wired into the CLI
  dispatcher so ``uv run qqa ...`` keeps working.

We deliberately do NOT exercise ``run()``'s end-to-end anneal — the
real runner is tested by ``test_bench_discs_runner.py``; here we only
verify argv construction and the output-path contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from qqa import bench
from qqa.cli import build_parser, main


def test_default_results_dir_is_bench_results():
    assert Path("bench_results") == bench.DEFAULT_RESULTS_DIR


def test_list_suites_returns_nested_mapping():
    # list_suites may be empty on CI before setup; the contract is
    # "dict of dict of list of str". We accept an empty dict.
    catalog = bench.list_suites()
    assert isinstance(catalog, dict)
    for fam, types in catalog.items():
        assert isinstance(fam, str)
        assert isinstance(types, dict)
        for gt, subs in types.items():
            assert isinstance(gt, str)
            assert isinstance(subs, list)
            for s in subs:
                assert isinstance(s, str)


def test_resolve_suite_delegates_to_runner(monkeypatch):
    called = {}

    def fake(suite):
        called["suite"] = suite
        return [("mis", "er", "800")]

    fake_module = mock.Mock()
    fake_module._resolve_suite = fake

    with mock.patch.object(bench, "_load_bench_discs", return_value=fake_module):
        out = bench.resolve_suite("mis-er-800")

    assert called["suite"] == "mis-er-800"
    assert out == [("mis", "er", "800")]


def test_plot_relative_output_is_normalised_under_results_dir(tmp_path):
    # Write a minimal JSON that plot_benchmarks can load.
    payload = {
        "backend": "qqa",
        "suite": "demo",
        "device": "cpu",
        "results": [
            {
                "problem": "mis",
                "graph_type": "er",
                "subset": "800",
                "n": 1,
                "n_feasible": 1,
                "mean_ratio": 1.0,
                "instances": [
                    {"ratio": 1.0, "feasible": True, "best_known": 1.0, "objective": 1.0}
                ],
            }
        ],
    }
    j = tmp_path / "results.json"
    j.write_text(json.dumps(payload))

    with mock.patch.object(bench, "DEFAULT_RESULTS_DIR", tmp_path / "out"):
        written = bench.plot([j], output="report.png", title="unit test")

    assert written is not None
    assert (tmp_path / "out" / "report.png").is_file()


def test_plot_absolute_output_is_respected(tmp_path):
    payload = {
        "backend": "qqa",
        "suite": "demo",
        "device": "cpu",
        "results": [
            {
                "problem": "ea3d",
                "graph_type": "gaussian",
                "subset": "L4",
                "n": 1,
                "n_feasible": 1,
                "mean_ratio": 0.9,
                "instances": [
                    {
                        "ratio": 0.9,
                        "feasible": True,
                        "best_known": -10.0,
                        "objective": -9.0,
                    }
                ],
            }
        ],
    }
    j = tmp_path / "results.json"
    j.write_text(json.dumps(payload))
    out = tmp_path / "abs_report.png"

    written = bench.plot([j], output=out)
    assert written == out
    assert out.is_file()


# --------------------------------------------------------------------------- #
# CLI wiring                                                                  #
# --------------------------------------------------------------------------- #


def test_cli_parser_exposes_bench_subcommands():
    parser = build_parser()
    # argparse stores subparsers on the _SubParsersAction; walk actions to find it.
    subparsers = None
    for action in parser._actions:  # type: ignore[attr-defined]
        if action.__class__.__name__ == "_SubParsersAction":
            subparsers = action
            break
    assert subparsers is not None
    names = set(subparsers.choices)
    assert {"bench", "bench-list", "bench-run", "bench-plot"} <= names


def test_cli_bench_list_runs(capsys, monkeypatch):
    # Pretend the catalog is tiny and deterministic so the test doesn't
    # depend on whether ./data has been populated.
    fake = {"mis": {"er": ["800"]}, "coloring": {"myciel": [""]}}
    monkeypatch.setattr(bench, "list_suites", lambda: fake)
    rc = main(["bench-list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mis" in out and "coloring" in out


def test_cli_bench_list_as_suites(capsys, monkeypatch):
    fake = {"mis": {"er": ["800"]}, "coloring": {"myciel": [""]}}
    monkeypatch.setattr(bench, "list_suites", lambda: fake)
    rc = main(["bench-list", "--as-suites"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mis-er-800" in out
    assert "coloring-myciel" in out


def test_cli_bench_plot_smoke(tmp_path, capsys):
    payload = {
        "backend": "qqa",
        "suite": "demo",
        "device": "cpu",
        "results": [
            {
                "problem": "mis",
                "graph_type": "er",
                "subset": "800",
                "n": 1,
                "n_feasible": 1,
                "mean_ratio": 1.0,
                "instances": [
                    {"ratio": 1.0, "feasible": True, "best_known": 1.0, "objective": 1.0}
                ],
            }
        ],
    }
    j = tmp_path / "results.json"
    j.write_text(json.dumps(payload))
    out = tmp_path / "r.png"

    rc = main(["bench-plot", str(j), "--output", str(out)])
    assert rc == 0
    assert out.is_file()


@pytest.mark.parametrize(
    "argv",
    [
        ["bench-run"],  # suite defaults to "all"
        ["bench-list"],  # always works even with empty data tree
    ],
)
def test_cli_help_does_not_crash(argv):
    parser = build_parser()
    ns = parser.parse_args(argv)
    assert ns.command == argv[0]
