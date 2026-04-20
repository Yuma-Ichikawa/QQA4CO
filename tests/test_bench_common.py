"""Tests for the shared bench-runner helpers in ``scripts/_bench_common.py``.

These lock the contracts that ``bench_discs.py`` and
``bench_factorization.py`` rely on: argparse defaults, kwarg extraction
shape, and JSON output schema. If any of these change, both bench
scripts (and any future bench script using the helpers) must be updated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Load `scripts/_bench_common.py` as a top-level module without polluting
# `sys.modules` permanently. We use importlib instead of patching
# `sys.path` so the test is hermetic — adjacent test files can run in
# any order without picking up a stray `_bench_common` from sys.path.
_BENCH_COMMON_PATH = Path(__file__).resolve().parents[1] / "scripts" / "_bench_common.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("_bench_common_test", _BENCH_COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_common_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("_bench_common_test", None)


def test_add_qqa_hp_args_paper_defaults(bench):
    """Defaults must be the paper-aligned smoke values, not arbitrary."""
    p = argparse.ArgumentParser()
    bench.add_qqa_hp_args(p)
    args = p.parse_args([])
    assert args.learning_rate == pytest.approx(1.0)
    assert args.temp == pytest.approx(1e-3)
    assert args.curve_rate == 4
    assert args.gamma_min == pytest.approx(-2.0)
    assert args.gamma_max == pytest.approx(0.1)
    assert args.div_param == pytest.approx(0.2)


def test_add_qqa_hp_args_overrides(bench):
    p = argparse.ArgumentParser()
    bench.add_qqa_hp_args(p)
    args = p.parse_args(
        [
            "--learning-rate",
            "0.1",
            "--temp",
            "5e-4",
            "--curve-rate",
            "8",
            "--gamma-min",
            "-3",
            "--gamma-max",
            "0.5",
            "--div-param",
            "0.4",
        ]
    )
    assert args.learning_rate == pytest.approx(0.1)
    assert args.curve_rate == 8


def test_qqa_hp_kwargs_shape(bench):
    """``qqa_hp_kwargs`` must return *exactly* the kwargs ``run_qqa_anneal``
    accepts (any divergence breaks the bench scripts at call time)."""
    p = argparse.ArgumentParser()
    bench.add_qqa_hp_args(p)
    kwargs = bench.qqa_hp_kwargs(p.parse_args([]))
    expected_keys = {
        "learning_rate",
        "temp",
        "curve_rate",
        "gamma_min",
        "gamma_max",
        "div_param",
    }
    assert set(kwargs.keys()) == expected_keys


def test_run_qqa_anneal_kwargs_match_qqa_hp_kwargs(bench):
    """Static contract: ``run_qqa_anneal`` must accept every key
    ``qqa_hp_kwargs`` produces (plus ``device, sol_size, num_epochs``).

    Reflection-based check: if a refactor adds a key to one without the
    other, this test fails immediately.
    """
    import inspect

    p = argparse.ArgumentParser()
    bench.add_qqa_hp_args(p)
    hp_keys = set(bench.qqa_hp_kwargs(p.parse_args([])).keys())

    sig = inspect.signature(bench.run_qqa_anneal)
    accepted = set(sig.parameters.keys()) - {"problem"}
    missing = hp_keys - accepted
    assert not missing, (
        f"run_qqa_anneal does not accept HP kwargs: {missing}. "
        "Update either add_qqa_hp_args/qqa_hp_kwargs or run_qqa_anneal "
        "to keep them in sync."
    )


def test_dump_results_json_creates_parent(bench, tmp_path):
    out = tmp_path / "nested" / "results.json"
    payload = {"hello": "world", "values": [1, 2, 3]}
    bench.dump_results_json(out, payload)
    assert out.is_file()
    assert json.loads(out.read_text()) == payload


def test_setup_device_resolves_auto(bench):
    """``auto`` must collapse to ``cpu`` (CUDA absent in CI by default)."""
    import torch

    args = argparse.Namespace(device="auto")
    resolved = bench.setup_device(args)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolved == expected
    assert args.device == expected


def test_setup_device_passes_through_explicit(bench):
    args = argparse.Namespace(device="cpu")
    assert bench.setup_device(args) == "cpu"
