"""Regression tests for portable artifacts and strict public runtime boundaries."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

import qqa
from qqa.benchmarking.registry import audit_benchmark_snapshot
from qqa.hybrid.feedback import ExactFeedbackBus, LinearCut
from qqa.model import VariableBlock
from qqa.result import (
    CandidateRecord,
    ConstraintReport,
    ConstraintViolation,
    CoordinateSpace,
    FeasibilityStatus,
    Provenance,
    ResourceReport,
    SolveResult,
    SolveStatus,
    TimingReport,
)
from qqa.runtime import (
    Checkpoint,
    export_result_package,
    load_checkpoint,
    save_checkpoint,
    verify_result_package,
)
from qqa.schedule import ExponentialBGSchedule, SigmoidBGSchedule


def _result() -> SolveResult:
    return SolveResult(
        SolveStatus.FEASIBLE,
        torch.tensor([0.0, 1.0]),
        0.0,
        0.0,
        0.0,
        True,
        ConstraintReport.unconstrained(),
        TimingReport(0.1),
        ResourceReport("cpu"),
        Provenance("qqa", 0, "fast"),
    )


def _rewrite_package(path: Path, updates: dict[str, object]) -> None:
    with zipfile.ZipFile(path) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    for name, value in updates.items():
        members[name] = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    replacement = path.with_suffix(".replacement")
    with zipfile.ZipFile(replacement, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    replacement.replace(path)


def test_result_package_manifest_must_cover_every_payload(tmp_path: Path) -> None:
    target = export_result_package(
        _result(),
        tmp_path / "result.qqapkg",
        model_summary={"name": "portable-model"},
        model_fingerprint="abc",
    )
    assert verify_result_package(target).result_status == "feasible"

    with zipfile.ZipFile(target) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    manifest["files"] = {}
    _rewrite_package(target, {"manifest.json": manifest})
    with pytest.raises(ValueError, match="cover every payload"):
        verify_result_package(target)


def test_result_package_revalidates_payload_and_manifest_status(tmp_path: Path) -> None:
    target = export_result_package(
        _result(),
        tmp_path / "result.qqapkg",
        model_summary={"name": "portable-model"},
        model_fingerprint="abc",
    )
    with zipfile.ZipFile(target) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    unsafe_summary = {"source": "/private/model.mps"}
    encoded = json.dumps(unsafe_summary, sort_keys=True, separators=(",", ":")).encode()
    manifest["files"]["model-summary.json"] = hashlib.sha256(encoded).hexdigest()
    _rewrite_package(
        target,
        {"model-summary.json": unsafe_summary, "manifest.json": manifest},
    )
    with pytest.raises(ValueError, match="local path or private endpoint"):
        verify_result_package(target)

    target = export_result_package(
        _result(),
        tmp_path / "status.qqapkg",
        model_summary={"name": "portable-model"},
        model_fingerprint="abc",
    )
    with zipfile.ZipFile(target) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    manifest["result_status"] = "optimal"
    _rewrite_package(target, {"manifest.json": manifest})
    with pytest.raises(ValueError, match="status does not match"):
        verify_result_package(target)


def test_feedback_bus_snapshots_do_not_share_mutable_state() -> None:
    coefficients = torch.tensor([1.0, -2.0])
    cut = LinearCut(torch.tensor([0, 1]), coefficients, "<=", 1.0)
    metadata = {"nested": {"iteration": 1}}
    bus = ExactFeedbackBus(maximum_cuts=1, maximum_no_goods=1)
    published = bus.publish(
        lp_primal=torch.tensor([0.25, 0.75]),
        cuts=(cut,),
        no_goods=((0, 1),),
        metadata=metadata,
    )

    coefficients[0] = 99.0
    cut.coefficients[1] = 99.0
    metadata["nested"]["iteration"] = 99
    published.lp_primal[0] = 99.0
    published.cuts[0].coefficients[0] = 99.0
    published.metadata["nested"]["iteration"] = 99

    snapshot = bus.snapshot()
    assert snapshot.version == 1
    assert snapshot.lp_primal.tolist() == pytest.approx([0.25, 0.75])
    assert snapshot.cuts[0].coefficients.tolist() == pytest.approx([1.0, -2.0])
    assert snapshot.metadata == {"nested": {"iteration": 1}}
    assert len(bus.constraints()) == 1
    assert len(bus.no_good_factors()) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maximum_cuts": 1.5}, "positive"),
        ({"maximum_no_goods": True}, "positive"),
    ],
)
def test_feedback_bus_rejects_ambiguous_bounds(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExactFeedbackBus(**kwargs)


def test_feedback_bus_rejects_nonbinary_no_goods_and_nonfinite_cuts() -> None:
    bus = ExactFeedbackBus()
    with pytest.raises(ValueError, match="zeros and ones"):
        bus.publish(no_goods=((0, 2),))
    with pytest.raises(ValueError, match="finite"):
        LinearCut(torch.tensor([0]), torch.tensor([math.nan]), "<=", 1.0)
    with pytest.raises(ValueError, match="indices must be integers"):
        LinearCut(torch.tensor([0.5]), torch.tensor([1.0]), "<=", 1.0)
    with pytest.raises(ValueError, match="finite"):
        LinearCut(torch.tensor([0]), torch.tensor([1.0]), "<=", True)
    with pytest.raises(TypeError, match="lp_primal"):
        bus.publish(lp_primal=[0.25, 0.75])

    snapshot = bus.publish(no_goods=((np.int64(0), np.int64(1)),))
    assert snapshot.no_goods == ((0, 1),)


@pytest.mark.parametrize("schedule_type", [ExponentialBGSchedule, SigmoidBGSchedule])
@pytest.mark.parametrize("shape", [1e-300, 1e6])
def test_extreme_finite_schedule_shapes_remain_finite(schedule_type, shape: float) -> None:
    schedule = schedule_type(-2.0, 0.5, shape)
    values = [schedule(epoch, 11) for epoch in range(11)]
    assert all(math.isfinite(value) for value in values)
    assert values[0] == pytest.approx(-2.0)
    assert values[-1] == pytest.approx(0.5)
    assert values == sorted(values)


@pytest.mark.parametrize(
    "options",
    [
        {"curve_rate": 2.0},
        {"polish": 1},
        {"memory_fraction": True},
        {"device": ""},
        {"schedule": ""},
    ],
)
def test_solver_config_rejects_ambiguous_public_values(options: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        qqa.SolverConfig(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"record_history": 1},
        {"return_population": 0},
        {"learning_rate": True},
        {"schedule": "linear"},
    ],
)
def test_anneal_rejects_ambiguous_public_values(options: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        qqa.anneal(qqa.MaxCut(__import__("networkx").cycle_graph(3)), num_epochs=0, **options)


@pytest.mark.parametrize("value", [True, math.nan, math.inf, "not-a-number"])
def test_anneal_rejects_invalid_schedule_outputs(value: object) -> None:
    with pytest.raises(ValueError, match="schedule must return"):
        qqa.anneal(
            qqa.MaxCut(__import__("networkx").cycle_graph(3)),
            schedule=lambda _epoch, _epochs: value,
            sol_size=2,
            num_epochs=0,
            record_history=False,
            verbose=False,
        )


def test_anneal_evaluates_a_stateful_schedule_once_per_epoch() -> None:
    calls: list[tuple[int, int]] = []

    def schedule(epoch: int, num_epochs: int) -> float:
        calls.append((epoch, num_epochs))
        return float(epoch)

    qqa.anneal(
        qqa.MaxCut(__import__("networkx").cycle_graph(3)),
        schedule=schedule,
        sol_size=2,
        num_epochs=2,
        record_history=False,
        verbose=False,
        polish=False,
    )
    assert calls == [(0, 2), (1, 2)]


def test_variable_categories_require_an_integer() -> None:
    with pytest.raises(ValueError, match="integer"):
        VariableBlock("choice", "categorical", (2,), categories=2.5)


def test_result_contract_rejects_ambiguous_types() -> None:
    with pytest.raises(TypeError, match="selected"):
        CandidateRecord(
            "raw",
            "solver",
            CoordinateSpace.ORIGINAL,
            1.0,
            FeasibilityStatus.FEASIBLE,
            selected=1,
        )
    with pytest.raises(TypeError, match="satisfied"):
        ConstraintViolation("row", 0.0, 0.0, 1e-6, 1)
    with pytest.raises(ValueError, match="integers"):
        ResourceReport("cpu", peak_host_memory_bytes=1.5)
    with pytest.raises(ValueError, match="profile"):
        Provenance("qqa", 0, "")

    result = _result()
    result.feasible = 1
    with pytest.raises(TypeError, match="feasible"):
        result.__post_init__()


def test_checkpoint_manifest_rejects_ambiguous_schema_and_checksum(tmp_path: Path) -> None:
    target = save_checkpoint(
        Checkpoint("abc", {"profile": "fast"}, 0, {"x": torch.arange(2)}, {}),
        tmp_path / "state.qqacp",
    )
    with zipfile.ZipFile(target) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
        manifest = json.loads(members["manifest.json"])

    manifest["schema_version"] = True
    members["manifest.json"] = json.dumps(manifest).encode()
    replacement = target.with_suffix(".replacement")
    with zipfile.ZipFile(replacement, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    replacement.replace(target)
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(target)

    manifest["schema_version"] = 1
    manifest["tensor_checksums"]["x"] = "not-a-checksum"
    members["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(replacement, "w") as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    replacement.replace(target)
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(target)


def test_failed_benchmark_download_removes_unique_temporary_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    download_module = importlib.import_module("qqa.benchmarking.download")

    class FailingOpener:
        def open(self, request, *, timeout):
            del request, timeout
            raise OSError("network unavailable")

    monkeypatch.setattr(
        download_module.urllib.request,
        "build_opener",
        lambda *_handlers: FailingOpener(),
    )
    target = tmp_path / "benchmark.zip"
    with pytest.raises(RuntimeError, match="public benchmark file"):
        download_module._download(
            "https://example.org/benchmark.zip",
            target,
            overwrite=True,
        )
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))


def test_benchmark_redirect_handler_rejects_cross_origin() -> None:
    download_module = importlib.import_module("qqa.benchmarking.download")
    handler = download_module._SameOriginRedirectHandler(("example.org", 443))
    with pytest.raises(download_module.urllib.error.HTTPError, match="Cross-origin"):
        handler.redirect_request(
            None,
            None,
            302,
            "redirect",
            {},
            "https://other.example/benchmark.zip",
        )


@pytest.mark.parametrize("name", ["../outside.txt", "folder\\outside.txt", "/outside.txt"])
def test_benchmark_archive_rejects_nonportable_members(tmp_path: Path, name: str) -> None:
    download_module = importlib.import_module("qqa.benchmarking.download")
    archive = tmp_path / "benchmark.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name, b"payload")
    with pytest.raises(ValueError, match="Unsafe or duplicate"):
        download_module._safe_extract(archive, tmp_path / "extract")


@pytest.mark.parametrize(
    "url",
    [
        "http://miplib.zib.de/downloads/benchmark.zip",
        "https://user@miplib.zib.de/downloads/benchmark.zip",
        "https://miplib.zib.de/downloads/benchmark.zip?token=value",
        "https://miplib.zib.de:444/downloads/benchmark.zip",
    ],
)
def test_snapshot_audit_requires_unambiguous_official_https_origin(
    tmp_path: Path,
    url: str,
) -> None:
    source = tmp_path / "benchmark.zip"
    source.write_bytes(b"benchmark")
    payload = {
        "library": "miplib",
        "snapshot": "MIPLIB-2017-benchmark-v2/solu-v36",
        "files": [
            {
                "name": source.name,
                "url": url,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "size": source.stat().st_size,
            }
        ],
    }
    (tmp_path / "snapshot.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="official origin"):
        audit_benchmark_snapshot(tmp_path)
