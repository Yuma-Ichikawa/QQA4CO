"""Persistent torch.export cache tests for sparse QUBO evaluation."""

from __future__ import annotations

import json

import torch

from qqa.compile import SparseQUBO, compile_sparse_qubo_aot


def _qubo() -> SparseQUBO:
    return SparseQUBO(
        torch.tensor([-1.0, 0.5, 0.25]),
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([2.0, -0.75]),
        0.125,
    )


def test_export_cache_has_eager_parity_and_reuses_artifact(tmp_path) -> None:
    qubo = _qubo()
    example = torch.rand((2, 3), requires_grad=True)
    first = compile_sparse_qubo_aot(qubo, example, cache_dir=tmp_path)
    assert first.cache_hit is False
    assert torch.allclose(first(example), qubo.energy(example))
    first(example).sum().backward()
    assert example.grad is not None
    assert torch.isfinite(example.grad).all()

    second = qubo.compile_aot(torch.rand((1, 3)), cache_dir=tmp_path)
    assert second.cache_hit is True
    assert second.key == first.key
    values = torch.rand((4, 3))
    assert torch.allclose(second(values), qubo.energy(values))

    manifest = json.loads(first.artifact.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest == {"backend": "export", "format": 1, "key": first.key}


def test_export_cache_dynamic_batch_parity(tmp_path) -> None:
    qubo = _qubo()
    artifact = qubo.compile_aot(torch.rand((1, 3)), cache_dir=tmp_path)
    values = torch.rand((4, 3))
    assert torch.allclose(artifact(values), qubo.energy(values))
