"""Nightly CUDA contract checks; skipped on ordinary CPU CI runners."""

from __future__ import annotations

import gc
import math

import networkx as nx
import pytest
import torch

import qqa
from qqa.compile import SparseQUBO

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


def test_cuda_sparse_energy_and_gradient_match_cpu():
    generator = torch.Generator().manual_seed(7)
    model = SparseQUBO(
        torch.randn(32, generator=generator),
        torch.randint(0, 32, (2, 96), generator=generator),
        torch.randn(96, generator=generator),
    )
    values = torch.rand((8, 32), generator=generator)
    cuda_model = model.to("cuda")
    torch.testing.assert_close(cuda_model.energy(values.cuda()).cpu(), model.energy(values))
    torch.testing.assert_close(cuda_model.gradient(values.cuda()).cpu(), model.gradient(values))


def test_cuda_qqa_compile_and_bfloat16_paths_are_finite():
    problem = qqa.MaxCut(nx.path_graph(12), device="cuda")
    qqa.fix_seed(11)
    compiled = qqa.anneal(
        problem,
        sol_size=8,
        num_epochs=4,
        learning_rate=0.05,
        compile_core=True,
        polish=False,
        verbose=False,
    )
    qqa.fix_seed(11)
    reduced = qqa.anneal(
        problem,
        sol_size=8,
        num_epochs=4,
        learning_rate=0.05,
        mixed_precision="bf16",
        polish=False,
        verbose=False,
    )
    assert math.isfinite(compiled.best_obj)
    assert math.isfinite(reduced.best_obj)


def test_cuda_reproducible_profile_and_no_live_allocation_growth():
    problem = qqa.MaximumIndependentSet(nx.path_graph(10), device="cuda")
    torch.cuda.empty_cache()
    initial_allocated = torch.cuda.memory_allocated()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        results = []
        for _ in range(2):
            results.append(
                qqa.solve(
                    problem,
                    profile="reproducible",
                    replicas=8,
                    epochs=5,
                    polish=False,
                    device="cuda",
                    seed=23,
                )
            )
        assert results[0].best_obj == results[1].best_obj
        assert torch.equal(results[0].solution, results[1].solution)
        del results
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=True)
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= initial_allocated + 1_048_576
