"""Noise-tolerant sparse-core regression gate for pull requests."""

from __future__ import annotations

import json
import statistics
from time import perf_counter

import torch

from qqa.compile import SparseQUBO


def _median_runtime(function, *, repeats: int = 7) -> float:
    function()
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        function()
        samples.append(perf_counter() - started)
    return statistics.median(samples)


def main() -> int:
    generator = torch.Generator().manual_seed(20260827)
    variables = 1024
    edges = 4096
    left = torch.randint(0, variables - 1, (edges,), generator=generator)
    right = torch.maximum(
        left + 1,
        torch.randint(1, variables, (edges,), generator=generator),
    ).clamp_max(variables - 1)
    edge_index = torch.stack((left, right))
    qubo = SparseQUBO(
        torch.randn(variables, generator=generator),
        edge_index,
        torch.randn(edges, generator=generator),
    )
    values = torch.rand((16, variables), generator=generator)
    dense = qubo.to_dense()
    dense.fill_diagonal_(0.0)

    def sparse_energy():
        return qubo.energy(values)

    def dense_energy():
        return (values * qubo.linear).sum(dim=-1) + torch.einsum(
            "bi,ij,bj->b", values, dense, values
        )

    sparse_result = sparse_energy()
    dense_result = dense_energy()
    if not torch.allclose(sparse_result, dense_result, atol=2e-4, rtol=2e-5):
        raise SystemExit("sparse/dense energy parity regression")
    sparse_time = _median_runtime(sparse_energy)
    dense_time = _median_runtime(dense_energy)
    sparse_storage = qubo.linear.numel() + qubo.edge_index.numel() + qubo.edge_weight.numel()
    dense_storage = dense.numel()
    payload = {
        "schema_version": 1,
        "variables": variables,
        "edges": edges,
        "median_sparse_seconds": sparse_time,
        "median_dense_seconds": dense_time,
        "runtime_ratio": sparse_time / max(dense_time, 1e-12),
        "storage_ratio": sparse_storage / dense_storage,
    }
    print(json.dumps(payload, sort_keys=True))
    # The broad runtime band tolerates noisy shared CI hosts; the structural
    # storage gate is deterministic and catches accidental dense fallback.
    if payload["runtime_ratio"] > 4.0 or payload["storage_ratio"] > 0.02:
        raise SystemExit("sparse-core performance regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
