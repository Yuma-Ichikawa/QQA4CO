from __future__ import annotations

import pytest
import torch

from qqa.compile import SparseQUBO
from qqa.gpu import (
    CUDAGraphStep,
    cuda_graphs_available,
    pack_binary,
    packed_hamming_distance,
    sparse_qubo_energy_gradient,
    unpack_binary,
)
from qqa.gpu.ops import sparse_qubo_energy


@pytest.mark.parametrize("word_bits", [8, 16, 32])
def test_bitpack_roundtrip_and_hamming(word_bits: int):
    values = torch.tensor([[0, 1, 1, 0, 1], [1, 1, 0, 0, 1]], dtype=torch.bool)
    packed = pack_binary(values, word_bits=word_bits)
    assert torch.equal(unpack_binary(packed, 5, word_bits=word_bits), values)
    assert packed_hamming_distance(packed[0], packed[1]).item() == 2


def test_bitpack_rejects_nonbinary_values():
    with pytest.raises(ValueError, match="binary"):
        pack_binary(torch.tensor([0.0, 0.5, 1.0]))


def test_sparse_custom_energy_gradient_and_autograd_match_reference():
    qubo = SparseQUBO(
        torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64),
        torch.tensor([[0, 0, 1], [1, 2, 2]]),
        torch.tensor([1.25, -0.75, 0.5], dtype=torch.float64),
        0.2,
    )
    values = torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.1, 0.6]], dtype=torch.float64)
    energy, gradient = sparse_qubo_energy_gradient(
        values,
        qubo.linear,
        qubo.edge_index,
        qubo.edge_weight,
        qubo.constant,
        implementation="torch",
    )
    assert torch.allclose(energy, qubo.energy(values))
    assert torch.allclose(gradient, qubo.gradient(values))
    differentiable = values.clone().requires_grad_(True)
    sparse_qubo_energy(
        differentiable,
        qubo.linear,
        qubo.edge_index,
        qubo.edge_weight,
        qubo.constant,
        implementation="torch",
    ).sum().backward()
    assert torch.allclose(differentiable.grad, qubo.gradient(values))


def test_registered_sparse_operator_passes_opcheck():
    if not hasattr(torch.library, "opcheck"):
        pytest.skip("torch.library.opcheck is unavailable in this PyTorch release.")
    values = torch.rand(2, 4, dtype=torch.float64, requires_grad=True)
    linear = torch.rand(4, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 0, 1], [1, 3, 2]])
    weights = torch.rand(3, dtype=torch.float64, requires_grad=True)
    torch.library.opcheck(
        torch.ops.qqa4co.sparse_qubo_energy_gradient.default,
        (values, linear, edges, weights, 0.25),
    )


def test_cuda_graph_step_reports_unavailable_on_cpu():
    if cuda_graphs_available():
        pytest.skip("CUDA-specific replay is covered by the nightly test below.")
    with pytest.raises(RuntimeError, match="CUDA Graphs"):
        CUDAGraphStep(lambda value: value + 1, (torch.ones(2),))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_step_replays_static_function():
    device = torch.device("cuda")
    graph = CUDAGraphStep(lambda value: value.square() + 2, (torch.ones(8, device=device),))
    output = graph.replay(torch.arange(8, device=device, dtype=torch.float32), clone_output=True)
    assert torch.equal(output.cpu(), torch.arange(8, dtype=torch.float32).square() + 2)
