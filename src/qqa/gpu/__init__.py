"""Optional GPU acceleration primitives with portable PyTorch fallbacks."""

from qqa.gpu.bitpack import pack_binary, packed_hamming_distance, unpack_binary
from qqa.gpu.cuda_graphs import CUDAGraphStep, cuda_graphs_available
from qqa.gpu.ops import sparse_qubo_energy_gradient

__all__ = [
    "CUDAGraphStep",
    "cuda_graphs_available",
    "pack_binary",
    "packed_hamming_distance",
    "sparse_qubo_energy_gradient",
    "unpack_binary",
]
