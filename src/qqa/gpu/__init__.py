"""Optional GPU acceleration primitives with portable PyTorch fallbacks."""

from qqa.gpu.bitpack import pack_binary, packed_hamming_distance, unpack_binary
from qqa.gpu.cuda_graphs import CUDAGraphStep, cuda_graphs_available
from qqa.gpu.factors import CompiledFactorGraph, compile_factor_graph, segmented_sum
from qqa.gpu.local_search import (
    GPULocalSearchResult,
    binary_flip_delta,
    gpu_k_flip_search,
    gpu_two_opt,
)
from qqa.gpu.ops import sparse_qubo_energy_gradient
from qqa.gpu.repair import assignment_repair, exact_k_repair, one_hot_repair
from qqa.gpu.telemetry import KernelAutotuner, KernelProfile, profile_kernel

__all__ = [
    "CUDAGraphStep",
    "CompiledFactorGraph",
    "GPULocalSearchResult",
    "KernelAutotuner",
    "KernelProfile",
    "assignment_repair",
    "binary_flip_delta",
    "compile_factor_graph",
    "cuda_graphs_available",
    "pack_binary",
    "exact_k_repair",
    "gpu_k_flip_search",
    "gpu_two_opt",
    "one_hot_repair",
    "packed_hamming_distance",
    "sparse_qubo_energy_gradient",
    "segmented_sum",
    "profile_kernel",
    "unpack_binary",
]
