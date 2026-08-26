"""Safe opt-in CUDA Graph capture for static tensor callables."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

import torch

T = TypeVar("T", torch.Tensor, tuple[torch.Tensor, ...])


def cuda_graphs_available() -> bool:
    return bool(torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph"))


class CUDAGraphStep(Generic[T]):
    """Capture and replay a fixed-shape CUDA tensor function.

    Inputs are copied into persistent buffers before replay.  Outputs remain
    graph-owned tensors; callers that retain them across subsequent replays
    must request ``clone_output=True``.
    """

    def __init__(
        self,
        function: Callable[..., T],
        example_inputs: tuple[torch.Tensor, ...],
        *,
        warmup: int = 3,
        state_tensors: tuple[torch.Tensor, ...] = (),
    ) -> None:
        if not cuda_graphs_available():
            raise RuntimeError("CUDA Graphs require an available CUDA device.")
        if not example_inputs or any(not item.is_cuda for item in example_inputs):
            raise ValueError("example_inputs must be a non-empty tuple of CUDA tensors.")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("warmup must be a positive integer.")
        self._function = function
        self._inputs = tuple(item.detach().clone() for item in example_inputs)
        if any(item.device != self._inputs[0].device for item in state_tensors):
            raise ValueError("state_tensors must share the captured CUDA device.")
        state_snapshots = tuple(item.detach().clone() for item in state_tensors)
        stream = torch.cuda.Stream(device=self._inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream(self._inputs[0].device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                function(*self._inputs)
        torch.cuda.current_stream(self._inputs[0].device).wait_stream(stream)
        with torch.no_grad():
            for state, snapshot in zip(state_tensors, state_snapshots, strict=True):
                state.copy_(snapshot)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._output = function(*self._inputs)
        with torch.no_grad():
            for state, snapshot in zip(state_tensors, state_snapshots, strict=True):
                state.copy_(snapshot)

    def replay(self, *inputs: torch.Tensor, clone_output: bool = False) -> T:
        if len(inputs) != len(self._inputs):
            raise ValueError("Replay input count differs from capture input count.")
        for source, target in zip(inputs, self._inputs, strict=True):
            if (
                source.shape != target.shape
                or source.dtype != target.dtype
                or source.device != target.device
            ):
                raise ValueError("Replay inputs must preserve captured shape, dtype, and device.")
            target.copy_(source)
        self._graph.replay()
        if not clone_output:
            return self._output
        if torch.is_tensor(self._output):
            return self._output.clone()
        return tuple(item.clone() for item in self._output)


__all__ = ["CUDAGraphStep", "cuda_graphs_available"]
