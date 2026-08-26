"""Portable bit packing for binary populations and archive operations."""

from __future__ import annotations

import torch


def _validate_word_bits(word_bits: int) -> None:
    if isinstance(word_bits, bool) or not isinstance(word_bits, int) or not 1 <= word_bits <= 32:
        raise ValueError("word_bits must be an integer in [1, 32].")


def pack_binary(values: torch.Tensor, *, word_bits: int = 32) -> torch.Tensor:
    """Pack the final binary dimension into non-negative ``int64`` words.

    ``int64`` is used as the storage tensor because PyTorch implements its
    shift/XOR operations consistently on CPU and CUDA.  Limiting each word to
    32 payload bits avoids signed overflow and keeps the SWAR popcount exact.
    """
    _validate_word_bits(word_bits)
    tensor = torch.as_tensor(values)
    if tensor.ndim == 0 or tensor.shape[-1] == 0:
        raise ValueError("values must have a non-empty final binary dimension.")
    if tensor.dtype is torch.bool:
        bits = tensor.to(torch.int64)
    else:
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError("values must contain only binary 0/1 entries.")
        bits = tensor.to(torch.int64)
    size = bits.shape[-1]
    words = (size + word_bits - 1) // word_bits
    padded = torch.zeros(
        (*bits.shape[:-1], words * word_bits), dtype=torch.int64, device=bits.device
    )
    padded[..., :size] = bits
    grouped = padded.reshape(*bits.shape[:-1], words, word_bits)
    shifts = torch.arange(word_bits, dtype=torch.int64, device=bits.device)
    return (grouped << shifts).sum(dim=-1)


def unpack_binary(
    packed: torch.Tensor,
    size: int,
    *,
    word_bits: int = 32,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Invert :func:`pack_binary`, trimming padding to ``size`` bits."""
    _validate_word_bits(word_bits)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer.")
    words = torch.as_tensor(packed)
    if words.ndim == 0 or words.shape[-1] < (size + word_bits - 1) // word_bits:
        raise ValueError("packed does not contain enough words for size.")
    if words.dtype != torch.int64 or torch.any(words < 0):
        raise ValueError("packed must be a non-negative int64 tensor.")
    shifts = torch.arange(word_bits, dtype=torch.int64, device=words.device)
    bits = ((words.unsqueeze(-1) >> shifts) & 1).reshape(*words.shape[:-1], -1)[..., :size]
    return bits.to(dtype=dtype)


def _popcount32(words: torch.Tensor) -> torch.Tensor:
    value = words
    value = value - ((value >> 1) & 0x55555555)
    value = (value & 0x33333333) + ((value >> 2) & 0x33333333)
    value = (value + (value >> 4)) & 0x0F0F0F0F
    value = value + (value >> 8)
    value = value + (value >> 16)
    return value & 0x3F


def packed_hamming_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return Hamming distance over the final packed-word dimension."""
    lhs = torch.as_tensor(left)
    rhs = torch.as_tensor(right, device=lhs.device)
    if lhs.dtype != torch.int64 or rhs.dtype != torch.int64:
        raise TypeError("packed Hamming inputs must use int64 storage.")
    if lhs.shape[-1] != rhs.shape[-1]:
        raise ValueError("packed inputs must have the same word count.")
    return _popcount32(torch.bitwise_xor(lhs, rhs)).sum(dim=-1)


__all__ = ["pack_binary", "packed_hamming_distance", "unpack_binary"]
