"""Adaptive node-local encodings for bounded general integers."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import numpy as np

EncodingKind = Literal["binary", "categorical", "order", "radix", "local"]


@dataclass(frozen=True, slots=True)
class IntegerEncodingPlan:
    lower: int
    upper: int
    kind: EncodingKind
    radices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError("Integer encoding requires lower < upper.")
        if self.kind == "radix" and (not self.radices or prod(self.radices) < self.cardinality):
            raise ValueError("Radices do not cover the integer domain.")

    @property
    def cardinality(self) -> int:
        return self.upper - self.lower + 1

    @property
    def encoded_size(self) -> int:
        if self.kind == "binary":
            return 1
        if self.kind == "categorical":
            return self.cardinality
        if self.kind == "order":
            return self.cardinality - 1
        if self.kind == "radix":
            return sum(self.radices)
        return 1


def choose_integer_encoding(
    lower: int,
    upper: int,
    *,
    local_lower: int | None = None,
    local_upper: int | None = None,
    categorical_limit: int = 8,
    order_limit: int = 32,
    radix: int = 8,
) -> IntegerEncodingPlan:
    """Choose an encoding after applying SCIP's node-local domain."""
    effective_lower = max(lower, local_lower) if local_lower is not None else lower
    effective_upper = min(upper, local_upper) if local_upper is not None else upper
    if effective_lower >= effective_upper:
        raise ValueError("Effective integer domain must contain at least two values.")
    cardinality = effective_upper - effective_lower + 1
    if effective_lower == 0 and effective_upper == 1:
        return IntegerEncodingPlan(effective_lower, effective_upper, "binary")
    if cardinality <= categorical_limit:
        return IntegerEncodingPlan(effective_lower, effective_upper, "categorical")
    if cardinality <= order_limit:
        return IntegerEncodingPlan(effective_lower, effective_upper, "order")
    remaining = cardinality
    radices: list[int] = []
    coverage = 1
    while coverage < remaining:
        radices.append(radix)
        coverage *= radix
    return IntegerEncodingPlan(effective_lower, effective_upper, "radix", tuple(radices))


def encode_integer(value: int, plan: IntegerEncodingPlan) -> np.ndarray:
    if value < plan.lower or value > plan.upper:
        raise ValueError("Integer value is outside encoding domain.")
    offset = value - plan.lower
    if plan.kind in {"binary", "local"}:
        return np.asarray([offset], dtype=np.float64)
    if plan.kind == "categorical":
        encoded = np.zeros(plan.cardinality, dtype=np.float64)
        encoded[offset] = 1.0
        return encoded
    if plan.kind == "order":
        encoded = np.zeros(plan.cardinality - 1, dtype=np.float64)
        encoded[:offset] = 1.0
        return encoded
    digits = []
    remaining = offset
    for radix in plan.radices:
        digit = remaining % radix
        remaining //= radix
        one_hot = np.zeros(radix, dtype=np.float64)
        one_hot[digit] = 1.0
        digits.append(one_hot)
    return np.concatenate(digits)


def decode_integer(encoded: np.ndarray, plan: IntegerEncodingPlan) -> int:
    values = np.asarray(encoded, dtype=np.float64)
    if values.shape != (plan.encoded_size,):
        raise ValueError("Encoded vector has the wrong size.")
    if plan.kind in {"binary", "local"}:
        offset = int(round(float(values[0])))
    elif plan.kind == "categorical":
        offset = int(np.argmax(values))
    elif plan.kind == "order":
        offset = int(np.count_nonzero(values >= 0.5))
    else:
        offset = 0
        multiplier = 1
        start = 0
        for radix in plan.radices:
            offset += int(np.argmax(values[start : start + radix])) * multiplier
            multiplier *= radix
            start += radix
    return min(plan.upper, plan.lower + offset)


__all__ = [
    "EncodingKind",
    "IntegerEncodingPlan",
    "choose_integer_encoding",
    "decode_integer",
    "encode_integer",
]
