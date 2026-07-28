"""Typed variable declarations and tensor packing for mixed optimisation."""

from __future__ import annotations

import keyword
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

import torch

VariableKind = Literal["binary", "integer", "real"]


class VariableSpec(Protocol):
    """Structural type shared by all variable declarations."""

    name: str
    size: int
    kind: VariableKind
    lower: float
    upper: float


def _validate_name_and_size(name: str, size: int) -> None:
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"Variable name must be a non-keyword Python identifier, got {name!r}.")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"Variable size must be a positive integer, got {size!r}.")


@dataclass(frozen=True, slots=True)
class BinaryVariable:
    """One or more variables in ``{0, 1}``."""

    name: str
    size: int = 1
    kind: ClassVar[Literal["binary"]] = "binary"
    lower: ClassVar[float] = 0.0
    upper: ClassVar[float] = 1.0

    def __post_init__(self) -> None:
        _validate_name_and_size(self.name, self.size)


@dataclass(frozen=True, slots=True)
class IntegerVariable:
    """One or more bounded integer variables."""

    name: str
    lower: int
    upper: int
    size: int = 1
    kind: ClassVar[Literal["integer"]] = "integer"

    def __post_init__(self) -> None:
        _validate_name_and_size(self.name, self.size)
        if (
            not isinstance(self.lower, int)
            or isinstance(self.lower, bool)
            or not isinstance(self.upper, int)
            or isinstance(self.upper, bool)
        ):
            raise TypeError("IntegerVariable bounds must be integers.")
        if self.lower >= self.upper:
            raise ValueError(
                f"IntegerVariable lower must be < upper, got [{self.lower}, {self.upper}]."
            )


@dataclass(frozen=True, slots=True)
class RealVariable:
    """One or more bounded real-valued variables."""

    name: str
    lower: float
    upper: float
    size: int = 1
    kind: ClassVar[Literal["real"]] = "real"

    def __post_init__(self) -> None:
        _validate_name_and_size(self.name, self.size)
        lower = float(self.lower)
        upper = float(self.upper)
        if not torch.isfinite(torch.tensor([lower, upper])).all().item():
            raise ValueError("RealVariable bounds must be finite.")
        if lower >= upper:
            raise ValueError(
                f"RealVariable lower must be < upper, got [{self.lower}, {self.upper}]."
            )


class VariableSpace:
    """Flatten typed variables into a stable tensor layout.

    The solver works on a dense ``(..., D)`` tensor for GPU efficiency while
    user objectives receive a mapping such as ``{"units": tensor, ...}``.
    """

    def __init__(self, variables: list[VariableSpec] | tuple[VariableSpec, ...]):
        if not variables:
            raise ValueError("At least one variable must be declared.")
        self.variables = tuple(variables)
        names = [variable.name for variable in self.variables]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Variable names must be unique; duplicates: {duplicates}.")

        self._slices: dict[str, slice] = {}
        lower: list[float] = []
        upper: list[float] = []
        kinds: list[VariableKind] = []
        offset = 0
        for variable in self.variables:
            stop = offset + variable.size
            self._slices[variable.name] = slice(offset, stop)
            lower.extend([float(variable.lower)] * variable.size)
            upper.extend([float(variable.upper)] * variable.size)
            kinds.extend([variable.kind] * variable.size)
            offset = stop

        self.dimension = offset
        self.kinds = tuple(kinds)
        # Keep canonical bounds in float64. Casting a large integer bound to
        # float32 here would irreversibly round it even when a model explicitly
        # requests float64 later.
        self._lower_cpu = torch.tensor(lower, dtype=torch.float64)
        self._upper_cpu = torch.tensor(upper, dtype=torch.float64)
        self._bounds_cache: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        self._discrete_indices = [index for index, kind in enumerate(self.kinds) if kind != "real"]

    def _bounds_like(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(tensor.device), tensor.dtype)
        if key not in self._bounds_cache:
            self._bounds_cache[key] = (
                self._lower_cpu.to(device=tensor.device, dtype=tensor.dtype),
                self._upper_cpu.to(device=tensor.device, dtype=tensor.dtype),
            )
        return self._bounds_cache[key]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Map normalised solver coordinates from ``[0, 1]`` to user units."""
        self._check_last_dimension(latent)
        lower, upper = self._bounds_like(latent)
        return lower + (upper - lower) * latent.clamp(0.0, 1.0)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Map user-unit values into normalised solver coordinates."""
        self._check_last_dimension(values)
        lower, upper = self._bounds_like(values)
        return ((values - lower) / (upper - lower)).clamp(0.0, 1.0)

    def project(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor and enforce every declared variable domain."""
        values = self.decode(latent)
        if self._discrete_indices:
            values = values.clone()
            values[..., self._discrete_indices] = values[..., self._discrete_indices].round()
        return values

    def unpack(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return named zero-copy views into a user-unit tensor."""
        self._check_last_dimension(values)
        named: dict[str, torch.Tensor] = {}
        for variable in self.variables:
            part = values[..., self._slices[variable.name]]
            named[variable.name] = part.squeeze(-1) if variable.size == 1 else part
        return named

    def pack(
        self,
        values: dict[str, float | int | list[float] | torch.Tensor],
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Pack one named solution into the solver's stable flat layout."""
        missing = sorted(set(self._slices) - set(values))
        unknown = sorted(set(values) - set(self._slices))
        if missing or unknown:
            raise ValueError(f"Invalid solution keys; missing={missing}, unknown={unknown}.")
        parts: list[torch.Tensor] = []
        for variable in self.variables:
            part = torch.as_tensor(values[variable.name], device=device, dtype=dtype).reshape(-1)
            if part.numel() != variable.size:
                raise ValueError(
                    f"{variable.name!r} expects {variable.size} value(s), got {part.numel()}."
                )
            parts.append(part)
        packed = torch.cat(parts)
        self.validate(packed)
        return packed

    def validate(self, values: torch.Tensor, *, atol: float = 1e-6) -> None:
        """Validate shape, bounds, and integrality of a user-unit solution."""
        self._check_last_dimension(values)
        lower, upper = self._bounds_like(values)
        if torch.any(values < lower - atol) or torch.any(values > upper + atol):
            raise ValueError("Solution contains values outside declared bounds.")
        if self._discrete_indices and not torch.allclose(
            values[..., self._discrete_indices],
            values[..., self._discrete_indices].round(),
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError("Binary/integer solution values must be integral.")

    def describe(self) -> list[dict[str, object]]:
        """Return JSON-friendly variable metadata in tensor-column order."""
        rows: list[dict[str, object]] = []
        for variable in self.variables:
            sl = self._slices[variable.name]
            rows.append(
                {
                    "name": variable.name,
                    "kind": variable.kind,
                    "size": variable.size,
                    "lower": variable.lower,
                    "upper": variable.upper,
                    "start": sl.start,
                    "stop": sl.stop,
                }
            )
        return rows

    def _check_last_dimension(self, tensor: torch.Tensor) -> None:
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(tensor).__name__}.")
        if tensor.ndim < 1 or tensor.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected tensor with last dimension {self.dimension}, "
                f"got shape {tuple(tensor.shape)}."
            )

    def __getstate__(self) -> dict:
        """Avoid serialising device-specific bound caches with a model."""
        state = self.__dict__.copy()
        state["_bounds_cache"] = {}
        return state


# Concise aliases for mathematical model declarations.
Binary = BinaryVariable
Integer = IntegerVariable
Real = RealVariable
