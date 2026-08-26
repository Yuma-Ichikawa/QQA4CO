"""Online linear-UCB solver selector with portable numeric state."""

from __future__ import annotations

import math

import torch

from qqa.model import ModelIR, VariableDomain


def model_features(model: ModelIR) -> torch.Tensor:
    """Return scale-stable structural features for adaptive backend selection."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
    factors = len(model.objective.factors) + sum(
        len(row.expression.factors) for row in model.constraints
    )
    discrete = sum(
        block.size
        for block in model.variables
        if block.domain
        in {
            VariableDomain.BINARY,
            VariableDomain.SPIN,
            VariableDomain.INTEGER,
            VariableDomain.CATEGORICAL,
            VariableDomain.PERMUTATION,
        }
    )
    return torch.tensor(
        [
            1.0,
            math.log1p(model.num_variables),
            math.log1p(factors),
            math.log1p(len(model.constraints)),
            discrete / model.num_variables,
        ],
        dtype=torch.float64,
    )


class OnlineSolverSelector:
    """Per-backend ridge regressors with an upper-confidence exploration term."""

    def __init__(
        self,
        backends: tuple[str, ...] | list[str],
        *,
        num_features: int = 5,
        ridge: float = 1.0,
        exploration: float = 0.5,
    ) -> None:
        names = tuple(backends)
        if not names or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("backends must contain unique non-empty names.")
        if num_features < 1 or ridge <= 0 or exploration < 0:
            raise ValueError("num_features/ridge must be positive and exploration non-negative.")
        self.backends = names
        self.num_features = num_features
        self.exploration = float(exploration)
        self._covariance = {
            name: ridge * torch.eye(num_features, dtype=torch.float64) for name in names
        }
        self._reward = {name: torch.zeros(num_features, dtype=torch.float64) for name in names}
        self._observations = {name: 0 for name in names}

    def scores(self, features: torch.Tensor) -> dict[str, float]:
        vector = torch.as_tensor(features, dtype=torch.float64).reshape(-1)
        if vector.numel() != self.num_features or not bool(torch.isfinite(vector).all()):
            raise ValueError("features must be a finite aligned vector.")
        result: dict[str, float] = {}
        for name in self.backends:
            inverse_vector = torch.linalg.solve(self._covariance[name], vector)
            estimate = torch.dot(inverse_vector, self._reward[name])
            confidence = torch.sqrt(torch.dot(vector, inverse_vector).clamp_min(0))
            result[name] = float((estimate + self.exploration * confidence).item())
        return result

    def select(self, features: torch.Tensor) -> str:
        scores = self.scores(features)
        return max(self.backends, key=lambda name: (scores[name], -self.backends.index(name)))

    def update(self, backend: str, features: torch.Tensor, reward: float) -> None:
        if backend not in self._covariance:
            raise KeyError(f"Unknown backend {backend!r}.")
        vector = torch.as_tensor(features, dtype=torch.float64).reshape(-1)
        if vector.numel() != self.num_features or not bool(torch.isfinite(vector).all()):
            raise ValueError("features must be a finite aligned vector.")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite.")
        self._covariance[backend] += torch.outer(vector, vector)
        self._reward[backend] += float(reward) * vector
        self._observations[backend] += 1

    @property
    def observations(self) -> dict[str, int]:
        return dict(self._observations)


__all__ = ["OnlineSolverSelector", "model_features"]
