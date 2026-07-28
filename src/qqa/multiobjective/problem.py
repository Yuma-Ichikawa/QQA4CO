"""Declarative multi-objective mixed-variable models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from qqa.mixed.problem import BatchFunction, Constraint, MixedProblem, _batch_vector
from qqa.mixed.variables import VariableSpec

Direction = Literal["min", "max"]


@dataclass(frozen=True, slots=True)
class Objective:
    """One named objective and its optimisation direction."""

    function: BatchFunction
    name: str
    direction: Direction = "min"
    unit: str = ""

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("Objective function must be callable.")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Objective name must be a non-empty string.")
        if self.direction not in ("min", "max"):
            raise ValueError("Objective direction must be 'min' or 'max'.")


class MultiObjectiveProblem(MixedProblem):
    """Mixed binary/integer/real model with two or more objectives."""

    def __init__(
        self,
        variables: Sequence[VariableSpec],
        objectives: Sequence[Objective],
        *,
        constraints: Sequence[Constraint] = (),
        name: str = "multi-objective-problem",
        dtype: torch.dtype = torch.float32,
    ):
        objectives = tuple(objectives)
        if len(objectives) < 2:
            raise ValueError("At least two objectives are required.")
        names = [objective.name for objective in objectives]
        if len(set(names)) != len(names):
            raise ValueError("Objective names must be unique.")
        self.objectives = objectives
        super().__init__(
            variables,
            objectives[0].function,
            constraints=constraints,
            name=name,
            objective_label=objectives[0].name,
            objective_unit=objectives[0].unit,
            dtype=dtype,
        )

    @property
    def num_objectives(self) -> int:
        return len(self.objectives)

    def objective_matrix(self, values: torch.Tensor, *, minimize: bool = False) -> torch.Tensor:
        """Return shape ``(population, objectives)``.

        With ``minimize=True``, maximisation columns are sign-flipped so all
        columns follow a common minimisation convention.
        """
        values = self._ensure_batched(values)
        named = self.space.unpack(values)
        columns = [
            _batch_vector(
                objective.function(named),
                batch_size=values.shape[0],
                label=f"objective {objective.name!r}",
                like=values,
            )
            for objective in self.objectives
        ]
        matrix = torch.stack(columns, dim=1)
        if minimize:
            signs = matrix.new_tensor(
                [1.0 if objective.direction == "min" else -1.0 for objective in self.objectives]
            )
            matrix = matrix * signs
        return matrix

    def loss_fn(self, values: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """Reject accidental single-objective annealing."""
        raise TypeError(
            "MultiObjectiveProblem has no single loss. Use problem.solve_pareto() "
            "or qqa.pareto_anneal(problem)."
        )

    def solve_pareto(self, **kwargs):
        """Find a Pareto front with one parallel QQA run."""
        from qqa.multiobjective.solver import pareto_anneal

        return pareto_anneal(self, **kwargs)


__all__ = ["Direction", "MultiObjectiveProblem", "Objective"]
