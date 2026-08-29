"""Study/Trial orchestration backed by the QQA black-box optimiser."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from qqa.blackbox.evaluation import EvaluationDatabase
from qqa.blackbox.problem import BlackBoxProblem
from qqa.blackbox.solver import BlackBoxResult, blackbox_optimize
from qqa.runtime.security import validate_portable_payload


class TrialState(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    PRUNED = "pruned"
    FAILED = "failed"


@dataclass(slots=True)
class Trial:
    number: int
    point: dict[str, Any]
    packed: torch.Tensor
    state: TrialState = TrialState.RUNNING
    value: float | None = None
    violations: tuple[float, ...] = ()
    seed: int = 0
    fidelity: str = "default"
    replicate: int = 0
    user_attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool | None:
        if self.state is not TrialState.COMPLETE:
            return None
        return all(value <= 1e-10 for value in self.violations)


class Study:
    """A resumable campaign whose acquisition batches are selected by QQA."""

    def __init__(
        self,
        problem: BlackBoxProblem,
        *,
        name: str | None = None,
        storage: EvaluationDatabase | str | Path | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(problem, BlackBoxProblem):
            raise TypeError("problem must be a BlackBoxProblem.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        self.problem = problem
        self.name = name or problem.name
        if not self.name:
            raise ValueError("Study name must be non-empty.")
        validate_portable_payload(self.name)
        self.storage = (
            storage
            if isinstance(storage, EvaluationDatabase) or storage is None
            else EvaluationDatabase(storage)
        )
        self.seed = seed
        self.trials: list[Trial] = []
        self.result: BlackBoxResult | None = None

    def optimize(self, *, budget: int, **kwargs: Any) -> BlackBoxResult:
        """Run or resume the campaign with QQA diverse-batch acquisition."""
        options = dict(kwargs)
        options.setdefault("acquisition_optimizer", "qqa")
        options.setdefault("seed", self.seed)
        if self.storage is not None:
            options.setdefault("evaluation_database", self.storage)
        if self.result is not None:
            options.setdefault("resume_from", self.result)
        result = blackbox_optimize(self.problem, budget=budget, **options)
        self.result = result
        self.trials = []
        for number, (packed, value, violations) in enumerate(
            zip(result.points, result.values, result.violations, strict=True)
        ):
            self.trials.append(
                Trial(
                    number,
                    self.problem._named_point(packed),
                    packed.detach().clone(),
                    TrialState.COMPLETE,
                    float(value.item()),
                    tuple(float(item) for item in violations.tolist()),
                    seed=self.seed,
                )
            )
        return result

    def ask(self, *, fidelity: str = "default", replicate: int = 0) -> Trial:
        """Reserve a deterministic projected point for distributed ask/tell use."""
        if not isinstance(fidelity, str) or not fidelity:
            raise ValueError("fidelity must be a non-empty string.")
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise ValueError("replicate must be a non-negative integer.")
        number = len(self.trials)
        engine = torch.quasirandom.SobolEngine(
            self.problem.space.dimension,
            scramble=True,
            seed=self.seed + number,
        )
        packed = self.problem.space.project(engine.draw(1).to(torch.float64))[0]
        trial = Trial(
            number,
            self.problem._named_point(packed),
            packed,
            seed=self.seed,
            fidelity=fidelity,
            replicate=replicate,
        )
        self.trials.append(trial)
        return trial

    def tell(
        self,
        trial: Trial,
        *,
        value: float | None = None,
        violations: tuple[float, ...] = (),
        state: TrialState = TrialState.COMPLETE,
    ) -> Trial:
        """Complete a reserved trial with strict finite observations."""
        if not any(candidate is trial for candidate in self.trials) or trial.state is not TrialState.RUNNING:
            raise ValueError("trial must be one running trial owned by this study.")
        state = TrialState(state)
        if state is TrialState.COMPLETE:
            if value is None or not math.isfinite(value):
                raise ValueError("A completed trial requires a finite value.")
            if len(violations) != len(self.problem.constraints) or any(
                not math.isfinite(item) or item < 0 for item in violations
            ):
                raise ValueError("Trial violations must be finite, non-negative, and aligned.")
        trial.value = value
        trial.violations = tuple(violations)
        trial.state = state
        return trial

    @property
    def best_trial(self) -> Trial:
        completed = [trial for trial in self.trials if trial.state is TrialState.COMPLETE]
        if not completed:
            raise RuntimeError("The study has no completed trials.")
        sign = 1.0 if self.problem.direction == "min" else -1.0
        return min(
            completed,
            key=lambda trial: (
                not bool(trial.feasible),
                sum(trial.violations),
                sign * float(trial.value),
                trial.number,
            ),
        )


def create_study(
    problem: BlackBoxProblem,
    *,
    name: str | None = None,
    storage: EvaluationDatabase | str | Path | None = None,
    seed: int = 0,
) -> Study:
    return Study(problem, name=name, storage=storage, seed=seed)


__all__ = ["Study", "Trial", "TrialState", "create_study"]
