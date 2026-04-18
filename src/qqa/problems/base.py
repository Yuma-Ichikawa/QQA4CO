"""Abstract problem base classes.

Every problem class in QQA exposes:

* ``loss_fn(x)`` — the (continuous or discrete) objective, vectorised over
  the leading batch dimension that ``qqa.anneal`` uses for the parallel
  population.
* ``relaxation`` — a :class:`~qqa.relaxation.Relaxation` instance describing
  how the variable is represented during annealing.

Binary QUBO problems return losses of shape ``(B,)`` for a single graph, or
``(B, I)`` for batched-instance variants. Categorical and spin problems
return losses of shape ``(B,)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from qqa.relaxation import Relaxation


class COProblem(ABC):
    """Abstract base class for any combinatorial optimisation problem."""

    relaxation: Relaxation

    @abstractmethod
    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        ...


class QUBOProblem(COProblem):
    """Abstract base for QUBO problems that expose a Q matrix."""

    @abstractmethod
    def generate_qubo_matrix(self) -> torch.Tensor:  # pragma: no cover - abstract
        ...
