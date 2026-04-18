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

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        """Problem-specific, human-readable breakdown of a discrete solution.

        The default implementation evaluates :meth:`loss_fn` and reports the
        raw loss. Concrete subclasses should override to return a dict with
        ``label`` / ``value`` / ``unit`` / ``feasible`` / ``extra`` so the
        dashboard can display e.g. *"IS size: 22"* instead of *"loss: -22"*.
        """
        import torch as _torch  # local import to keep the abstract base thin

        with _torch.no_grad():
            x = x_disc if x_disc.ndim >= 1 else x_disc.unsqueeze(0)
            val = self.loss_fn(x if x.ndim > 1 else x.unsqueeze(0))
            if val.ndim > 0:
                val = val.reshape(-1)[0]
        return {
            "label": "loss",
            "value": float(val.item()),
            "unit": "",
            "feasible": True,
            "extra": {},
        }


class QUBOProblem(COProblem):
    """Abstract base for QUBO problems that expose a Q matrix."""

    @abstractmethod
    def generate_qubo_matrix(self) -> torch.Tensor:  # pragma: no cover - abstract
        ...
