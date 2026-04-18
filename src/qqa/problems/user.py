"""User-defined problem helper.

Drop-in wrapper that lets a user plug an arbitrary differentiable loss into
:func:`qqa.anneal` without having to subclass :class:`COProblem`::

    import torch, qqa

    J = torch.randn(50, 50); J = (J + J.T) / 2; J.fill_diagonal_(0)
    problem = qqa.UserProblem(
        num_vars=50,
        variable_kind="spin",
        loss_fn=lambda s: -0.5 * torch.einsum("bi,ij,bj->b", s, J, s),
    )
    result = qqa.anneal(problem, sol_size=128, num_epochs=1000)

Three variable kinds are supported:

* ``"binary"`` — ``x \\in [0, 1]`` (rounded to ``{0, 1}``).
* ``"spin"``   — ``s = 2 \\, \\text{clip}(x, 0, 1) - 1 \\in [-1, +1]`` and
  projected to ``\\{-1, +1\\}``.
* ``"categorical"`` — one-hot simplex ``x \\in \\Delta^K`` per variable.

For categorical kinds the ``num_vars`` argument sets the number of categorical
variables and ``num_category`` must also be passed.

Loss functions must accept a tensor whose leading axis is the parallel batch
``B = sol_size`` and return a tensor of shape ``(B,)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from qqa.problems.base import COProblem
from qqa.relaxation import (
    BinaryRelaxation,
    CategoricalRelaxation,
    Relaxation,
    SpinRelaxation,
)

VariableKind = Literal["binary", "spin", "categorical"]


class UserProblem(COProblem):
    """Wrap an arbitrary loss function as a :class:`COProblem`.

    Args:
        num_vars: Number of discrete variables (``N``).
        loss_fn: Callable mapping a batched tensor to a ``(B,)`` loss tensor.
            For ``"binary"`` / ``"spin"`` the input has shape ``(B, N)``.
            For ``"categorical"`` the input has shape ``(B, N, K)``.
        variable_kind: ``"binary"``, ``"spin"``, or ``"categorical"``.
        num_category: Required when ``variable_kind == "categorical"``.
        relaxation: Advanced — pass a custom :class:`Relaxation` instance to
            override the default chosen from ``variable_kind``.
        name: Optional display name used by the GUI/CLI.
        device: Torch device hint (only used by QQA's allocation path; the
            loss itself is device-agnostic as long as any constants it
            captures live on the right device).
    """

    def __init__(
        self,
        num_vars: int,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        variable_kind: VariableKind = "binary",
        num_category: int | None = None,
        relaxation: Relaxation | None = None,
        name: str = "user-problem",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_vars = int(num_vars)
        self.num_nodes = int(num_vars)
        self.num_node = int(num_vars)
        self.num_spins = int(num_vars)
        self._loss_fn = loss_fn
        self.variable_kind = variable_kind
        self.device = device
        self.name = name

        if relaxation is not None:
            self.relaxation = relaxation
        elif variable_kind == "binary":
            self.relaxation = BinaryRelaxation()
        elif variable_kind == "spin":
            self.relaxation = SpinRelaxation()
        elif variable_kind == "categorical":
            if num_category is None or num_category < 2:
                raise ValueError("variable_kind='categorical' requires num_category >= 2.")
            self.num_category = int(num_category)
            self.relaxation = CategoricalRelaxation()
        else:
            raise ValueError(
                f"Unknown variable_kind={variable_kind!r}; expected "
                "'binary', 'spin', or 'categorical'."
            )

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self._loss_fn(x)

    def __repr__(self) -> str:
        return (
            f"UserProblem(name={self.name!r}, num_vars={self.num_vars}, kind={self.variable_kind})"
        )


def user_problem_from_source(
    source: str,
    num_vars: int,
    variable_kind: VariableKind = "binary",
    num_category: int | None = None,
    name: str = "inline",
    device: str | torch.device = "cpu",
    extra_globals: dict | None = None,
) -> UserProblem:
    """Build a :class:`UserProblem` by ``exec``-ing a Python snippet.

    The snippet must define a callable named ``loss_fn`` (single argument, the
    batched configuration tensor, returning a ``(B,)`` loss tensor). The
    namespace has ``torch``, ``np`` (numpy), and any ``extra_globals`` entries
    pre-loaded, and the defined ``loss_fn`` is closed over that namespace.

    .. warning::
       This executes arbitrary Python code. Only use with trusted input
       (e.g. the local GUI or CLI on your own machine).
    """
    import numpy as np

    ns: dict = {"torch": torch, "np": np}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(source, f"<{name}>", "exec"), ns)  # noqa: S102
    if "loss_fn" not in ns or not callable(ns["loss_fn"]):
        raise ValueError("The snippet must define a callable named `loss_fn(x) -> tensor`.")
    return UserProblem(
        num_vars=num_vars,
        loss_fn=ns["loss_fn"],
        variable_kind=variable_kind,
        num_category=num_category,
        name=name,
        device=device,
    )


def load_problem_from_file(path: str | os.PathLike[str]) -> COProblem:  # noqa: F821
    """Load a user-provided problem from a Python file.

    The file must either define a top-level variable ``problem`` that is a
    :class:`COProblem` instance, or a callable ``make_problem()`` / ``build()``
    that returns one.
    """
    import importlib.util
    import os

    p = os.fspath(path)
    spec = importlib.util.spec_from_file_location("_qqa_user_problem", p)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    if hasattr(mod, "problem") and isinstance(mod.problem, COProblem):
        return mod.problem
    for attr in ("make_problem", "build", "build_problem"):
        fn = getattr(mod, attr, None)
        if callable(fn):
            obj = fn()
            if isinstance(obj, COProblem):
                return obj
            raise TypeError(
                f"{attr}() in {p} must return a qqa.COProblem; got {type(obj).__name__}."
            )
    raise AttributeError(
        f"{p} must define `problem` (a COProblem) or `make_problem()` / `build()`."
    )
