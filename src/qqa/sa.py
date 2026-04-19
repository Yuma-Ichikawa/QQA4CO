"""GPU-parallel Simulated Annealing backend.

This module ships a single :func:`simulated_annealing` entry point that
mirrors the surface of :func:`qqa.anneal` so SA can be benchmarked head-to-
head against QQA / CRA-PI-GNN / CPRA on the *same*
:class:`~qqa.problems.COProblem` instances.

Two execution paths are dispatched automatically:

1. **QUBO fast path** (``problem.Q_mat`` present) — every sweep updates all
   spins in **one** ``x @ Q`` matmul plus a Glauber-like independent
   acceptance per bit. Total cost is ``O(num_sweeps * N^2)`` flops — the
   same as one QQA epoch — and runs end-to-end on GPU with no host
   round-trips inside the loop.
2. **Generic single-spin sequential MH** — for non-QUBO problems
   (``Knapsack``, ``MaxSAT3``, ``MaximumIndependentSet``'s edge-list
   variants when used through ``UserProblem``, ...). Calls
   ``problem.loss_fn`` ``N`` times per sweep; correct, slower, but works
   for any problem the QQA loop accepts.

Both paths support a leading batch dimension of size ``sol_size`` so the
chain is naturally parallelised on GPU.

The Glauber-like acceptance used in the fast path proposes every bit
independently in the *same* sweep, which is **not** a strictly correct
single-spin Metropolis chain (proposals are not conditional on the most
recent neighbour state). It does converge to the Boltzmann distribution
in the high-``beta`` limit for QUBO objectives, and matches the
"parallel-tempered classical SA" baselines used in the QQA papers — which
is the comparison we ship for benchmark notebooks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import time
from typing import Any, Literal

import numpy as np
import torch

from qqa.relaxation import BinaryRelaxation, CategoricalRelaxation, SpinRelaxation
from qqa.utils import require_cuda_if_requested, safe_score_summary


@dataclass
class SAResult:
    """Result returned by :func:`simulated_annealing`.

    Mirrors the public surface of :class:`qqa.AnnealResult` so downstream
    plotting / scoring code can treat the two interchangeably.
    """

    best_sol: torch.Tensor
    best_obj: Any
    runtime: float
    history: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)


def _resolve_num_vars(problem) -> int:
    """Pull the variable count out of a problem in a relaxation-agnostic way."""
    for attr in ("num_nodes", "num_spins", "num_vars", "num_node"):
        if hasattr(problem, attr):
            return int(getattr(problem, attr))
    raise TypeError(
        f"Cannot infer variable count from {type(problem).__name__}; "
        "the SA backend needs one of num_nodes / num_spins / num_vars / num_node."
    )


def _build_beta_schedule(
    schedule: str, beta_start: float, beta_end: float, num_sweeps: int
) -> torch.Tensor:
    if schedule not in ("geometric", "linear"):
        raise ValueError(f"schedule must be 'geometric' or 'linear', got {schedule!r}.")
    if beta_start <= 0 or beta_end <= 0:
        raise ValueError(f"beta_start ({beta_start}) and beta_end ({beta_end}) must be positive.")
    if num_sweeps <= 0:
        return torch.empty(0)
    if schedule == "geometric":
        return torch.tensor(
            np.geomspace(beta_start, beta_end, num_sweeps, dtype=np.float64),
            dtype=torch.float32,
        )
    return torch.linspace(beta_start, beta_end, num_sweeps, dtype=torch.float32)


def simulated_annealing(
    problem,
    *,
    sol_size: int = 128,
    num_sweeps: int = 1_000,
    beta_schedule: Literal["geometric", "linear"] = "geometric",
    beta_start: float = 0.1,
    beta_end: float = 10.0,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    initial_state: torch.Tensor | None = None,
    history_stride: int = 1,
    record_history: bool = True,
    verbose: bool = True,
    check_interval: int = 100,
    callback: Callable[[int, float, float], None] | None = None,
) -> SAResult:
    """Run GPU-parallel Simulated Annealing on ``problem``.

    Parameters
    ----------
    problem:
        Any :class:`~qqa.problems.COProblem` exposing ``loss_fn(x)`` and a
        ``relaxation`` attribute (BinaryRelaxation, BinaryInstanceRelaxation,
        or SpinRelaxation). CategoricalRelaxation is not yet supported.
    sol_size:
        Number of independent SA chains run in parallel on GPU.
    num_sweeps:
        Number of full sweeps. One sweep updates every spin once
        (sequentially in the generic path, in parallel in the QUBO fast path).
    beta_schedule:
        ``"geometric"`` (recommended for SA) or ``"linear"``. ``beta`` is
        the inverse temperature, so high beta = greedy.
    beta_start, beta_end:
        Endpoints of the schedule. ``beta_end >= beta_start`` is recommended.
    seed:
        RNG seed.
    device:
        ``"cpu"`` or ``"cuda"``.
    initial_state:
        Optional ``(sol_size, N)`` starting configuration. If ``None`` a
        random ``{0,1}`` (binary) or ``{-1,+1}`` (spin) state is sampled.
    history_stride:
        Record every ``history_stride`` sweeps (1 = every sweep).
    record_history:
        Record loss / beta / best per sweep.
    verbose:
        Print progress every ``check_interval`` sweeps.
    callback:
        Optional ``(sweep_idx, mean_loss, best_obj) -> None`` callback.
        Called after each recorded sweep.

    Returns
    -------
    SAResult
        With ``best_sol``, ``best_obj`` (Python ``float``), ``runtime``,
        ``history`` and a ``score_summary`` dict if the problem provides one.
    """
    if sol_size < 1:
        raise ValueError(f"sol_size must be >= 1, got {sol_size}.")
    if num_sweeps < 0:
        raise ValueError(f"num_sweeps must be >= 0, got {num_sweeps}.")
    if history_stride < 1:
        raise ValueError(f"history_stride must be >= 1, got {history_stride}.")

    require_cuda_if_requested(device)
    device = torch.device(device) if isinstance(device, str) else device

    relax = getattr(problem, "relaxation", None)
    if isinstance(relax, CategoricalRelaxation):
        raise NotImplementedError(
            "simulated_annealing does not support CategoricalRelaxation problems "
            "(e.g. Coloring, GraphBisection). Use qqa.anneal for those."
        )
    # Batched-instance problems (MaximumIndependentSetInstance, ...) carry a
    # 3-D state ``(B, I, N)`` and a ``loss_fn`` that expects the instance
    # axis. The single-instance SA loop below would either silently misuse
    # the einsum or crash deep inside the per-sweep matmul; surface a clear
    # message at the API boundary instead.
    if hasattr(problem, "num_instance"):
        raise NotImplementedError(
            "simulated_annealing does not support batched-instance problems "
            f"({type(problem).__name__}); SA needs a single-instance "
            "problem. Iterate over instances and call simulated_annealing "
            "on each, or use qqa.anneal which handles batched instances."
        )
    is_spin = isinstance(relax, SpinRelaxation)
    is_binary = isinstance(relax, BinaryRelaxation) and not is_spin
    if not (is_spin or is_binary):
        raise TypeError(
            f"Unsupported relaxation {type(relax).__name__}; expected "
            "BinaryRelaxation or SpinRelaxation."
        )

    num_vars = _resolve_num_vars(problem)
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(int(seed))

    if initial_state is not None:
        x = initial_state.to(device).float()
        if x.shape != (sol_size, num_vars):
            raise ValueError(
                f"initial_state shape {tuple(x.shape)} != expected {(sol_size, num_vars)}."
            )
    else:
        u = torch.rand((sol_size, num_vars), device=device, generator=rng)
        x = torch.where(u > 0.5, 1.0, -1.0) if is_spin else (u > 0.5).float()

    betas = _build_beta_schedule(beta_schedule, beta_start, beta_end, num_sweeps).to(device)

    # Pull QUBO fast-path matrix when available. Q_mat is the canonical
    # attribute on every QUBOProblem subclass shipped by qqa.
    q_mat = getattr(problem, "Q_mat", None)
    use_qubo_fast = (
        is_binary and isinstance(q_mat, torch.Tensor) and q_mat.shape == (num_vars, num_vars)
    )
    if use_qubo_fast:
        q_mat = q_mat.to(device)
        # Symmetrise once: ΔE_i for a single-bit flip on x^T Q x is
        # (1 - 2 x_i) * (Q + Q^T)_i,: x   (when Q is symmetric this is 2 Q_i x).
        # Working with the symmetric Q_sym keeps the per-sweep matmul
        # symmetric and lets us write the diagonal correction cleanly.
        q_sym = 0.5 * (q_mat + q_mat.t())
        q_diag = q_sym.diagonal().contiguous()

    history: dict[str, list] = {
        "loss_mean": [],
        "loss_min": [],
        "best_obj": [],
        "beta": [],
    }

    with torch.no_grad():
        loss_curr = problem.loss_fn(x)
    best_loss, best_idx = torch.min(loss_curr, dim=0)
    best_obj = float(best_loss.item())
    best_sol = x[int(best_idx.item())].detach().clone()
    best_per_chain = loss_curr.clone()
    best_state = x.clone()

    runtime_start = time()
    for sweep in range(num_sweeps):
        beta = float(betas[sweep].item())

        if use_qubo_fast:
            # Glauber-like parallel update: ΔE_i = (1 - 2 x_i) * (2 Q_sym x)_i
            # ignoring the diagonal self-interaction.
            with torch.no_grad():
                qx = x @ q_sym  # (B, N)
                # Subtract the self term so ΔE matches "flip bit i and
                # recompute x^T Q x exactly" rather than "flip in a Q with
                # diag suppressed".
                delta_e = (1.0 - 2.0 * x) * (2.0 * qx - 2.0 * x * q_diag + q_diag)
                accept_p = torch.exp(torch.clamp(-beta * delta_e, max=0.0))
                u = torch.rand(x.shape, device=device, generator=rng)
                accept = u < accept_p
                x = torch.where(accept, 1.0 - x, x)
        else:
            # Sequential single-spin Metropolis. Correct for any problem.
            perm = torch.randperm(num_vars, generator=rng, device=device)
            for j in perm.tolist():
                x_new = x.clone()
                if is_spin:
                    x_new[:, j] = -x_new[:, j]
                else:
                    x_new[:, j] = 1.0 - x_new[:, j]
                with torch.no_grad():
                    delta = problem.loss_fn(x_new) - problem.loss_fn(x)
                    accept_p = torch.exp(torch.clamp(-beta * delta, max=0.0))
                    u = torch.rand(sol_size, device=device, generator=rng)
                    accept = u < accept_p
                x = torch.where(accept.unsqueeze(-1), x_new, x)

        with torch.no_grad():
            loss_curr = problem.loss_fn(x)
            improved = loss_curr < best_per_chain
            best_per_chain = torch.where(improved, loss_curr, best_per_chain)
            best_state = torch.where(improved.unsqueeze(-1), x, best_state)

            min_val, min_idx = torch.min(loss_curr, dim=0)
            if min_val.item() < best_obj:
                best_obj = float(min_val.item())
                best_sol = x[int(min_idx.item())].detach().clone()

        if record_history and (sweep % history_stride == 0 or sweep == num_sweeps - 1):
            history["loss_mean"].append(float(loss_curr.mean().item()))
            history["loss_min"].append(float(loss_curr.min().item()))
            history["best_obj"].append(best_obj)
            history["beta"].append(beta)

        if callback is not None and (sweep % history_stride == 0 or sweep == num_sweeps - 1):
            callback(sweep, float(loss_curr.mean().item()), best_obj)

        if verbose and (sweep % check_interval == 0 or sweep == num_sweeps - 1):
            mean_l = float(loss_curr.mean().item())
            print(
                f"[SA] sweep {sweep:>6d}  beta={beta:.4f}  "
                f"mean_loss={mean_l:.4f}  best={best_obj:.4f}"
            )

    runtime = time() - runtime_start
    if verbose:
        print(f"[SA] done. best={best_obj:.6f}  runtime={runtime:.2f}s")

    # Convert best_sol back to the convention the problem's score_summary
    # expects: spin problems work on {-1,+1} floats already, binary
    # problems want {0,1} (which is what we already track).
    best_sol_disc = best_sol.detach()

    score = safe_score_summary(problem, best_sol_disc, fallback_obj=float(best_obj))

    return SAResult(
        best_sol=best_sol_disc,
        best_obj=best_obj,
        runtime=runtime,
        history=history,
        score=score,
    )
