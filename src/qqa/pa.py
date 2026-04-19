"""GPU-parallel Population Annealing (PA) backend.

Population Annealing is a sequential-Monte-Carlo cousin of SA that augments
each temperature step with a *resampling* of the population proportional to
Boltzmann reweighting factors. References:

* Hukushima & Iba (2003), *Population annealing and its application to a
  spin glass*, AIP Conf. Proc. 690.
* Machta (2010), *Population annealing with weighted averages*, PRE 82.
* Wang, Machta, Katzgraber (2015), *Comparing Monte Carlo methods for
  finding ground states of Ising spin glasses*, PRE 92.

The algorithm — for a population of ``R`` replicas at inverse temperatures
``β_0 < β_1 < ... < β_T``:

1. Initialise replicas at ``β_0`` (random or warm-started).
2. For each step ``t = 0..T-1``:

   a. **Reweight**: compute ``w_r ∝ exp(-Δβ_t · E_r)`` with
      ``Δβ_t = β_{t+1} - β_t``, in log-space for numerical stability.
   b. **Resample** the population to size ``R`` (multinomial or systematic
      resampling) according to ``w``.
   c. **Equilibrate** at the new temperature with ``K`` MCMC sweeps.

The MCMC primitives (QUBO Glauber sweep, generic single-spin MH sweep)
are shared with :func:`qqa.simulated_annealing` via private helpers in
:mod:`qqa.sa`, so the per-sweep dynamics stay byte-identical.

Compared to running ``R`` independent SA chains at the same compute,
resampling concentrates the population on low-energy regions early, which
is empirically very effective on rugged spin-glass landscapes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import time
from typing import Any, Literal

import torch

from qqa.sa import (
    _build_beta_schedule,
    _qubo_glauber_sweep,
    _seq_mh_sweep,
    _validate_chain_problem,
)
from qqa.utils import require_cuda_if_requested, safe_score_summary


@dataclass
class PAResult:
    """Result returned by :func:`population_annealing`.

    Mirrors :class:`qqa.AnnealResult` / :class:`qqa.SAResult` so it can be
    plugged into the same plotting and scoring helpers.

    Attributes
    ----------
    best_sol:
        ``(num_vars,)`` tensor in the problem's native discrete encoding
        (``{0,1}`` for binary, ``{-1,+1}`` for spin).
    best_obj:
        Best ``loss_fn`` value seen across the entire run.
    runtime:
        Wall-clock seconds spent inside the MCMC loop.
    history:
        Per-temperature-step diagnostics: ``loss_mean``, ``loss_min``,
        ``best_obj``, ``beta``, ``ess`` (effective sample size after
        reweighting, in ``[1, R]``).
    score:
        ``problem.score_summary`` of ``best_sol`` if defined, otherwise
        ``{"objective": best_obj}``.
    """

    best_sol: torch.Tensor
    best_obj: float
    runtime: float
    history: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)


def _systematic_resample_indices(weights: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Low-variance systematic resampling.

    Splits ``[0, 1)`` into ``R`` equal strata and draws one uniform sample
    per stratum; returns the indices into ``weights`` whose cumulative
    distribution covers each stratum boundary. Variance is provably
    smaller than multinomial resampling — see Doucet & Johansen (2008),
    *A Tutorial on Particle Filtering and Smoothing*.
    """
    r = weights.shape[0]
    cumulative = torch.cumsum(weights, dim=0)
    # Numerical guard: force the last bin to exactly 1.0 so searchsorted
    # never overshoots due to floating-point error.
    cumulative = cumulative / cumulative[-1].clamp(min=torch.finfo(weights.dtype).tiny)
    u0 = torch.rand(1, device=weights.device, generator=rng).item()
    grid = (torch.arange(r, device=weights.device, dtype=weights.dtype) + u0) / r
    return torch.searchsorted(cumulative, grid).clamp_(max=r - 1)


def _multinomial_resample_indices(weights: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Plain multinomial resampling. Higher variance, simpler to reason about."""
    r = weights.shape[0]
    return torch.multinomial(weights, r, replacement=True, generator=rng)


def _effective_sample_size(weights: torch.Tensor) -> float:
    """Kish's effective sample size, in ``[1, R]``."""
    s1 = weights.sum()
    s2 = (weights * weights).sum()
    if s2.item() == 0.0:
        return float(weights.shape[0])
    return float((s1 * s1 / s2).item())


def population_annealing(
    problem,
    *,
    sol_size: int = 128,
    num_temps: int = 100,
    sweeps_per_temp: int = 10,
    beta_schedule: Literal["geometric", "linear"] = "geometric",
    beta_start: float = 0.1,
    beta_end: float = 10.0,
    resample: Literal["systematic", "multinomial"] = "systematic",
    seed: int | None = None,
    device: str | torch.device = "cpu",
    initial_state: torch.Tensor | None = None,
    history_stride: int = 1,
    record_history: bool = True,
    verbose: bool = True,
    check_interval: int = 10,
    callback: Callable[[int, float, float, float], None] | None = None,
) -> PAResult:
    """Run GPU-parallel Population Annealing (with resampling) on ``problem``.

    Parameters
    ----------
    problem:
        Single-instance :class:`~qqa.problems.COProblem` exposing
        ``loss_fn(x)`` and a ``relaxation`` attribute
        (:class:`~qqa.relaxation.BinaryRelaxation` or
        :class:`~qqa.relaxation.SpinRelaxation`).
        :class:`~qqa.relaxation.CategoricalRelaxation` and batched-instance
        problems are rejected with :class:`NotImplementedError`.
    sol_size:
        Population size ``R`` (number of replicas carried through resampling).
    num_temps:
        Number of inverse-temperature steps ``T``. The first step initialises
        at ``beta_start`` with no reweighting; subsequent steps reweight by
        ``Δβ`` and then run ``sweeps_per_temp`` MCMC sweeps.
    sweeps_per_temp:
        Number of MCMC sweeps performed *after* each resampling step.
        ``num_temps * sweeps_per_temp`` is the natural compute analogue of
        SA's ``num_sweeps`` for budget-matched comparisons.
    beta_schedule:
        ``"geometric"`` (recommended) or ``"linear"``.
    beta_start, beta_end:
        Endpoints of the inverse-temperature schedule.
    resample:
        ``"systematic"`` (low-variance, default) or ``"multinomial"``.
    seed, device, initial_state, history_stride, record_history, verbose,
    check_interval, callback:
        Same semantics as :func:`qqa.simulated_annealing`. ``callback`` is
        ``(step_idx, mean_loss, best_obj, ess) -> None``.

    Returns
    -------
    PAResult
    """
    if sol_size < 1:
        raise ValueError(f"sol_size must be >= 1, got {sol_size}.")
    if num_temps < 1:
        raise ValueError(f"num_temps must be >= 1, got {num_temps}.")
    if sweeps_per_temp < 0:
        raise ValueError(f"sweeps_per_temp must be >= 0, got {sweeps_per_temp}.")
    if history_stride < 1:
        raise ValueError(f"history_stride must be >= 1, got {history_stride}.")
    if resample not in ("systematic", "multinomial"):
        raise ValueError(f"resample must be 'systematic' or 'multinomial', got {resample!r}.")

    require_cuda_if_requested(device)
    device = torch.device(device) if isinstance(device, str) else device

    is_spin, is_binary, num_vars = _validate_chain_problem(problem)
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

    betas = _build_beta_schedule(beta_schedule, beta_start, beta_end, num_temps).to(device)

    q_mat = getattr(problem, "Q_mat", None)
    use_qubo_fast = (
        is_binary and isinstance(q_mat, torch.Tensor) and q_mat.shape == (num_vars, num_vars)
    )
    if use_qubo_fast:
        q_mat = q_mat.to(device)
        q_sym = 0.5 * (q_mat + q_mat.t())
        q_diag = q_sym.diagonal().contiguous()

    history: dict[str, list[Any]] = {
        "loss_mean": [],
        "loss_min": [],
        "best_obj": [],
        "beta": [],
        "ess": [],
    }

    with torch.no_grad():
        loss_curr = problem.loss_fn(x)
    min_val, min_idx = torch.min(loss_curr, dim=0)
    best_obj = float(min_val.item())
    best_sol = x[int(min_idx.item())].detach().clone()

    runtime_start = time()
    prev_beta = float(betas[0].item())
    for step in range(num_temps):
        beta = float(betas[step].item())
        delta_beta = beta - prev_beta  # 0.0 on the first step → no reweighting

        if delta_beta > 0.0:
            # Numerically-stable Boltzmann reweighting:
            # log w_r = -Δβ · (E_r - min E),  w_r = exp(log w_r)
            with torch.no_grad():
                log_w = -delta_beta * (loss_curr - loss_curr.min())
                weights = torch.exp(log_w)
                ess = _effective_sample_size(weights)
                if resample == "systematic":
                    idx = _systematic_resample_indices(weights, rng)
                else:
                    idx = _multinomial_resample_indices(weights, rng)
                x = x[idx].contiguous()
                loss_curr = loss_curr[idx].contiguous()
        else:
            ess = float(sol_size)

        with torch.no_grad():
            for _ in range(sweeps_per_temp):
                if use_qubo_fast:
                    x = _qubo_glauber_sweep(x, q_sym, q_diag, beta, rng)
                else:
                    x = _seq_mh_sweep(x, problem, beta, num_vars, is_spin, rng)
            loss_curr = problem.loss_fn(x)

            min_val, min_idx = torch.min(loss_curr, dim=0)
            if min_val.item() < best_obj:
                best_obj = float(min_val.item())
                best_sol = x[int(min_idx.item())].detach().clone()

        if record_history and (step % history_stride == 0 or step == num_temps - 1):
            history["loss_mean"].append(float(loss_curr.mean().item()))
            history["loss_min"].append(float(loss_curr.min().item()))
            history["best_obj"].append(best_obj)
            history["beta"].append(beta)
            history["ess"].append(float(ess))

        if callback is not None and (step % history_stride == 0 or step == num_temps - 1):
            callback(step, float(loss_curr.mean().item()), best_obj, float(ess))

        if verbose and (step % check_interval == 0 or step == num_temps - 1):
            print(
                f"[PA] step {step:>5d}/{num_temps}  beta={beta:.4f}  "
                f"mean_loss={float(loss_curr.mean().item()):.4f}  "
                f"best={best_obj:.4f}  ess={ess:.1f}/{sol_size}"
            )

        prev_beta = beta

    runtime = time() - runtime_start
    if verbose:
        print(f"[PA] done. best={best_obj:.6f}  runtime={runtime:.2f}s")

    best_sol_disc = best_sol.detach()
    score = safe_score_summary(problem, best_sol_disc, fallback_obj=float(best_obj))
    return PAResult(
        best_sol=best_sol_disc,
        best_obj=best_obj,
        runtime=runtime,
        history=history,
        score=score,
    )
