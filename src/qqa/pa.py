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

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import torch

from qqa.polish import apply_polish_if_improves
from qqa.sa import (
    _build_beta_schedule,
    _interaction_color_classes,
    _qubo_seq_glauber_sweep,
    _seq_mh_sweep,
    _sparse_colored_metropolis_sweep,
    _validate_chain_problem,
)
from qqa.utils import require_cuda_if_requested, resolve_device, safe_score_summary


@dataclass
class PAResult:
    """Result returned by :func:`population_annealing`.

    Mirrors :class:`qqa.AnnealResult` / :class:`qqa.SAResult` so it can be
    plugged into the same plotting and scoring helpers, and adds three
    PA-specific extensions: the equilibrium population at ``β_end``, an
    unbiased free-energy estimator, and an optional genealogy dump.

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
        Per-temperature-step diagnostics. Always populated:
        ``loss_mean``, ``loss_min``, ``best_obj``, ``beta``,
        ``ess`` (effective sample size after reweighting, in ``[1, R]``),
        ``log_z_ratio`` (per-step ``ln Z(β_t) - ln Z(β_{t-1})``,
        Hukushima–Iba estimator, **with the resampling correction implicit**
        because we average the unnormalised weights over the current
        population), ``log_z`` (running **absolute** ``ln Z(β_t)``,
        anchored at ``ln Z(0) = N · ln 2``), ``free_energy_density``
        (``F(β_t) / N`` from ``log_z``).
    score:
        ``problem.score_summary`` of ``best_sol`` if defined, otherwise
        ``{"objective": best_obj}``.
    final_x:
        ``(sol_size, num_vars)`` population at ``β_end``. After enough
        equilibration sweeps these are approximately Boltzmann samples at
        the final inverse temperature — use them to estimate observables
        rather than ``best_sol`` (which is the running min, not a sample).
    final_loss:
        ``(sol_size,)`` energies of ``final_x``.
    log_z:
        Final absolute ``ln Z(β_end)`` estimate, anchored at
        ``ln Z(0) = N · ln 2`` for both binary and spin encodings (each
        of the ``N`` variables has two states under the uniform prior).
    free_energy:
        ``F(β_end) = -ln Z(β_end) / β_end`` (absolute, includes the
        ``N · ln 2`` reference).
    free_energy_density:
        ``F(β_end) / N``.
    genealogy:
        Present when ``record_genealogy=True``. A dict with keys
        ``parents`` (list of ``(R,)`` long tensors, one per resampling
        step, holding the *index into the previous population* each
        replica was copied from) and ``ancestors`` (``(R,)`` long tensor
        with the **root** ancestor index in the initial population for
        each surviving replica, propagated through every resampling
        step). Use ``ancestors`` directly or chain ``parents`` to draw
        a family tree / count surviving founders / visualise clonal
        sweeps.
    polished_sol:
        ``(num_vars,)`` tensor — greedy-1-flip-locally-optimal version
        of ``best_sol``. Populated iff :func:`population_annealing` is
        called with ``polish=True`` on a QUBO problem (mirrors the
        :class:`qqa.AnnealResult.polished_sol` contract). ``best_sol``
        / ``best_obj`` / ``score`` are all overwritten by the polished
        result whenever it strictly improves the QUBO loss, so reading
        ``best_sol`` works unconditionally. ``None`` on non-QUBO
        problems or when ``polish=False``.
    """

    best_sol: torch.Tensor
    best_obj: float
    runtime: float
    history: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)
    final_x: torch.Tensor | None = None
    final_loss: torch.Tensor | None = None
    log_z: float | None = None
    free_energy: float | None = None
    free_energy_density: float | None = None
    genealogy: dict | None = None
    polished_sol: torch.Tensor | None = None
    diagnostics: dict = field(default_factory=dict)


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
    record_genealogy: bool = False,
    polish: bool = True,
    verbose: bool = True,
    check_interval: int = 10,
    callback: Callable[[int, float, float, float], None] | None = None,
    time_limit: float | None = None,
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
    record_genealogy:
        If ``True``, store per-step parent indices and the running
        root-ancestor map on the result, so callers can reconstruct the
        family tree of the population. Costs ``O(R · num_temps)`` integer
        memory; off by default.
    polish:
        If ``True`` (default), run :func:`qqa.polish.greedy_one_flip`
        on ``best_sol`` once the MCMC loop has finished, mirroring the
        post-processing :func:`qqa.anneal` applies by default. Noop on
        non-QUBO problems (no ``Q_mat``). Overwrites ``best_sol`` /
        ``best_obj`` / ``score`` whenever the polish strictly improves
        the QUBO loss, so PA and PQQA are directly comparable on the
        same problem without the caller remembering which post-
        processing is active.

    Returns
    -------
    PAResult
        With the equilibrium population (``final_x`` / ``final_loss``)
        and a free-energy estimate (``log_z`` / ``free_energy`` /
        ``free_energy_density``) populated. The free-energy estimator is
        Hukushima–Iba's
        ``Σ_t  ln (1/R) Σ_r exp(-Δβ_t · E_r^{(t)})``, where ``E_r^{(t)}``
        is the energy of replica ``r`` *just before* the resampling at
        step ``t``. Resampling cancels in expectation because we average
        the unnormalised weights — this is the property the textbooks
        flag as "PA gives free energies for free".
    """
    if sol_size < 1:
        raise ValueError(f"sol_size must be >= 1, got {sol_size}.")
    if num_temps < 1:
        raise ValueError(f"num_temps must be >= 1, got {num_temps}.")
    if sweeps_per_temp < 0:
        raise ValueError(f"sweeps_per_temp must be >= 0, got {sweeps_per_temp}.")
    if history_stride < 1:
        raise ValueError(f"history_stride must be >= 1, got {history_stride}.")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and non-negative or None.")
    if resample not in ("systematic", "multinomial"):
        raise ValueError(f"resample must be 'systematic' or 'multinomial', got {resample!r}.")
    # PA is only well-defined for non-decreasing β: reweighting by exp(-Δβ E)
    # with Δβ < 0 would up-weight high-energy replicas, which is the opposite
    # of what PA wants. Reject upfront rather than silently skipping resampling.
    if beta_end < beta_start:
        raise ValueError(
            f"Population annealing requires beta_end >= beta_start, got "
            f"beta_start={beta_start}, beta_end={beta_end}. Use SA if you "
            "actually want a heating schedule."
        )

    device = resolve_device(device)
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

    sparse_qubo = getattr(problem, "sparse_qubo", None)
    use_sparse_fast = bool(
        is_binary
        and sparse_qubo is not None
        and getattr(sparse_qubo, "num_variables", None) == num_vars
        and callable(getattr(sparse_qubo, "gradient", None))
    )
    if use_sparse_fast:
        if sparse_qubo is None:  # Defensive guard for dynamically typed problem objects.
            raise RuntimeError("Sparse QUBO dispatch lost its sparse representation.")
        color_classes = _interaction_color_classes(sparse_qubo.edge_index, num_vars, device=device)
    else:
        color_classes = ()
    q_mat = None if use_sparse_fast else getattr(problem, "Q_mat", None)
    use_qubo_fast = (
        not use_sparse_fast
        and is_binary
        and isinstance(q_mat, torch.Tensor)
        and q_mat.shape == (num_vars, num_vars)
    )
    if use_qubo_fast:
        if not isinstance(q_mat, torch.Tensor):
            raise RuntimeError("Dense QUBO dispatch requires a tensor Q_mat.")
        q_mat = q_mat.to(device)
        q_sym = 0.5 * (q_mat + q_mat.t())
        q_diag = q_sym.diagonal().contiguous()

    history: dict[str, list[Any]] = {
        "loss_mean": [],
        "loss_min": [],
        "best_obj": [],
        "beta": [],
        "ess": [],
        # Free-energy diagnostics. ``log_z_ratio[t] = ln Z(β_t) - ln Z(β_{t-1})``
        # for t >= 1, and the implicit ``β = 0 → β_start`` jump at t = 0
        # (so ``log_z[t] = ln Z(β_t) - ln Z(0)`` is comparable across runs).
        "log_z_ratio": [],
        "log_z": [],
        "free_energy_density": [],
    }
    # Free-energy reference: at β = 0 every configuration is equally likely,
    # so ``Z(0) = 2^N`` and ``ln Z(0) = N · ln 2`` for both binary {0,1}
    # and spin {-1,+1} encodings (each variable has two states).
    log_z_zero = float(num_vars) * math.log(2.0)

    # Genealogy buffers — only allocated when the caller asks for them so
    # the default-path memory footprint stays at O(R · N) floats.
    parents_log: list[torch.Tensor] = []
    ancestors: torch.Tensor | None = None
    if record_genealogy:
        ancestors = torch.arange(sol_size, device=device, dtype=torch.long)

    with torch.no_grad():
        loss_curr = problem.loss_fn(x)
    min_val, min_idx = torch.min(loss_curr, dim=0)
    best_obj = float(min_val.item())
    best_sol = x[int(min_idx.item())].detach().clone()
    log_z_running = 0.0  # accumulator for ln Z(β_t) - ln Z(0)

    runtime_start = perf_counter()
    deadline_reached = False
    prev_beta = 0.0  # virtual β=0 reference, so step 0 carries the β=0 → β_start jump
    for step in range(num_temps):
        if time_limit is not None and perf_counter() - runtime_start >= time_limit:
            deadline_reached = True
            break
        beta = float(betas[step].item())
        delta_beta = beta - prev_beta

        # ----- Reweighting + free-energy update + resampling --------------
        # We always compute the log-weights (even at step 0, where Δβ
        # is the full jump from β=0 to β_start) because Hukushima–Iba's
        # free-energy estimator needs the average unnormalised weight at
        # *every* annealing step — not only the ones that resample.
        if delta_beta > 0.0:
            with torch.no_grad():
                # Numerically-stable: factor out the min so that the largest
                # log-weight is 0 and exp() never under/over-flows. The
                # additive constant is reintroduced on the log-Z side.
                e_min = float(loss_curr.min().item())
                log_w_shift = -delta_beta * (loss_curr - e_min)
                weights = torch.exp(log_w_shift)
                # ln(1/R Σ w_r) = logsumexp(log_w_shift) - ln R - Δβ · e_min.
                log_z_step = (
                    float(torch.logsumexp(log_w_shift, dim=0).item())
                    - math.log(float(sol_size))
                    - delta_beta * e_min
                )
                ess = _effective_sample_size(weights)
                if resample == "systematic":
                    idx = _systematic_resample_indices(weights, rng)
                else:
                    idx = _multinomial_resample_indices(weights, rng)
                x = x[idx].contiguous()
                loss_curr = loss_curr[idx].contiguous()
                if record_genealogy:
                    parents_log.append(idx.detach().clone())
                    if ancestors is None:
                        raise RuntimeError("Genealogy tracking was not initialised.")
                    ancestors = ancestors[idx].contiguous()
        else:
            log_z_step = 0.0
            ess = float(sol_size)

        log_z_running += log_z_step

        # ----- Equilibration sweeps at the new temperature ----------------
        with torch.no_grad():
            for _ in range(sweeps_per_temp):
                if use_sparse_fast:
                    x = _sparse_colored_metropolis_sweep(
                        x,
                        sparse_qubo,
                        color_classes,
                        betas[step],
                        rng,
                    )
                elif use_qubo_fast:
                    x = _qubo_seq_glauber_sweep(x, q_sym, q_diag, beta, rng)
                else:
                    x = _seq_mh_sweep(x, problem, beta, num_vars, is_spin, rng)
                # Track best after EVERY sweep (matches SA semantics): a low-
                # energy transient that vanishes by the end of the K-sweep
                # batch would otherwise be invisible to ``best_obj``.
                loss_curr = problem.loss_fn(x)
                min_val, min_idx = torch.min(loss_curr, dim=0)
                if min_val.item() < best_obj:
                    best_obj = float(min_val.item())
                    best_sol = x[int(min_idx.item())].detach().clone()

        # F(β_t) = -ln Z(β_t) / β_t. ln Z(β_t) = ln Z(0) + log_z_running.
        free_energy_density = -(log_z_zero + log_z_running) / (beta * float(num_vars))

        if record_history and (step % history_stride == 0 or step == num_temps - 1):
            history["loss_mean"].append(float(loss_curr.mean().item()))
            history["loss_min"].append(float(loss_curr.min().item()))
            history["best_obj"].append(best_obj)
            history["beta"].append(beta)
            history["ess"].append(float(ess))
            history["log_z_ratio"].append(float(log_z_step))
            history["log_z"].append(float(log_z_zero + log_z_running))
            history["free_energy_density"].append(float(free_energy_density))

        if callback is not None and (step % history_stride == 0 or step == num_temps - 1):
            callback(step, float(loss_curr.mean().item()), best_obj, float(ess))

        if verbose and (step % check_interval == 0 or step == num_temps - 1):
            print(
                f"[PA] step {step:>5d}/{num_temps}  beta={beta:.4f}  "
                f"mean_loss={float(loss_curr.mean().item()):.4f}  "
                f"best={best_obj:.4f}  ess={ess:.1f}/{sol_size}  "
                f"f={free_energy_density:.4f}"
            )

        prev_beta = beta

    runtime = perf_counter() - runtime_start
    deadline_reached |= time_limit is not None and runtime >= time_limit
    if verbose:
        print(
            f"[PA] done. best={best_obj:.6f}  "
            f"f(β_end)={-(log_z_zero + log_z_running) / (float(betas[-1].item()) * float(num_vars)):.6f}  "
            f"runtime={runtime:.2f}s"
        )

    best_sol_disc = best_sol.detach()

    # Default-on greedy 1-flip polish. Shared with :func:`qqa.anneal` via
    # ``qqa.polish.apply_polish_if_improves``; noop when ``Q_mat`` is absent
    # (spin / categorical / batched problems). PA therefore matches PQQA's
    # post-processing contract by construction.
    best_sol_disc, best_obj, polished_sol = apply_polish_if_improves(
        problem, best_sol_disc, best_obj, polish=polish and not deadline_reached
    )

    score = safe_score_summary(problem, best_sol_disc, fallback_obj=float(best_obj))

    beta_final = float(betas[-1].item())
    log_z_total = log_z_zero + log_z_running
    free_energy = -log_z_total / beta_final
    f_density = free_energy / float(num_vars)

    genealogy: dict | None = None
    if record_genealogy:
        genealogy = {
            "parents": parents_log,
            "ancestors": ancestors.detach().clone() if ancestors is not None else None,
            "betas": [float(b.item()) for b in betas],
        }

    return PAResult(
        best_sol=best_sol_disc,
        best_obj=best_obj,
        runtime=runtime,
        history=history,
        score=score,
        final_x=x.detach().clone(),
        final_loss=loss_curr.detach().clone(),
        log_z=float(log_z_total),
        free_energy=float(free_energy),
        free_energy_density=float(f_density),
        genealogy=genealogy,
        polished_sol=polished_sol,
        diagnostics={"deadline_reached": deadline_reached, "time_limit": time_limit},
    )
