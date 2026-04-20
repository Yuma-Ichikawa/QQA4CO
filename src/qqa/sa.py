"""GPU-parallel Simulated Annealing backend.

This module ships a single :func:`simulated_annealing` entry point that
mirrors the surface of :func:`qqa.anneal` so SA can be benchmarked head-to-
head against QQA / CRA-PI-GNN / CPRA on the *same*
:class:`~qqa.problems.COProblem` instances.

Two execution paths are dispatched automatically:

1. **QUBO fast path** (``problem.Q_mat`` present) — :func:`_qubo_seq_glauber_sweep`
   runs *sequential* single-bit Glauber/Metropolis updates inside one
   sweep, but every replica is updated in parallel on GPU. The per-bit
   ΔE is computed from the relevant column of ``Q`` in O(N) per bit, so a
   full sweep is O(N²) flops per replica with **zero host round-trips**.
   This is the textbook-correct sampler that Boltzmann-equilibrates for
   any symmetric QUBO.
2. **Generic single-spin sequential MH** — for non-QUBO problems
   (``Knapsack``, ``MaxSAT3``, ``MaximumIndependentSet``'s edge-list
   variants when used through ``UserProblem``, ...). Calls
   ``problem.loss_fn`` ``N`` times per sweep; correct, slower, but works
   for any problem the QQA loop accepts.

Both paths support a leading batch dimension of size ``sol_size`` so the
chain is naturally parallelised on GPU.

History note (kept for reproducibility of the QQA papers): the previous
fast path proposed every bit *in parallel* against the same pre-sweep
state. That is **not** a valid single-spin chain on coupled QUBOs and
caused catastrophic mode-locking on MIS / MaxCut style problems
(empirically: 3-regular MIS oscillates between all-zeros and all-ones
forever because every bit independently sees a favourable single-flip
ΔE). The biased baseline is preserved as :func:`_qubo_parallel_metropolis_sweep`
for paper-reproduction tests, but it is *no longer* the default sampler.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import time
from typing import Any, Literal

import numpy as np
import torch

from qqa.polish import greedy_one_flip
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
    polished_sol: torch.Tensor | None = None


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


# ---------------------------------------------------------------------------
# Single-sweep MCMC primitives, shared with qqa.pa.population_annealing.
# Both routines mutate ``x`` out-of-place and return the new tensor; using
# them keeps SA / PA in lock-step on the per-sweep dynamics so any future
# ΔE bug-fix automatically lands on both.
# ---------------------------------------------------------------------------


def _qubo_seq_glauber_sweep(
    x: torch.Tensor,
    q_sym: torch.Tensor,
    q_diag: torch.Tensor,
    beta: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """One **sequential** single-bit Metropolis sweep on a binary QUBO.

    Updates the bits in a random permutation. For each bit ``j`` the
    single-flip ΔE is computed from the *current* state via the cached
    pre-sweep field ``qx = x @ q_sym``, which is incrementally updated
    after each accepted flip via the rank-1 identity

        Δ(qx) = q_sym[:, j] · Δx_j   (with Δx_j = ±1)

    so the per-bit work is O(N) instead of O(N²). The full sweep cost
    stays at O(N²) per replica — we just dropped the per-bit matrix-
    vector slice. The ``sol_size`` batch dimension is handled natively
    on GPU with no host round-trips inside the loop.

    Acceptance is independent across replicas (Metropolis,
    ``p = min(1, exp(-β ΔE))``). The function never mutates its input
    tensor — a single clone is taken up-front so callers can rely on
    pure value semantics.
    """
    # One up-front clone so we never mutate the caller's tensor (and so
    # subsequent in-loop ``x[:, j] = …`` assignments are safe regardless
    # of how the caller obtained ``x``).
    x = x.clone()
    sol_size, num_vars = x.shape
    perm = torch.randperm(num_vars, generator=rng, device=x.device)
    # Pre-compute the field qx = x @ q_sym once, then update it
    # incrementally as bits flip.
    qx = x @ q_sym  # shape (sol_size, num_vars)
    two_q_diag = 2.0 * q_diag
    for j in perm.tolist():
        x_j = x[:, j]
        qx_j = qx[:, j]
        # ΔE_j = (1 - 2 x_j) · (2 (Q x)_j - 2 x_j Q_jj + Q_jj)
        delta_e = (1.0 - 2.0 * x_j) * (2.0 * qx_j - x_j * two_q_diag[j] + q_diag[j])
        accept_p = torch.exp(torch.clamp(-beta * delta_e, max=0.0))
        u = torch.rand(sol_size, device=x.device, generator=rng)
        flip = u < accept_p
        # Δx_j = (new - old) = (1 - x_j) - x_j   when flipped, else 0.
        # Using x_j ∈ {0, 1}: Δx_j = +1 if x_j was 0, −1 if x_j was 1.
        # Pack that into a per-replica scalar that's 0 when flip=False.
        delta_x_j = torch.where(flip, 1.0 - 2.0 * x_j, torch.zeros_like(x_j))
        x[:, j] = x_j + delta_x_j
        # Rank-1 update of the cached field.
        qx = qx + delta_x_j.unsqueeze(1) * q_sym[j, :].unsqueeze(0)
    return x


# Legacy alias kept for one release so external users get a clear
# DeprecationWarning if they were importing the buggy parallel sweep
# directly. The default sampler is now the sequential version above.
def _qubo_glauber_sweep(
    x: torch.Tensor,
    q_sym: torch.Tensor,
    q_diag: torch.Tensor,
    beta: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Deprecated alias — forwards to :func:`_qubo_seq_glauber_sweep`.

    The previous implementation did parallel single-flip proposals on the
    same pre-sweep state, which mode-locks on MIS / MaxCut. Use
    :func:`_qubo_seq_glauber_sweep` (correct) or
    :func:`_qubo_parallel_metropolis_sweep` (paper-reproduction baseline).
    """
    return _qubo_seq_glauber_sweep(x, q_sym, q_diag, beta, rng)


def _qubo_parallel_metropolis_sweep(
    x: torch.Tensor,
    q_sym: torch.Tensor,
    q_diag: torch.Tensor,
    beta: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """One **fully parallel** Metropolis sweep on a binary QUBO.

    .. warning::
       This is the historical "parallel-tempered classical SA" baseline.
       Every bit is proposed against the *same* pre-sweep ``x`` and
       accepted independently — which **breaks detailed balance on any
       coupled QUBO** and produces deterministic 0^N ↔ 1^N oscillation
       on regular-graph MIS at any temperature. It is retained only for
       reproducibility of the QQA-paper baselines; do **not** use it as
       a sampler.
    """
    qx = x @ q_sym
    delta_e = (1.0 - 2.0 * x) * (2.0 * qx - 2.0 * x * q_diag + q_diag)
    accept_p = torch.exp(torch.clamp(-beta * delta_e, max=0.0))
    u = torch.rand(x.shape, device=x.device, generator=rng)
    return torch.where(u < accept_p, 1.0 - x, x)


def _seq_mh_sweep(
    x: torch.Tensor,
    problem,
    beta: float,
    num_vars: int,
    is_spin: bool,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sequential single-spin Metropolis-Hastings sweep.

    Correct for *any* ``problem`` exposing ``loss_fn(x)`` — N+1 loss
    evaluations per sweep, far slower than the QUBO fast path but
    relaxation-agnostic.
    """
    perm = torch.randperm(num_vars, generator=rng, device=x.device)
    sol_size = x.shape[0]
    for j in perm.tolist():
        x_new = x.clone()
        if is_spin:
            x_new[:, j] = -x_new[:, j]
        else:
            x_new[:, j] = 1.0 - x_new[:, j]
        delta = problem.loss_fn(x_new) - problem.loss_fn(x)
        accept_p = torch.exp(torch.clamp(-beta * delta, max=0.0))
        u = torch.rand(sol_size, device=x.device, generator=rng)
        x = torch.where((u < accept_p).unsqueeze(-1), x_new, x)
    return x


def _validate_chain_problem(problem) -> tuple[bool, bool, int]:
    """Shared validation for SA / PA — returns ``(is_spin, is_binary, num_vars)``.

    Raises ``NotImplementedError`` for categorical / batched-instance /
    structured-shape binary problems (the chain backends only handle
    single-instance binary or spin relaxations whose latent is a flat
    ``(B, N)`` tensor) and ``TypeError`` for any other unsupported
    relaxation.
    """
    relax = getattr(problem, "relaxation", None)
    if isinstance(relax, CategoricalRelaxation):
        raise NotImplementedError(
            f"{type(problem).__name__} uses a CategoricalRelaxation, which the "
            "chain-based backends (SA / PA) do not support. Use qqa.anneal."
        )
    if hasattr(problem, "num_instance"):
        raise NotImplementedError(
            "Chain-based backends (SA / PA) do not support batched-instance "
            f"problems ({type(problem).__name__}); iterate over instances or "
            "use qqa.anneal which handles batched instances natively."
        )
    # Some binary problems (TSP's penalty-method formulation) override the
    # latent shape via ``BinaryRelaxation(shape_fn=...)`` to lift a flat
    # ``(B, N*N)`` latent into a position×city grid. ``loss_fn`` then uses
    # einsum / masked sums that only accept the multi-axis layout, so the
    # chain primitives (which sample a flat ``(B, N)`` state) would blow up
    # inside ``loss_fn`` with an opaque einsum error. Reject at the API
    # boundary so users see an actionable message instead.
    if isinstance(relax, BinaryRelaxation) and getattr(relax, "_shape_fn", None) is not None:
        raise NotImplementedError(
            f"{type(problem).__name__} uses a structured BinaryRelaxation "
            "(non-flat latent shape), which the chain-based backends "
            "(SA / PA) cannot sample into. Use qqa.anneal, which honours "
            "the relaxation's ``shape_fn`` natively."
        )
    is_spin = isinstance(relax, SpinRelaxation)
    is_binary = isinstance(relax, BinaryRelaxation) and not is_spin
    if not (is_spin or is_binary):
        raise TypeError(
            f"Unsupported relaxation {type(relax).__name__}; expected "
            "BinaryRelaxation or SpinRelaxation."
        )
    return is_spin, is_binary, _resolve_num_vars(problem)


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
    polish: bool = True,
) -> SAResult:
    """Run GPU-parallel Simulated Annealing on ``problem``.

    Parameters
    ----------
    problem:
        Any single-instance :class:`~qqa.problems.COProblem` exposing
        ``loss_fn(x)`` and a ``relaxation`` attribute
        (:class:`~qqa.relaxation.BinaryRelaxation` or
        :class:`~qqa.relaxation.SpinRelaxation`).
        :class:`~qqa.relaxation.CategoricalRelaxation` and batched-instance
        problems (those exposing ``num_instance``) are rejected at the API
        boundary with :class:`NotImplementedError`; iterate over instances
        and call :func:`simulated_annealing` per instance, or use
        :func:`qqa.anneal` which handles both natively.
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

        with torch.no_grad():
            if use_qubo_fast:
                x = _qubo_seq_glauber_sweep(x, q_sym, q_diag, beta, rng)
            else:
                x = _seq_mh_sweep(x, problem, beta, num_vars, is_spin, rng)

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

    # Default-on greedy 1-flip polish — mirrors ``qqa.anneal`` / ``population_annealing``
    # so every QQA4CO backend exposes the same post-processing contract.
    polished_sol: torch.Tensor | None = None
    if polish and getattr(problem, "Q_mat", None) is not None:
        polished_sol = greedy_one_flip(problem, best_sol_disc)
        with torch.no_grad():
            pol_obj = float(problem.loss_fn(polished_sol.unsqueeze(0)).item())
        if pol_obj < best_obj:
            best_obj = pol_obj
            best_sol_disc = polished_sol.detach()

    score = safe_score_summary(problem, best_sol_disc, fallback_obj=float(best_obj))

    return SAResult(
        best_sol=best_sol_disc,
        best_obj=best_obj,
        runtime=runtime,
        history=history,
        score=score,
        polished_sol=polished_sol,
    )
