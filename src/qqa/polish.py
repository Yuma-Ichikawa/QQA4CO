"""Monotone local-search polishing across QUBO, spin, and categorical domains.

A free, deterministic improvement applied **after** annealing:

* :func:`greedy_one_flip` uses exact incremental deltas for QUBOs, including
  non-symmetric user matrices;
* :func:`greedy_spin_flip` uses the exact local field of a quadratic Ising
  model;
* :func:`greedy_categorical_move` evaluates one-site category changes in
  vectorised chunks for arbitrary categorical objectives.

Usage (default-on inside :func:`qqa.anneal` and the PI-GNN trainers; rarely
needed by hand)::

    polished = qqa.polish.greedy_one_flip(problem, result.best_sol)
"""

from __future__ import annotations

import torch


def _polish_solution(problem, best_sol: torch.Tensor) -> torch.Tensor | None:
    """Dispatch to the strongest safe local neighbourhood for the domain."""
    if getattr(problem, "sparse_qubo", None) is not None:
        return greedy_one_flip(problem, best_sol)
    if getattr(problem, "Q_mat", None) is not None:
        return greedy_one_flip(problem, best_sol)

    # Local import keeps this low-level helper independent of problem modules.
    from qqa.relaxation import CategoricalRelaxation, SpinRelaxation

    relaxation = getattr(problem, "relaxation", None)
    if isinstance(relaxation, SpinRelaxation):
        return greedy_spin_flip(problem, best_sol)
    if isinstance(relaxation, CategoricalRelaxation):
        return greedy_categorical_move(problem, best_sol)
    return None


@torch.no_grad()
def apply_polish_if_improves(
    problem,
    best_sol: torch.Tensor | None,
    best_obj: float,
    *,
    polish: bool = True,
) -> tuple[torch.Tensor | None, float, torch.Tensor | None]:
    """Run a domain-aware local search and hot-swap iff it improves.

    Returns ``(best_sol, best_obj, polished_sol)`` where:

    * ``polished_sol`` is the raw greedy result whenever the polish was
      attempted (kept for introspection, surfaced on
      ``AnnealResult.polished_sol`` / ``SAResult.polished_sol`` / …),
      or ``None`` when polishing was skipped (``polish=False``, unsupported
      relaxation, invalid solution shape, or ``best_sol is None``).
    * ``best_sol`` / ``best_obj`` are **replaced** only when the polish
      strictly improved the objective — preserving the "monotone free
      improvement" contract every QQA4CO backend expects.

    Centralising the pattern here keeps :func:`qqa.anneal`,
    :func:`qqa.simulated_annealing`, :func:`qqa.population_annealing` and
    the PI-GNN trainers in lock-step; any future tweak (e.g. a smarter
    multi-flip polish) lands once.
    """
    if not polish or best_sol is None:
        return best_sol, best_obj, None
    polished = _polish_solution(problem, best_sol)
    if polished is None:
        return best_sol, best_obj, None
    pol_obj = float(problem.loss_fn(polished.unsqueeze(0)).item())
    if pol_obj < best_obj:
        return polished, pol_obj, polished
    return best_sol, best_obj, polished


@torch.no_grad()
def greedy_one_flip(
    problem,
    bits: torch.Tensor,
    *,
    max_iters: int | None = None,
) -> torch.Tensor:
    """Return ``bits`` walked down to a 1-flip-locally optimal QUBO solution.

    For ``f(x) = x^\\top Q x`` and ``x ∈ {0,1}^N``, flipping bit ``i`` by
    ``d_i = 1 - 2x_i`` changes the loss by

    .. math::

        \\Delta_i = d_i\\left((Qx)_i + (Q^\\mathsf{T}x)_i\\right) + Q_{ii}.

    This form is correct even when a user supplies a non-symmetric QUBO
    matrix; only its symmetric part affects the quadratic form. Each
    iteration flips ``argmin_i Δ_i`` iff ``Δ < 0``. ``Qx`` and
    ``Q^T x`` are updated incrementally from one column and one row, yielding
    ``O(N)`` work per accepted flip and ``O(N · #flips)`` total.

    Returns ``bits`` unchanged when the problem has no ``Q_mat`` attribute
    (Spin / Categorical / batched-instance problems) — making the routine
    safe to invoke unconditionally as a post-processing step.
    """
    sparse = getattr(problem, "sparse_qubo", None)
    if sparse is not None and bits.ndim == 1 and bits.numel():
        from qqa.compile import SparseQUBO
        from qqa.local import sparse_qubo_descent

        if isinstance(sparse, SparseQUBO):
            return sparse_qubo_descent(
                sparse,
                bits,
                max_flips=max_iters,
            ).solution.to(device=bits.device, dtype=bits.dtype)
    Q = getattr(problem, "Q_mat", None)
    if Q is None or bits.numel() == 0 or bits.dim() != 1 or Q.dim() != 2:
        return bits
    x = bits.detach().to(Q.device, dtype=Q.dtype).clone()
    diag = torch.diagonal(Q)
    Qx = Q @ x
    Qtx = Q.T @ x
    cap = max_iters if max_iters is not None else 20 * x.numel()
    for _ in range(cap):
        delta = (1.0 - 2.0 * x) * (Qx + Qtx) + diag
        best_i = int(torch.argmin(delta).item())
        if float(delta[best_i].item()) >= 0.0:
            break
        sign = 1.0 - 2.0 * x[best_i]
        Qx = Qx + sign * Q[:, best_i]
        Qtx = Qtx + sign * Q[best_i, :]
        x[best_i] = 1.0 - x[best_i]
    return x.to(bits.dtype)


@torch.no_grad()
def greedy_spin_flip(
    problem,
    spins: torch.Tensor,
    *,
    max_iters: int | None = None,
) -> torch.Tensor | None:
    """Return a 1-spin-stable solution for a quadratic Ising energy.

    For

    .. math::

        E(s)=-\\tfrac12s^TJs-h^Ts,

    with symmetric ``J`` and zero diagonal, flipping spin ``i`` changes the
    energy by ``2 s_i ((J s)_i + h_i)``. The local field is updated with one
    matrix column after every accepted flip, so each iteration is ``O(N)``.
    Problems without a finite square ``J`` are left to their annealer because
    a generic p-spin or neural loss has no equivalent exact incremental field.
    """
    coupling = getattr(problem, "J", None)
    if (
        not torch.is_tensor(coupling)
        or spins.ndim != 1
        or coupling.ndim != 2
        or coupling.shape != (spins.numel(), spins.numel())
        or not torch.isfinite(coupling).all()
    ):
        return None
    x = spins.detach().to(coupling.device, dtype=coupling.dtype).clone()
    symmetric = 0.5 * (coupling + coupling.T)
    diagonal = torch.diagonal(symmetric)
    field = symmetric @ x - diagonal * x
    external = getattr(problem, "h", None)
    if torch.is_tensor(external):
        field = field + external.to(device=x.device, dtype=x.dtype)
    cap = max_iters if max_iters is not None else 20 * x.numel()
    for _ in range(cap):
        delta = 2.0 * x * field
        best_i = int(torch.argmin(delta).item())
        if float(delta[best_i].item()) >= -1e-10:
            break
        old_spin = x[best_i].clone()
        x[best_i] = -old_spin
        field = field - 2.0 * old_spin * symmetric[:, best_i]
        # The diagonal never contributes to Ising energy because s_i²=1.
        field[best_i] = field[best_i] + 2.0 * old_spin * diagonal[best_i]
    return x.to(device=spins.device, dtype=spins.dtype)


@torch.no_grad()
def greedy_categorical_move(
    problem,
    one_hot: torch.Tensor,
    *,
    max_iters: int | None = None,
    chunk_size: int = 256,
) -> torch.Tensor | None:
    """Descend over all one-site category changes in vectorised chunks.

    This objective-agnostic neighbourhood covers coloring, graph partition,
    and custom categorical models without assuming a graph-specific formula.
    The accepted objective is non-increasing. A small tabu set permits neutral
    plateau moves (essential for even-cycle 2-coloring) without cycling;
    :func:`apply_polish_if_improves` still replaces the public incumbent only
    after a strict net improvement. Candidate chunks bound peak memory while
    using the problem's batched GPU/CPU loss.
    """
    if one_hot.ndim != 2 or one_hot.shape[0] == 0 or one_hot.shape[1] < 2:
        return None
    x = one_hot.detach().clone()
    nodes, categories = x.shape
    current = float(problem.loss_fn(x.unsqueeze(0)).reshape(-1)[0].item())
    cap = max_iters if max_iters is not None else 4 * nodes
    moves = nodes * categories
    visited = {bytes(torch.argmax(x, dim=1).to(torch.int32).cpu().numpy())}
    for _ in range(cap):
        best_loss = current
        best_candidate = None
        plateau_candidate = None
        for start in range(0, moves, chunk_size):
            stop = min(start + chunk_size, moves)
            move_ids = torch.arange(start, stop, device=x.device)
            sites = torch.div(move_ids, categories, rounding_mode="floor")
            choices = move_ids.remainder(categories)
            candidates = x.unsqueeze(0).expand(stop - start, -1, -1).clone()
            rows = torch.arange(stop - start, device=x.device)
            candidates[rows, sites] = 0
            candidates[rows, sites, choices] = 1
            losses = problem.loss_fn(candidates).reshape(-1)
            value, index = torch.min(losses, dim=0)
            candidate_loss = float(value.item())
            if candidate_loss < best_loss - 1e-10:
                best_loss = candidate_loss
                best_candidate = candidates[int(index.item())].clone()
            if best_candidate is None and plateau_candidate is None:
                tied = torch.where(
                    torch.isclose(losses, losses.new_tensor(current), atol=1e-10, rtol=0.0)
                )[0]
                for tied_index in tied.tolist():
                    candidate = candidates[tied_index]
                    key = bytes(torch.argmax(candidate, dim=1).to(torch.int32).cpu().numpy())
                    if key not in visited:
                        plateau_candidate = candidate.clone()
                        break
        if best_candidate is not None:
            x = best_candidate
            current = best_loss
        elif plateau_candidate is not None:
            x = plateau_candidate
        else:
            break
        visited.add(bytes(torch.argmax(x, dim=1).to(torch.int32).cpu().numpy()))
    return x
