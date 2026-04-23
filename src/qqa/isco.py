r"""GPU-parallel iSCO backend (Sun et al., ICML 2023).

Faithful implementation of the annealed discrete-Langevin sampler proposed
by **Sun, Goshvadi, Nova, Schuurmans, Dai**, *Revisiting Sampling for
Combinatorial Optimization*, ICML 2023
(https://proceedings.mlr.press/v202/sun23c.html), cross-checked against

* https://github.com/google-research/discs (``samplers/path_auxiliary.py``,
  ``samplers/dmala.py``);
* https://github.com/ruqizhang/discrete-langevin (``samplers.py``).

Unlike the initial port of this module, this revision implements the
paper's **Algorithm 1** and **Appendix C (PAS-MH-Step)** in full:

1. **Variable-length path** ``L ~ Poisson(μ)`` truncated to ``L ≥ 1``
   (paper Eq. 31 — ``α(L) ∝ μ^L e^{-μ}/L! · 1{L>0}``).
2. **Without-replacement** index sampling of ``L`` sites from
   ``w_j ∝ g(exp(-Δ_j/τ))`` via the Gumbel top-`L` trick (Eq. 28).
   The locally-balanced function is ``g(z) = √z`` (Zanella 2020),
   so ``log w_j = -Δ_j / (2τ)``.
3. **Path-auxiliary Metropolis-Hastings** on the ordered permutation
   ``σ`` (Eq. 30):

   .. math::

      A = \min\!\Bigl\{1,\;\frac{\pi(y)\,q_y(\sigma_r)}
                                 {\pi(x)\,q_x(\sigma)}\Bigr\},

   where ``q_x(σ) = ∏_k w_{σ_k} / (W_x − Σ_{m<k} w_{σ_m})``
   (Plackett-Luce) and ``σ_r`` is ``σ`` reversed.  For binary variables
   the per-site flip distribution ``q^j_x(y_j)`` is deterministic (a
   single alternative value), so the inner product of Eq. 30 collapses.
3. **Adaptive path length**: ``μ ← clip(μ + 0.001 · (Ā − 0.574), 1, N)``
   after every inner-loop batch (Eq. 31), tracking the optimal
   acceptance rate 0.574 from Sun et al. 2022.
4. **Two-level annealing loop**: ``num_steps`` outer temperature updates
   (``m`` in Algorithm 1) × ``num_inner`` inner MH steps each (``n``).
5. **Exponential temperature schedule** (``schedule="exp"`` — paper §5
   default).  ``"lin"`` reproduces the literal form of Algorithm 1, and
   ``"geom"`` is an alias for ``"exp"`` kept for compatibility with
   earlier benchmarks.

**Algorithmic summary.** For QUBO ``f(x) = x^⊤Q x`` on binary ``x``,
the exact one-flip energy delta is

.. math::

   \Delta_i(x) = (1 - 2x_i)\bigl(Q_{ii} + 2\,[(Qx)_i - Q_{ii}x_i]\bigr),

so ``f(σ_i x) − f(x) = Δ_i``.  The multi-flip energy change for the full
path ``J`` is computed exactly from the post-flip ``Qy`` cache (no
Taylor approximation required — QUBO is exactly quadratic).

**API parity with QQA4CO.**  The public entry point
:func:`discrete_langevin` mirrors :func:`qqa.simulated_annealing`
(``sol_size`` = ``num_chains``, ``polish=True`` by default, identical
history / callback conventions).  Both single-instance (``problem.Q_mat``)
and batched-instance (``problem.Q_tensor``) QUBO problems are supported.
Spin / Categorical / structured-shape relaxations are rejected at the
API boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import time
from typing import Any, Literal

import numpy as np
import torch

from qqa.polish import apply_polish_if_improves
from qqa.relaxation import BinaryRelaxation, CategoricalRelaxation, SpinRelaxation
from qqa.utils import require_cuda_if_requested, safe_score_summary

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class ISCOResult:
    """Result returned by :func:`discrete_langevin`.

    Attributes mirror :class:`qqa.SAResult` / :class:`qqa.PAResult` so the
    same plotting / scoring / benchmark helpers work unchanged. iSCO-
    specific diagnostics (``accept_rate``, ``num_steps``, ``num_inner``,
    ``num_chains``, ``t_max_used``, ``mu_final``, ``mean_path_length``)
    live alongside the standard fields.

    For **single-instance** problems ``best_sol`` is ``(N,)`` and
    ``best_obj`` is a Python ``float``; for **batched-instance** problems
    ``best_sol`` is ``(I, N)`` and ``best_obj`` is a NumPy array of
    length ``I``.
    """

    best_sol: torch.Tensor
    best_obj: Any
    runtime: float
    history: dict = field(default_factory=dict)
    score: dict = field(default_factory=dict)
    polished_sol: torch.Tensor | None = None
    accept_rate: float = 0.0
    num_steps: int = 0
    num_inner: int = 1
    num_chains: int = 0
    t_max_used: float = 0.0
    mu_final: float = 0.0
    mean_path_length: float = 0.0
    num_instance: int = 1


# ---------------------------------------------------------------------------
# Validation + temperature schedule + μ-schedule utilities
# ---------------------------------------------------------------------------


def _validate_qubo_problem(problem) -> tuple[bool, int]:
    """Return ``(is_batched, num_vars)`` when ``problem`` is a binary QUBO.

    Rejects categorical / spin / structured-shape / non-QUBO problems
    with a helpful :class:`NotImplementedError`.
    """
    relax = getattr(problem, "relaxation", None)
    if isinstance(relax, CategoricalRelaxation):
        raise NotImplementedError(
            f"{type(problem).__name__} uses a CategoricalRelaxation; "
            "iSCO requires a binary QUBO problem. Categorical iSCO "
            "(Sun et al. 2023 §3.2.1 Eq. 17) is not implemented in this "
            "backend — use qqa.anneal for categorical problems."
        )
    if isinstance(relax, SpinRelaxation):
        raise NotImplementedError(
            f"{type(problem).__name__} uses a SpinRelaxation; iSCO "
            "requires a binary {0,1} QUBO. Use qqa.simulated_annealing "
            "or qqa.population_annealing for spin problems."
        )
    q_mat = getattr(problem, "Q_mat", None)
    q_tensor = getattr(problem, "Q_tensor", None)
    num_instance = int(getattr(problem, "num_instance", 1) or 1)
    is_batched = num_instance > 1 or q_tensor is not None
    if (
        not is_batched
        and isinstance(relax, BinaryRelaxation)
        and getattr(relax, "_shape_fn", None) is not None
    ):
        raise NotImplementedError(
            f"{type(problem).__name__} uses a structured BinaryRelaxation "
            "(non-flat latent shape); iSCO cannot sample into it. Use "
            "qqa.anneal, which honours the relaxation's ``shape_fn`` natively."
        )
    if is_batched:
        if q_tensor is None:
            raise TypeError(
                "iSCO on a batched-instance problem requires `Q_tensor` of "
                f"shape (I, N, N). Got {type(problem).__name__} without "
                "Q_tensor."
            )
        if q_tensor.dim() != 3 or q_tensor.shape[-1] != q_tensor.shape[-2]:
            raise ValueError(f"Q_tensor must be (I, N, N); got {tuple(q_tensor.shape)}.")
        return True, int(q_tensor.shape[-1])

    if q_mat is None:
        raise TypeError(
            "iSCO requires a QUBO problem exposing `Q_mat` (single-instance) "
            f"or `Q_tensor` (batched). Got {type(problem).__name__} without "
            "either. Use qqa.anneal if your problem has no QUBO form."
        )
    if q_mat.dim() != 2 or q_mat.shape[0] != q_mat.shape[1]:
        raise ValueError(f"Q_mat must be square (N, N); got {tuple(q_mat.shape)}.")
    return False, int(q_mat.shape[0])


def _temperature_schedule(
    num_steps: int,
    t_max: float,
    t_min: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    kind: str = "exp",
) -> torch.Tensor:
    """Return a ``(num_steps,)`` tensor of temperatures used at each outer step.

    - ``kind='exp'`` / ``'geom'`` (default; Sun et al. 2023 §5):
      :math:`T_i = T_{\\max} (T_{\\min}/T_{\\max})^{i/(m-1)}`.
    - ``kind='lin'`` (Sun et al. Algorithm 1 literal):
      :math:`T_i = T_{\\max}(1 - i/m) + T_{\\min}\\cdot i/m`  —  a linear
      interpolation that reaches ``t_min`` at the last step (the paper
      writes ``τ_0 (1 − (i+1)/m)``; we clamp to ``t_min`` to avoid a
      divide-by-zero in the MH acceptance).
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    if t_max <= 0 or t_min <= 0:
        raise ValueError(f"temperatures must be positive (got t_max={t_max}, t_min={t_min}).")
    if kind in ("exp", "geom"):
        if num_steps == 1:
            return torch.tensor([t_min], device=device, dtype=dtype)
        ratio = (t_min / t_max) ** (1.0 / (num_steps - 1))
        t = torch.empty(num_steps, device=device, dtype=dtype)
        t[0] = t_max
        for i in range(1, num_steps):
            t[i] = t[i - 1] * ratio
        return t
    if kind == "lin":
        return torch.linspace(t_max, t_min, steps=num_steps, device=device, dtype=dtype)
    raise ValueError(f"Unknown schedule kind '{kind}'. Use 'exp', 'geom', or 'lin'.")


def _auto_t_max(
    abs_delta: torch.Tensor,
    t_min: float,
    quantile: float,
    gen: torch.Generator | None,
) -> float:
    """Calibrate ``t_max`` from the ``quantile`` of the initial |Δ| distribution.

    With the SQRT-balanced proposal ``τ = 1/(2T)``, keeping the initial
    acceptance ≈0.5 requires ``T ≈ |Δ|/2``.  We pick the 95 %-quantile
    by default (same recipe as DISCS' ``dmala.py``).
    """
    flat = abs_delta.flatten()
    if flat.numel() == 0:
        return max(float(t_min) * 10.0, 1.0)
    MAX_Q = 1 << 23  # 8 M; well below the 2**24 sort-backend cap.
    if flat.numel() > MAX_Q:
        idx = torch.randint(0, flat.numel(), (MAX_Q,), device=flat.device, generator=gen)
        flat = flat[idx]
    q = float(torch.quantile(flat, quantile).item())
    return max(q, float(t_min) * 10.0, 1.0)


# ---------------------------------------------------------------------------
# Path-auxiliary helpers (Gumbel top-L, Plackett-Luce log-prob, path utils)
# ---------------------------------------------------------------------------


def _gumbel_topL(
    log_w: torch.Tensor,
    L_max: int,
    *,
    gen: torch.Generator | None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ordered top-``L_max`` indices drawn without replacement from
    ``Categorical(softmax(log_w, dim=-1))`` (Gumbel-Max / Plackett-Luce).

    ``log_w`` is ``(*, N)``; optional ``mask`` of the same shape masks out
    sites that must not be picked (logit → −inf).
    """
    if mask is not None:
        log_w = torch.where(mask.bool(), log_w, torch.full_like(log_w, -1e30))
    u = torch.rand(log_w.shape, device=log_w.device, dtype=log_w.dtype, generator=gen)
    u = u.clamp_min_(1e-38)
    gumbel = -torch.log(-torch.log(u))
    z = log_w + gumbel
    _, topk = torch.topk(z, L_max, dim=-1)  # (*, L_max)
    return topk


def _plackett_luce_logprob(
    log_w: torch.Tensor,
    sigma: torch.Tensor,
    L_per_chain: torch.Tensor,
    L_max: int,
) -> torch.Tensor:
    """Compute ``log q(σ)`` for Plackett-Luce sampling without replacement.

    Parameters
    ----------
    log_w : ``(*, N)``
        Unnormalised log-weights.
    sigma : ``(*, L_max)``
        Ordered permutation indices; only positions ``< L_per_chain[*]``
        are considered valid.
    L_per_chain : ``(*,)``
        Valid path length for each chain (int tensor).
    L_max : int

    Returns
    -------
    ``(*,)`` log-probability of the ordered path (sum over valid
    positions; invalid positions contribute 0).
    """
    dtype = log_w.dtype
    device = log_w.device
    chosen_log_w = torch.gather(log_w, dim=-1, index=sigma)  # (*, L_max)
    G = torch.logsumexp(log_w, dim=-1, keepdim=False)  # (*,)
    log_remaining = torch.empty_like(chosen_log_w)
    log_remaining[..., 0] = G
    # Clamp ``diff`` strictly below 0 so ``log1p(-exp(·))`` stays finite.
    # CRITICAL: the threshold must be representable in the working dtype.
    # ``-1e-12`` round-trips to ``0.0`` in ``float32`` (whose epsilon is
    # ~1.19e-7), making the clamp a no-op and sending the recursion into
    # ``log(0) = -inf`` whenever ``sigma`` contains repeated indices (which
    # happens for every reverse path with ``L_per_chain < L_max`` because
    # ``_reverse_path`` clamps the tail to ``sigma[0]``).  Using ``-1e-6``
    # is safe in float32 and keeps the recursion in finite arithmetic.
    eps_clamp = -1e-6 if dtype == torch.float32 else -1e-12
    for k in range(1, L_max):
        diff = chosen_log_w[..., k - 1] - log_remaining[..., k - 1]
        diff = diff.clamp(max=eps_clamp)
        log_remaining[..., k] = log_remaining[..., k - 1] + torch.log1p(-torch.exp(diff))
    log_q_per_step = chosen_log_w - log_remaining  # (*, L_max)
    shape_prefix = log_w.shape[:-1]
    k_idx = torch.arange(L_max, device=device).view((1,) * len(shape_prefix) + (L_max,))
    mask = k_idx < L_per_chain.unsqueeze(-1)  # (*, L_max)
    # Zero masked positions via ``where`` (not ``* mask.to(dtype)``) so that
    # any non-finite ``log_q_per_step`` entry at a masked tail position
    # cannot poison the sum through ``inf * 0 = NaN``.
    log_q_per_step = torch.where(mask, log_q_per_step, torch.zeros_like(log_q_per_step))
    return log_q_per_step.sum(dim=-1)


def _reverse_path(sigma: torch.Tensor, L_per_chain: torch.Tensor, L_max: int) -> torch.Tensor:
    """Return ``σ`` reversed per chain: ``σ_r[k] = σ[L − 1 − k]`` for
    ``k < L_per_chain`` (invalid positions filled with ``σ[0]`` — they are
    masked out downstream)."""
    k_idx = torch.arange(L_max, device=sigma.device).view((1,) * (sigma.dim() - 1) + (L_max,))
    rev_k = L_per_chain.unsqueeze(-1) - 1 - k_idx  # (*, L_max)
    rev_k_clamped = rev_k.clamp(min=0)
    return torch.gather(sigma, dim=-1, index=rev_k_clamped)


# ---------------------------------------------------------------------------
# Single-instance kernel
# ---------------------------------------------------------------------------


@torch.no_grad()
def _isco_single(
    problem,
    *,
    sol_size: int,
    num_steps: int,
    num_inner: int,
    t_max: float | None,
    t_min: float,
    schedule: str,
    mu0: float,
    target_accept: float,
    mu_step: float,
    device: torch.device,
    seed: int | None,
    initial_state: torch.Tensor | None,
    t_max_quantile: float,
    record_history: bool,
    history_stride: int,
    verbose: bool,
    check_interval: int,
    callback: Callable[[int, float, float], None] | None,
) -> tuple[
    torch.Tensor,
    float,
    dict,
    int,
    int,
    float,
    float,
    float,
]:
    """Run iSCO on a single-instance QUBO.

    Returns
    -------
    ``(best_bits, best_obj, history, accepted, total, t_max_used,
      mu_final, mean_path_length)``.
    """
    Q = problem.Q_mat.to(device=device).float()
    Q = 0.5 * (Q + Q.T)  # symmetrise for safety
    N = Q.shape[0]
    diag = torch.diagonal(Q).contiguous()

    if seed is not None:
        gen: torch.Generator | None = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    else:
        gen = None

    S = int(sol_size)
    if initial_state is not None:
        x = initial_state.to(device=device, dtype=Q.dtype)
        if x.shape != (S, N):
            raise ValueError(f"initial_state shape {tuple(x.shape)} != expected {(S, N)}.")
        x = x.clone()
    else:
        x = torch.bernoulli(torch.full((S, N), 0.5, device=device, dtype=Q.dtype), generator=gen)
    Qx = x @ Q  # (S, N)
    energies = (x * Qx).sum(dim=-1)  # (S,)

    if t_max is None:
        delta_init = (1.0 - 2.0 * x) * (diag + 2.0 * (Qx - diag * x))
        t_max_eff = _auto_t_max(delta_init.abs(), t_min, t_max_quantile, gen)
    else:
        t_max_eff = float(t_max)
    temps = _temperature_schedule(
        num_steps, t_max_eff, t_min, device=device, dtype=Q.dtype, kind=schedule
    )

    accepted = 0
    total = 0
    path_len_sum = 0.0  # for mean path length diagnostic
    path_len_count = 0
    mu = float(mu0)

    best_chain = int(torch.argmin(energies).item())
    best_bits = x[best_chain].clone()
    best_obj = float(energies[best_chain].item())

    history: dict[str, list] = {
        "loss_mean": [],
        "loss_min": [],
        "best_obj": [],
        "temp": [],
        "accept_rate_cum": [],
        "mu": [],
        "mean_L": [],
    }

    for t_idx in range(num_steps):
        T = float(temps[t_idx].item())
        inv_T = 1.0 / max(T, 1e-12)
        inv2T = 0.5 * inv_T

        batch_accepted = 0
        batch_total = 0
        batch_path_len = 0.0
        batch_path_count = 0

        for _inner in range(num_inner):
            # Forward proposal weights
            delta = (1.0 - 2.0 * x) * (diag + 2.0 * (Qx - diag * x))  # (S, N)
            log_w = -delta * inv2T  # log w_j = -Δ_j / (2τ) (g = √·)

            # Sample per-chain path length L_s ~ Poisson(μ) truncated at L≥1.
            mu_tensor = torch.full((S,), mu, device=device, dtype=Q.dtype)
            L_s = torch.poisson(mu_tensor, generator=gen).to(torch.long).clamp_(min=1, max=N)
            L_max = int(L_s.max().item())

            # Gumbel top-L without replacement → ordered σ (S, L_max).
            sigma = _gumbel_topL(log_w, L_max, gen=gen)

            # Apply flips to get y. Gumbel top-L guarantees distinct cols
            # within a chain, so we can safely scatter-flip per k.
            y = x.clone()
            k_idx = torch.arange(L_max, device=device)
            mask_path = k_idx.unsqueeze(0) < L_s.unsqueeze(-1)  # (S, L_max)
            for k in range(L_max):
                active = mask_path[:, k]
                if not bool(active.any().item()):
                    continue
                s_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
                cols = sigma[s_idx, k]
                y[s_idx, cols] = 1.0 - y[s_idx, cols]

            # Post-flip cache Qy (single matmul; Q symmetric so x@Q = Q@x).
            Qy = y @ Q
            energies_y = (y * Qy).sum(dim=-1)  # (S,)
            dE = energies_y - energies  # (S,)

            # Reverse proposal weights at y
            delta_y = (1.0 - 2.0 * y) * (diag + 2.0 * (Qy - diag * y))
            log_w_y = -delta_y * inv2T
            sigma_rev = _reverse_path(sigma, L_s, L_max)

            log_q_fwd = _plackett_luce_logprob(log_w, sigma, L_s, L_max)
            log_q_rev = _plackett_luce_logprob(log_w_y, sigma_rev, L_s, L_max)

            log_alpha = -dE * inv_T + log_q_rev - log_q_fwd
            u = torch.rand(S, device=device, generator=gen, dtype=Q.dtype).clamp_min_(1e-38)
            accept = torch.log(u) < log_alpha  # (S,)

            # Commit accepted flips; restore x (and Qx, energies) for rejected chains.
            x = torch.where(accept.unsqueeze(-1), y, x)
            Qx = torch.where(accept.unsqueeze(-1), Qy, Qx)
            energies = torch.where(accept, energies_y, energies)

            batch_accepted += int(accept.sum().item())
            batch_total += S
            batch_path_len += float(L_s.to(Q.dtype).sum().item())
            batch_path_count += S

            # Track running best-ever across the whole trajectory.
            step_min_val, step_min_idx = torch.min(energies, dim=0)
            if float(step_min_val.item()) < best_obj:
                best_obj = float(step_min_val.item())
                best_bits = x[int(step_min_idx.item())].clone()

        accepted += batch_accepted
        total += batch_total
        path_len_sum += batch_path_len
        path_len_count += batch_path_count

        # μ-adaptation (Eq. 31). Use batch-mean acceptance rate over the
        # inner loop so μ responds to the most recent temperature level.
        A_bar = batch_accepted / max(1, batch_total)
        mu = max(1.0, min(float(N), mu + mu_step * (A_bar - target_accept)))

        should_record = record_history and (t_idx % history_stride == 0 or t_idx == num_steps - 1)
        if should_record:
            history["loss_mean"].append(float(energies.mean().item()))
            history["loss_min"].append(float(energies.min().item()))
            history["best_obj"].append(best_obj)
            history["temp"].append(T)
            history["accept_rate_cum"].append(accepted / max(1, total))
            history["mu"].append(mu)
            history["mean_L"].append(batch_path_len / max(1, batch_path_count))

        if callback is not None and should_record:
            callback(t_idx, float(energies.mean().item()), best_obj)

        if verbose and (t_idx % check_interval == 0 or t_idx == num_steps - 1):
            print(
                f"[iSCO] step {t_idx:>6d}  T={T:.4f}  μ={mu:.2f}  "
                f"meanL={batch_path_len / max(1, batch_path_count):.2f}  "
                f"mean_E={float(energies.mean().item()):.4f}  "
                f"best={best_obj:.4f}  "
                f"acc={accepted / max(1, total):.3f}"
            )

    mean_path_length = path_len_sum / max(1, path_len_count)
    return (
        best_bits,
        best_obj,
        history,
        accepted,
        total,
        t_max_eff,
        mu,
        mean_path_length,
    )


# ---------------------------------------------------------------------------
# Batched-instance kernel
# ---------------------------------------------------------------------------


@torch.no_grad()
def _isco_batched(
    problem,
    *,
    sol_size: int,
    num_steps: int,
    num_inner: int,
    t_max: float | None,
    t_min: float,
    schedule: str,
    mu0: float,
    target_accept: float,
    mu_step: float,
    device: torch.device,
    seed: int | None,
    t_max_quantile: float,
    record_history: bool,
    history_stride: int,
    verbose: bool,
    check_interval: int,
    callback: Callable[[int, float, float], None] | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict,
    int,
    int,
    float,
    float,
    float,
]:
    """Run iSCO on a batched-instance QUBO (``Q_tensor`` of shape ``(I, N, N)``).

    Returns
    -------
    ``(best_bits (I, N), best_obj (I,), history, accepted, total,
      t_max_used, mu_final, mean_path_length)``.  ``initial_state`` is
    not supported for batched problems — there is no canonical per-
    instance seed and :func:`qqa.anneal` follows the same policy.
    """
    Q = problem.Q_tensor.to(device=device).float()
    Q = 0.5 * (Q + Q.transpose(-1, -2))
    I, N, _ = Q.shape  # noqa: E741 - paper-faithful: I = #instances, mirrors (S, I, N) tensor convention used throughout this kernel
    diag = torch.diagonal(Q, dim1=-2, dim2=-1).contiguous()

    pm = getattr(problem, "pad_mask", None)
    if pm is None:
        mask_var = torch.ones((I, N), dtype=Q.dtype, device=device)
    else:
        mask_var = pm.to(device=device, dtype=Q.dtype)
        if mask_var.dim() == 3 and mask_var.shape[0] == 1:
            mask_var = mask_var.squeeze(0)

    if seed is not None:
        gen: torch.Generator | None = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    else:
        gen = None

    S = int(sol_size)
    x = torch.bernoulli(torch.full((S, I, N), 0.5, device=device, dtype=Q.dtype), generator=gen)
    x = x * mask_var.unsqueeze(0)
    Qx = torch.einsum("inm,sim->sin", Q, x)
    energies = (x * Qx).sum(dim=-1)  # (S, I)

    if t_max is None:
        delta_init = (1.0 - 2.0 * x) * (diag.unsqueeze(0) + 2.0 * (Qx - diag.unsqueeze(0) * x))
        abs_d = (delta_init.abs() * mask_var.unsqueeze(0)).flatten()
        mask_flat = mask_var.unsqueeze(0).expand(S, I, N).flatten()
        nonzero = abs_d[mask_flat > 0]
        if nonzero.numel() == 0:
            t_max_eff = max(float(t_min) * 10.0, 1.0)
        else:
            t_max_eff = _auto_t_max(nonzero, t_min, t_max_quantile, gen)
    else:
        t_max_eff = float(t_max)
    temps = _temperature_schedule(
        num_steps, t_max_eff, t_min, device=device, dtype=Q.dtype, kind=schedule
    )

    accepted = 0
    total = 0
    path_len_sum = 0.0
    path_len_count = 0
    mu = float(mu0)

    # Valid-variable count per instance (for μ clamp).
    N_per_inst = mask_var.sum(dim=-1).clamp(min=1)  # (I,)

    best_obj = energies.min(dim=0).values.clone()  # (I,)
    best_bits_per_inst = torch.empty((I, N), device=device, dtype=Q.dtype)
    min_chain = energies.argmin(dim=0)  # (I,)
    for i in range(I):
        best_bits_per_inst[i] = x[int(min_chain[i].item()), i]

    history: dict[str, list] = {
        "loss_mean": [],
        "loss_min": [],
        "best_obj": [],
        "temp": [],
        "accept_rate_cum": [],
        "mu": [],
        "mean_L": [],
    }

    mask_var_b = mask_var.unsqueeze(0).bool()  # (1, I, N)

    for t_idx in range(num_steps):
        T = float(temps[t_idx].item())
        inv_T = 1.0 / max(T, 1e-12)
        inv2T = 0.5 * inv_T

        batch_accepted = 0
        batch_total = 0
        batch_path_len = 0.0
        batch_path_count = 0

        for _inner in range(num_inner):
            delta = (1.0 - 2.0 * x) * (diag.unsqueeze(0) + 2.0 * (Qx - diag.unsqueeze(0) * x))
            log_w = -delta * inv2T  # (S, I, N)

            mu_tensor = torch.full((S, I), mu, device=device, dtype=Q.dtype)
            L_s = torch.poisson(mu_tensor, generator=gen).to(torch.long)
            # Clamp per-instance to the valid-variable count (so μ doesn't
            # ask for more flips than exist on ragged instances).
            L_s = L_s.clamp_(min=1)
            L_s = torch.minimum(L_s, N_per_inst.view(1, I).expand(S, I).to(torch.long))
            L_max = int(L_s.max().item())

            sigma = _gumbel_topL(log_w, L_max, gen=gen, mask=mask_var_b.expand(S, I, N))

            y = x.clone()
            k_arange = torch.arange(L_max, device=device)
            mask_path = k_arange.view(1, 1, L_max) < L_s.unsqueeze(-1)  # (S, I, L_max)
            # Flip loop (L_max iterations; within each iteration distinct
            # cols per (s, i) are guaranteed by Gumbel top-L).
            for k in range(L_max):
                active = mask_path[..., k]  # (S, I)
                if not bool(active.any().item()):
                    continue
                cols = sigma[..., k]  # (S, I)
                # Build a safe index tensor: we only read/write where active.
                # Use torch.gather then conditional update.
                cols_safe = cols.clamp(min=0, max=N - 1)
                x_at = torch.gather(y, dim=-1, index=cols_safe.unsqueeze(-1)).squeeze(-1)
                new_at = torch.where(active, 1.0 - x_at, x_at)
                y.scatter_(dim=-1, index=cols_safe.unsqueeze(-1), src=new_at.unsqueeze(-1))

            Qy = torch.einsum("inm,sim->sin", Q, y)
            energies_y = (y * Qy).sum(dim=-1)  # (S, I)
            dE = energies_y - energies

            delta_y = (1.0 - 2.0 * y) * (diag.unsqueeze(0) + 2.0 * (Qy - diag.unsqueeze(0) * y))
            log_w_y = -delta_y * inv2T
            sigma_rev = _reverse_path(sigma, L_s, L_max)

            log_q_fwd = _plackett_luce_logprob(log_w, sigma, L_s, L_max)
            log_q_rev = _plackett_luce_logprob(log_w_y, sigma_rev, L_s, L_max)

            log_alpha = -dE * inv_T + log_q_rev - log_q_fwd
            u = torch.rand(S, I, device=device, generator=gen, dtype=Q.dtype).clamp_min_(1e-38)
            accept = torch.log(u) < log_alpha  # (S, I)

            accept_b = accept.unsqueeze(-1).bool()
            x = torch.where(accept_b, y, x)
            Qx = torch.where(accept_b, Qy, Qx)
            energies = torch.where(accept, energies_y, energies)

            batch_accepted += int(accept.sum().item())
            batch_total += S * I
            batch_path_len += float(L_s.to(Q.dtype).sum().item())
            batch_path_count += S * I

            step_min_vals, step_min_idx = torch.min(energies, dim=0)  # (I,), (I,)
            improved = step_min_vals < best_obj
            if bool(improved.any().item()):
                best_obj = torch.where(improved, step_min_vals, best_obj)
                sel = x[step_min_idx, torch.arange(I, device=device)]  # (I, N)
                best_bits_per_inst = torch.where(improved.unsqueeze(-1), sel, best_bits_per_inst)

        accepted += batch_accepted
        total += batch_total
        path_len_sum += batch_path_len
        path_len_count += batch_path_count

        A_bar = batch_accepted / max(1, batch_total)
        mu = max(1.0, min(float(N), mu + mu_step * (A_bar - target_accept)))

        should_record = record_history and (t_idx % history_stride == 0 or t_idx == num_steps - 1)
        if should_record:
            history["loss_mean"].append(float(energies.mean().item()))
            history["loss_min"].append(float(energies.min().item()))
            history["best_obj"].append(float(best_obj.mean().item()))
            history["temp"].append(T)
            history["accept_rate_cum"].append(accepted / max(1, total))
            history["mu"].append(mu)
            history["mean_L"].append(batch_path_len / max(1, batch_path_count))

        if callback is not None and should_record:
            callback(t_idx, float(energies.mean().item()), float(best_obj.mean().item()))

        if verbose and (t_idx % check_interval == 0 or t_idx == num_steps - 1):
            print(
                f"[iSCO] step {t_idx:>6d}  T={T:.4f}  μ={mu:.2f}  "
                f"meanL={batch_path_len / max(1, batch_path_count):.2f}  "
                f"best(mean over I)={float(best_obj.mean().item()):.4f}  "
                f"acc={accepted / max(1, total):.3f}"
            )

    mean_path_length = path_len_sum / max(1, path_len_count)
    return (
        best_bits_per_inst,
        best_obj,
        history,
        accepted,
        total,
        t_max_eff,
        mu,
        mean_path_length,
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def discrete_langevin(
    problem,
    *,
    sol_size: int = 128,
    num_steps: int = 3000,
    num_inner: int = 1,
    t_max: float | None = None,
    t_min: float = 0.01,
    schedule: Literal["exp", "geom", "lin"] = "exp",
    mu0: float = 1.0,
    target_accept: float = 0.574,
    mu_step: float = 0.001,
    device: str | torch.device = "cpu",
    seed: int | None = None,
    initial_state: torch.Tensor | None = None,
    t_max_quantile: float = 0.95,
    record_history: bool = True,
    history_stride: int = 1,
    verbose: bool = True,
    check_interval: int = 100,
    callback: Callable[[int, float, float], None] | None = None,
    polish: bool = True,
) -> ISCOResult:
    """Run GPU-parallel iSCO on ``problem`` (Sun et al., ICML 2023).

    Implements Algorithm 1 + Appendix C of **Sun, Goshvadi, Nova,
    Schuurmans, Dai**, *Revisiting Sampling for Combinatorial
    Optimization*, ICML 2023 (pmlr-v202-sun23c).  Each outer iteration
    lowers the temperature once; ``num_inner`` MH steps are run at each
    temperature.  Every MH step samples a Poisson-length path ``L ~
    Poisson(μ), L≥1``, picks ``L`` sites without replacement via
    Gumbel-top-L with logits ``-Δ/(2τ)``, flips them, and accepts via
    the path-auxiliary MH correction (Eq. 30).  ``μ`` is adapted to
    track the target acceptance rate 0.574.

    Parameters
    ----------
    problem:
        Binary QUBO problem exposing either ``Q_mat`` ``(N, N)`` (single
        instance) or ``Q_tensor`` ``(I, N, N)`` (batched instances).
    sol_size:
        Number of parallel chains — the ``num_chains`` knob in the
        paper.  DISCS defaults are 128-1024.
    num_steps:
        Outer annealing steps = number of distinct temperatures in the
        schedule.  (Paper's ``m`` in Algorithm 1.)
    num_inner:
        MH steps at each temperature.  (Paper's ``n`` in Algorithm 1.)
        Total MH steps per chain = ``num_steps · num_inner``.
    t_max, t_min:
        Initial / final temperature.  Leave ``t_max=None`` for the
        DISCS-style auto-calibration from the 95 %-quantile of
        ``|Δ|`` on the initial Bernoulli sample.
    schedule:
        ``"exp"`` (default; §5 paper) / ``"geom"`` (alias) /
        ``"lin"`` (Algorithm 1 literal linear decay).
    mu0:
        Initial Poisson mean for path length.  Paper uses ``μ_0 = 1``.
    target_accept:
        Target batch-mean acceptance rate for μ-adaptation (Eq. 31;
        paper 0.574 — the Sun et al. 2022 Langevin optimum).
    mu_step:
        Step size of the μ-update ``μ ← μ + mu_step · (Ā − target)``.
        Paper ``mu_step = 0.001``.
    device, seed, initial_state, record_history, history_stride,
    verbose, check_interval, callback:
        Same semantics as :func:`qqa.simulated_annealing`.
        ``initial_state`` applies only to single-instance runs.
    polish:
        If ``True`` (default), run :func:`qqa.polish.greedy_one_flip`
        via :func:`qqa.polish.apply_polish_if_improves` on the best
        single-instance incumbent after annealing.

    Returns
    -------
    ISCOResult
        ``best_sol`` / ``best_obj`` / ``runtime`` / ``history`` /
        ``score`` / ``polished_sol`` (standard QQA4CO surface) plus the
        iSCO-specific ``accept_rate``, ``num_steps``, ``num_inner``,
        ``num_chains``, ``t_max_used``, ``mu_final``, ``mean_path_length``,
        ``num_instance``.

    Examples
    --------
    Single-instance MIS (drop-in replacement for :func:`qqa.anneal`) ::

        import networkx as nx
        import qqa

        qqa.fix_seed(0)
        g = nx.random_regular_graph(d=3, n=200, seed=0)
        problem = qqa.MaximumIndependentSet(g, penalty=2)
        result = qqa.discrete_langevin(
            problem, sol_size=256, num_steps=500, num_inner=4, device="cuda",
        )
        print(-int(result.best_obj), result.accept_rate, result.mu_final)
    """
    if sol_size < 1:
        raise ValueError(f"sol_size must be >= 1, got {sol_size}.")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    if num_inner < 1:
        raise ValueError(f"num_inner must be >= 1, got {num_inner}.")
    if history_stride < 1:
        raise ValueError(f"history_stride must be >= 1, got {history_stride}.")
    if mu0 < 1.0:
        raise ValueError(f"mu0 must be >= 1.0, got {mu0}.")
    if not (0.0 < target_accept < 1.0):
        raise ValueError(f"target_accept must be in (0, 1), got {target_accept}.")

    require_cuda_if_requested(device)
    device_t = torch.device(device) if isinstance(device, str) else device

    is_batched, _ = _validate_qubo_problem(problem)

    t0 = time()
    if is_batched:
        if initial_state is not None:
            initial_state = None  # silently ignored, mirrors qqa.anneal
        (
            best_bits,
            best_obj_t,
            history,
            accepted,
            total,
            t_max_eff,
            mu_final,
            mean_L,
        ) = _isco_batched(
            problem,
            sol_size=sol_size,
            num_steps=num_steps,
            num_inner=num_inner,
            t_max=t_max,
            t_min=t_min,
            schedule=schedule,
            mu0=mu0,
            target_accept=target_accept,
            mu_step=mu_step,
            device=device_t,
            seed=seed,
            t_max_quantile=t_max_quantile,
            record_history=record_history,
            history_stride=history_stride,
            verbose=verbose,
            check_interval=check_interval,
            callback=callback,
        )
        runtime = time() - t0
        best_sol_disc = best_bits.to(torch.uint8).cpu()
        best_obj_np = best_obj_t.detach().cpu().numpy().astype(np.float64)

        score: dict = {}
        from qqa.problems.base import COProblem  # noqa: PLC0415

        if type(problem).score_summary is not COProblem.score_summary:
            try:
                score = problem.score_summary(best_sol_disc)
            except Exception as exc:  # noqa: BLE001 - surface but never abort
                score = {"label": "loss", "feasible": False, "extra": {"error": str(exc)}}

        if verbose:
            print(
                f"[iSCO] done. best(mean over I)={float(best_obj_np.mean()):.6f}  "
                f"acc={accepted / max(1, total):.3f}  μ={mu_final:.2f}  "
                f"runtime={runtime:.2f}s"
            )

        return ISCOResult(
            best_sol=best_sol_disc,
            best_obj=best_obj_np,
            runtime=runtime,
            history=history,
            score=score,
            polished_sol=None,
            accept_rate=accepted / max(1, total),
            num_steps=num_steps,
            num_inner=num_inner,
            num_chains=sol_size,
            t_max_used=t_max_eff,
            mu_final=mu_final,
            mean_path_length=mean_L,
            num_instance=int(best_bits.shape[0]),
        )

    (
        best_bits,
        best_obj,
        history,
        accepted,
        total,
        t_max_eff,
        mu_final,
        mean_L,
    ) = _isco_single(
        problem,
        sol_size=sol_size,
        num_steps=num_steps,
        num_inner=num_inner,
        t_max=t_max,
        t_min=t_min,
        schedule=schedule,
        mu0=mu0,
        target_accept=target_accept,
        mu_step=mu_step,
        device=device_t,
        seed=seed,
        initial_state=initial_state,
        t_max_quantile=t_max_quantile,
        record_history=record_history,
        history_stride=history_stride,
        verbose=verbose,
        check_interval=check_interval,
        callback=callback,
    )
    runtime = time() - t0

    best_sol_disc = best_bits.detach()
    best_sol_disc, best_obj, polished_sol = apply_polish_if_improves(
        problem, best_sol_disc, best_obj, polish=polish
    )
    score = safe_score_summary(problem, best_sol_disc, fallback_obj=float(best_obj))

    if verbose:
        print(
            f"[iSCO] done. best={best_obj:.6f}  "
            f"acc={accepted / max(1, total):.3f}  μ={mu_final:.2f}  "
            f"meanL={mean_L:.2f}  runtime={runtime:.2f}s"
        )

    return ISCOResult(
        best_sol=best_sol_disc,
        best_obj=float(best_obj),
        runtime=runtime,
        history=history,
        score=score,
        polished_sol=polished_sol,
        accept_rate=accepted / max(1, total),
        num_steps=num_steps,
        num_inner=num_inner,
        num_chains=sol_size,
        t_max_used=t_max_eff,
        mu_final=mu_final,
        mean_path_length=mean_L,
        num_instance=1,
    )


#: Paper-faithful alias. Sun et al. (ICML 2023) call the method iSCO; we
#: expose both names so citers can use whichever matches their paper.
isco_anneal = discrete_langevin


__all__ = ["ISCOResult", "discrete_langevin", "isco_anneal"]
