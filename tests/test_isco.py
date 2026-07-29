"""Smoke + correctness tests for the iSCO backend (Sun et al. ICML 2023).

Covers the paper-faithful Algorithm 1 + Appendix C (PAS-MH-Step)
implementation in ``qqa.isco``: Poisson-length multi-flip paths,
Plackett-Luce without-replacement proposals, path-auxiliary MH
correction, and μ-adaptation to target 0.574 acceptance.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

import qqa


def test_isco_is_exported_on_public_api():
    """The discrete-Langevin entry point and result type live on qqa."""
    assert hasattr(qqa, "discrete_langevin")
    assert hasattr(qqa, "isco_anneal")
    assert hasattr(qqa, "ISCOResult")
    assert qqa.isco_anneal is qqa.discrete_langevin
    for name in ("discrete_langevin", "isco_anneal", "ISCOResult"):
        assert name in qqa.__all__


def test_isco_finds_mis_on_path_graph():
    """P_6 has MIS size 3 → QUBO optimum -3. iSCO + polish should reach it."""
    qqa.fix_seed(0)
    g = nx.path_graph(6)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)

    result = qqa.discrete_langevin(
        problem,
        sol_size=64,
        num_steps=400,
        t_max=1.0,
        t_min=0.01,
        seed=0,
        device="cpu",
        verbose=False,
    )

    assert result.best_obj <= -3.0 + 1e-6, f"got best_obj={result.best_obj}"
    assert result.best_sol.shape == (6,)
    assert result.runtime > 0.0
    assert 0.0 <= result.accept_rate <= 1.0
    assert result.num_chains == 64
    assert result.num_steps == 400
    assert result.num_inner == 1
    assert result.t_max_used > 0.0
    assert result.mu_final >= 1.0
    assert result.mean_path_length >= 1.0
    # Default history keys match the SA surface, with the iSCO-specific
    # "temp" replacing SA's "beta" plus μ / mean-L diagnostics.
    for key in ("loss_mean", "loss_min", "best_obj", "temp", "accept_rate_cum", "mu", "mean_L"):
        assert key in result.history
    assert len(result.history["loss_mean"]) == 400


def test_isco_finds_maxcut_on_k4():
    """K_4 has Max-Cut = 4."""
    qqa.fix_seed(0)
    g = nx.complete_graph(4)
    problem = qqa.MaxCut(g)

    result = qqa.discrete_langevin(
        problem,
        sol_size=64,
        num_steps=300,
        t_max=1.0,
        t_min=0.01,
        seed=0,
        device="cpu",
        verbose=False,
    )

    assert result.best_obj <= -4.0 + 1e-6, f"got best_obj={result.best_obj}"


def test_isco_incumbent_is_one_flip_locally_optimal():
    """Default polish=True must leave the incumbent at a 1-flip QUBO minimum."""
    qqa.fix_seed(1)
    g = nx.erdos_renyi_graph(n=14, p=0.4, seed=7)
    problem = qqa.MaxCut(g)

    result = qqa.discrete_langevin(
        problem,
        sol_size=32,
        num_steps=500,
        t_max=1.0,
        t_min=0.005,
        seed=1,
        device="cpu",
        verbose=False,
    )

    x = result.best_sol.to(problem.Q_mat.dtype)
    E = float(problem.loss_fn(x.unsqueeze(0)).item())
    for i in range(x.numel()):
        y = x.clone()
        y[i] = 1.0 - y[i]
        E_y = float(problem.loss_fn(y.unsqueeze(0)).item())
        assert E_y >= E - 1e-6, (
            f"iSCO + polish left a non-locally-optimal incumbent: "
            f"flipping bit {i} drops E from {E:.3f} to {E_y:.3f}."
        )


def test_isco_auto_calibrates_t_max_when_none():
    """t_max=None triggers quantile-based calibration; t_max_used must be > 0."""
    qqa.fix_seed(0)
    g = nx.path_graph(6)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)

    result = qqa.discrete_langevin(
        problem,
        sol_size=32,
        num_steps=50,
        t_max=None,
        t_min=0.01,
        seed=0,
        device="cpu",
        verbose=False,
    )
    assert result.t_max_used > 0.01


def test_isco_rejects_categorical_problem():
    """Coloring uses CategoricalRelaxation; iSCO is QUBO-only."""
    g = nx.cycle_graph(5)
    problem = qqa.Coloring(g, num_category=3)
    with pytest.raises(NotImplementedError):
        qqa.discrete_langevin(problem, sol_size=2, num_steps=1, verbose=False)


def test_isco_rejects_spin_problem():
    """SK is spin ({-1,+1}); iSCO requires {0,1} QUBO."""
    problem = qqa.SherringtonKirkpatrick(N=8, seed=0)
    with pytest.raises(NotImplementedError):
        qqa.discrete_langevin(problem, sol_size=2, num_steps=1, verbose=False)


def test_isco_validates_initial_state_shape():
    """Wrong-shape initial_state raises a clear ValueError."""
    qqa.fix_seed(0)
    g = nx.path_graph(4)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)
    bad = torch.zeros((3, 5))  # sol_size=3 but N=4 variables
    with pytest.raises(ValueError, match="initial_state"):
        qqa.discrete_langevin(
            problem,
            sol_size=3,
            num_steps=10,
            initial_state=bad,
            verbose=False,
        )


def test_isco_batched_smoke():
    """Batched MIS: Q_tensor of shape (I, N, N) is dispatched to the batched kernel."""
    qqa.fix_seed(0)
    graphs = [nx.path_graph(5) for _ in range(3)]
    problem = qqa.MaximumIndependentSetInstance(graphs, penalty=2.0)

    result = qqa.discrete_langevin(
        problem,
        sol_size=16,
        num_steps=100,
        t_max=1.0,
        t_min=0.01,
        seed=0,
        device="cpu",
        verbose=False,
    )

    assert result.num_instance == 3
    assert result.best_sol.shape[0] == 3
    assert isinstance(result.best_obj, np.ndarray)
    assert result.best_obj.shape == (3,)
    assert 0.0 <= result.accept_rate <= 1.0
    assert result.mu_final >= 1.0


def test_isco_callback_is_invoked():
    """The callback should fire at every history-recording tick."""
    qqa.fix_seed(0)
    g = nx.path_graph(5)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)

    seen: list[tuple[int, float, float]] = []

    def cb(step_idx: int, mean_loss: float, best_obj: float) -> None:
        seen.append((step_idx, mean_loss, best_obj))

    qqa.discrete_langevin(
        problem,
        sol_size=8,
        num_steps=20,
        t_max=1.0,
        t_min=0.01,
        seed=0,
        device="cpu",
        verbose=False,
        callback=cb,
        history_stride=5,
    )
    steps = [s for s, _, _ in seen]
    assert steps == [0, 5, 10, 15, 19]


# -----------------------------------------------------------------------
# Paper-faithful behaviour (Algorithm 1 + Appendix C, Sun et al. 2023)
# -----------------------------------------------------------------------


def test_isco_samples_multi_flip_paths():
    """Paper §3.2.2 / Eq 31: L~Poisson(μ) is typically >1 on non-trivial graphs.

    With a 20-node MaxCut problem and μ adapted toward 0.574 acceptance,
    the mean path length should exceed 1 in most cases (the whole point
    of iSCO over R=1 PAS / DMALA is that it flips multiple sites per
    step when the local landscape allows).
    """
    qqa.fix_seed(7)
    g = nx.erdos_renyi_graph(n=20, p=0.3, seed=7)
    problem = qqa.MaxCut(g)

    result = qqa.discrete_langevin(
        problem,
        sol_size=64,
        num_steps=600,
        num_inner=2,
        t_max=1.0,
        t_min=0.01,
        mu0=1.0,
        seed=0,
        device="cpu",
        verbose=False,
    )
    # μ should move off the initial value and track non-trivial L.
    assert result.mean_path_length > 1.0, (
        f"iSCO did not generate multi-flip paths (mean_L={result.mean_path_length:.3f})"
    )


def test_isco_mu_adapts_toward_target_acceptance():
    """Paper Eq 31: μ ← clip(μ + 0.001·(Ā − 0.574), 1, N).

    Starting μ=1, running long enough should move μ toward a value that
    keeps the realised acceptance rate close to (but not exceeding)
    the 0.574 target.  We demand only that the acceptance rate ends up
    in a reasonable band around the target — μ-adaptation is stochastic
    and the tolerance reflects that.
    """
    qqa.fix_seed(11)
    g = nx.erdos_renyi_graph(n=30, p=0.25, seed=11)
    problem = qqa.MaxCut(g)

    result = qqa.discrete_langevin(
        problem,
        sol_size=64,
        num_steps=2000,
        num_inner=1,
        t_max=2.0,
        t_min=0.1,
        mu0=1.0,
        target_accept=0.574,
        mu_step=0.01,  # bigger step so adaptation converges faster in the test
        seed=0,
        device="cpu",
        verbose=False,
    )
    assert 0.2 <= result.accept_rate <= 0.9, (
        f"acceptance rate {result.accept_rate:.3f} is far from target "
        "0.574; μ-adaptation is miscalibrated."
    )


def test_isco_num_inner_doubles_total_mh_steps():
    """num_steps × num_inner = total MH steps per chain."""
    qqa.fix_seed(0)
    g = nx.path_graph(6)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)

    result = qqa.discrete_langevin(
        problem,
        sol_size=16,
        num_steps=50,
        num_inner=3,
        t_max=1.0,
        t_min=0.01,
        seed=0,
        device="cpu",
        verbose=False,
    )
    assert result.num_steps == 50
    assert result.num_inner == 3
    # History length tracks outer steps only (one record per temperature).
    assert len(result.history["loss_mean"]) == 50


def test_plackett_luce_logprob_handles_repeated_indices_in_float32():
    """Regression: ``_reverse_path`` clamps the masked tail of ``sigma_rev``
    to ``sigma[0]``, so any chain with ``L_per_chain < L_max`` evaluates the
    Plackett-Luce log-prob with repeated indices.

    A previous version used ``diff.clamp(max=-1e-12)`` to guard
    ``log1p(-exp(diff))``; ``-1e-12`` round-trips to ``0.0`` in float32
    (machine eps ≈ 1.2e-7), so ``exp(0) - 1 = 0``, ``log1p(0) = -inf``,
    and ``inf * mask = NaN`` propagated into the MH acceptance ratio,
    silently breaking detailed balance for any multi-flip path.

    This test guards against that regression for both float32 and
    float64 by checking the closed-form Plackett-Luce log-prob.
    """
    from qqa.isco import _plackett_luce_logprob

    torch.manual_seed(0)
    log_w = torch.randn(4)

    sigma_rev = torch.tensor([2, 2, 2, 2])
    L_per_chain = torch.tensor(1)

    expected = log_w[2] - torch.logsumexp(log_w, dim=-1)

    for dtype in (torch.float32, torch.float64):
        log_q = _plackett_luce_logprob(log_w.to(dtype), sigma_rev, L_per_chain, L_max=4)
        assert torch.isfinite(log_q), (
            f"_plackett_luce_logprob returned non-finite log-prob for "
            f"L_s=1 < L_max=4 in {dtype}: {log_q.item()}"
        )
        assert torch.allclose(
            log_q, expected.to(dtype), atol=1e-5 if dtype is torch.float32 else 1e-12
        ), (
            f"masked tail leaked into the sum (dtype={dtype}): "
            f"got {log_q.item()}, expected {expected.item()}"
        )


def test_plackett_luce_logprob_excludes_ragged_padding_from_denominator():
    """The MH proposal ratio must use the same mask as ragged sampling.

    With one valid site, selecting that site is deterministic and therefore
    has log-probability zero. An unmasked padding logit would incorrectly
    enter the denominator and bias the target distribution.
    """
    from qqa.isco import _plackett_luce_logprob

    log_w = torch.tensor(
        [
            [[-2.0, 50.0, -30.0], [0.4, -0.1, 0.8]],
            [[3.0, -40.0, 20.0], [-0.7, 0.2, 1.1]],
        ],
        dtype=torch.float64,
    )
    mask = torch.tensor(
        [[[True, False, False], [True, True, True]]],
    )
    sigma = torch.tensor(
        [
            [[0], [2]],
            [[0], [1]],
        ],
    )
    lengths = torch.ones((2, 2), dtype=torch.long)

    actual = _plackett_luce_logprob(
        log_w,
        sigma,
        lengths,
        L_max=1,
        mask=mask,
    )

    torch.testing.assert_close(actual[:, 0], torch.zeros(2, dtype=torch.float64))
    expected_second = torch.stack(
        [
            log_w[0, 1, 2] - torch.logsumexp(log_w[0, 1], dim=0),
            log_w[1, 1, 1] - torch.logsumexp(log_w[1, 1], dim=0),
        ]
    )
    torch.testing.assert_close(actual[:, 1], expected_second)


def test_isco_ragged_kernel_passes_sampling_mask_to_mh_ratio(monkeypatch):
    """Forward and reverse proposal probabilities must share the sampling mask."""
    import qqa.isco as isco_module

    original = isco_module._plackett_luce_logprob
    observed_masks: list[torch.Tensor] = []

    def checked_logprob(log_w, sigma, lengths, max_length, *, mask=None):
        assert mask is not None
        observed_masks.append(mask.detach().cpu())
        return original(
            log_w,
            sigma,
            lengths,
            max_length,
            mask=mask,
        )

    monkeypatch.setattr(isco_module, "_plackett_luce_logprob", checked_logprob)
    problem = qqa.MaximumIndependentSetInstance(
        [nx.path_graph(3), nx.path_graph(6)],
        penalty=2.0,
    )
    qqa.discrete_langevin(
        problem,
        sol_size=4,
        num_steps=2,
        t_max=1.0,
        t_min=0.5,
        seed=0,
        verbose=False,
    )

    assert len(observed_masks) == 4  # forward + reverse for both MH steps
    assert all((~mask).any() for mask in observed_masks)


def test_isco_detailed_balance_on_tiny_qubo():
    """Empirical Boltzmann-stationarity check on a tiny enumerable QUBO.

    Runs iSCO at a fixed temperature with variable Poisson path lengths
    and compares the empirical state visitation frequency to the exact
    Boltzmann distribution via total-variation distance.

    A correctly-implemented Metropolis-Hastings chain with the
    path-auxiliary correction (Eq. 30) must converge to the target
    distribution.  A broken implementation (e.g. NaNs in the proposal
    log-prob silently rejecting all moves of one parity) drives this
    metric to ``O(1)``; a correct one keeps TV well below ``0.02``
    after 4000 MH steps × 200 chains.
    """
    from qqa.isco import (
        _gumbel_topL,
        _plackett_luce_logprob,
        _reverse_path,
    )

    torch.manual_seed(0)
    N = 4
    Q = torch.randn(N, N) * 0.8
    Q = 0.5 * (Q + Q.T)
    diag = torch.diagonal(Q).contiguous()

    # Exact Boltzmann at T=1.
    states = torch.tensor(
        [[float((k >> i) & 1) for i in range(N)] for k in range(2**N)],
    )
    energies_all = torch.einsum("ki,ij,kj->k", states, Q, states).numpy()
    log_p = -energies_all
    log_p -= log_p.max()
    p_theo = np.exp(log_p)
    p_theo /= p_theo.sum()

    # iSCO inner-loop replica at T=1.
    S = 200
    n_inner = 4000
    burn = 2000
    gen = torch.Generator(device="cpu").manual_seed(123)
    x = torch.bernoulli(torch.full((S, N), 0.5), generator=gen)
    Qx = x @ Q
    energies = (x * Qx).sum(dim=-1)
    mu_value = 2.0
    samples: list[torch.Tensor] = []

    inv_T = 1.0
    inv2T = 0.5
    for it in range(n_inner):
        delta = (1.0 - 2.0 * x) * (diag + 2.0 * (Qx - diag * x))
        log_w = -delta * inv2T
        L_s = (
            torch.poisson(torch.full((S,), mu_value), generator=gen)
            .to(torch.long)
            .clamp_(min=1, max=N)
        )
        L_max = int(L_s.max().item())

        sigma = _gumbel_topL(log_w, L_max, gen=gen)
        y = x.clone()
        k_idx = torch.arange(L_max)
        mask_path = k_idx.unsqueeze(0) < L_s.unsqueeze(-1)
        for k in range(L_max):
            active = mask_path[:, k]
            if not bool(active.any().item()):
                continue
            s_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            cols = sigma[s_idx, k]
            y[s_idx, cols] = 1.0 - y[s_idx, cols]

        Qy = y @ Q
        energies_y = (y * Qy).sum(dim=-1)
        dE = energies_y - energies

        delta_y = (1.0 - 2.0 * y) * (diag + 2.0 * (Qy - diag * y))
        log_w_y = -delta_y * inv2T
        sigma_rev = _reverse_path(sigma, L_s, L_max)

        log_q_fwd = _plackett_luce_logprob(log_w, sigma, L_s, L_max)
        log_q_rev = _plackett_luce_logprob(log_w_y, sigma_rev, L_s, L_max)
        assert torch.isfinite(log_q_fwd).all()
        assert torch.isfinite(log_q_rev).all()

        log_alpha = -dE * inv_T + log_q_rev - log_q_fwd
        u = torch.rand(S, generator=gen).clamp_min_(1e-38)
        accept = torch.log(u) < log_alpha
        x = torch.where(accept.unsqueeze(-1), y, x)
        Qx = torch.where(accept.unsqueeze(-1), Qy, Qx)
        energies = torch.where(accept, energies_y, energies)

        if it >= burn:
            samples.append(x.clone())

    H = torch.stack(samples, dim=0).numpy().astype(int).reshape(-1, N)
    codes = (H << np.arange(N)).sum(axis=-1)
    counts = np.bincount(codes, minlength=2**N)
    p_emp = counts / counts.sum()
    tv = 0.5 * float(np.abs(p_emp - p_theo).sum())
    assert tv < 0.02, (
        f"iSCO MH chain not converging to Boltzmann (TV={tv:.4f}); "
        "Plackett-Luce / detailed balance is broken."
    )


def test_isco_schedules_produce_valid_results():
    """All three schedules should reach the MIS optimum on P_6."""
    qqa.fix_seed(0)
    g = nx.path_graph(6)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)
    for schedule in ("exp", "geom", "lin"):
        result = qqa.discrete_langevin(
            problem,
            sol_size=32,
            num_steps=300,
            t_max=1.0,
            t_min=0.01,
            schedule=schedule,
            seed=0,
            device="cpu",
            verbose=False,
        )
        assert result.best_obj <= -3.0 + 1e-6, (
            f"schedule={schedule} failed: best_obj={result.best_obj}"
        )
