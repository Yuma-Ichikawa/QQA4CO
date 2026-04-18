"""Tests for the optional CRA-PI-GNN (PyTorch Geometric) backend.

The whole module is wrapped in a single ``importorskip("torch_geometric")``
so contributors and CI environments without PyG installed simply skip
these tests instead of failing — the dev extra deliberately omits PyG to
keep CPU CI light.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import qqa

pytest.importorskip("torch_geometric")

from qqa.pignn import GCNNet, train_cra_pi_gnn  # noqa: E402


def _small_mis_problem(n: int = 30, d: int = 3, seed: int = 0):
    qqa.fix_seed(seed)
    g = nx.random_regular_graph(d=d, n=n, seed=seed)
    return qqa.MaximumIndependentSet(g, penalty=2, device="cpu")


def test_train_cra_pi_gnn_returns_anneal_result():
    problem = _small_mis_problem(n=20)
    result = train_cra_pi_gnn(
        problem,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=300,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    assert isinstance(result, qqa.AnnealResult)
    assert torch.is_tensor(result.best_sol)
    assert result.best_sol.shape == (20,)
    assert isinstance(result.best_obj, float)
    assert result.runtime > 0.0
    assert result.score["feasible"] is True


def test_train_cra_pi_gnn_finds_nontrivial_independent_set():
    """Sanity check: a 50-node 3-regular graph should yield |IS| >= 10.

    The (1/2 - log d / log d) Caro-Wei bound for d-regular graphs at d=3
    gives |IS| / N ~ 0.36, so |IS| ~ 18 is typical. We assert >= 10 to
    leave generous slack for stochastic seed effects on a tiny graph.
    """
    problem = _small_mis_problem(n=50, d=3, seed=0)
    result = train_cra_pi_gnn(
        problem,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=4000,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    assert result.score["feasible"] is True, result.score
    assert result.score["value"] >= 10, result.score


def test_pignn_rejects_non_graph_problem():
    qqa.fix_seed(0)
    problem = qqa.SherringtonKirkpatrick(N=20, seed=0, device="cpu")
    with pytest.raises(TypeError, match="graph-based"):
        train_cra_pi_gnn(problem, num_epochs=5, verbose=False)


def test_pignn_anneal_disabled_keeps_reg_param_zero():
    problem = _small_mis_problem(n=20)
    result = train_cra_pi_gnn(
        problem,
        learning_rate=1e-3,
        annealing=False,
        init_reg_param=0.0,
        num_epochs=50,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    assert (result.history["reg_param"] == 0.0).all()


def test_pignn_history_arrays_shapes():
    problem = _small_mis_problem(n=20)
    n_epoch = 80
    result = train_cra_pi_gnn(
        problem,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=n_epoch,
        # Set patience > num_epochs so early stopping cannot truncate the
        # history; otherwise this test becomes flaky on tiny problems
        # where loss/penalty plateau within the first dozen epochs.
        patience=n_epoch + 10,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    for k in ("loss", "cost", "reg_term", "reg_param"):
        assert k in result.history
        assert len(result.history[k]) == n_epoch


def test_pignn_smoke_maxcut():
    """CRA-PI-GNN must not be MIS-specific: it should also tackle MaxCut."""
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=20, seed=0)
    problem = qqa.MaxCut(g, device="cpu")
    result = train_cra_pi_gnn(
        problem,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=300,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    assert torch.is_tensor(result.best_sol)
    assert result.best_sol.shape == (20,)
    # Cut size for a 3-regular N=20 graph cannot exceed 30 edges; the
    # trivial all-zero/all-one assignment already cuts 0 edges. Insist on
    # something better than trivial so we know the solver actually moved.
    assert result.score["value"] >= 10


def test_pignn_smoke_vertex_cover():
    """``VertexCover`` stores its graph on ``problem.graph`` (not ``nx_graph``).

    Regression guard: the original ``extract_nx_graph`` only checked for
    ``nx_graph`` and silently broke for VertexCover / GraphBisection even
    though the CLI advertised them as supported. Keep this test cheap.
    """
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=20, seed=0)
    problem = qqa.VertexCover(g, penalty=4.0, device="cpu")
    result = train_cra_pi_gnn(
        problem,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=300,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    assert torch.is_tensor(result.best_sol)
    assert result.best_sol.shape == (20,)


def test_pignn_smoke_graph_bisection():
    """``GraphBisection`` also stores its graph on ``problem.graph``."""
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=20, seed=0)
    problem = qqa.GraphBisection(g, balance_penalty=1.0, device="cpu")
    result = train_cra_pi_gnn(
        problem,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=300,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    assert torch.is_tensor(result.best_sol)
    assert result.best_sol.shape == (20,)


def test_gcn_net_forward_shape():
    g = nx.cycle_graph(8)
    from qqa.pignn.graph import nx_to_edge_index

    edge_index = nx_to_edge_index(g)
    net = GCNNet(num_nodes=8)
    out = net(edge_index)
    assert out.shape == (8,)
    assert torch.all((out >= 0) & (out <= 1))


def test_pignn_cli_branch(tmp_path):
    """Smoke-test the CLI dispatch for ``--backend pignn``."""
    import sys
    from subprocess import run

    cmd = [
        sys.executable,
        "-m",
        "qqa.cli",
        "solve",
        "--problem",
        "mis",
        "--backend",
        "pignn",
        "--size",
        "20",
        "--epochs",
        "300",
        "--pignn-init-reg-param",
        "-2",
        "--pignn-annealing-rate",
        "5e-4",
        "--learning-rate",
        "1e-3",
        "--quiet",
    ]
    proc = run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "backend    : pignn" in proc.stdout
    assert "best_obj" in proc.stdout
