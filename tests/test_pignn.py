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

from qqa.pignn import GCNNet, train_cpra_pi_gnn, train_cra_pi_gnn  # noqa: E402


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


def test_pignn_smoke_maxclique():
    """``MaxClique`` was advertised as supported but had no smoke test.

    Regression guard: the CLI's ``_PIGNN_SUPPORTED_KINDS`` whitelist
    includes ``maxclique``, so each whitelisted kind gets at least one
    cheap end-to-end test (cf. tasks/lessons.md "Always check ALL
    declared supported entry points").
    """
    qqa.fix_seed(0)
    g = nx.erdos_renyi_graph(n=20, p=0.5, seed=0)
    problem = qqa.MaxClique(g, penalty=2, device="cpu")
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


# ---------------------------------------------------------------------------
# CPRA backend (multi-head trainer)
# ---------------------------------------------------------------------------


def test_gcnnet_multi_head_shapes():
    """``num_replicas`` controls the trailing output dim, BC at R=1."""
    g = nx.cycle_graph(8)
    from qqa.pignn.graph import nx_to_edge_index

    edge_index = nx_to_edge_index(g)

    net1 = GCNNet(num_nodes=8, num_replicas=1)
    out1 = net1(edge_index)
    assert out1.shape == (8,)

    net4 = GCNNet(num_nodes=8, num_replicas=4)
    out4 = net4(edge_index)
    assert out4.shape == (8, 4)
    assert torch.all((out4 >= 0) & (out4 <= 1))


def test_gcnnet_rejects_zero_replicas():
    with pytest.raises(ValueError, match="num_replicas"):
        GCNNet(num_nodes=4, num_replicas=0)


def test_train_cpra_pi_gnn_returns_anneal_result_with_replicas():
    problem = _small_mis_problem(n=20)
    R = 3
    result = train_cpra_pi_gnn(
        problem,
        num_replicas=R,
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
    extras = result.score.get("extra") or {}
    assert "replicas" in extras and len(extras["replicas"]) == R
    for record in extras["replicas"]:
        assert {"replica", "obj", "score", "sol"} <= set(record)
        assert record["sol"].shape == (20,)


def test_train_cpra_pi_gnn_penalty_diversification():
    """Three penalty levels => three matching MIS instances => R replica scores."""
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=24, seed=0)
    base = qqa.MaximumIndependentSet(g, penalty=2, device="cpu")
    levels = [1.5, 2.0, 3.0]
    replica_problems = [qqa.MaximumIndependentSet(g, penalty=p, device="cpu") for p in levels]
    result = train_cpra_pi_gnn(
        base,
        num_replicas=len(levels),
        replica_problems=replica_problems,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=600,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    records = result.score["extra"]["replicas"]
    assert len(records) == 3
    # At least one replica must reach a non-trivial independent set.
    # (At very small N some replicas may collapse to all-zero on this many
    # epochs — the contract is "diversified candidates", not "all good".)
    assert any(int(record["sol"].sum()) >= 1 for record in records)
    # The best replica must always be non-trivial.
    assert int(result.best_sol.sum()) >= 1


def test_train_cpra_pi_gnn_variation_diversification_separates_replicas():
    """vari_param > 0 should pull replicas away from each other on MaxCut."""
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=20, seed=0)
    problem = qqa.MaxCut(g, device="cpu")
    R = 3
    result = train_cpra_pi_gnn(
        problem,
        num_replicas=R,
        vari_param=0.4,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=400,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    sols = [r["sol"] for r in result.score["extra"]["replicas"]]
    pairwise = [(sols[i] != sols[j]).any().item() for i in range(R) for j in range(i + 1, R)]
    assert any(pairwise), "All CPRA replicas collapsed to identical solutions."


def test_train_cpra_pi_gnn_history_includes_per_replica_obj():
    problem = _small_mis_problem(n=16)
    R = 3
    n_epoch = 50
    result = train_cpra_pi_gnn(
        problem,
        num_replicas=R,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=n_epoch,
        patience=n_epoch + 10,
        check_interval=10_000,
        verbose=False,
        seed=0,
    )
    assert "per_replica_obj" in result.history
    arr = result.history["per_replica_obj"]
    assert arr.shape == (n_epoch, R)


def test_train_cpra_pi_gnn_rejects_mismatched_replica_problems():
    problem = _small_mis_problem(n=16)
    with pytest.raises(ValueError, match="num_replicas"):
        train_cpra_pi_gnn(
            problem,
            num_replicas=3,
            replica_problems=[problem, problem],
            num_epochs=5,
            verbose=False,
        )


def test_pignn_silently_migrates_problem_to_device():
    """API users routinely build a problem on CPU then train on a different
    "cpu" string vs torch.device("cpu") — ensure no spurious errors.

    Also guards the wider regression: building a CPU problem and asking
    for ``device="cpu"`` (a fresh torch.device handle) must not raise.
    The on-GPU variant requires CUDA so we don't exercise it in CI;
    `_ensure_problem_on_device` is the implementation.
    """
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=16, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2, device="cpu")
    # Pass the device as a torch.device handle, not the string the problem
    # was built with — this exercises the device-equality branch in
    # `_ensure_problem_on_device`.
    result = train_cra_pi_gnn(
        problem,
        learning_rate=1e-3,
        init_reg_param=-2.0,
        annealing_rate=5e-4,
        num_epochs=80,
        check_interval=10_000,
        device=torch.device("cpu"),
        verbose=False,
        seed=0,
    )
    assert result.best_sol.shape == (16,)


def test_train_cpra_pi_gnn_rejects_negative_vari_param():
    problem = _small_mis_problem(n=16)
    with pytest.raises(ValueError, match="vari_param"):
        train_cpra_pi_gnn(
            problem,
            num_replicas=2,
            vari_param=-0.1,
            num_epochs=5,
            verbose=False,
        )


def test_cpra_build_replica_problems_handles_vertex_cover():
    """``_build_replica_problems`` had a hidden bug where it pulled the
    graph from ``base_problem.nx_graph`` directly, which is ``None`` for
    ``VertexCover`` (graph stored on ``problem.graph`` instead). This
    forced the CLI's ``--backend cpra --problem vertex_cover
    --cpra-penalty-levels ...`` path to crash. Regression guard.
    """
    from argparse import Namespace

    from qqa.cli import _build_replica_problems

    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=16, seed=0)
    base = qqa.VertexCover(g, penalty=2.0, device="cpu")
    args = Namespace(
        problem="vertex_cover",
        cpra_penalty_levels="1.0,2.0",
        cpra_num_replicas=2,
        device="cpu",
    )
    replicas = _build_replica_problems(args, base)
    assert replicas is not None
    assert len(replicas) == 2
    for r in replicas:
        assert isinstance(r, qqa.VertexCover)


def test_cpra_cli_branch():
    """Smoke-test the CLI dispatch for ``--backend cpra``."""
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
        "cpra",
        "--cpra-num-replicas",
        "3",
        "--cpra-penalty-levels",
        "1.5,2.0,3.0",
        "--size",
        "20",
        "--epochs",
        "200",
        "--pignn-init-reg-param",
        "-2",
        "--pignn-annealing-rate",
        "5e-4",
        "--learning-rate",
        "1e-3",
        "--quiet",
    ]
    proc = run(cmd, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "backend    : cpra" in proc.stdout
    assert "best_obj" in proc.stdout
