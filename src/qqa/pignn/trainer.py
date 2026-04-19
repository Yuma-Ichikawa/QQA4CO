"""CRA-PI-GNN trainer (PyTorch Geometric).

Faithful port of the ``fit_model`` loop from the reference
NeurIPS 2024 implementation [#cra]_, with two intentional changes:

1. The graph backend is :mod:`torch_geometric` instead of DGL, so the
   trainer runs on every PyTorch-supported GPU (including NVIDIA
   Blackwell / ``sm_100`` for which DGL has no prebuilt wheel as of
   April 2026).
2. The loss / penalty mathematics are expressed via the QQA primitives
   :meth:`qqa.problems.QUBOProblem.loss_fn` and
   :meth:`qqa.relaxation.BinaryRelaxation.penalty`, so the function
   returns a :class:`qqa.AnnealResult` and is a drop-in alternative to
   :func:`qqa.anneal`. Numerically the loss is identical to the original
   paper for ``curve_rate=2`` (and for any even ``curve_rate``):

   .. math::

      L(p; \\gamma) \\;=\\; p^\\top Q\\, p
                    \\;+\\; \\gamma \\sum_i \\bigl(1 - (1 - 2p_i)^c\\bigr)

   matches CRA's ``cost + reg_param * Σ (1 - (2p - 1)^c)`` because
   ``(1 - 2p)^c = (2p - 1)^c`` for even ``c``.

.. [#cra] Y. Ichikawa, "Controlling Continuous Relaxation for
   Combinatorial Optimization," NeurIPS 2024.
   https://github.com/Yuma-Ichikawa/CRA4CO
"""

from __future__ import annotations

import contextlib
from time import time

import networkx as nx
import numpy as np
import torch

from qqa.annealing import AnnealResult
from qqa.pignn._import import require_pyg
from qqa.pignn.graph import extract_nx_graph, nx_to_edge_index
from qqa.pignn.model import GCNNet, default_in_feats
from qqa.relaxation import BinaryRelaxation
from qqa.utils import fix_seed, require_cuda_if_requested, safe_score_summary


def _ensure_problem_on_device(problem, device: torch.device) -> None:
    """Move every Tensor attribute of ``problem`` to ``device`` in-place.

    Background: a QQA problem instance is created with a ``device=...``
    kwarg that puts ``Q_mat`` (and friends) on that device once. If the
    user later trains the PyG backend on a different device — e.g.
    ``MaximumIndependentSet(g, device='cpu')`` followed by
    ``train_cra_pi_gnn(problem, device='cuda')`` — every forward pass
    raises a cryptic CUDA / CPU ``einsum`` mismatch. The CLI sidesteps
    this because ``_build_problem`` always passes ``args.device``, but
    Python-API users hit it routinely. Rather than failing late, we
    silently migrate any ``torch.Tensor`` attribute exposed by the
    problem to the requested device.

    Subclasses that override ``__setattr__`` keep their custom logic
    because we use ``setattr``; pure-data classes get straightforward
    ``.to(device)`` migration.
    """
    for name in dir(problem):
        if name.startswith("_"):
            continue
        try:
            val = getattr(problem, name)
        except Exception:  # noqa: BLE001 — properties may raise on uninit attrs
            continue
        if torch.is_tensor(val) and val.device != device:
            # read-only / property-backed tensors are skipped silently;
            # if the device truly is wrong the loss_fn will surface it.
            with contextlib.suppress(AttributeError, TypeError):
                setattr(problem, name, val.to(device))


def train_cra_pi_gnn(
    problem,
    *,
    nx_graph: nx.Graph | None = None,
    in_feats: int | None = None,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    annealing: bool = True,
    init_reg_param: float = -20.0,
    annealing_rate: float = 1e-3,
    curve_rate: int = 2,
    num_epochs: int = 100_000,
    tol: float = 1e-4,
    patience: int = 1000,
    early_stop_disc_patience: int | None = None,
    check_interval: int = 1000,
    device: str | torch.device = "cpu",
    seed: int | None = None,
    verbose: bool = True,
) -> AnnealResult:
    """Train a CRA-PI-GNN solver and return a :class:`qqa.AnnealResult`.

    Parameters
    ----------
    problem:
        A graph-based binary QUBO problem from :mod:`qqa` — typically
        :class:`~qqa.MaximumIndependentSet`, :class:`~qqa.MaxClique`,
        :class:`~qqa.MaxCut`, :class:`~qqa.VertexCover`, or
        :class:`~qqa.GraphBisection`. Anything that exposes
        ``problem.nx_graph`` and ``problem.loss_fn`` works.
    nx_graph:
        Override for ``problem.nx_graph`` (rare; only needed for custom
        problems that store the graph elsewhere).
    in_feats, hidden_dim:
        GCN widths. Both default to ``floor(sqrt(N))``, matching the
        reference paper.
    dropout:
        Dropout probability between the two GCN layers. ``0`` (default)
        reproduces the headline numbers in the paper.
    learning_rate, weight_decay:
        AdamW hyper-parameters. Defaults match the reference.
    annealing:
        If ``False`` the trainer reduces to vanilla PI-GNN
        (``reg_param = 0`` for every epoch).
    init_reg_param, annealing_rate:
        CRA schedule: ``reg_param = init_reg_param + epoch * annealing_rate``.
        Set ``init_reg_param < 0`` so the early-epoch loss landscape is
        concave (encourages exploration); ``annealing_rate > 0`` linearly
        ramps it through 0 toward the discrete-favouring regime.
    curve_rate:
        Penalty exponent (must be even). Defaults to 2.
    num_epochs:
        Hard upper bound on gradient steps. Early stopping (see
        ``tol`` / ``patience``) usually terminates earlier.
    tol, patience:
        Stop when both the loss *and* the penalty change by less than
        ``tol`` for ``patience`` consecutive epochs.
    check_interval:
        How often the verbose log is printed.
    device:
        Torch device. Strings like ``"cuda"`` are validated up-front to
        give a clear error if CUDA is unavailable.
    seed:
        If supplied, calls :func:`qqa.fix_seed` before allocating the
        model and embedding (so the run is reproducible).
    verbose:
        If ``True`` print periodic progress and a final summary.

    Notes
    -----
    The defaults here match the **NeurIPS 2024 paper** and are tuned for
    large instances (``N >= 1000``). For small graphs (``N <= 200``) the
    paper defaults severely under-converge: try
    ``init_reg_param=-2.0, annealing_rate=5e-4, learning_rate=1e-3``
    (or even ``learning_rate=1e-2``) instead. This is a standard PI-GNN
    quirk — the optimal annealing schedule is sub-linear in problem size
    because the cost / penalty magnitudes scale very differently.

    Returns
    -------
    qqa.AnnealResult
        With ``best_sol`` of shape ``(N,)`` (rounded ``{0, 1}`` tensor),
        ``best_obj`` the float QUBO loss on that solution, and
        ``history`` containing per-epoch arrays
        ``loss``, ``cost``, ``reg_term``, ``reg_param``.

    Raises
    ------
    TypeError
        If ``problem`` is not graph-based (no ``nx_graph`` attribute and
        no ``nx_graph`` override).
    RuntimeError
        If ``device`` requests CUDA but ``torch.cuda.is_available()`` is
        ``False``.
    ValueError
        On obviously-wrong arguments (``curve_rate`` odd, ``num_epochs``
        negative, ...).
    """
    require_pyg()

    if curve_rate % 2 != 0:
        raise ValueError(
            f"curve_rate must be even (got {curve_rate}); odd exponents make "
            "the penalty 1-(1-2p)^c non-convex and asymmetric."
        )
    if num_epochs < 0:
        raise ValueError(f"num_epochs must be >= 0, got {num_epochs}.")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}.")

    require_cuda_if_requested(device)
    device = torch.device(device) if isinstance(device, str) else device

    if seed is not None:
        fix_seed(seed)

    g = extract_nx_graph(problem, override=nx_graph)
    num_nodes = g.number_of_nodes()
    edge_index = nx_to_edge_index(g, device=device)
    _ensure_problem_on_device(problem, device)

    in_feats_resolved = default_in_feats(num_nodes) if in_feats is None else in_feats
    hidden_resolved = in_feats_resolved if hidden_dim is None else hidden_dim
    model = GCNNet(
        num_nodes=num_nodes,
        in_feats=in_feats_resolved,
        hidden_dim=hidden_resolved,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    relax = BinaryRelaxation()
    reg_param = float(init_reg_param if annealing else 0.0)

    history: dict[str, list[float]] = {
        "loss": [],
        "cost": [],
        "reg_term": [],
        "reg_param": [],
    }

    # Track best (lowest) discrete QUBO loss across all epochs so a transient
    # spike late in annealing cannot wipe out a good early solution.
    best_obj = float("inf")
    best_bits: torch.Tensor | None = None
    prev_loss = float("inf")
    prev_pen = float("inf")
    stagnant_steps = 0
    best_obj_so_far = float("inf")
    disc_stagnant = 0

    runtime_start = time()
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        probs = model(edge_index)  # (N,)
        # ``problem.loss_fn`` expects a leading batch dim of size B; we feed
        # B=1 here because CRA-PI-GNN keeps a single (deterministic given
        # the embedding) candidate per epoch — there is no parallel
        # population in the reference implementation.
        cost = problem.loss_fn(probs.unsqueeze(0)).sum()
        reg_term = relax.penalty(probs.unsqueeze(0), curve_rate).sum()
        loss = cost + reg_param * reg_term
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bits = (probs >= 0.5).to(probs.dtype)
            disc_obj = float(problem.loss_fn(bits.unsqueeze(0)).item())
            if disc_obj < best_obj:
                best_obj = disc_obj
                best_bits = bits.detach().clone()

        loss_v = float(loss.item())
        cost_v = float(cost.item())
        pen_v = float(reg_term.item())
        history["loss"].append(loss_v)
        history["cost"].append(cost_v)
        history["reg_term"].append(pen_v)
        history["reg_param"].append(reg_param)

        # Early stopping: both objective *and* penalty must be quasi-stationary.
        # This mirrors the reference loop's joint convergence test.
        if abs(prev_loss - loss_v) < tol and abs(prev_pen - pen_v) < tol:
            stagnant_steps += 1
            if stagnant_steps >= patience:
                if verbose:
                    print(
                        f"[CRA-PI-GNN] Early stop at epoch {epoch} (stagnant for {patience} steps)."
                    )
                break
        else:
            stagnant_steps = 0
        prev_loss = loss_v
        prev_pen = pen_v

        # Optional discrete-best early stop. Reflects "the integer answer
        # has settled" rather than the continuous loss having settled.
        if early_stop_disc_patience is not None:
            if best_obj < best_obj_so_far - tol:
                best_obj_so_far = best_obj
                disc_stagnant = 0
            else:
                disc_stagnant += 1
                if disc_stagnant >= int(early_stop_disc_patience):
                    if verbose:
                        print(
                            f"[CRA-PI-GNN] Early stop at epoch {epoch} (discrete best "
                            f"unchanged for {early_stop_disc_patience} epochs)."
                        )
                    break

        if annealing:
            reg_param += float(annealing_rate)

        if verbose and (epoch % check_interval == 0 or epoch == num_epochs - 1):
            _print_progress(epoch, best_obj, loss_v, cost_v, pen_v, reg_param)

    runtime = time() - runtime_start

    # In the (vanishingly rare) case that ``num_epochs == 0``, fall back
    # to the model's t=0 prediction so ``best_sol`` is always a real
    # tensor and the AnnealResult contract is preserved.
    if best_bits is None:
        with torch.no_grad():
            probs = model(edge_index)
            best_bits = (probs >= 0.5).to(probs.dtype)
            best_obj = float(problem.loss_fn(best_bits.unsqueeze(0)).item())

    score = safe_score_summary(problem, best_bits, fallback_obj=float(best_obj))

    if verbose:
        print("\n" + "=" * 30 + " [FINAL] " + "=" * 30)
        print(f"  BEST LOSS : {best_obj}")
        print(f"  RUN TIME  : {runtime:.2f} s")
        print("=" * 69)

    return AnnealResult(
        best_sol=best_bits,
        best_obj=best_obj,
        runtime=runtime,
        history={k: np.asarray(v) for k, v in history.items()},
        callbacks=[],
        score=score,
    )


def _print_progress(
    epoch: int,
    best_obj: float,
    loss_v: float,
    cost_v: float,
    pen_v: float,
    reg_param: float,
) -> None:
    print("\n" + "=" * 30 + " [LOG] " + "=" * 32)
    print(f"[ EPOCH {epoch} ]")
    print(f"  Best Loss So Far : {best_obj}")
    print(f"  Loss             : {loss_v:.4f}")
    print(f"  Cost  (x^T Q x)  : {cost_v:.4f}")
    print(f"  Reg term         : {pen_v:.4f}")
    print(f"  reg_param (gamma): {reg_param:.4f}")
    print("=" * 69)


def train_cpra_pi_gnn(
    problem,
    *,
    num_replicas: int = 4,
    replica_problems: list | None = None,
    vari_param: float = 0.0,
    nx_graph: nx.Graph | None = None,
    in_feats: int | None = None,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    annealing: bool = True,
    init_reg_param: float = -20.0,
    annealing_rate: float = 1e-3,
    curve_rate: int = 2,
    num_epochs: int = 100_000,
    tol: float = 1e-4,
    patience: int = 1000,
    early_stop_disc_patience: int | None = None,
    check_interval: int = 1000,
    device: str | torch.device = "cpu",
    seed: int | None = None,
    verbose: bool = True,
) -> AnnealResult:
    """Train a **CPRA** multi-head PI-GNN solver and return an :class:`AnnealResult`.

    CPRA (*Continual Parallel Relaxation Annealing*) is the multi-replica
    extension of CRA-PI-GNN introduced by Ichikawa & Iwashita,
    *Transactions on Machine Learning Research*, 2025
    (`OpenReview <https://openreview.net/forum?id=ix33zd5zCw>`_).
    A single shared GCN backbone produces ``R`` continuous solutions in
    one forward pass, and the loss combines a per-replica QUBO term, the
    standard CRA penalty, and an optional inter-replica diversity term.

    Two diversification regimes are supported:

    1. **Penalty diversification** — supply ``replica_problems`` (length
       ``num_replicas``) where each problem instance differs in some
       hyperparameter (e.g. ``MaximumIndependentSet(g, penalty=p_r)`` for
       a sweep of ``p_r``). One training run yields one solution per
       penalty level, much cheaper than independent runs.
    2. **Variation diversification** — leave ``replica_problems=None`` so
       every replica solves the same ``problem``, but set
       ``vari_param > 0`` to add the diversity term
       ``-R · Σ_i std_r(p_{i,r})`` (sign chosen so the loss
       *decreases* when between-replica spread *grows*). Replicas then
       converge to structurally different solutions.

    Parameters
    ----------
    problem:
        Base graph-based problem (used for graph extraction and as the
        default ``score_summary`` provider when ``replica_problems`` is
        ``None``).
    num_replicas:
        Number of parallel continuous solutions ``R``. Defaults to 4.
    replica_problems:
        Optional list of ``num_replicas`` problem instances. When
        provided, ``replica_problems[r].loss_fn`` evaluates the cost for
        replica ``r``. Must share the same underlying graph as
        ``problem`` (only the QUBO ``Q_mat`` may differ — typically via
        a different penalty weight). When ``None``, all replicas use
        ``problem.loss_fn``.
    vari_param:
        Coefficient of the diversity term. ``0`` (default) is pure
        penalty diversification; positive values reward inter-replica
        spread (used for variation diversification on a fixed problem).
    nx_graph, in_feats, hidden_dim, dropout, learning_rate,
    weight_decay, annealing, init_reg_param, annealing_rate, curve_rate,
    num_epochs, tol, patience, check_interval, device, seed, verbose:
        Identical semantics to :func:`train_cra_pi_gnn`.

    Returns
    -------
    qqa.AnnealResult
        ``best_sol`` — the discrete ``(N,)`` assignment of the best
        replica (lowest QUBO objective on its own ``loss_fn``).
        ``best_obj`` — that replica's float objective.
        ``history`` — per-epoch ``loss``, ``mean_cost``, ``reg_term``,
        ``vari_term``, ``reg_param`` arrays plus a ``per_replica_obj``
        array of shape ``(epochs, R)`` for downstream visualisation.
        ``score['extra']['replicas']`` — list of
        ``{replica, obj, score, sol}`` dicts so the caller can inspect
        every diversified solution, not only the best one.

    Raises
    ------
    ValueError
        On invalid ``num_replicas``, ``vari_param`` sign, or a
        ``replica_problems`` list whose length does not match
        ``num_replicas``.

    Notes
    -----
    * **Backbone vs. CPRA4CO.** The reference CPRA implementation uses
      DGL ``GraphSAGE``; this port reuses :class:`GCNNet` (a 2-layer
      ``GCNConv`` stack) for full parity with :func:`train_cra_pi_gnn`
      so head-to-head ablations across the two solvers measure the
      training objective, not the message-passing op.
    * **Best-tracking.** The reference CPRA loop returns the *final*
      iteration's discretised bits rather than the best-so-far solution.
      This trainer deliberately tracks the running best per replica —
      the QQA-side convention — to avoid losing a good early-epoch
      solution to a transient late spike.
    * **History keys differ from** :func:`train_cra_pi_gnn`.
      ``train_cra_pi_gnn`` reports per-epoch ``"cost"``; CPRA reports
      ``"mean_cost"`` (per-replica average) because the raw cost scales
      linearly with R and is harder to compare across runs. When you
      mix the two trainers in a single plot, normalise by R yourself.
    * **Replica collapse with** ``vari_param=0`` **and**
      ``replica_problems=None``. With identical losses on every replica
      and shared embedding+backbone gradients, the R output channels
      drift toward the same fixed point. They start different (random
      init) and remain visibly distinct for the first few hundred
      epochs, but eventually collapse. For real variation
      diversification on a fixed problem, set ``vari_param > 0`` (e.g.
      ``0.1`` to ``0.5`` works in practice).
    """
    require_pyg()

    if int(num_replicas) < 1:
        raise ValueError(f"num_replicas must be >= 1, got {num_replicas}.")
    if curve_rate % 2 != 0:
        raise ValueError(
            f"curve_rate must be even (got {curve_rate}); odd exponents make "
            "the penalty 1-(1-2p)^c non-convex and asymmetric."
        )
    if num_epochs < 0:
        raise ValueError(f"num_epochs must be >= 0, got {num_epochs}.")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}.")
    if vari_param < 0.0:
        raise ValueError(
            f"vari_param must be >= 0 (got {vari_param}); negative values would "
            "actively *collapse* the replicas, defeating the purpose of CPRA."
        )

    if replica_problems is not None:
        if len(replica_problems) != int(num_replicas):
            raise ValueError(
                f"len(replica_problems)={len(replica_problems)} must equal "
                f"num_replicas={num_replicas}."
            )
        # Validate every replica's variable count matches the base problem so
        # the einsum inside loss_fn doesn't crash deep in the training loop
        # with an opaque "shape mismatch" message.
        base_n = getattr(problem, "num_nodes", None)
        if base_n is not None:
            for i, rp in enumerate(replica_problems):
                rp_n = getattr(rp, "num_nodes", None)
                if rp_n is not None and rp_n != base_n:
                    raise ValueError(
                        f"replica_problems[{i}].num_nodes={rp_n} does not "
                        f"match base problem.num_nodes={base_n}; CPRA expects "
                        "every replica to live on the same graph."
                    )
        problems_per_replica = list(replica_problems)
        # When all replicas share the *same Python object*, fall back to the
        # vectorised single-problem path below. (User typically passes a list
        # of distinct instances, so this is a small extra safety check.)
        single_problem = all(rp is problem for rp in problems_per_replica)
    else:
        problems_per_replica = [problem] * int(num_replicas)
        single_problem = True

    require_cuda_if_requested(device)
    device = torch.device(device) if isinstance(device, str) else device

    if seed is not None:
        fix_seed(seed)

    g = extract_nx_graph(problem, override=nx_graph)
    num_nodes = g.number_of_nodes()
    edge_index = nx_to_edge_index(g, device=device)
    _ensure_problem_on_device(problem, device)
    for rp in problems_per_replica:
        if rp is not problem:
            _ensure_problem_on_device(rp, device)

    in_feats_resolved = default_in_feats(num_nodes) if in_feats is None else in_feats
    hidden_resolved = in_feats_resolved if hidden_dim is None else hidden_dim
    model = GCNNet(
        num_nodes=num_nodes,
        in_feats=in_feats_resolved,
        hidden_dim=hidden_resolved,
        dropout=dropout,
        num_replicas=int(num_replicas),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    relax = BinaryRelaxation()
    reg_param = float(init_reg_param if annealing else 0.0)

    # Penalty-diversified CPRA typically passes R copies of the same QUBO
    # problem with different ``penalty`` coefficients. When every replica
    # exposes a per-replica ``Q_mat`` of identical shape, we can stack them
    # into a single ``(R, N, N)`` tensor and replace the Python R-step loop
    # with one batched einsum — which is several times faster on GPU because
    # the launch overhead of R small einsums dominates the actual flops.
    _q_stack: torch.Tensor | None = None
    if not single_problem:
        q_mats = [getattr(p, "Q_mat", None) for p in problems_per_replica]
        if (
            all(q is not None for q in q_mats)
            and all(isinstance(q, torch.Tensor) for q in q_mats)
            and len({tuple(q.shape) for q in q_mats}) == 1
        ):
            _q_stack = torch.stack([q.to(device) for q in q_mats], dim=0)

    history: dict[str, list] = {
        "loss": [],
        "mean_cost": [],
        "reg_term": [],
        "vari_term": [],
        "reg_param": [],
        "per_replica_obj": [],
    }

    best_obj_per_replica = [float("inf")] * int(num_replicas)
    best_bits_per_replica: list[torch.Tensor | None] = [None] * int(num_replicas)
    prev_loss = float("inf")
    prev_pen = float("inf")
    stagnant_steps = 0
    # Optional: stop once the discrete best across all replicas hasn't
    # improved for ``early_stop_disc_patience`` epochs. Disabled by default
    # to preserve the historical ``num_epochs``-fixed behaviour.
    best_overall_so_far = float("inf")
    disc_stagnant = 0

    runtime_start = time()
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)
        probs = model(edge_index)  # (N, R) when R >= 2
        if probs.dim() == 1:
            # GCNNet squeezes to (N,) when num_replicas == 1.
            probs = probs.unsqueeze(-1)

        # Per-replica continuous cost. Three paths in order of decreasing
        # speed: shared single problem (one einsum), per-replica QUBO with
        # identical Q_mat shape (one batched einsum across R), then the
        # generic Python loop fallback for non-QUBO problems.
        if single_problem:
            cost = problem.loss_fn(probs.t()).sum()
        elif _q_stack is not None:
            p_t = probs.t()  # (R, N)
            cost = torch.einsum("rn,rnm,rm->", p_t, _q_stack, p_t)
        else:
            per_replica_costs = [
                problems_per_replica[r].loss_fn(probs[:, r].unsqueeze(0)).sum()
                for r in range(int(num_replicas))
            ]
            cost = torch.stack(per_replica_costs).sum()

        # CRA penalty: BinaryRelaxation.penalty operates element-wise on the
        # last dim, so passing (R, N) gives the correct per-replica term.
        # We sum across both replicas and nodes.
        reg_term = relax.penalty(probs.t(), curve_rate).sum()

        # CPRA diversity term (only meaningful when R >= 2).
        if int(num_replicas) >= 2 and vari_param != 0.0:
            std_per_node = probs.std(dim=1)
            vari_term = -float(num_replicas) * std_per_node.sum()
        else:
            vari_term = torch.zeros((), device=probs.device, dtype=probs.dtype)

        loss = cost + reg_param * reg_term + vari_param * vari_term
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bits_all = (probs >= 0.5).to(probs.dtype)  # (N, R)
            if single_problem:
                # One batched einsum + one cuda->cpu transfer for the whole
                # replica vector — much cheaper than the R-step loop below.
                disc_objs = problem.loss_fn(bits_all.t()).cpu().tolist()
            elif _q_stack is not None:
                b = bits_all.t()  # (R, N)
                disc_objs = torch.einsum("rn,rnm,rm->r", b, _q_stack, b).cpu().tolist()
            else:
                disc_objs = [
                    float(problems_per_replica[r].loss_fn(bits_all[:, r].unsqueeze(0)).item())
                    for r in range(int(num_replicas))
                ]
            for r, disc_obj in enumerate(disc_objs):
                if disc_obj < best_obj_per_replica[r]:
                    best_obj_per_replica[r] = disc_obj
                    best_bits_per_replica[r] = bits_all[:, r].detach().clone()

        loss_v = float(loss.item())
        mean_cost_v = float(cost.item()) / float(num_replicas)
        pen_v = float(reg_term.item())
        vari_v = float(vari_term.item())
        history["loss"].append(loss_v)
        history["mean_cost"].append(mean_cost_v)
        history["reg_term"].append(pen_v)
        history["vari_term"].append(vari_v)
        history["reg_param"].append(reg_param)
        history["per_replica_obj"].append(disc_objs)

        if abs(prev_loss - loss_v) < tol and abs(prev_pen - pen_v) < tol:
            stagnant_steps += 1
            if stagnant_steps >= patience:
                if verbose:
                    print(f"[CPRA] Early stop at epoch {epoch} (stagnant for {patience} steps).")
                break
        else:
            stagnant_steps = 0
        prev_loss = loss_v
        prev_pen = pen_v

        # Optional second early-stop on the *discrete* best across replicas.
        # Useful for problems where the continuous loss keeps drifting after
        # the integer solution has clearly stabilised.
        if early_stop_disc_patience is not None:
            current_best = min(best_obj_per_replica)
            if current_best < best_overall_so_far - tol:
                best_overall_so_far = current_best
                disc_stagnant = 0
            else:
                disc_stagnant += 1
                if disc_stagnant >= int(early_stop_disc_patience):
                    if verbose:
                        print(
                            f"[CPRA] Early stop at epoch {epoch} (discrete best "
                            f"unchanged for {early_stop_disc_patience} epochs)."
                        )
                    break

        if annealing:
            reg_param += float(annealing_rate)

        if verbose and (epoch % check_interval == 0 or epoch == num_epochs - 1):
            best_overall = min(best_obj_per_replica)
            _print_progress(epoch, best_overall, loss_v, mean_cost_v, pen_v, reg_param)

    runtime = time() - runtime_start

    # Fallback for num_epochs == 0 — same contract as train_cra_pi_gnn.
    if all(b is None for b in best_bits_per_replica):
        with torch.no_grad():
            probs = model(edge_index)
            if probs.dim() == 1:
                probs = probs.unsqueeze(-1)
            bits_all = (probs >= 0.5).to(probs.dtype)
            if single_problem:
                fallback_objs = problem.loss_fn(bits_all.t()).cpu().tolist()
            else:
                fallback_objs = [
                    float(problems_per_replica[r].loss_fn(bits_all[:, r].unsqueeze(0)).item())
                    for r in range(int(num_replicas))
                ]
            for r, obj_r in enumerate(fallback_objs):
                best_bits_per_replica[r] = bits_all[:, r].detach().clone()
                best_obj_per_replica[r] = obj_r

    # Determine the overall best replica.
    best_replica = int(min(range(int(num_replicas)), key=lambda r: best_obj_per_replica[r]))
    best_bits = best_bits_per_replica[best_replica]
    best_obj = best_obj_per_replica[best_replica]

    # Per-replica score summaries.
    replica_records: list[dict] = []
    for r in range(int(num_replicas)):
        bits_r = best_bits_per_replica[r]
        score_r = safe_score_summary(
            problems_per_replica[r],
            bits_r,
            fallback_obj=float(best_obj_per_replica[r]),
        )
        replica_records.append(
            {
                "replica": r,
                "obj": float(best_obj_per_replica[r]),
                "score": score_r,
                "sol": bits_r.detach().cpu(),
            }
        )

    # Top-level score reflects the best replica, with all replicas in extra.
    score: dict = dict(replica_records[best_replica]["score"])
    extra = dict(score.get("extra") or {})
    extra["best_replica"] = best_replica
    extra["replicas"] = replica_records
    score["extra"] = extra

    if verbose:
        print("\n" + "=" * 30 + " [FINAL] " + "=" * 30)
        print(f"  BEST REPLICA : {best_replica}")
        print(f"  BEST LOSS    : {best_obj}")
        print(f"  ALL OBJS     : {[f'{o:.4f}' for o in best_obj_per_replica]}")
        print(f"  RUN TIME     : {runtime:.2f} s")
        print("=" * 69)

    return AnnealResult(
        best_sol=best_bits,
        best_obj=best_obj,
        runtime=runtime,
        history={k: np.asarray(v, dtype=float) for k, v in history.items()},
        callbacks=[],
        score=score,
    )
