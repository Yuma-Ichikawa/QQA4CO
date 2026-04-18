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

from time import time

import networkx as nx
import numpy as np
import torch

from qqa.annealing import AnnealResult
from qqa.pignn._import import require_pyg
from qqa.pignn.graph import extract_nx_graph, nx_to_edge_index
from qqa.pignn.model import GCNNet, default_in_feats
from qqa.relaxation import BinaryRelaxation


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

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled torch build, or pass device='cpu'."
        )
    device = torch.device(device) if isinstance(device, str) else device

    if seed is not None:
        from qqa.utils import fix_seed

        fix_seed(seed)

    g = extract_nx_graph(problem, override=nx_graph)
    num_nodes = g.number_of_nodes()
    edge_index = nx_to_edge_index(g, device=device)

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

    runtime_start = time()
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
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

    score: dict = {}
    try:
        score = problem.score_summary(best_bits)
    except Exception as exc:  # noqa: BLE001 - mirror qqa.anneal's contract
        score = {
            "label": "loss",
            "value": float(best_obj),
            "unit": "",
            "feasible": False,
            "extra": {"error": str(exc)},
        }

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
