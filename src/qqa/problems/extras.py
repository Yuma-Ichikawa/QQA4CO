"""Extra problem catalog for QQA.

Eight classic discrete-optimization problems that plug into the unified
``qqa.anneal`` loop alongside the graph / spin / categorical problems
already shipped.

Design
------
* Every class exposes ``loss_fn(x) -> (B,)`` that is **minimised** by QQA
  (so e.g. Knapsack returns the *negative* value of the packed items).
* Every class exposes ``score_summary(x_disc) -> dict`` returning a
  human-readable breakdown of the best solution so the dashboard can
  print e.g. *"packed value: 138 / 150, feasible: True"*.
* Variable sizing attributes are chosen to match the relaxation
  that each problem consumes:
    - :class:`BinaryRelaxation`     expects ``num_nodes``.
    - :class:`SpinRelaxation`       expects ``num_spins``.
    - :class:`CategoricalRelaxation` expects ``num_node`` + ``num_category``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qqa.problems.base import COProblem, normalize_graph
from qqa.relaxation import BinaryRelaxation, CategoricalRelaxation, SpinRelaxation

__all__ = [
    "Knapsack",
    "NumberPartitioning",
    "MaxSAT3",
    "VertexCover",
    "GraphBisection",
    "TSP",
    "QAP",
    "NQueens",
]


# ---------------------------------------------------------------------------
# Utility: deterministic instance generators
# ---------------------------------------------------------------------------


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ensure_batched(x: torch.Tensor, var_ndim: int) -> torch.Tensor:
    """Return ``x`` with a leading batch dim. ``var_ndim`` is the intrinsic
    rank of a *single* solution (1 for binary / spin, 2 for categorical)."""
    if x.ndim == var_ndim:
        return x.unsqueeze(0)
    if x.ndim == var_ndim + 1:
        return x
    raise ValueError(f"Expected ndim in {{{var_ndim}, {var_ndim + 1}}}, got {x.ndim}")


# ---------------------------------------------------------------------------
# Binary problems
# ---------------------------------------------------------------------------


class NumberPartitioning(COProblem):
    r"""Classic number partitioning of ``N`` positive integers.

    Given ``a_1, ..., a_N`` the goal is to split them into two subsets
    whose sums are as close as possible, i.e. minimise

    .. math::
        \Bigl(\sum_i a_i s_i\Bigr)^2,
        \quad s_i \in \{-1, +1\}.

    Uses :class:`SpinRelaxation` internally so ``loss_fn`` receives
    :math:`s \in [-1, 1]^{B\times N}` directly.
    """

    def __init__(
        self,
        N: int = 64,
        seed: int | None = 0,
        max_value: int = 100,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.N = N
        self.num_spins = N
        self.num_nodes = N  # compatibility for UI/CLI size displays
        self.max_value = max_value
        self.seed = seed
        self.device = device
        rng = _rng(seed)
        self.values = torch.as_tensor(
            rng.integers(1, max_value + 1, size=N), dtype=torch.float32, device=device
        )
        self.relaxation = SpinRelaxation()

    def loss_fn(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, N), spins in [-1, 1].
        diff = torch.einsum("bi,i->b", s, self.values)
        return diff * diff

    def score_summary(self, s_disc: torch.Tensor) -> dict:
        s = _ensure_batched(s_disc, 1)
        with torch.no_grad():
            diff = torch.einsum("bi,i->b", s.float(), self.values)
        best_idx = int(torch.argmin(diff.abs()).item())
        best_diff = float(diff[best_idx].item())
        return {
            "label": "|Σ aᵢ sᵢ|",
            "value": abs(best_diff),
            "unit": "",
            "feasible": True,
            "extra": {"signed_diff": best_diff, "N": self.N},
        }


class Knapsack(COProblem):
    r"""0/1 knapsack.

    Maximise :math:`\sum_i v_i x_i` subject to
    :math:`\sum_i w_i x_i \le C`. We minimise

    .. math::
        -\sum_i v_i x_i + \lambda\,\bigl[\max(0, \sum_i w_i x_i - C)\bigr]^2

    so solutions that respect the capacity dominate.
    """

    def __init__(
        self,
        N: int = 40,
        capacity_ratio: float = 0.5,
        penalty: float | None = None,
        seed: int | None = 0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.N = N
        self.num_nodes = N
        self.device = device
        rng = _rng(seed)
        values = rng.integers(1, 100, size=N)
        weights = rng.integers(1, 100, size=N)
        self.values = torch.as_tensor(values, dtype=torch.float32, device=device)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.capacity = float(weights.sum()) * capacity_ratio
        # Default penalty keeps the violation term comparable to the value term.
        self.penalty = float(values.max()) * 2 if penalty is None else float(penalty)
        self.relaxation = BinaryRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.einsum("bi,i->b", x, self.values)
        weight = torch.einsum("bi,i->b", x, self.weights)
        overflow = torch.clamp(weight - self.capacity, min=0.0)
        return -value + self.penalty * overflow * overflow

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x_disc = _ensure_batched(x_disc, 1)
        with torch.no_grad():
            value = torch.einsum("bi,i->b", x_disc.float(), self.values)
            weight = torch.einsum("bi,i->b", x_disc.float(), self.weights)
        feas = weight <= self.capacity + 1e-6
        if feas.any():
            value_feas = value.clone()
            value_feas[~feas] = -float("inf")
            idx = int(torch.argmax(value_feas).item())
            best_value = float(value[idx].item())
            best_weight = float(weight[idx].item())
            feasible = True
        else:
            idx = int(torch.argmin(weight).item())
            best_value = float(value[idx].item())
            best_weight = float(weight[idx].item())
            feasible = False
        return {
            "label": "packed value",
            "value": best_value,
            "unit": "",
            "feasible": feasible,
            "extra": {"weight": best_weight, "capacity": self.capacity},
        }


class VertexCover(COProblem):
    r"""Minimum vertex cover on an undirected graph.

    Select a minimum-size vertex subset that touches every edge; we use
    the QUBO form

    .. math::
        H = \sum_i x_i + \lambda \sum_{(u,v)\in E} (1 - x_u)(1 - x_v)
    """

    def __init__(
        self,
        graph,
        penalty: float = 4.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        graph = normalize_graph(graph)
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.device = device
        self.penalty = penalty
        edges = list(graph.edges())
        if edges:
            edge_idx = torch.as_tensor(edges, dtype=torch.long, device=device)
        else:
            edge_idx = torch.zeros((0, 2), dtype=torch.long, device=device)
        self.edge_u = edge_idx[:, 0]
        self.edge_v = edge_idx[:, 1]
        self.num_edges = edge_idx.shape[0]
        self.relaxation = BinaryRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        cover_size = x.sum(dim=-1)
        if self.num_edges == 0:
            return cover_size
        xu = x[:, self.edge_u]
        xv = x[:, self.edge_v]
        uncovered = (1 - xu) * (1 - xv)
        return cover_size + self.penalty * uncovered.sum(dim=-1)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x_disc = _ensure_batched(x_disc, 1)
        with torch.no_grad():
            xd = x_disc.float()
            sizes = xd.sum(dim=-1)
            if self.num_edges == 0:
                violations = torch.zeros_like(sizes)
            else:
                xu = xd[:, self.edge_u]
                xv = xd[:, self.edge_v]
                violations = ((1 - xu) * (1 - xv)).sum(dim=-1)
        feas = violations <= 0.5
        if feas.any():
            s = sizes.clone()
            s[~feas] = float("inf")
            idx = int(torch.argmin(s).item())
            feasible = True
        else:
            idx = int(torch.argmin(violations).item())
            feasible = False
        return {
            "label": "cover size",
            "value": int(sizes[idx].item()),
            "unit": "",
            "feasible": feasible,
            "extra": {
                "uncovered_edges": int(violations[idx].item()),
                "num_edges": int(self.num_edges),
            },
        }


class GraphBisection(COProblem):
    r"""Balanced graph bisection.

    Partition vertices into two equal-size sets minimising the cut:

    .. math::
        H = \sum_{(u,v)\in E} (x_u - x_v)^2
            + \lambda \bigl(\sum_i x_i - N/2 \bigr)^2
    """

    def __init__(
        self,
        graph,
        balance_penalty: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        graph = normalize_graph(graph)
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.device = device
        self.balance_penalty = balance_penalty
        edges = list(graph.edges())
        if edges:
            edge_idx = torch.as_tensor(edges, dtype=torch.long, device=device)
        else:
            edge_idx = torch.zeros((0, 2), dtype=torch.long, device=device)
        self.edge_u = edge_idx[:, 0]
        self.edge_v = edge_idx[:, 1]
        self.num_edges = edge_idx.shape[0]
        self.target = self.num_nodes / 2.0
        self.relaxation = BinaryRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_edges == 0:
            cut = torch.zeros(x.shape[0], device=x.device)
        else:
            xu = x[:, self.edge_u]
            xv = x[:, self.edge_v]
            cut = ((xu - xv) ** 2).sum(dim=-1)
        balance = (x.sum(dim=-1) - self.target) ** 2
        return cut + self.balance_penalty * balance

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x_disc = _ensure_batched(x_disc, 1)
        with torch.no_grad():
            xd = x_disc.float()
            sizes = xd.sum(dim=-1)
            balance = (sizes - self.target).abs()
            if self.num_edges == 0:
                cut = torch.zeros_like(sizes)
            else:
                xu = xd[:, self.edge_u]
                xv = xd[:, self.edge_v]
                cut = ((xu - xv) ** 2).sum(dim=-1)
        feas = balance <= 0.5
        if feas.any():
            c = cut.clone()
            c[~feas] = float("inf")
            idx = int(torch.argmin(c).item())
            feasible = True
        else:
            idx = int(torch.argmin(balance).item())
            feasible = False
        return {
            "label": "cut size",
            "value": int(cut[idx].item()),
            "unit": "",
            "feasible": feasible,
            "extra": {
                "partition_sizes": (
                    int(sizes[idx].item()),
                    self.num_nodes - int(sizes[idx].item()),
                ),
            },
        }


@dataclass(frozen=True)
class SATClause:
    """Signed literals of a 3-SAT clause. ``lits[k] = (var_idx, sign)``."""

    lits: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


class MaxSAT3(COProblem):
    r"""Random 3-SAT / MaxSAT.

    Minimise the number of unsatisfied 3-CNF clauses. A clause is
    represented by three signed literals. Let :math:`L_i(x)` equal
    :math:`x_i` when the literal is positive and :math:`1 - x_i` otherwise.
    The clause is violated iff
    :math:`(1-L_1)(1-L_2)(1-L_3) = 1`, which we sum across all clauses as
    the (exact) loss.
    """

    def __init__(
        self,
        N: int = 40,
        ratio: float = 4.0,
        seed: int | None = 0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.N = N
        self.num_nodes = N
        self.num_clauses = max(1, int(round(ratio * N)))
        self.device = device
        rng = _rng(seed)

        vars_idx = np.empty((self.num_clauses, 3), dtype=np.int64)
        signs = np.empty((self.num_clauses, 3), dtype=np.float32)
        for m in range(self.num_clauses):
            chosen = rng.choice(N, size=3, replace=False)
            vars_idx[m] = chosen
            signs[m] = rng.choice([-1.0, 1.0], size=3)

        # Store as (M, 3) tensors and precompute an (M, 3) "positive mask"
        # with +1 for positive literals and -1 for negatives (used in forward).
        self.clause_vars = torch.as_tensor(vars_idx, dtype=torch.long, device=device)
        self.clause_signs = torch.as_tensor(signs, dtype=torch.float32, device=device)
        self.relaxation = BinaryRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N) in [0, 1].
        # Gather per-literal x values: shape (B, M, 3).
        B = x.shape[0]
        x_lit = x[:, self.clause_vars]  # broadcasts (B, M, 3)
        sgn = self.clause_signs.unsqueeze(0).expand(B, -1, -1)
        # When sign=+1, literal value = x; when -1, literal value = 1-x.
        lit_val = torch.where(sgn > 0, x_lit, 1.0 - x_lit)
        unsat = (1.0 - lit_val).prod(dim=2)  # 1 iff all three literals are 0
        return unsat.sum(dim=1)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x_disc = _ensure_batched(x_disc, 1)
        with torch.no_grad():
            x_lit = x_disc[:, self.clause_vars]
            sgn = self.clause_signs.unsqueeze(0).expand(x_disc.shape[0], -1, -1)
            lit_val = torch.where(sgn > 0, x_lit, 1.0 - x_lit)
            unsat = (1.0 - lit_val).prod(dim=2).sum(dim=1)
        idx = int(torch.argmin(unsat).item())
        num_unsat = int(unsat[idx].item())
        return {
            "label": "clauses satisfied",
            "value": self.num_clauses - num_unsat,
            "unit": f"/ {self.num_clauses}",
            "feasible": num_unsat == 0,
            "extra": {"ratio": self.num_clauses / self.N},
        }


# ---------------------------------------------------------------------------
# Categorical / permutation problems
# ---------------------------------------------------------------------------


class TSP(COProblem):
    r"""Symmetric travelling salesperson — solved with the **penalty method**.

    Variable encoding follows Lucas (2014) "Ising formulations of many NP
    problems": ``x ∈ {0, 1}^{N×N}`` with ``x[t, i] = 1`` iff city ``i``
    occupies tour position ``t``. Three additive terms make every
    constraint visible inside ``loss_fn``:

    * **distance** — :math:`\sum_t \sum_{i \neq j} d_{ij}\, x_{t,i}\, x_{t+1,j}`
      (cyclic, so position ``N-1`` wraps back to ``0``).
    * **row penalty** — :math:`\lambda_r \sum_t (\sum_i x_{t,i} - 1)^2` so
      every position holds exactly one city.
    * **column penalty** — :math:`\lambda_c \sum_i (\sum_t x_{t,i} - 1)^2`
      so every city appears exactly once.

    Unlike the previous CategoricalRelaxation-based formulation (where the
    row constraint was enforced by softmax inside the relaxation, hiding
    it from the loss), this version uses :class:`BinaryRelaxation` so QQA
    sees both penalties as gradients and the user can tune their weights
    from the Streamlit sidebar — a textbook penalty-method setup.
    """

    def __init__(
        self,
        N: int = 8,
        seed: int | None = 0,
        row_penalty: float = 5.0,
        col_penalty: float = 5.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.N = N
        self.num_node = N  # positions (kept for backwards compat)
        self.num_category = N  # cities
        self.num_nodes = N * N  # BinaryRelaxation reads num_nodes
        self.device = device
        self.row_penalty = float(row_penalty)
        self.col_penalty = float(col_penalty)
        rng = _rng(seed)
        self.coords = torch.as_tensor(rng.random((N, 2)), dtype=torch.float32, device=device)
        diff = self.coords.unsqueeze(0) - self.coords.unsqueeze(1)
        self.distance = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
        # Two penalty terms ⇒ both constraints are *gradients*, not hard
        # constraints embedded in a softmax.  The shape_fn lifts the
        # default (sol_size, num_nodes) latent into a position×city grid.
        self.relaxation = BinaryRelaxation(shape_fn=lambda sol_size, p: (sol_size, p.N, p.N))

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) ∈ [0, 1]
        x_next = torch.roll(x, shifts=-1, dims=1)
        tour = torch.einsum("bti,ij,btj->b", x, self.distance, x_next)
        row_pen = ((x.sum(dim=2) - 1.0) ** 2).sum(dim=1)
        col_pen = ((x.sum(dim=1) - 1.0) ** 2).sum(dim=1)
        return tour + self.row_penalty * row_pen + self.col_penalty * col_pen

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x_disc = _ensure_batched(x_disc, 2)
        with torch.no_grad():
            # ``x_disc`` lives in {0, 1}^{B×T×C} but the penalty method does
            # not guarantee a valid permutation matrix per replica. We
            # report feasibility of the *raw* projection and additionally
            # snap each replica to its closest permutation via the
            # Hungarian algorithm so the dashboard can always show a real
            # tour.
            idx_raw = torch.argmax(x_disc, dim=2)  # (B, T)
            row_counts = x_disc.sum(dim=2)  # (B, T)
            col_counts = torch.zeros(x_disc.shape[0], self.N, device=x_disc.device)
            col_counts.scatter_add_(1, idx_raw, torch.ones_like(idx_raw, dtype=torch.float32))
            row_violation = ((row_counts - 1.0) ** 2).sum(dim=1)
            col_violation = ((col_counts - 1.0) ** 2).sum(dim=1)
            total_violation = row_violation + col_violation

            # Snap each replica's continuous matrix to a permutation:
            # max trace assignment ≡ ``linear_sum_assignment(-x)``.
            from scipy.optimize import linear_sum_assignment as _lsa  # noqa: PLC0415

            x_np = x_disc.detach().cpu().numpy().astype(np.float32)
            B = x_np.shape[0]
            snapped_lens = torch.empty(B)
            snapped_idx = torch.empty(B, self.N, dtype=torch.long)
            for b in range(B):
                row_ind, col_ind = _lsa(-x_np[b])
                # ``row_ind`` is just 0..N-1 already; ``col_ind[t]`` is the
                # city assigned to position ``t``.
                snapped_idx[b] = torch.as_tensor(col_ind, dtype=torch.long)
                coords = self.coords.detach().cpu()[col_ind]
                coords_next = torch.roll(coords, shifts=-1, dims=0)
                seg = torch.sqrt(((coords - coords_next) ** 2).sum(dim=-1) + 1e-12)
                snapped_lens[b] = seg.sum()

            # Pick the replica with the shortest *snapped* tour, breaking
            # ties by preferring those with the smallest raw violation so
            # users see "good" tours even when the penalty has fully
            # converged.
            best = int(torch.argmin(snapped_lens).item())
            raw_feasible = bool(total_violation[best].item() < 0.5)

            # Replace ``best_sol`` in-place with a clean one-hot tour so
            # downstream visualisation (TSP solution renderer) sees a
            # valid permutation matrix.  Mutation is intentional — the
            # caller passes ``best_sol`` directly and wants the cleaned
            # version surfaced via ``result.best_sol``.
            cleaned = torch.zeros_like(x_disc[best])
            cleaned[torch.arange(self.N), snapped_idx[best]] = 1.0
            x_disc[best] = cleaned

        return {
            # The displayed tour is *always* a valid permutation thanks to
            # the Hungarian snap, so the dashboard label says "feasible".
            # Whether the raw penalty optimiser itself converged to a
            # permutation is reported separately in ``extra.raw_feasible``.
            "label": "tour length (Hungarian-snapped)" if not raw_feasible else "tour length",
            "value": float(snapped_lens[best].item()),
            "unit": "",
            "feasible": True,
            "extra": {
                "unique_cities": int((col_counts[best] > 0).sum().item()),
                "row_violation": float(row_violation[best].item()),
                "col_violation": float(col_violation[best].item()),
                "raw_feasible": raw_feasible,
                "snapped": not raw_feasible,
            },
        }


class QAP(COProblem):
    r"""Quadratic assignment problem (random flow / distance matrices).

    Minimise :math:`\sum_{i,j} F_{ij} D_{\pi(i)\pi(j)}` where ``π`` assigns
    facilities to locations. Encoded as a CategoricalRelaxation with
    ``x[:, i, k] = 1`` iff facility ``i`` goes to location ``k``.
    """

    def __init__(
        self,
        N: int = 10,
        seed: int | None = 0,
        column_penalty: float = 5.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.N = N
        self.num_node = N
        self.num_category = N
        self.num_nodes = N
        self.device = device
        self.column_penalty = column_penalty
        rng = _rng(seed)
        F = rng.integers(0, 10, size=(N, N)).astype(np.float32)
        D = rng.integers(0, 10, size=(N, N)).astype(np.float32)
        np.fill_diagonal(F, 0)
        np.fill_diagonal(D, 0)
        F = (F + F.T) / 2
        D = (D + D.T) / 2
        self.F = torch.as_tensor(F, device=device)
        self.D = torch.as_tensor(D, device=device)
        self.relaxation = CategoricalRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        # cost = Σ_{ij} F_ij D_kl x[i,k] x[j,l]
        xD = torch.einsum("bik,kl->bil", x, self.D)
        cost_per_fac = torch.einsum("bil,bjl,ij->b", xD, x, self.F)
        col_sum = x.sum(dim=1)
        col_pen = ((col_sum - 1.0) ** 2).sum(dim=1)
        return cost_per_fac + self.column_penalty * col_pen

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x_disc = _ensure_batched(x_disc, 2)
        with torch.no_grad():
            assign = torch.argmax(x_disc, dim=2)  # (B, N)
            B = x_disc.shape[0]
            costs = torch.zeros(B, device=x_disc.device)
            for b in range(B):
                a = assign[b]
                # Permutation-indexed D.
                D_perm = self.D[a[:, None], a[None, :]]
                costs[b] = (self.F * D_perm).sum()
            # Count collisions (non-permutation).
            col_counts = torch.zeros(B, self.N, device=x_disc.device)
            col_counts.scatter_add_(1, assign, torch.ones_like(assign, dtype=torch.float32))
            duplicates = ((col_counts - 1.0).clamp(min=0.0)).sum(dim=1)
        feas = duplicates < 0.5
        if feas.any():
            c = costs.clone()
            c[~feas] = float("inf")
            best = int(torch.argmin(c).item())
            feasible = True
        else:
            best = int(torch.argmin(duplicates).item())
            feasible = False
        return {
            "label": "assignment cost",
            "value": float(costs[best].item()),
            "unit": "",
            "feasible": feasible,
            "extra": {"duplicated_locations": int(duplicates[best].item())},
        }


class NQueens(COProblem):
    r"""Place ``N`` non-attacking queens on an ``N×N`` board.

    Each row is a simplex (handled by :class:`CategoricalRelaxation`) so
    row constraints are free; we penalise column-, diagonal- and
    anti-diagonal conflicts via pair-counts.
    """

    def __init__(self, N: int = 10, device: str | torch.device = "cpu"):
        super().__init__()
        self.N = N
        self.num_node = N
        self.num_category = N
        self.num_nodes = N
        self.device = device
        self.relaxation = CategoricalRelaxation()

        rows = torch.arange(N, device=device)
        cols = torch.arange(N, device=device)
        # Precompute one-hot indicators per (row, col) for diagonal and
        # anti-diagonal lines. diagonals indexed by (r - c), anti by (r + c).
        diag_idx = rows[:, None] - cols[None, :] + (N - 1)  # ∈ [0, 2N-2]
        anti_idx = rows[:, None] + cols[None, :]
        self._diag_idx = diag_idx
        self._anti_idx = anti_idx
        self._num_diag = 2 * N - 1

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, R, C). Rows already simplex-normalised, so row pair-counts
        # are 1·0 = 0 in the discrete limit. Penalise columns, diagonals,
        # anti-diagonals.
        B = x.shape[0]
        col_counts = x.sum(dim=1)  # (B, C)
        col_pen = (col_counts * (col_counts - 1)).sum(dim=1)

        diag_counts = torch.zeros(B, self._num_diag, device=x.device, dtype=x.dtype)
        anti_counts = torch.zeros(B, self._num_diag, device=x.device, dtype=x.dtype)
        diag_flat = self._diag_idx.reshape(-1)
        anti_flat = self._anti_idx.reshape(-1)
        x_flat = x.reshape(B, -1)
        diag_counts = diag_counts.scatter_add(1, diag_flat.unsqueeze(0).expand(B, -1), x_flat)
        anti_counts = anti_counts.scatter_add(1, anti_flat.unsqueeze(0).expand(B, -1), x_flat)
        diag_pen = (diag_counts * (diag_counts - 1)).sum(dim=1)
        anti_pen = (anti_counts * (anti_counts - 1)).sum(dim=1)
        return col_pen + diag_pen + anti_pen

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        x_disc = _ensure_batched(x_disc, 2)
        with torch.no_grad():
            conflicts = self.loss_fn(x_disc)
        idx = int(torch.argmin(conflicts).item())
        conf = int(conflicts[idx].item())
        return {
            "label": "attacking pairs",
            "value": conf,
            "unit": "",
            "feasible": conf == 0,
            "extra": {"board": self.N},
        }
