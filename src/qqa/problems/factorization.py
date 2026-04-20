"""Integer-factorization Ising/QUBO benchmark.

Direct implementation of the planted-solution Ising model described in
Hen, *Planted-solution SAT and Ising benchmarks from integer
factorization*, arXiv:2604.09837 (2026).

Given a semi-prime ``N = p * q`` with bit-lengths ``n_p, n_q``, we encode
the long-multiplication ``N = pq`` as a circuit of AND and XOR clauses:

* **Partial products** ``a_{ij} = p_i ∧ q_j`` are formed for all
  ``i ∈ [0, n_p)`` and ``j ∈ [0, n_q)``.
* Each output column ``k = i + j`` collects the partial products that
  map to it (and any incoming carries from column ``k − 1``).
* Inside a column, pairs are reduced with **half-adders**:
  ``sum = x ⊕ y`` stays in the column and ``carry = x ∧ y`` is pushed
  to ``k + 1`` (Eq. 2 of the paper).
* The single bit that survives in column ``k`` is **pinned** to the
  known constant ``N_k = (N >> k) & 1``.

Each AND clause becomes the three-spin gadget of Eq. (10) of the paper,

.. math::
    E_\\wedge(s_i, s_j, s_k) = 3 - s_i - s_j + s_i s_j
        + 2 s_k - 2 s_i s_k - 2 s_j s_k,

and each XOR clause becomes the four-spin gadget of Eq. (11),

.. math::
    E_\\oplus(s_i, s_j, s_k, s_a) = 4 - s_i - s_j + s_i s_j
        + s_k - s_i s_k - s_j s_k
        + 2 s_a - 2 s_i s_a - 2 s_j s_a + 2 s_k s_a,

where the auxiliary spin's planted value ``s_a^* = +1 iff s_i = s_j = +1``
(Eq. 12). Pinned spins are *substituted* (not penalised) to keep the
problem tight: pinning ``s_p = ±1`` folds ``h[p] s_p`` into the constant
and ``J[p, j] s_p`` into ``h[j]``, then drops index ``p``. The result is
a strictly QUBO problem ``loss(x) = x^T Q x + offset`` whose **global
minimum equals zero** by construction (planted solution).

Solving the QUBO recovers the bits of ``p`` and ``q``, and ``score_summary``
returns the decoded factor pair, the bit-Hamming distance to the planted
configuration, and the residual energy.

This implementation does **not** apply the preprocessing pipeline of
Sec. III of the paper; problem sizes therefore scale as :math:`O(d^4)`
in the symmetric case ``n_p = n_q = d``. For QQA-scale benchmarks
(``d ≤ 10``) this is comfortably within reach.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from qqa.problems.base import QUBOProblem
from qqa.relaxation import BinaryRelaxation

# --------------------------------------------------------------------------- #
# Number-theory helpers                                                       #
# --------------------------------------------------------------------------- #


def _is_probable_prime(n: int, *, witnesses: Sequence[int] = (2, 3, 5, 7, 11, 13)) -> bool:
    """Deterministic Miller–Rabin for ``n < 3 * 10^18`` (more than enough
    for the bit-lengths we benchmark).
    """
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def random_prime(bit_length: int, rng: np.random.Generator) -> int:
    """Sample a prime with **exactly** ``bit_length`` bits."""
    if bit_length < 2:
        raise ValueError("bit_length must be ≥ 2 (the smallest prime is 2 = 0b10).")
    lo = 1 << (bit_length - 1)
    hi = (1 << bit_length) - 1
    while True:
        candidate = int(rng.integers(lo, hi + 1))
        if candidate % 2 == 0:
            candidate += 1
            if candidate > hi:
                continue
        if _is_probable_prime(candidate):
            return candidate


def random_semiprime(
    bit_length: int,
    *,
    seed: int | None = None,
    distinct: bool = True,
) -> tuple[int, int]:
    """Pick a pair ``(p, q)`` of equal-bit-length primes.

    With ``distinct=True`` the function rejects ``p == q`` so that the
    factorization is non-trivially two-fold degenerate (Sec. II.D of the
    paper); with ``distinct=False`` the perfect-square ``p == q`` case is
    allowed.
    """
    rng = np.random.default_rng(seed)
    p = random_prime(bit_length, rng)
    while True:
        q = random_prime(bit_length, rng)
        if not distinct or p != q:
            return (p, q) if p <= q else (q, p)


# --------------------------------------------------------------------------- #
# Compilation: (p, q) -> Ising clauses                                        #
# --------------------------------------------------------------------------- #


@dataclass
class _Compilation:
    """Intermediate clause + planted-spin record produced by ``_compile``."""

    n_vars: int
    """Total number of Ising spins (free + pinned + auxiliary)."""

    planted: list[int]
    """``planted[i] ∈ {0, 1}``: planted Boolean value for spin ``i``."""

    pinned: dict[int, int]
    """``{spin_idx: 0 or 1}`` for spins fixed to a known constant
    (the bits of ``N`` and the input bits if the user wants to pin them)."""

    and_clauses: list[tuple[int, int, int]]
    """``(s_i, s_j, s_k)`` triples enforcing ``s_k ↔ s_i ∧ s_j``."""

    xor_clauses: list[tuple[int, int, int, int]]
    """``(s_i, s_j, s_k, s_a)`` quadruples enforcing ``s_k ↔ s_i ⊕ s_j``;
    ``s_a`` is the auxiliary spin whose planted value is ``s_i ∧ s_j``."""

    p_bits: list[int]
    """Spin indices of the bits of ``p`` (``p_bits[i]`` is the spin for
    ``p_i``)."""

    q_bits: list[int]
    """Spin indices of the bits of ``q``."""


def _compile(p: int, q: int) -> _Compilation:
    """Lower the multiplication ``p * q = N`` into AND / XOR clauses."""
    n_p = max(2, p.bit_length())
    n_q = max(2, q.bit_length())
    N = p * q
    n_out = n_p + n_q  # number of output bit columns

    planted: list[int] = []
    pinned: dict[int, int] = {}
    and_clauses: list[tuple[int, int, int]] = []
    xor_clauses: list[tuple[int, int, int, int]] = []

    def new_spin(value: int) -> int:
        idx = len(planted)
        planted.append(value & 1)
        return idx

    p_bits = [new_spin((p >> i) & 1) for i in range(n_p)]
    q_bits = [new_spin((q >> j) & 1) for j in range(n_q)]

    # Partial products live in columns indexed by k = i + j.
    columns: list[list[int]] = [[] for _ in range(n_out)]
    for i in range(n_p):
        for j in range(n_q):
            pp_val = ((p >> i) & 1) * ((q >> j) & 1)
            pp_idx = new_spin(pp_val)
            and_clauses.append((p_bits[i], q_bits[j], pp_idx))
            columns[i + j].append(pp_idx)

    # Half-adder reduction: pop two, push (sum, carry).
    #
    # Optimisation: each half-adder produces ``carry = x ∧ y`` (Eq. 2 of
    # the paper), which is *exactly* the value that Eq. (12) prescribes
    # for the XOR gadget's auxiliary ("the auxiliary records the AND of
    # the two XOR inputs").  We therefore allocate ONE spin per
    # half-adder for the joint carry + XOR-aux role and feed it to BOTH
    # clauses; this saves one spin per half-adder (≈ 33% of all spins
    # for d ≥ 4) and is exactly what the paper's Sec. III preprocessing
    # would discover via union-find.
    for k in range(n_out):
        col = columns[k]
        while len(col) >= 2:
            x = col.pop(0)
            y = col.pop(0)
            xv, yv = planted[x], planted[y]
            sum_idx = new_spin(xv ^ yv)
            carry_idx = new_spin(xv & yv)
            xor_clauses.append((x, y, sum_idx, carry_idx))
            and_clauses.append((x, y, carry_idx))
            col.insert(0, sum_idx)
            if k + 1 < n_out:
                columns[k + 1].append(carry_idx)

        # Pin the single surviving bit (or 0 if the column collapsed) to N_k.
        n_k = (N >> k) & 1
        if col:
            (final_idx,) = col
            if planted[final_idx] != n_k:
                raise RuntimeError(
                    f"internal: column {k} planted bit {planted[final_idx]} ≠ N_k={n_k}"
                )
            pinned[final_idx] = n_k
        else:
            if n_k != 0:
                raise RuntimeError(f"internal: column {k} has no spin but N_k={n_k} ≠ 0")

    # Pin the high bits of p and q (no enforcement — the optimiser is free
    # to permute them; pinning the leading bit anchors the known
    # bit-length without breaking the (p, q) <-> (q, p) symmetry that the
    # paper highlights in Sec. II.D).
    pinned[p_bits[n_p - 1]] = 1  # p has exactly n_p bits => MSB is 1
    pinned[q_bits[n_q - 1]] = 1

    return _Compilation(
        n_vars=len(planted),
        planted=planted,
        pinned=pinned,
        and_clauses=and_clauses,
        xor_clauses=xor_clauses,
        p_bits=p_bits,
        q_bits=q_bits,
    )


# --------------------------------------------------------------------------- #
# Ising assembly (gadgets) → free QUBO                                        #
# --------------------------------------------------------------------------- #


def _add_and_gadget(h: np.ndarray, J: np.ndarray, i: int, j: int, k: int) -> float:
    """Accumulate Eq. (10): returns the constant contribution to ``E_0``."""
    h[i] -= 1.0
    h[j] -= 1.0
    h[k] += 2.0
    J[i, j] += 1.0
    J[i, k] -= 2.0
    J[j, k] -= 2.0
    return 3.0


def _add_xor_gadget(h: np.ndarray, J: np.ndarray, i: int, j: int, k: int, a: int) -> float:
    """Accumulate Eq. (11): returns the constant contribution to ``E_0``."""
    h[i] -= 1.0
    h[j] -= 1.0
    h[k] += 1.0
    h[a] += 2.0
    J[i, j] += 1.0
    J[i, k] -= 1.0
    J[j, k] -= 1.0
    J[i, a] -= 2.0
    J[j, a] -= 2.0
    J[k, a] += 2.0
    return 4.0


def _build_qubo_from_compilation(c: _Compilation) -> tuple[np.ndarray, float, list[int]]:
    """Build the symmetric upper-triangular Ising ``(h, J)`` from clauses,
    fold the pinned spins as constants, then convert to QUBO ``Q`` with
    ``s = 2 x − 1``.

    Returns ``(Q, offset, free_indices)``. The Boolean variable ``i`` of
    ``Q`` corresponds to spin ``free_indices[i]`` of the original
    compilation (so the planted Boolean assignment is
    ``[c.planted[free_indices[i]] for i in range(len(free_indices))]``).
    """
    n = c.n_vars
    h = np.zeros(n, dtype=np.float64)
    J = np.zeros((n, n), dtype=np.float64)
    e0 = 0.0
    for i, j, k in c.and_clauses:
        e0 += _add_and_gadget(h, J, i, j, k)
    for i, j, k, a in c.xor_clauses:
        e0 += _add_xor_gadget(h, J, i, j, k, a)

    # Symmetrise J so the substitution loop below is simpler.
    J = 0.5 * (J + J.T)

    # ----- Substitute pinned spins (fold s_p = ±1 into h, e_0). -----------
    pinned = c.pinned
    is_pinned = np.zeros(n, dtype=bool)
    s_pin = np.zeros(n, dtype=np.float64)
    for idx, b in pinned.items():
        is_pinned[idx] = True
        s_pin[idx] = 2.0 * b - 1.0  # 0 → −1, 1 → +1

    # h[p] s_p and J[p, q] s_p s_q rules:
    #   for each pinned p: e_0 += h[p] * s_p; for each j: h[j] += 2 J[p, j] s_p
    #   for pinned p AND pinned q (p < q): e_0 += 2 J[p, q] s_p s_q
    pinned_idx = np.where(is_pinned)[0]
    for p in pinned_idx:
        e0 += float(h[p] * s_pin[p])
    for p in pinned_idx:
        for j in range(n):
            if j == p or is_pinned[j]:
                continue
            h[j] += 2.0 * J[p, j] * s_pin[p]
    # Pin–pin couplings (counted once via the upper triangle).
    for ii in range(len(pinned_idx)):
        for jj in range(ii + 1, len(pinned_idx)):
            p, q = pinned_idx[ii], pinned_idx[jj]
            e0 += 2.0 * J[p, q] * s_pin[p] * s_pin[q]

    free = [i for i in range(n) if not is_pinned[i]]
    h_free = h[free]
    J_free = J[np.ix_(free, free)]

    # ----- Convert (h, J) on free spins → Boolean QUBO -----------------------
    # Storage convention: J_free is symmetric with zero diagonal.  We
    # treat the Ising Hamiltonian as the bilinear form
    #     H_Ising(s) = e_0 + h^T s + s^T J_sym s,
    # where the full sum s^T J_sym s = Σ_{i,j} J_sym[i,j] s_i s_j double-
    # counts each unordered pair (because J_sym[i,j] = J_sym[j,i]).  Each
    # add_*_gadget call deposits the *bare* coupling once into the upper
    # triangle (e.g. J[i,j] += 1.0 for the AND term s_i s_j); the
    # `0.5 * (J + J.T)` symmetrisation then arranges that the bilinear
    # sum reproduces the original gadget energy.  Substituting s = 2 x − 1
    # and using J_sym.diag() = 0:
    #   h^T s              = 2 h^T x − Σ h
    #   s^T J_sym s        = 4 x^T J_sym x − 4 (J_sym · 1)^T x + Σ J_sym
    # ⇒  Q  = 4 J_sym + diag(2 h − 4 J_sym · 1)
    #    e₀_QUBO = e_0 − Σ h + Σ J_sym
    nf = len(free)
    Q = 4.0 * J_free.copy()
    np.fill_diagonal(Q, 0.0)  # off-diagonal block of Q
    row_sum = J_free.sum(axis=1)  # (nf,) — note: J_free.diag() == 0
    diag_lin = 2.0 * h_free - 4.0 * row_sum
    for i in range(nf):
        Q[i, i] = float(diag_lin[i])

    offset = float(e0 - h_free.sum() + J_free.sum())
    return Q, offset, free


# --------------------------------------------------------------------------- #
# Public problem class                                                        #
# --------------------------------------------------------------------------- #


class IntegerFactorizationIsing(QUBOProblem):
    """Planted-solution factorization Ising/QUBO (Hen 2026).

    Parameters
    ----------
    p, q : int
        Two primes whose product ``N = p * q`` is the planted semi-prime.
        The class itself does *not* check primality; pass any positive
        integers to fuzz the construction. Bit-lengths must be ≥ 2.
    device : str | torch.device
        Where to place the QUBO ``Q_mat`` tensor.

    Notes
    -----
    The QUBO loss ``x^T Q x + offset`` evaluates to **zero** on the
    planted Boolean assignment (which encodes ``(p, q)``) and to a
    strictly positive integer on every other binary configuration
    (gap ≥ 2, paper Sec. II.C). Decoded factors are recovered with
    :meth:`decode_factors`.

    Examples
    --------
    >>> import qqa
    >>> prob = qqa.IntegerFactorizationIsing(p=11, q=13)  # N = 143
    >>> prob.num_nodes  # number of free Boolean variables
    ...
    """

    def __init__(self, p: int, q: int, *, device: str | torch.device = "cpu"):
        super().__init__()
        if p < 2 or q < 2:
            raise ValueError("p and q must be ≥ 2.")
        self.p = int(p)
        self.q = int(q)
        self.N = self.p * self.q
        self.n_p = max(2, self.p.bit_length())
        self.n_q = max(2, self.q.bit_length())
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        compilation = _compile(self.p, self.q)
        Q_np, offset, free_indices = _build_qubo_from_compilation(compilation)
        self._compilation = compilation
        self.free_indices = free_indices
        self.offset = float(offset)
        self.num_nodes = len(free_indices)
        self.num_and = len(compilation.and_clauses)
        self.num_xor = len(compilation.xor_clauses)
        self.Q_mat = torch.from_numpy(Q_np).to(dtype=torch.float32, device=self.device)
        self.relaxation = BinaryRelaxation()

        # Cache the planted (free-variable) Boolean assignment for tests.
        self.planted_x = torch.tensor(
            [compilation.planted[i] for i in free_indices],
            dtype=torch.float32,
            device=self.device,
        )
        # Pinned spin values, indexed by the *original* spin id.
        self._pinned = dict(compilation.pinned)
        self._planted_full = list(compilation.planted)
        self._p_bits = list(compilation.p_bits)
        self._q_bits = list(compilation.q_bits)

    # ------------------------------------------------------------------ #
    # COProblem interface                                                #
    # ------------------------------------------------------------------ #

    def generate_qubo_matrix(self) -> torch.Tensor:
        """Return the cached QUBO ``Q_mat`` (built once in ``__init__``).

        The factorization QUBO is *not* derived from a single graph the
        way MIS / MaxCut are, so we precompute ``Q_mat`` from the AND/XOR
        gadgets in the constructor and simply expose it here.
        """
        return self.Q_mat

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        """``x^T Q x + offset`` for ``x`` of shape ``(B, N)``."""
        quad = torch.einsum("bi,ij,bj->b", x, self.Q_mat, x)
        return quad + self.offset

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        """Decode ``(p̂, q̂)``, report planted-distance / energy / `N̂`."""
        x = x_disc if x_disc.ndim == 2 else x_disc.unsqueeze(0)
        with torch.no_grad():
            xb = x.float().round().clamp_(0.0, 1.0)
            energies = torch.einsum("bi,ij,bj->b", xb, self.Q_mat, xb) + self.offset
            hamming = (xb != self.planted_x).sum(dim=-1)

        # Pick the lowest-energy replica.
        idx = int(torch.argmin(energies).item())
        chosen = xb[idx].cpu().numpy().astype(np.int64)

        # Reconstruct full spin vector by interleaving free and pinned spins.
        full = np.zeros(self._compilation.n_vars, dtype=np.int64)
        for orig, b in self._pinned.items():
            full[orig] = b
        for new_i, orig in enumerate(self.free_indices):
            full[orig] = int(chosen[new_i])
        p_hat = sum(int(full[self._p_bits[i]]) << i for i in range(self.n_p))
        q_hat = sum(int(full[self._q_bits[j]]) << j for j in range(self.n_q))
        n_hat = p_hat * q_hat

        feasible = bool(energies[idx].item() <= 1e-6) and (n_hat == self.N)
        return {
            "label": "factorisation gap",
            "value": float(energies[idx].item()),
            "unit": "energy above planted",
            "feasible": feasible,
            "extra": {
                "p": int(self.p),
                "q": int(self.q),
                "N": int(self.N),
                "p_hat": int(p_hat),
                "q_hat": int(q_hat),
                "N_hat": int(n_hat),
                "matches_planted": bool(n_hat == self.N),
                "hamming_to_planted": int(hamming[idx].item()),
                "num_and_clauses": int(self.num_and),
                "num_xor_clauses": int(self.num_xor),
                "num_free_spins": int(self.num_nodes),
                "n_p": int(self.n_p),
                "n_q": int(self.n_q),
            },
        }

    # ------------------------------------------------------------------ #
    # Convenience                                                        #
    # ------------------------------------------------------------------ #

    def decode_factors(self, x: torch.Tensor) -> tuple[int, int]:
        """Decode ``(p̂, q̂)`` from a single Boolean configuration."""
        flat = x.view(-1).round().clamp_(0.0, 1.0).long().cpu().numpy()
        if flat.size != self.num_nodes:
            raise ValueError(f"decode_factors expected length {self.num_nodes}, got {flat.size}")
        full = np.zeros(self._compilation.n_vars, dtype=np.int64)
        for orig, b in self._pinned.items():
            full[orig] = b
        for new_i, orig in enumerate(self.free_indices):
            full[orig] = int(flat[new_i])
        p_hat = sum(int(full[self._p_bits[i]]) << i for i in range(self.n_p))
        q_hat = sum(int(full[self._q_bits[j]]) << j for j in range(self.n_q))
        return p_hat, q_hat

    @classmethod
    def from_random_semiprime(
        cls,
        bit_length: int,
        *,
        seed: int | None = None,
        distinct: bool = True,
        device: str | torch.device = "cpu",
    ) -> IntegerFactorizationIsing:
        """Sample ``(p, q)`` of equal bit-length and build the QUBO."""
        p, q = random_semiprime(bit_length, seed=seed, distinct=distinct)
        return cls(p, q, device=device)


def random_factorization_problems(
    bit_length: int,
    num_instances: int,
    *,
    seed: int | None = 0,
    distinct: bool = True,
    device: str | torch.device = "cpu",
) -> list[IntegerFactorizationIsing]:
    """Build a benchmark suite of ``num_instances`` factorization problems.

    Each instance uses a different random prime pair of the requested
    bit-length. The list is suitable for sequential benchmarking via
    ``scripts/bench_factorization.py``; batched-instance solving is not
    yet supported because instances of even the same bit-length have
    different free-variable counts.
    """
    if num_instances <= 0:
        raise ValueError("num_instances must be ≥ 1.")
    rng_seed = 0 if seed is None else int(seed)
    out: list[IntegerFactorizationIsing] = []
    for k in range(num_instances):
        p, q = random_semiprime(bit_length, seed=rng_seed + k, distinct=distinct)
        out.append(IntegerFactorizationIsing(p, q, device=device))
    return out


__all__ = [
    "IntegerFactorizationIsing",
    "random_semiprime",
    "random_prime",
    "random_factorization_problems",
]
