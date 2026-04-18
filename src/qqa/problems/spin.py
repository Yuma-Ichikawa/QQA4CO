"""Physics-flavoured spin problems for QQA.

All classes in this module use :class:`~qqa.relaxation.SpinRelaxation`, so the
``x`` tensor fed in during annealing lives in ``[0, 1]`` while
``problem.loss_fn`` sees the transformed spin ``s = 2x - 1 \\in [-1, +1]``
(and exactly ``\\pm 1`` after rounding).

Energies follow physics conventions (lower is better):

* Ising 1D: ``E = -sum_<i,j> J_{ij} s_i s_j - h sum_i s_i``
* Edwards-Anderson / SK / Hopfield: ``E = -0.5 s^T J s`` with symmetric ``J``
  and ``diag(J) = 0`` (so the full sum equals ``-sum_<i,j> J_{ij} s_i s_j``).
* Binary perceptron: a smooth surrogate for the number of mis-classified
  teacher-student patterns.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from qqa.problems.base import COProblem
from qqa.relaxation import SpinRelaxation


class SpinProblem(COProblem):
    """Base class for ``\\pm 1`` spin problems.

    Subclasses must populate ``self.num_spins`` and attach a
    :class:`SpinRelaxation`. Most subclasses also build a symmetric coupling
    matrix ``self.J`` and rely on :meth:`quadratic_energy`.
    """

    num_spins: int
    J: torch.Tensor | None
    h: torch.Tensor | None
    device: str | torch.device

    @property
    def num_nodes(self) -> int:  # compat with binary/categorical conventions
        return self.num_spins

    def quadratic_energy(self, s: torch.Tensor) -> torch.Tensor:
        """Compute ``E = -0.5 s^T J s - h . s`` for a batch of spin configs.

        ``s`` has shape ``(B, N)``; returns a 1D tensor of shape ``(B,)``.
        """
        e = -0.5 * torch.einsum("bi,ij,bj->b", s, self.J, s)
        if self.h is not None:
            e = e - torch.einsum("bi,i->b", s, self.h)
        return e

    def loss_fn(self, s: torch.Tensor) -> torch.Tensor:
        return self.quadratic_energy(s)

    def score_summary(self, s_disc: torch.Tensor) -> dict:
        s = s_disc if s_disc.ndim == 2 else s_disc.unsqueeze(0)
        with torch.no_grad():
            e = self.loss_fn(s.float())
        idx = int(torch.argmin(e).item())
        e_best = float(e[idx].item())
        return {
            "label": "energy / spin",
            "value": e_best / self.num_spins,
            "unit": "",
            "feasible": True,
            "extra": {"total_energy": e_best, "N": self.num_spins},
        }


# ---------------------------------------------------------------------------
# 1D Ising model
# ---------------------------------------------------------------------------


class Ising1D(SpinProblem):
    """One-dimensional Ising chain with nearest-neighbour coupling ``J``.

    Energy: ``E = -J sum_i s_i s_{i+1} - h sum_i s_i``.

    For ``J > 0`` and ``h = 0`` with periodic boundaries, the ground state is
    all spins aligned with energy ``-J * N``.

    Args:
        N: Number of spins.
        J: Uniform nearest-neighbour coupling strength.
        h: Uniform external field.
        periodic: Whether to close the chain (``s_N = s_0``).
    """

    def __init__(
        self,
        N: int,
        J: float = 1.0,
        h: float = 0.0,
        periodic: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_spins = int(N)
        self.device = device
        self.periodic = periodic
        self.J_value = float(J)
        self.h_value = float(h)

        J_mat = torch.zeros((N, N))
        for i in range(N - 1):
            J_mat[i, i + 1] = J
            J_mat[i + 1, i] = J
        if periodic and N > 1:
            J_mat[0, N - 1] = J
            J_mat[N - 1, 0] = J
        self.J = J_mat.to(device)
        self.h = torch.full((N,), float(h), device=device) if h != 0.0 else None
        self.relaxation = SpinRelaxation()


# ---------------------------------------------------------------------------
# Edwards-Anderson (D-dimensional, nearest-neighbour, random J)
# ---------------------------------------------------------------------------


def _lattice_neighbors(shape: tuple[int, ...], periodic: bool):
    """Yield pairs of flat indices ``(i, j)`` for nearest-neighbour bonds of a
    hyper-cubic lattice with the given ``shape``. Each undirected bond is
    yielded exactly once.
    """
    dims = len(shape)
    strides: list[int] = [1] * dims
    for d in range(dims - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]

    def flat(idx: tuple[int, ...]) -> int:
        return sum(i * s for i, s in zip(idx, strides, strict=True))

    ranges = [range(s) for s in shape]
    from itertools import product

    for idx in product(*ranges):
        for d in range(dims):
            nxt = list(idx)
            if idx[d] + 1 < shape[d]:
                nxt[d] = idx[d] + 1
                yield flat(idx), flat(tuple(nxt))
            elif periodic and shape[d] > 2:
                nxt[d] = 0
                yield flat(idx), flat(tuple(nxt))


class EdwardsAnderson(SpinProblem):
    """Edwards-Anderson spin-glass on a hyper-cubic lattice.

    Only nearest-neighbour bonds are coupled, with ``J_{ij} \\sim N(0, \\sigma^2)``
    drawn once at construction time. The energy is
    ``E = -0.5 s^T J s`` with symmetric ``J``.

    Args:
        L: Lattice side length (``N = L ** dim`` spins).
        dim: Spatial dimension (``2`` or ``3``). Default ``3`` matches the
            classical 3D-EA benchmark.
        seed: RNG seed for the couplings.
        periodic: Whether to use periodic boundary conditions.
        sigma: Standard deviation of the Gaussian couplings.
    """

    def __init__(
        self,
        L: int,
        dim: int = 3,
        seed: int = 0,
        periodic: bool = True,
        sigma: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.L = int(L)
        self.dim = int(dim)
        self.periodic = periodic
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.device = device

        N = L**dim
        self.num_spins = N
        shape = (L,) * dim

        rng = np.random.default_rng(seed)
        J_mat = torch.zeros((N, N))
        for i, j in _lattice_neighbors(shape, periodic):
            val = float(rng.normal(0.0, sigma))
            J_mat[i, j] = val
            J_mat[j, i] = val
        self.J = J_mat.to(device)
        self.h = None
        self.relaxation = SpinRelaxation()

    @classmethod
    def from_couplings_txt(
        cls,
        path: str | Path,
        N: int,
        device: str | torch.device = "cpu",
    ) -> EdwardsAnderson:
        """Load an EA instance from a text file of ``i j J_ij`` rows.

        Compatible with the ``couplings_L{L}_R1_seed{seed}.txt`` format
        produced by related projects: rows of ``i j J_ij`` with 0-based
        indices. No metadata is assumed; ``N`` must be provided.
        """
        obj = cls.__new__(cls)
        super(EdwardsAnderson, obj).__init__()
        obj.num_spins = int(N)
        obj.device = device
        obj.sigma = float("nan")
        obj.seed = -1
        obj.L = int(round(N ** (1 / 3)))
        obj.dim = 3
        obj.periodic = True

        J_mat = torch.zeros((N, N))
        data = np.loadtxt(str(path))
        if data.ndim == 1:
            data = data[None, :]
        for row in data:
            i = int(row[0])
            j = int(row[1])
            val = float(row[2])
            J_mat[i, j] = val
            J_mat[j, i] = val
        obj.J = J_mat.to(device)
        obj.h = None
        obj.relaxation = SpinRelaxation()
        return obj


# ---------------------------------------------------------------------------
# Sherrington-Kirkpatrick (mean-field spin glass)
# ---------------------------------------------------------------------------


class SherringtonKirkpatrick(SpinProblem):
    """Sherrington-Kirkpatrick mean-field spin glass.

    All-to-all couplings with ``J_{ij} \\sim N(0, 1/N)`` for ``i \\ne j`` and
    ``J_{ii} = 0``. Energy: ``E = -0.5 s^T J s``.

    The standard normalisation ``J_{ij} \\sim N(0, 1/N)`` makes the typical
    ground-state energy density ``e_0 = E_0 / N`` converge to
    ``\\approx -0.7632`` (Parisi).
    """

    def __init__(
        self,
        N: int,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_spins = int(N)
        self.seed = int(seed)
        self.device = device

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(N)
        J_upper = rng.normal(0.0, scale, size=(N, N)).astype(np.float32)
        J_sym = np.triu(J_upper, k=1)
        J_sym = J_sym + J_sym.T
        self.J = torch.from_numpy(J_sym).to(device)
        self.h = None
        self.relaxation = SpinRelaxation()


# ---------------------------------------------------------------------------
# Binary (discrete) perceptron — teacher/student learning problem
# ---------------------------------------------------------------------------


class BinaryPerceptron(SpinProblem):
    """Teacher-student binary perceptron in the storage formulation.

    Patterns ``\\xi^\\mu \\in \\{-1, +1\\}^N`` are drawn uniformly, and a
    teacher ``s^* \\in \\{-1, +1\\}^N`` generates labels
    ``\\sigma^\\mu = sign(\\frac{1}{\\sqrt N} \\xi^\\mu \\cdot s^*)``.

    The learning loss is the number of patterns on which the student ``s``
    disagrees with the teacher. During annealing this is replaced by a
    smooth surrogate ``\\sum_\\mu \\sigma(-k z^\\mu)`` where
    ``z^\\mu = \\sigma^\\mu \\frac{1}{\\sqrt N} \\xi^\\mu \\cdot s`` and
    ``\\sigma`` is the logistic sigmoid, so gradients are available. After
    rounding, :meth:`error_count` returns the exact number of errors.

    Args:
        N: Input dimension.
        alpha: Loading ``M / N`` (so ``M = round(alpha * N)`` patterns).
        seed: RNG seed for patterns and teacher.
        sharpness: Sigmoid steepness ``k``. Larger ``k`` approaches the step
            loss but is harder to optimise.
    """

    def __init__(
        self,
        N: int,
        alpha: float = 0.5,
        seed: int = 0,
        sharpness: float = 10.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_spins = int(N)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.sharpness = float(sharpness)
        self.device = device

        M = max(1, int(round(alpha * N)))
        self.num_patterns = M
        rng = np.random.default_rng(seed)
        xi = rng.choice([-1.0, 1.0], size=(M, N)).astype(np.float32)
        s_teacher = rng.choice([-1.0, 1.0], size=(N,)).astype(np.float32)
        labels = np.sign(xi @ s_teacher)
        labels[labels == 0] = 1.0

        # Combine xi and sigma into a signed pattern so the surrogate is
        # simply sigmoid(-k * (xi_signed @ s) / sqrt(N)).
        xi_signed = (xi * labels[:, None]).astype(np.float32)

        self.xi = torch.from_numpy(xi).to(device)
        self.teacher = torch.from_numpy(s_teacher).to(device)
        self.labels = torch.from_numpy(labels).to(device)
        self.xi_signed = torch.from_numpy(xi_signed).to(device)

        # No J/h — loss is defined directly.
        self.J = None
        self.h = None
        self.relaxation = SpinRelaxation()

    def _field(self, s: torch.Tensor) -> torch.Tensor:
        """``z^\\mu = (\\xi_signed @ s) / sqrt(N)``, shape ``(B, M)``."""
        return torch.einsum("mi,bi->bm", self.xi_signed, s) / (self.num_spins**0.5)

    def loss_fn(self, s: torch.Tensor) -> torch.Tensor:
        z = self._field(s)
        return torch.sigmoid(-self.sharpness * z).sum(dim=1)

    def error_count(self, s: torch.Tensor) -> torch.Tensor:
        """Exact number of mis-classified patterns for each config in ``s``.

        A pattern is counted as an error when the pre-activation ``z`` is
        strictly negative; exact ties (``z == 0``) occur only for the rare
        case where the teacher inner product vanishes and are treated as
        correct.
        """
        with torch.no_grad():
            z = self._field(s)
            return (z < 0).sum(dim=1)

    def score_summary(self, s_disc: torch.Tensor) -> dict:
        s = s_disc if s_disc.ndim == 2 else s_disc.unsqueeze(0)
        with torch.no_grad():
            errs = self.error_count(s.float())
        idx = int(torch.argmin(errs).item())
        err = int(errs[idx].item())
        return {
            "label": "patterns classified",
            "value": self.num_patterns - err,
            "unit": f"/ {self.num_patterns}",
            "feasible": err == 0,
            "extra": {"alpha": self.alpha, "N": self.num_spins},
        }


# ---------------------------------------------------------------------------
# Hopfield associative memory
# ---------------------------------------------------------------------------


class HopfieldMemory(SpinProblem):
    """Hopfield associative-memory model with Hebbian couplings.

    Given ``P`` patterns ``\\xi^\\mu \\in \\{-1, +1\\}^N``, the couplings are
    ``J_{ij} = (1/N) \\sum_\\mu \\xi^\\mu_i \\xi^\\mu_j`` for ``i \\ne j`` and
    ``J_{ii} = 0``. Energy: ``E = -0.5 s^T J s``.

    At a stored pattern ``s = \\xi^\\mu`` and low loading ``\\alpha = P/N``,
    ``E \\approx -N/2``.

    Args:
        N: Number of spins.
        patterns: Either an integer ``P`` (sample ``P`` random ``\\pm 1``
            patterns) or a pre-computed tensor/array of shape ``(P, N)``.
        seed: RNG seed used when ``patterns`` is an integer.
    """

    def __init__(
        self,
        N: int,
        patterns: int | np.ndarray | torch.Tensor | Sequence[Sequence[int]] = 1,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_spins = int(N)
        self.seed = int(seed)
        self.device = device

        if isinstance(patterns, int):
            rng = np.random.default_rng(seed)
            xi = rng.choice([-1.0, 1.0], size=(patterns, N)).astype(np.float32)
        else:
            xi = np.asarray(patterns, dtype=np.float32)
            if xi.ndim != 2 or xi.shape[1] != N:
                raise ValueError(f"patterns must have shape (P, {N}); got {xi.shape}")
        self.patterns = torch.from_numpy(xi).to(device)
        self.num_patterns = xi.shape[0]

        J_np = (xi.T @ xi) / N
        np.fill_diagonal(J_np, 0.0)
        self.J = torch.from_numpy(J_np).to(device)
        self.h = None
        self.relaxation = SpinRelaxation()

    def overlap(self, s: torch.Tensor) -> torch.Tensor:
        """Normalised overlap with every stored pattern.

        Returns a tensor of shape ``(B, P)`` with
        ``m^\\mu_b = (1/N) \\sum_i \\xi^\\mu_i s_{b,i}``.
        """
        return torch.einsum("pi,bi->bp", self.patterns, s) / self.num_spins

    def score_summary(self, s_disc: torch.Tensor) -> dict:
        s = s_disc if s_disc.ndim == 2 else s_disc.unsqueeze(0)
        with torch.no_grad():
            e = self.loss_fn(s.float())
            ov = self.overlap(s.float()).abs().max(dim=1).values
        idx = int(torch.argmin(e).item())
        return {
            "label": "max |overlap|",
            "value": float(ov[idx].item()),
            "unit": "",
            "feasible": True,
            "extra": {
                "energy_per_spin": float(e[idx].item()) / self.num_spins,
                "patterns_stored": self.num_patterns,
            },
        }
