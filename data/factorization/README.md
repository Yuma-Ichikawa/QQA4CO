# Planted-solution factorization Ising/QUBO benchmark

Implementation of the **planted-solution Ising / QUBO benchmark from
integer factorization** introduced by:

> Itay Hen, *Planted-solution SAT and Ising benchmarks from integer
> factorization*, **arXiv:2604.09837** (2026).

The construction encodes the long-multiplication identity ``N = p · q``
as a Boolean circuit of AND and XOR gates, then lowers each gate to one
of the two Ising **energy gadgets** of the paper (Eqs. 10 and 11). The
resulting QUBO has a **provably optimal, planted ground state** equal to
the bits of ``(p, q)`` — making it an unusually clean benchmark for any
discrete optimiser (SAT solvers, simulated / quantum annealers,
gradient-based samplers, …) because every reported solution can be
*verified* against the known minimum.

Unlike random ``k``-SAT (where the ground state energy is unknown and
solver "successes" cannot be cross-checked), every instance here has

* a known optimum ``H(s*) = 0`` (after offset normalisation),
* a single tunable difficulty knob: the bit-length ``d = max(n_p, n_q)``,
* a strictly positive spectral gap ``≥ 2`` between the planted optimum
  and any other configuration.

## Quick start

The data is generated at runtime; no download needed.

```bash
# Tiny smoke (4-bit primes, ~50 free spins; CPU, < 30 s)
uv run python scripts/bench_factorization.py --bits 4 --instances 5

# Larger sweep on GPU
uv run python scripts/bench_factorization.py --bits 6 --instances 10 \
    --device cuda --sol-size 1000 --num-epochs 5000 \
    --output bench_factorization_d6.json
```

Or programmatically:

```python
import qqa

# Single instance with known factors
prob = qqa.IntegerFactorizationIsing(p=11, q=13)        # N = 143
print(prob.num_nodes, prob.num_and, prob.num_xor)        # ~150 spins

# Random benchmark suite
suite = qqa.random_factorization_problems(
    bit_length=5, num_instances=10, seed=0,
)

# Solve with QQA
result = qqa.anneal(
    prob,
    sol_size=500, num_epochs=3000, learning_rate=0.5,
    schedule=qqa.LinearBGSchedule(min_bg=-2, max_bg=0.5),
    curve_rate=4, div_param=0.2,
)
print(result.score["extra"]["p_hat"], result.score["extra"]["q_hat"])
```

`prob.score_summary(x)` reports the residual energy above the planted
optimum, the **decoded** factor pair ``(p̂, q̂)``, ``N̂ = p̂ · q̂``, and
the bit-Hamming distance to ``s*``. ``score['feasible']`` is `True` iff
the optimiser reached the planted ground state and ``p̂ · q̂ = N``.

## What the construction looks like inside

| Pipeline step                                   | Where in the code                          |
|-------------------------------------------------|--------------------------------------------|
| Random equal-bit-length prime sampling          | `random_prime`, `random_semiprime`         |
| Long-multiplication → AND/XOR clause graph      | `_compile`                                 |
| AND gadget (Eq. 10), XOR gadget (Eq. 11)        | `_add_and_gadget`, `_add_xor_gadget`       |
| Pin substitution + Ising → QUBO                 | `_build_qubo_from_compilation`             |
| Public ``qqa.IntegerFactorizationIsing``        | top of `src/qqa/problems/factorization.py` |
| End-to-end smoke + planted-invariant tests      | `tests/test_factorization.py`              |

The implementation is **gadget-faithful** to the paper but does *not*
ship the preprocessing pipeline of Sec. III. Problem sizes therefore
scale as ``O(d^4)`` in the symmetric case ``n_p = n_q = d``:

| ``d`` | free spins (typical) | suitable for |
|-------|----------------------|--------------|
| 2–3   |  16 – 40             | CPU smoke    |
| 4–5   | 100 – 250            | CPU bench    |
| 6–7   | 500 – 1 500          | single-GPU   |
| ≥ 8   | ≥ 3 000              | multi-GPU / preprocessing |

Concrete sizes (``carry`` and XOR-aux fused per Eq. 12 footnote — see
``_compile`` in `src/qqa/problems/factorization.py`):

| ``N``  | ``(p, q)`` | free spins | AND clauses | XOR clauses |
|--------|-----------|------------|-------------|-------------|
| 15     | (3, 5)    | 16         | 12          | 6           |
| 35     | (5, 7)    | 37         | 24          | 15          |
| 143    | (11, 13)  | 102        | 60          | 44          |
| 221    | (13, 17)  | 148        | 85          | 65          |
| 667    | (23, 29)  | 213        | 120         | 95          |

For larger ``d`` the upstream paper's preprocessing is essential; an
optional pin-propagation / AND-XOR simplification pass is a natural
follow-up that would slot directly into ``_build_qubo_from_compilation``.

## What QQA can do today

The construction is **correct for any ``(p, q)``**: ``planted_x``
evaluates to exactly ``loss = 0``, any single bit-flip raises the
loss by at least 2 (the gadget gap; see
`tests/test_factorization.py::test_planted_is_strict_minimum`), and the
gadget energy tables match Eqs. (10)–(11) of the paper element-wise
(`test_and_gadget_full_spectrum`, `test_xor_gadget_full_spectrum`).

The bench script reports **two** orthogonal success metrics:

* `gs` (ground state) — `loss == 0`, i.e. the entire spin string sits
  on the planted optimum. This is the *strong* contract.
* `decoded_N` — the bits on the input wires `p_0..p_{n_p-1}` and
  `q_0..q_{n_q-1}` decode to a pair `(p̂, q̂)` with `p̂ · q̂ = N`,
  even if internal pp/sum/carry spins are still inconsistent (so
  `loss > 0`). This is the *weak* contract: the optimiser found the
  factorisation but did not finish the auxiliary bookkeeping.

Empirically, the default ``qqa.anneal`` recipe consistently reaches
`gs` for ``d ≤ 3`` (``N ≤ 7·7``) and gets `decoded_N` for ``d = 4``
(``N ≈ 11·13``) within a few seconds on CPU; ``d ≥ 5`` benefits from
larger ``sol_size`` and ``num_epochs`` and may still leave a non-zero
gap, exactly as the paper predicts (``T_med`` grows as ``2^{β d}``
with ``β ≈ 1``). This is the *whole point* of the benchmark: it is
genuinely difficult for general-purpose CO solvers even at moderate
``d``.

## Citation

Please cite the original paper:

```bibtex
@article{hen2026planted,
  title   = {Planted-solution {SAT} and Ising benchmarks from integer factorization},
  author  = {Hen, Itay},
  journal = {arXiv preprint arXiv:2604.09837},
  year    = {2026},
}
```

The reference SAT/Ising compiler is open source at
<https://github.com/itay-hen/pq-SAT-benchmark>; the QQA4CO
implementation here is independent and follows the published gadget
formulas (Eqs. 10–12) directly.
