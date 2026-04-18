# Problem catalog

Every problem class exposes:

- `problem.loss_fn(x)` — the (continuous) objective used during annealing.
- `problem.relaxation` — a :class:`qqa.relaxation.Relaxation` describing how
  the variable is represented.

## Combinatorial problems (binary QUBO)

| Class                          | Variables | Loss (lower is better) |
| ------------------------------ | --------- | --------------------- |
| `MaximumIndependentSet(g)`     | $x \in \{0,1\}^N$ | $-\\|S\\| + p \cdot (\\text{violated edges})$ |
| `MaxClique(g)`                 | $x \in \{0,1\}^N$ | $-\\|S\\| + p \cdot (\\text{non-edges in } S)$ |
| `MaxCut(g)`                    | $x \in \{0,1\}^N$ | $x^T Q x$ (standard QUBO) |
| `MaximumIndependentSetInstance(gs, max_node)` | batched | per-instance MIS QUBO |

## Categorical problems (one-hot)

| Class | Variables | Loss |
| ----- | --------- | ---- |
| `BalancedGraphPartition(g, K)` | $x \in \Delta^K$ per node | edge cut + balance penalty |
| `Coloring(g, K)` | $x \in \Delta^K$ per node | $\\sum_{(i,j)\in E}\\sum_c x_{ic}x_{jc}$ |

## Spin-glass and physics problems

All spin problems use :class:`qqa.relaxation.SpinRelaxation`: internally
variables are stored in $[0,1]$ and mapped to $\\pm 1$ via $s = 2x - 1$.

### `Ising1D(N, J, h, periodic=True)`

One-dimensional chain with nearest-neighbour coupling.

$$ E(s) = -J\\sum_i s_i s_{i+1} - h\\sum_i s_i $$

### `EdwardsAnderson(L, dim=3, seed, periodic=True, sigma=1.0)`

Hyper-cubic lattice of side $L$ with Gaussian couplings $J_{ij}\\sim
\\mathcal{N}(0, \\sigma^2)$ on nearest-neighbour bonds only.

$$ E(s) = -\\tfrac{1}{2}\\sum_{i,j} J_{ij} s_i s_j $$

### `SherringtonKirkpatrick(N, seed)`

Mean-field spin glass: all-to-all couplings with $J_{ij}\\sim
\\mathcal{N}(0, 1/N)$ (symmetric, zero diagonal). The Parisi ground-state
energy density is $e_0 \\approx -0.7632$.

### `BinaryPerceptron(N, alpha, seed, sharpness)`

Teacher/student binary perceptron. The loss is a smooth surrogate for the
number of mis-classified patterns; the exact count is available via
`problem.error_count(s)`.

### `HopfieldMemory(N, patterns, seed)`

Hebbian couplings for $P$ random $\\pm 1$ patterns:

$$ J_{ij} = \\tfrac{1}{N}\\sum_\\mu \\xi^\\mu_i \\xi^\\mu_j,\\quad J_{ii}=0. $$

Exposes `problem.overlap(s)` which returns the normalised overlaps with
every stored pattern.
