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
| `MaximumIndependentSetInstance(graphs[, max_node, penalty])` | $(B, I, N)$ batched | per-instance MIS QUBO, mask-zeroed padding |
| `MaxCutInstance(graphs[, max_node])` | batched | per-instance Max-Cut QUBO, mask-zeroed padding |
| `MaxCliqueInstance(graphs[, max_node, penalty])` | batched | per-instance Max-Clique QUBO, mask-zeroed padding |

`*Instance` variants take `graphs: Sequence[nx.Graph]`, auto-derive
`max_node = max(g.number_of_nodes())` (override via the kwarg), and
attach a per-instance ``pad_mask`` that is multiplied into ``x`` inside
both ``loss_fn`` and ``score_summary``. The mask makes padded positions
strictly inert: anything the optimiser writes into ``x[..., n_i:]`` is
squashed to zero before the einsum, so heterogeneous-size batches solve
correctly without leaking padded variables into the reported objective.

`score_summary` for the batched classes returns per-instance arrays
(``np.ndarray`` of length ``num_instance``) of objective values and
feasibility flags. ``qqa.anneal`` exposes them as ``result.score`` so
downstream tools can read the per-instance feasibility / |S| / cut
without recomputing.

`penalty` may be either a scalar (broadcast to all instances) or a
sequence of length ``num_instance`` for heterogeneous penalty tuning.

The batched path is supported by ``qqa.anneal`` only — SA / PA backends
require a single dense ``Q_mat`` and intentionally reject
batched-instance problems.

## Classic CO (binary, problem-specific losses)

| Class | Inputs | Loss / feasibility |
| ----- | ------ | ------------------ |
| `Knapsack(values, weights, capacity)` | tensors of length $N$ | $-\sum v_i x_i + p\cdot\max(0, \sum w_i x_i - C)$ |
| `NumberPartitioning(numbers)` | length-$N$ tensor | $(\sum_i (2x_i-1)\,a_i)^2$ |
| `VertexCover(g)` | graph | $\sum_i x_i + p\cdot (\text{uncovered edges})$ |
| `GraphBisection(g)` | graph | edge cut with equal-size penalty |
| `MaxSAT3(clauses)` | list of length-3 literal triples | smoothed count of unsatisfied clauses |

## Categorical problems (one-hot)

| Class | Variables | Loss |
| ----- | --------- | ---- |
| `BalancedGraphPartition(g, K)` | $x \in \Delta^K$ per node | edge cut + balance penalty |
| `Coloring(g, K)` | $x \in \Delta^K$ per node | $\\sum_{(i,j)\in E}\\sum_c x_{ic}x_{jc}$ |

## Categorical permutation problems

Variables are $N \times N$ double-stochastic soft permutations that collapse
onto a hard permutation after annealing (Sinkhorn-style projection).

| Class | Inputs | Loss |
| ----- | ------ | ---- |
| `TSP(distances)` | $(N, N)$ tensor | tour length |
| `QAP(flows, distances)` | two $(N, N)$ tensors | $\sum_{ij} F_{ij} D_{\pi(i)\pi(j)}$ |
| `NQueens(N)` | integer | attack-count surrogate |

## User-defined problems

| Class / helper | Purpose |
| -------------- | ------- |
| `UserProblem(num_vars, variable_kind, loss_fn)` | wrap an arbitrary `loss_fn(x) -> (B,)` into a `COProblem` with the matching relaxation (`"binary"`, `"spin"`, `"categorical"`, or `"permutation"`) |
| `user_problem_from_source(src, name=...)` | parse a Python snippet that defines `problem = UserProblem(...)` or a `make_problem()` factory |
| `load_problem_from_file(path)` | file-based counterpart used by the CLI via `--problem-file` |

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
