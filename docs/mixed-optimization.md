# Mixed binary, integer, and real optimisation

QQA's typed mixed-domain API solves bounded nonlinear models containing any
combination of:

- `Binary(name, size)` for values in `{0, 1}`;
- `Integer(name, lower, upper, size)` for bounded integers;
- `Real(name, lower, upper, size)` for bounded reals.

All variables use a common normalised latent coordinate internally. Binary
variables receive the standard QQA discrete penalty, integer variables receive
a periodic grid penalty whose minima are the integer points, and real variables
remain continuous. The objective and constraints are evaluated in the physical
units declared by the user.

## Minimal real-valued problem

User functions must preserve the leading population dimension:

```python
import qqa

problem = qqa.MixedProblem(
    [qqa.Real("x", -5, 5), qqa.Real("y", -5, 5)],
    lambda v: (v["x"] - 1.25).square() + (v["y"] + 2.5).square(),
    name="convex-real",
)
result = problem.solve(num_epochs=300, verbose=False)
print(problem.unpack(result.best_sol))
```

## Pure integer problem

```python
problem = qqa.MixedProblem(
    [qqa.Integer("quantity", lower=-10, upper=10)],
    lambda v: (v["quantity"] - 3).square(),
)
result = problem.solve(verbose=False)
assert result.best_sol.item() == 3
```

## Constraints

Each `Constraint` has a direction, right-hand side, weight, unit scale, and
reporting tolerance:

```python
qqa.Constraint(
    lambda v: v["production"].sum(-1),
    sense=">=",
    rhs=100,
    weight=250,
    scale=100,
    tolerance=1e-3,
    name="demand",
)
```

The optimisation loss adds
`weight * (raw_violation / scale) ** 2`. Choose `scale` near the typical
magnitude of the constraint so weights remain interpretable. `result.best_obj`
is this penalised selection loss; `result.score["value"]` is the original
unpenalised objective and `result.score["feasible"]` reports whether every
constraint meets its tolerance.

`problem.solve()` also calibrates one global penalty multiplier from Sobol
samples so a large currency-valued objective cannot silently overwhelm a
small normalised constraint. Explicit weights still express the relative
importance of the constraint rows. For exact mixed-integer/nonlinear
refinement of a safe `ModelSpec`, install `qqa[scip]` and use
`qqa.hybrid.solve_spec_scip`.

## Warm starts in physical units

Use `problem.pack()` to create a validated solution and pass it directly:

```python
seed = problem.pack({"quantity": 3})
result = qqa.anneal(problem, initial_state=seed, verbose=False)
```

The mixed relaxation automatically encodes the physical values into its
normalised latent space.

## Large integer bounds and numerical precision

Models use `torch.float32` by default. If integer magnitudes exceed float32's
exact-integer range, request float64 explicitly so packing, projection, and
objective evaluation preserve every grid point:

```python
import torch

problem = qqa.MixedProblem(
    [qqa.Integer("identifier", 100_000_000, 100_000_010)],
    lambda v: (v["identifier"] - 100_000_003).square(),
    dtype=torch.float64,
)
```

Float64 is slower on many consumer GPUs, so use it only when the domain
requires the additional precision.

## Diagnostics and portable reports

```python
from qqa import visualization as viz

viz.plot_result_dashboard(result, problem, backend="plotly")
viz.plot_variable_solution(result, problem, backend="plotly")
viz.plot_constraint_diagnostics(result, problem, backend="plotly")
qqa.save_html_report(result, problem, "report.html")
```

The HTML report embeds Plotly and a JSON result payload. It opens offline and
can be archived or shared as a single file.

## GPU execution

Objectives and constraints should use PyTorch operations and reduce only over
variable axes (normally `dim=-1`). QQA then evaluates all replicas in one
vectorised CUDA graph:

```python
result = problem.solve(
    sol_size=4096,
    num_epochs=2000,
    mixed_precision="bf16",
    device="auto",
    verbose=False,
)
```

The mixed solver defaults to gradient clipping and adaptive basin recovery.
Recovery always keeps the incumbent and divides weak replicas between global
reinitialisation and local incumbent-centred jitter. The result records
`diagnostics["restart_events"]` and `history["restart_epochs"]`. Pass
`restart_patience=None` when reproducing the original uninterrupted dynamics.

The dedicated
[`09_mixed_integer_real_optimization.ipynb`](https://colab.research.google.com/github/Yuma-Ichikawa/QQA4CO/blob/main/examples/09_mixed_integer_real_optimization.ipynb)
notebook includes real, integer, and factory-planning examples plus brute-force
verification of the mixed optimum.
