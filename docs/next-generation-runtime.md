# Typed primal–dual runtime

QQA4CO 0.10 extends the typed, proof-aware runtime around QQA. The central
rule is that representation, differentiable primal search, repair, and exact
proof are separate capabilities. A factor being readable as `ModelIR` does not
automatically make it valid for every solver.

## Diagnose, plan, solve

The production entry point accepts a goal and a wall-clock duration. Run the
doctor first when a model comes from another system:

```python
import qqa

report = qqa.doctor(model, replicas=128)
print(report.explain())

result = qqa.solve(
    model,
    goal="feasible",  # best | feasible | prove | diverse | pareto
    budget="30s",  # ms, s, m, and h are accepted
    device="auto",
    seed=0,
)
```

The doctor checks finite QQA bounds, factor capabilities, coefficient dynamic
range, large coefficients, quadratic curvature, presolve contradictions,
decomposition structure, route support, estimated working memory, and an
initial replica/budget recommendation. It does not execute a solver.

`goal="prove"` selects the certification profile. It never turns a heuristic
claim into a proof: an optimal or infeasible status is returned only when the
selected exact backend reports it and the original-space solution passes
feasibility evaluation.

## Strict factor and bound semantics

Use `qqa.model.inspect_capabilities(model_ir)` to inspect every factor. The
machine-readable capability set includes `DIFFERENTIABLE`, `SUBGRADIENT`,
`PROX`, `GPU_KERNEL`, `PROPAGATE`, `SEPARATE`, `REPAIR`, `LOWER_BOUND`,
`EXACT_ENCODE`, and `PROOF_SAFE`.

Pure QQA requires a valid derivative/prox route and finite domains. Missing or
infinite integer/real bounds are never replaced by a guessed box. Native
all-different, table, logical, subtour, and scheduling factors can instead be
routed to CP/SAT/exact propagation when their capability says so. A custom
black-box factor must explicitly declare whether it is differentiable.

Feasibility has three states: `feasible`, `infeasible`, and `unknown`. An empty
constraint report is unknown unless it was explicitly created with
`ConstraintReport.unconstrained()`.

Termination and guarantee strength are independent. For example,
`limit_reached_with_incumbent` can still carry `verified_feasible`, while only
a proven terminal state may carry `exact`. The stable guarantee vocabulary is
`exact`, `certified_bound`, `verified_feasible`, `approximate`, `heuristic`, and
`unknown`. A valid exact-backend termination with no incumbent remains a typed
result—such as `infeasible_proven` or `limit_reached_no_incumbent`—and retains
any certified bound the backend supplied.

## Factor-Split QQA and the solve DAG

`qqa.model.compile_execution_plan(model_ir)` groups factors by concrete
backend registration. CUDA selects a fused backend only where a real lowering
exists; unsupported factors remain eager or fail explicitly. The returned
immutable plan exposes value/gradient and constraint operations without
changing the immutable `ModelIR`.

`qqa.plan(...)` now exposes a budgeted stage DAG. `qqa-primal` is always the
population primal-search stage. LP relaxation, repair/LNS, and exact
certification depend on it and are labelled as warm-state, feasibility, bound,
or proof roles. Exact budget allocation adapts to model scale and structure
rather than using one fixed percentage.

## Scale-stable heterogeneous QQA

The default stable profiles use robust median/MAD objective scaling. Replica
objectives are averaged; binary penalties are normalized by the relaxed
dimension; and diversity is dimension-normalized. Raw original-space
objectives remain separate in `SolveResult`.

The population contains convexification, exploration, discretization, noisy,
incumbent, LP-centred, conflict-avoiding, and global roles. Roles receive
different beta and learning-rate schedules. Coarse parallel-tempering exchange
swaps latent and optimizer state together. For sparse QUBOs, a Gershgorin
curvature bound sets the negative c=2 convexification strength and a
factor-degree diagonal preconditioner rescales gradients. Negative beta adds
curvature; it does not imply global convexity for an arbitrary objective.

The historical archive retains feasible and violation-ranked candidates while
enforcing solution diversity. It supplies archive-centred restarts and remains
available on the result for relinking, RINS-like neighborhoods, uncertainty
analysis, and a solution pool. Constraint multipliers use residual-balanced
Powell–Hestenes–Rockafellar updates rather than unbounded fixed penalty growth.

## GPU factor runtime

`qqa.gpu.compile_factor_graph` compiles supported linear, quadratic,
cardinality, and clause factors into device-resident parameter buffers, factor
offsets, factor-to-variable indices, type IDs, and constraint metadata.
Portable Torch kernels provide segmented sparse reductions, exact-k and
one-hot projection, batched assignment repair, exact binary flip deltas,
bounded GPU tabu/k-flip search, and batched 2-opt.

The sparse-QUBO core retains its fused custom operation, optional Triton
kernel, CUDA Graph replay, and AOTInductor cache. `KernelAutotuner` performs a
bounded first-use comparison. `profile_kernel` uses CUDA events and explicit
event synchronization, while CPU timing uses a monotonic wall clock. Callback
telemetry is buffered on the active device and transferred at solve end.

Distributed islands exchange only bounded elites. Replica, model, and scenario
partitioning remain distinct policies; QQA4CO does not all-gather the full
population as an implicit default.

Island deadlines include migration overhead. Elites are objective-ranked and
diversity-filtered, the historical incumbent survives all rounds, no migration
occurs after the final round, and the returned final population is always one
that was actually evaluated.

Integral algebraic columns are never scaled. Row scaling and continuous-column
scaling remain available, and `ScalingFactors.preserves_integrality` makes the
invariant machine-readable.

## Primal, dual, and exact feedback

For sparse linear algebraic models, `qqa.dual.solve_lp_relaxation` runs PDHG on
CPU or CUDA and returns primal/dual vectors, reduced costs, KKT residuals, an
optional infeasibility-ray candidate, and a dual bound only when the complete
dual expression is finite. `qqa.dual.crossover_lp` resolves the identical LP
with HiGHS dual simplex and returns a basic solution plus portable variable and
row basis statuses.

`ExactFeedbackBus` carries versioned LP primal vectors, multipliers, reduced
costs, fractionalities, branch scores, local bounds, incumbents, linear cuts,
and no-goods. Cuts convert to typed constraints; no-goods convert to clause
factors. The SCIP conditional heuristic publishes live node information and
receives QQA incumbents and neighborhoods through this boundary.

General QUBO persistency uses bounded exact probing for small components and a
valid termwise lower bound otherwise. It never labels a heuristic fixing as a
proof. Exact results carry `CertificateMetadata`; the metadata says
`solver-reported-*` unless an independently verifiable proof digest is
actually present. Native exact adapters run in disposable processes by
default so a solver ABI failure cannot terminate the caller.

## CP, SAT, decomposition, global and uncertain models

The opt-in exact layer includes:

- CP-SAT lowering for bounded binary/integer linear models, all-different,
  assignment, clauses, precedence, no-overlap, and cumulative resources;
- PySAT RC2 lowering for SAT and integral weighted MaxSAT;
- factor-graph decomposition and variable-separator detection;
- generic Benders, column-generation, and progressive-hedging controllers;
- McCormick envelopes and bounded spatial branch-and-bound with a caller-
  supplied valid relaxation bound;
- mean, worst-case, CVaR, chance, Wasserstein, phi-divergence, and moment-
  ambiguity factors, scenario reduction, SAA confidence intervals, and
  held-out validation;
- direct QQA acquisition optimization for black-box trust regions;
- confidence/OOD-gated learned planning that always falls back to a
deterministic planner outside its training distribution.

SAT/MaxSAT calls with a deadline run in a disposable process and terminate it
at the hard wall-clock limit. Sparse QUBO neighbourhood APIs distinguish
induced, conditioned, and connected-component subproblems, preventing dropped
boundary energy from being mistaken for conditioning.

## Study/Trial and Benchmark Hub

`qqa.create_study` provides resumable black-box campaigns with QQA selected by
default for diverse batch acquisition. Cache identity includes seed, fidelity,
replicate, and evaluator version; an optional evaluation timeout isolates and
terminates a failed simulator process.

Benchmark Hub TOML/JSON manifests declare tracks, instances, checksums,
budgets, seeds, solvers, and metrics without retaining local paths. Paired
comparisons report wins/ties/losses, median differences, and deterministic
bootstrap confidence intervals plus an exact paired sign test; `holm_adjust`
controls family-wise error across declared comparisons. The portable starter
manifest is `benchmarks/manifests/qqa-core.toml`.

Unsupported coefficients, domains, factors, or proof semantics raise an
explicit exception. No route silently rounds coefficients, drops rows, or
changes infinite bounds.

## Events, cockpit, and decisions

Every `SolveEvent` uses schema version 2 and a monotone sequence number.
Events cover solve boundaries, presolve, relaxation updates, candidate repair,
incumbents, dual bounds, cuts, replica exchange/restart, constraint residuals,
kernel profiles, and completion.

```python
from qqa.visuals import decision_explorer, plot_optimization_cockpit

figure, _ = plot_optimization_cockpit(result, backend="matplotlib")
rows = decision_explorer(result, model_ir)
```

The cockpit separates primal and dual progress, phase timing, constraint
residuals, and outcome. The decision explorer reports archive stability and
one-coordinate counterfactual objective/violation deltas. Neither view
reinterprets unknown feasibility as success.

## Checkpoint and portable result packages

```python
first = qqa.solve(
    problem,
    budget="10s",
    checkpoint_path="run.qqacp",
    checkpoint_interval=100,
)
resumed = qqa.solve(problem, budget="30s", resume_from="run.qqacp")
```

Checkpoints are atomic ZIP containers with JSON metadata, NumPy tensors, and a
SHA-256 checksum for every tensor. They contain no pickle or executable code.
The model fingerprint, optimizer state, CPU/CUDA RNG state, schedule state,
incumbent, latent population, and historical archive are verified before
resume. Dynamics-changing options must match; a run may extend its epoch
horizon. File paths are API arguments and are not written into provenance.

`qqa.runtime.export_result_package` creates an independently verifiable model
summary/result/event bundle. Portable-payload validation rejects absolute
paths, private endpoints, credentials, host/server/cluster fields, non-finite
numbers, and executable objects.

## Remote service and custom-code boundary

Install `qqa[service]` and construct `qqa.service.create_app()` for a bounded
FastAPI job service. It accepts only schema-validated ModelIR dictionaries and
a small allowlist of solve options. Jobs execute in a bounded process pool;
tracebacks and filesystem paths are not returned. Set a bearer token at the
deployment boundary and apply TLS, rate limits, request-size limits, and
network policy in the reverse proxy.

Python source and pickle are trusted-local features. They are denied by
default in the library, CLI, and shared GUI. A local operator must explicitly
set `trusted=True`, pass `--allow-unsafe-python`, or set
`QQA_ALLOW_CUSTOM=1`. Never enable that switch on a multi-tenant service.

## Honest capability boundary

These APIs are composable solver building blocks, not a claim of universal
commercial-solver parity. In particular, QQA4CO does not claim a native C++20
persistent factor runtime, a complete branch-cut-price implementation, a
proof-producing CDCL/PB checker, or globally valid MINLP bounds unless the
selected bound oracle supplies them. Benchmark claims must name the snapshot,
hash, hardware-neutral budget, baselines, seeds, feasibility policy, and
statistical interval. The audited registry verifies origins and hashes; it
does not manufacture performance claims.
