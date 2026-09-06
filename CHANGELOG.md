# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Revalidate the exact member set, complete checksum map, JSON payloads,
  portability rules, and result status when opening result packages; reject
  malformed checkpoint schemas and tensor checksum records before loading.
- Isolate tensors, cuts, no-good assignments, and metadata published through
  the exact-feedback bus so callers cannot mutate shared solver state.
- Enforce finite numeric values, strict booleans, portable strings, and
  integer memory counters across the backend-independent result contract.
- Reject non-finite custom schedule outputs and evaluate stateful schedules
  exactly once per epoch; keep exponential and sigmoid schedules finite at
  extreme shape parameters.
- Account for setup and verification time before deciding whether a bounded
  QQA benchmark callback is affordable, and permit explicit `none` values for
  the two structural size gates.
- Use unique atomic benchmark-download temporary files, restrict redirects to
  the original HTTPS origin, and reject unsafe or duplicate archive members.
- Validate dataset-manifest members within their declared subset and avoid
  opening records beyond an explicit loader limit.
- Preserve runtime invariant checks under Python optimized mode by replacing
  removable assertions with explicit failures at solver and recorder boundaries.

### Changed

- The weekly public QPLIB smoke comparison now verifies that at least one
  measured `sg-cqqa` run performed a real QQA intervention rather than merely
  registering the optional plugin.

## [0.11.0] - 2026-09-06

### Solver integrity and QQA core

- Enforce original-model variable domains, finite values, coordinate-space
  identity, feasibility-first candidate selection, and candidate/certificate
  linkage across pure QQA, repair, and optional exact completion.
- Centralise float64 original-model verification of objective finiteness,
  domains, bounds, integrality, structured variables, and constraints in
  `ModelIR.verify_solution` for every result-construction path.
- Apply replica learning-rate and preconditioner scales to the effective Adam
  update, normalise convexification consistently, preserve fixed per-coordinate
  bounds, and keep adaptive augmented-Lagrangian/archive state on device.
- Add structural sparse-factor presolve, complete adaptive/checkpoint state,
  observed event timestamps, static original-objective incumbents, and
  auditable stage execution diagnostics.
- Preserve recorded trajectories and prior restart/exchange counters and epoch
  masks across pickle-free checkpoint resume, with interrupted-versus-
  uninterrupted parity coverage.
- Preserve inward repair gradients at declared continuous bounds across
  PyTorch clamp-derivative changes while retaining exact forward bounds.

### Deadlines, hybrid execution, and observability

- Introduce one monotonic solve context shared by compile, warm-up, search,
  repair, baseline, and certification stages; record skipped deadline stages
  instead of silently overrunning the requested budget.
- Make I/O, presolve, and runtime facades lazy, avoid importing Torch in native
  SCIP-only workers, and include isolated interpreter/package startup in the
  strict end-to-end benchmark clock.
- Pass QQA incumbents into CP-SAT, make QUBO-to-SCIP certification reachable,
  distinguish capability, planning, and actual execution, and add an explicit
  opt-in mode that requires a QQA-generated primal.
- Keep unavailable cockpit values unknown, distinguish search merit from the
  mathematical objective, and remove the obsolete external Polyfill script
  from the documentation site.

### Public benchmark protocol

- Add the complete MIPLIB/QPLIB audit manifest with fixed public snapshot
  hashes, 1/10/30/300-second budgets, five seeds, and a direct aggressive-SCIP
  ablation.
- Add optional all-solver process isolation, model import inside the matched
  deadline, independent bypass execution, bounded native workers, original-
  coordinate solution hashes/values, stage timings, peak memory, target time,
  anytime ECDFs, instance-level bootstrap intervals, and normalized failure
  outcomes.
- Preserve path-free failure records and separate independent runs, equivalent
  baseline reuse, QQA activation, and QQA-attributable improvements.
- Add a `qqa benchmark publish` command for deterministic, path-checked
  compact/full MIPLIB and QPLIB artifacts, keep full solution vectors out of
  the compact form, and record portable original-model structure metadata.
- Align Python and CLI hybrid defaults on the conservative screened profile,
  bypass QQA when the integer core or remaining budget cannot reach its minimum,
  and retain portable native signal/exit diagnostics.
- Require a 20-second QQA reserve by default after short-budget screening;
  1- and 10-second comparisons stay on the matched SCIP path.
- Delay the optional Torch-backed heuristic runtime until a SCIP callback
  passes every cheap timing and structural gate, and report its one-time
  initialisation cost separately in heuristic diagnostics.
- Enforce the hybrid overhead cap against complete measured callback wall
  time, including state inspection, lazy numerical-runtime startup, candidate
  ranking, QQA, and completion instead of only QQA/completion substeps.
- Require a conservative cold-start allowance before loading the optional
  numerical runtime, keeping 1-, 10-, and 30-second default comparisons on the
  native SCIP path when the 5% overhead budget cannot absorb startup.
- Reserve bounded callback-deadline slack before QQA execution so epoch-level
  stopping and candidate post-processing do not consume the advertised
  plugin-overhead allowance.
- Synchronise CUDA work at explicit wall-clock deadline checks, report QQA
  epoch/deadline diagnostics, and use configurable core precision while SCIP
  validates completed candidates against the original model.
- Remove history and duplicate historical-archive work from the bounded hybrid
  callback while retaining its best/final population and constraint archive.
- Seed general-integer QQA cores with alternating LP-centred randomized
  rounding and broad lattice samples instead of pathological bound-only draws.
- Adopt the screened float64/four-candidate hybrid profile, which produced
  more final-quality wins without losses across its five-seed tuning campaign.
- Make balanced solver order invariant to campaign sharding by deriving its
  phase from the portable instance name and seed, including independently
  executed structural-bypass cells.
- Compare resumable campaign settings in their persisted JSON representation,
  so tuple-valued QPLIB type allow-lists resume without a false configuration
  mismatch after checkpoint serialization.
- Publish the path-free aggregate and verification report for the complete
  27,720-run MIPLIB/QPLIB campaign across four budgets and five seeds,
  including failures, cold-clock overruns, instance-level inference, measured
  QQA activation, and the absence of suite-wide SCIP dominance.

### Repository layout

- Consolidate historical benchmark aggregates in the documentation and remove
  the duplicate root result archive and Benchmark Hub manifest.
- Keep downloaded MIPLIB/QPLIB data under the ignored
  `data/public-benchmarks/` tree, consolidate all example notebooks under
  `examples/`, and move documentation assets and deployment guidance into
  `docs/`.
- Remove the tracked operating-system metadata file and add a repository
  layout regression test.

## [0.10.0] - 2026-08-29

### Semantic correctness and Factor-Split QQA

- Add strict termination statuses and an independent `GuaranteeLevel`; remove
  unknown-as-feasible presentation paths and label mathematical objective,
  internal energy, merit, feasibility, bound, and proof separately. Exact
  adapters preserve proven or deadline-limited no-incumbent outcomes instead
  of replacing valid solver termination with an exception.
- Preserve every discrete lattice during algebraic scaling; separate
  constraint tolerance, row scale, search weight, user priority, and adaptive
  augmented-Lagrangian state.
- Derive factor capabilities from concrete eager, fused GPU, and exact backend
  registrations and add immutable device-aware execution plans.
- Correct multi-device island elite, deadline, incumbent, exchange, and final-
  population semantics; enforce hard process-isolated SAT deadlines.
- Split sparse QUBO induced, conditioned, and component extraction and harden
  AOT caches with architecture/toolchain keys, locks, and artifact checksums.

### QQA-centred expansion

- Add a dependency- and budget-aware solve DAG with adaptive QQA/exact routing.
- Add Study/Trial black-box campaigns with QQA batch acquisition, full noisy-
  observation cache identity, evaluator versioning, and isolated timeouts.
- Add portable Benchmark Hub manifests and deterministic paired bootstrap
  summaries, plus a credential-free end-to-end notebook.

### Deployment and native-boundary reliability

- Preserve the complete top-level API through lazy exports so metadata and
  discovery commands no longer import Torch or optional solver stacks; retain
  historical submodule access such as `qqa.tex`.
- Bound `qqa doctor` accelerator probes and native benchmark workers, start
  exact-solver budgets only after optional backend imports, and avoid querying
  SCIP solution-stage metrics when its solve phase did not run.
- Stabilize the sparse performance gate with steady-state warmups while
  retaining the existing runtime and storage regression thresholds.

## [0.9.0] - 2026-08-28

### Mathematical and semantic foundations

- Normalize stable QQA dynamics across replicas and relaxed dimensions with
  robust median/MAD objective scaling while retaining raw original-space
  objectives; correct the negative-BG convexity and CRA/PI-GNN explanations.
- Add machine-readable factor capabilities, reject non-differentiable factors
  from pure QQA, remove guessed finite-bound fallbacks, and represent
  unevaluated feasibility as `unknown` rather than success.
- Standardize schema-v2 solver events, checksum-safe certificate metadata,
  CUDA-event timing, and trusted-local-only Python/pickle execution.

### Heterogeneous primal and GPU runtime

- Add heterogeneous replica roles, per-replica beta/learning-rate schedules,
  parallel-tempering exchange, curvature-aware c=2 convexification,
  factor-aware preconditioning, residual-balanced augmented Lagrangians,
  archive-centred restarts, multi-source warm states, and a historical
  quality/diversity archive with lineage.
- Add a device-resident typed factor graph for linear, quadratic, cardinality,
  and clause factors; segmented reductions; GPU exact-k, one-hot, assignment,
  k-flip/tabu, and 2-opt operations; bounded kernel autotuning and roofline
  telemetry; and no-host-sync event/archive callbacks.

### Primal-dual and exact integration

- Add sparse CPU/CUDA PDHG with primal, dual, reduced-cost, KKT, ray-candidate,
  and finite-only bound reporting, plus dual-simplex LP crossover and portable
  basis statuses.
- Add a versioned SCIP/QQA feedback bus for LP vectors, reduced costs,
  fractionalities, branch scores, local bounds, cuts, conflicts/no-goods, and
  incumbents; extend general QUBO probing/persistency and proof-aware result
  adaptation without treating heuristic evidence as a certificate.
- Preserve disposable-process isolation for native exact backends and expose
  bounded CP-SAT scheduling and PySAT SAT/MaxSAT runtimes.

### Broader models and product runtime

- Add decomposition/separator detection, generic Benders, column generation,
  progressive hedging, McCormick/spatial branch-and-bound, Wasserstein,
  phi-divergence and moment DRO, scenario reduction, SAA intervals, and held-
  out validation.
- Add confidence/OOD-gated learned planning, direct QQA black-box acquisition,
  Model Doctor, duration/goal one-call solving, Optimization Cockpit, Decision
  Explorer, domain templates, and an audited MIPLIB/QPLIB snapshot registry.
- Add atomic pickle-free checkpoints with model/config checks, optimizer and
  RNG state, adaptive schedule state, population/incumbent/archive restoration;
  checksum-protected result packages; and a bounded schema-only FastAPI job
  service with optional bearer authentication.
- Add the typed primal-dual runtime guide and a credential-free Colab notebook
  covering diagnosis, events, visualization, resume, and package verification.

## [0.8.1] - 2026-08-27

### Reliability and portability

- Expand CI type checking from six core files to all 151 source modules and
  isolate uv cache keys per Python version to prevent matrix upload races.
- Expose validated CSR, bound-array, variable-domain, and variable-type views
  on canonical model objects so MIPLIB, QPLIB, SCIP, HiGHS, and CP-SAT share
  one statically safe representation without changing accepted input types.
- Fix integer warm-start clipping to use supported PyTorch operations, retain
  partial PI-GNN replica solutions while filling missing replicas, and harden
  sparse/dense sampler dispatch and Pareto archive invariants.
- Strengthen safe-expression AST narrowing, LLM client request typing,
  mixed-variable declarations, application callables, benchmark result
  aggregation, and optional exact-backend boundaries.
- Keep CUDA validation explicitly opt-in and give every unsupported or
  unavailable optional path a deterministic guard instead of an ambiguous
  attribute or `None` failure.
- Synchronise the software citation metadata with v0.8.1 and include
  `CITATION.cff` in source distributions.

## [0.8.0] - 2026-08-27

### Advanced opt-in QQA runtime

- Add deterministic and stochastic straight-through binary estimators,
  entropic softmax, sparsemax/entmax categorical maps, endpoint-inclusive
  temperature annealing, and simplex-native mirror descent. Pure QQA with
  AdamW remains the default path.
- Keep adaptive replica-restart decisions, counters, and event masks on the
  accelerator until the final result transfer, and align adaptive schedule
  observations with explicit progress checkpoints.
- Fix structured warm starts so each relaxation restores its own latent
  geometry instead of applying an invalid universal unit-cube clamp.
- Add persistent, content-addressed `torch.export` and AOTInductor sparse-QUBO
  artifacts with atomic writes, compatibility manifests, dynamic batch
  support for export, autograd parity, and cache reuse.
- Fuse sparse-QUBO endpoint gathering and reduction into one indexed load and
  matrix-vector reduction, reducing the CPU regression-gate runtime while
  preserving gradients and arbitrary leading batch dimensions.

### Conditional hybrid portfolio

- Expand the opt-in SCIP-guided portfolio with graph-induced, local-branching,
  trust-region, conflict, gradient, history, pseudocost, RENS, RINS, and
  reduced-cost neighbourhoods.
- Extract bounded constraint-interaction graphs and node signals from SCIP
  state while retaining solver-independent interfaces and path-free results.

### Packaging and automation

- Refresh the fully resolved lock to the newest compatible releases, including
  PyTorch 2.13, NumPy 2.5, SciPy 1.18, Streamlit 1.62, Plotly 7, pandas 3,
  Ruff 0.16, and pytest 9; pin the Streamlit deployment to current CPU wheels.
- Test CPython 3.10 through 3.14, use Python 3.12 for Streamlit and automation,
  and update all GitHub Actions to immutable current release tags.
- Add an entrypoint-local Streamlit requirements bridge so Community Cloud
  reliably selects the reproducible CPU deployment file.
- Document the opt-in relaxation, mirror-descent, and persistent AOT APIs,
  plus the CPython-only runtime boundary imposed by official PyTorch wheels.

## [0.7.0] - 2026-08-27

### GPU runtime, search, and reliability

- Add analytic sparse-QUBO energy/gradient primitives, an optional Triton path,
  a dispatcher-registered custom operation with autograd checks, packed binary
  Hamming operations, and opt-in state-preserving CUDA Graph replay.
- Add real distributed QQA islands over NCCL/Gloo with diverse elite migration.
  Single-process pure QQA remains the default.
- Add tabu, k-flip, path-relinking, iterated, 2-opt/3-opt, FM, MIS-swap,
  Kempe-chain, and WalkSAT local searches behind explicit APIs.
- Add safe QUBO dominance/persistency/symmetry reductions, graph-cut
  persistency, and singleton-constraint bound tightening with reversible
  original-space restoration.
- Add opt-in factor-graph GNN warm starts, a LinUCB solver selector, and
  energy-guided discrete diffusion without changing default solver selection.
- Harden parsers with portable million-variable guards and ModelIR schema
  checks; add Hypothesis fuzzing, solver-result contracts, GPU operation checks,
  CUDA Graph tests, and distributed execution tests.
- Split GPU, distributed, learned, local-search, presolve, CLI command, and UI
  theme responsibilities into focused modules.
- Add a prominent live Streamlit route, CRA/CPRA paper links and BibTeX, an
  original project visual, and CPython/PyPy support guidance.

### Canonical QQA runtime and sparse portfolio

- Add strict `qqa.solve`, `qqa.plan`, and `qqa.inspect` entry points, a
  factor-based `ModelIR`, reversible presolve ledger, and unified
  `SolveResult` semantics for raw/repaired solutions, objectives, merit,
  violations, timing, resources, bounds, and proof.
- Move sparse graph objectives to edge-factor evaluation; add component
  decomposition, incremental local search, diverse archives, automatic
  repair, adaptive schedules, replica islands, and optional compiled QQA.
- Add structured categorical/permutation execution with softmax, Gumbel, and
  Sinkhorn relaxations. TSP binary penalties are explicit opt-in, and
  score/repair operations no longer mutate their input.
- Add portable JSON/OPB/CNF/WCNF/QUBO/Ising inputs, native scheduling/logical/
  network factors, integer encoding selection, uncertainty aggregation, and
  rolling-horizon warm-state transfer.
- Add deterministic portfolio planning, multiple SCIP neighbourhoods with UCB
  allocation, and process-isolated optional SCIP, HiGHS, CP-SAT, and cuOpt
  adapters. The default route remains pure QQA.
- Extend black-box search with persistent evaluation states, asynchronous
  scheduling, multiple trust regions, scalable random-feature surrogates, and
  optional QQA acquisition optimisation; extend Pareto search with additional
  scalarisations and many-objective indicators.
- Correct NormalizedCut empty partitions, maximize-direction hybrid selection,
  mixed result semantics, exact schedule endpoints, strict UI option handling,
  and single-run comparison claims. Add semantic, parser, exact-adapter, CUDA,
  type, package, performance, and scheduled public-benchmark CI checks.
- Remove generated large datasets and per-run campaign trajectories from the
  source tree; retain public fetch/checksum instructions and compact,
  machine-neutral benchmark summaries.

### MIPLIB/QPLIB and SCIP-guided conditional QQA

- Fix SG-CQQA device propagation so `device="auto"`, CPU, CUDA, and MPS
  selections reach the actual QQA core solve instead of only its RNG context.
  Move bounded core execution into the PySCIPOpt-independent numerical runtime
  and add a regression test for the device contract.
- Move the complete MIPLIB/QPLIB CLI adapter out of the core CLI module and
  make `qqa.benchmarking` a lazy facade. Benchmark parsing, SCIP-facing code,
  and result modules now load independently on first use.
- Make Plotly, pandas, and Streamlit genuine `plotly`/`gui` extras instead of
  duplicated core dependencies, while retaining the deployment-specific GUI
  requirements file. Align the local Ruff pre-commit hook with the CI lock.
- Add a sparse algebraic IR with infinite bounds, linear/quadratic objectives
  and constraints, original-space evaluation, reversible scaling, and
  path-free portable provenance.
- Add SCIP-backed MPS/MIPLIB and `pyqplib`-backed QPLIB importers, including
  lower-triangle quadratic cross-checks against the parser at nonzero points.
- Add adaptive integer encodings, constraint-wise PHR augmented Lagrangian,
  separate feasibility/objective archives, elastic continuous repair,
  RENS/RINS core selection, local branching, and sparse DC convexification.
- Add iterative SG-CQQA as a PySCIPOpt primal heuristic. An original-objective,
  active-row surrogate first tries cheap integral core moves; QQA is a
  conditional fallback when sufficient wall time remains. An in-place SCIP
  dive performs the first completion and a bounded sub-SCIP repair is
  available when needed; full solutions return through `trySol()`.
- Keep the default `import qqa`, `auto` routing, and CLI solver path pure QQA.
  Exact-solver and MIPLIB/QPLIB integrations now load lazily behind explicit
  `qqa.hybrid`, `qqa.benchmarking`, and `qqa.io` opt-in boundaries; add the
  aggregate `qqa[benchmark]` installation extra.
- Split the conditional heuristic into configuration/diagnostics, vectorised
  core-model, numerical runtime, and SCIP callback modules. Batch candidate
  ranking and population generation, and cache core tensors per device/dtype
  to avoid repeated Python dispatch and accelerator transfers.
- Add `qqa benchmark fetch|inspect|run|compare` for official MIPLIB/QPLIB snapshots,
  with total shared deadlines, incumbent trajectories, time to first feasible,
  primal integral, primal/dual gap, completion/acceptance rates, and snapshot
  hashes. Paired comparisons include an aggressive-SCIP ablation and multiple
  deterministic seeds; fast-surrogate and QQA-only candidate rates are
  reported separately.
- Add atomic, configuration-checked campaign checkpoints, anonymous
  continue-on-error records, selective failure retries, and deterministic
  path/private-host rejecting publication artifacts for full benchmark runs.
- Add validated shard merging with duplicate-instance/record rejection and
  complete aggregate recomputation for portable multi-GPU campaigns.
- Isolate every QPLIB native solve in a disposable worker so nonlinear solver
  crashes, bounded worker timeouts, and retained allocator state cannot
  terminate or progressively exhaust a resumable campaign; support the same
  safe path from scripts, notebooks, interactive Python, and the CLI.
- Bound conditional heuristic overhead to 10% of the shared SCIP budget in
  the Python configuration (5% in the conservative benchmark CLI), evaluate
  two fast candidates, and stop fast completion immediately after a true
  original-objective improvement.
- Add a bounded second-stage LNS repair: when fixing the rounded integer
  complement is infeasible, retain the QQA core assignment and release the
  complement, then retain only the highest-scored quarter if needed, inside
  the original completion budget.
- Stop later QQA calls after a first non-improving call by default, preserving
  SCIP time when the learned neighbourhood is not productive while retaining
  an explicit CLI ablation switch.
- Route continuous-only QPLIB models directly through the matched aggressive
  SCIP configuration, avoiding empty QQA plugin and completion-template setup.
- Apply the constraint-wise PHR augmented Lagrangian and feasibility archive
  to QPLIB/no-incumbent cores while retaining the cheaper static selected-row
  merit for the screened incumbent-guided linear-MIP path.
- Escalate from the primary fast hybrid path to QQA only after observing a
  continuously completable fast candidate; QQA-only ablations remain explicit.
- Evaluate nonlinear QPLIB incumbents monotonically in the original sparse
  model, independent of SCIP's objective auxiliary, and report directional
  reference error so a genuinely better incumbent is never penalised.
- Apply the same total QQA+SCIP wall-clock budget to the existing one-shot
  QUBO and safe symbolic SCIP hybrids.

### Universal optimisation studio

- Remove provider-specific endpoint, model, and legacy credential identifiers
  from the library, UI, documentation, and notebooks; API profiles now use
  only the generic `QQA_LLM_*` configuration.
- Harden optional SCIP capability detection, TeX/API trust boundaries, model
  size/type validation, finite mixed-variable inputs, Pareto result contracts,
  and black-box resume/output validation; stabilise zero-regularisation RBF
  fitting for duplicate observations.
- Add the unified `qqa.ask(...)`, `qqa ask`, and Streamlit **Ask QQA**
  entry points: natural language is compiled with a separate hardened system
  prompt, validated as an auditable `ModelSpec`, routed deterministically to
  pure QQA, Pareto QQA, or budgeted black-box search, and then solved.
  QQA+SCIP remains available only when explicitly selected.
- Add a Colab-ready natural-language optimisation walkthrough with a reviewed
  binary/integer/real production model, exact SCIP certification, one-run
  three-objective Pareto search, and constrained parallel black-box tuning.
- Add a cardinality-constrained binary/real portfolio application with
  risk/return/turnover objectives to the Python API, CLI, UI, and Colab
  studio.
- Replace inequality handling in Pareto search with a projected
  Powell–Hestenes–Rockafellar augmented Lagrangian; preserve objective-axis
  anchors, maintain a separate KKT multiplier vector per reference-direction
  replica, and split stagnation recovery between archive-centred and global
  restarts.
- Add knee-aware Pareto plots and a dedicated feasibility/archive/penalty/
  restart diagnostics dashboard.
- Model every black-box constraint with a shared-factorisation multi-output
  log-violation RBF and use joint probability of feasibility.
- Add realistic microgrid dispatch/Pareto and constrained process black-box
  application builders plus `qqa example list|run`.
- Add adaptive augmented-Lagrangian Pareto feasibility, stagnation restarts,
  chunked exact dominance, knee/weighted selection, hypervolume, and
  decision-variable DataFrame export.
- Add expected-improvement and probability-of-feasibility black-box
  acquisition, resumable campaigns, bounded local RBF fitting, GPU float32
  surrogate auto-selection, metadata, and full evaluation export.
- Add safe mixed nonlinear `ModelSpec` → PySCIPOpt compilation with QQA
  multi-starts, exact constraints, dual bounds, gaps, and proof status.
- Add `qqa doctor`, TeX file input, explicit QQA/SCIP routing, reviewed-model
  printing, and the Streamlit Universal Optimization Studio.

### Added

- **Universal optimisation layer**: one-shot reference-direction Pareto
  annealing for mixed variables, RBF/trust-region black-box optimisation,
  optional QQA→SCIP QUBO refinement with multiple primal starts and
  optimality gaps, plus interactive 2-D/3-D/many-objective and black-box
  diagnostics.
- **Safe TeX modelling and `qqa tex` CLI** using OpenAI-compatible Responses
  or Messages endpoints. Structured JSON is validated and interpreted by a
  restricted arithmetic AST without `eval`/`exec`; credentials are accepted
  only from the environment and local `.env` files are ignored.
- **First-class mixed-variable optimisation** through `qqa.MixedProblem`,
  typed `Binary` / `Integer` / `Real` declarations, differentiable named
  `Constraint` objects, integer-grid relaxation, physical-unit warm starts,
  constraint-aware scoring, pure-real and pure-integer support, and the
  `solve_mixed` convenience entry point.
- **Advanced result diagnostics and offline reports**:
  `plot_result_dashboard`, `plot_variable_solution`,
  `plot_constraint_diagnostics`, plus `save_html_report` for a single
  self-contained interactive artifact with embedded machine-readable JSON.
  Backend-neutral data extraction and plotting are separated under
  `qqa.visuals`.
- **Mixed optimisation Colab walkthrough** covering convex real, bounded
  integer, and practical binary/integer/real factory planning with brute-force
  verification.
- A `py.typed` marker now makes the package's inline annotations visible to
  downstream type checkers, matching the existing `Typing :: Typed`
  classifier.
- **First-class iSCO sampler support** via `qqa.discrete_langevin`
  (paper-faithful alias `qqa.isco_anneal`). Faithful, GPU-parallel
  implementation of **Algorithm 1 + Appendix C (PAS-MH-Step)** of
  Sun, Goshvadi, Nova, Schuurmans, Dai, *Revisiting Sampling for
  Combinatorial Optimization*, ICML 2023 (pmlr-v202-sun23c). Every
  MH step samples a Poisson-length path `L ~ Poisson(μ)` truncated
  at `L ≥ 1`, picks `L` sites without replacement via Gumbel-top-`L`
  with logits `−Δ_j/(2τ)`, applies the **path-auxiliary MH
  correction** over the ordered permutation σ (Eq. 30), and adapts μ
  toward the paper's 0.574 acceptance target (Eq. 31). Works on
  single-instance (`Q_mat`) and batched-instance (`Q_tensor`) QUBOs;
  spin / categorical / structured-shape relaxations are rejected at
  the API boundary with an actionable `NotImplementedError`. Returns
  an `ISCOResult` that mirrors `SAResult` / `PAResult`
  (`best_sol` / `best_obj` / `runtime` / `history` / `score` /
  `polished_sol`) plus iSCO-specific diagnostics
  (`accept_rate`, `mu_final`, `mean_path_length`, `t_max_used`).
  Cross-checked against the DISCS reference implementation
  (`samplers/path_auxiliary.py`) and the Zhang et al.
  `discrete-langevin` reference. See the new
  *iSCO baseline (Sun et al., ICML 2023)* section in the README and
  citations `sun2023revisiting` + `goshvadi2023discs`.
- **Empirical detailed-balance test for iSCO**
  (`tests/test_isco.py::test_isco_detailed_balance_on_tiny_qubo`)
  enumerates a 2^4-state QUBO, runs the full PAS-MH kernel for 4000
  inner steps × 200 chains at fixed temperature, and asserts
  TV(empirical, exact Boltzmann) < 0.02. Ships as a permanent
  guard against silent MH-correction regressions; offline sweep
  across `N ∈ {3, 4, 5} × seed ∈ {0, 7, 42} × μ ∈ {1, 2, 3} ×
  {float32, float64}` shows the post-fix sampler converges to
  TV ≤ 0.0064 in every cell.

### Fixed

- Ship benchmark runners and plotting support inside wheels instead of
  importing repository-only scripts.
- Preserve the generating reference-direction weight for every filtered and
  sorted Pareto solution.
- Correct iSCO's ragged batched Plackett–Luce probability by excluding padded
  path entries from both forward and reverse denominators.
- Preflight complete SCIP model compilation before QQA work, reject
  infeasible projected SCIP replacements, and make mixed-model penalty
  calibration solve-local, offset-invariant, and feasibility-first.
- Harden generated model validation with structural quotas and deterministic
  finite-value probes; reject unsafe LLM base URLs, oversized responses, and
  credential-forwarding redirects.
- Resolve Streamlit page links relative to the live page script so all five
  navigation entries work from both the home page and legacy `pages/` routes.
- Remove AdamW's implicit weight decay from QQA defaults, which biased latent
  binary coordinates toward zero independently of the objective.
- Correct greedy 1-flip QUBO deltas for non-symmetric user matrices by using
  both `Q @ x` and `Q.T @ x`.
- Extend monotone polishing beyond QUBOs with exact local-field Ising flips
  and vectorised one-site categorical moves.
- Rank QQA→SCIP mixed warm starts feasibility-first and guarantee that a
  numerically rejected start cannot worsen the returned feasible incumbent.
- Fix multi-objective `score_summary`, which previously reached the scalar
  `loss_fn` override and failed on a selected Pareto plan.
- Categorical relaxations now restore non-negative bounded coordinates after
  every AdamW step even at `temp=0`, normalise a valid simplex, and measure
  diversity in probability rather than arbitrary raw-logit scale.
- `AutoDivTuner` no longer divides a population standard deviation by
  `sol_size` a second time and now uses negative rather than positive
  feedback; adaptive diversity control works consistently across population
  sizes and moves the diversity weight in the correct direction.
- `anneal` rejects zero/negative `curve_rate`, invalid `check_interval`,
  negative temperature/learning rate, and out-of-range `div_param` with clear
  errors.
- Graph normalisation now accepts heterogeneous, non-orderable node labels
  such as integers mixed with strings.
- **iSCO `_plackett_luce_logprob` NaN bug (silent detailed-balance
  violation).** The Plackett-Luce log-prob recursion used
  `diff.clamp(max=-1e-12)` to keep `log1p(-exp(diff))` finite, but
  `-1e-12` round-trips to `0.0` in float32 (machine ε ≈ 1.19e-7),
  sending the recursion into `log(0) = -inf` whenever `sigma`
  contained the repeated indices that `_reverse_path` writes into
  the masked tail (i.e. **every chain with `L_per_chain < L_max`,
  which is every short chain in any batch with variable Poisson
  path length**). Subsequent summation via `* mask.to(dtype)` then
  produced `(-inf) * 0 = NaN`, making `log(u) < log_alpha` evaluate
  to `False` everywhere and silently rejecting every multi-flip
  proposal in the affected chain. Empirical TV(empirical, Boltzmann)
  on a 4-bit enumerable QUBO was ~0.51; after the fix it is
  ~0.001-0.002. Two surgical changes: (a) dtype-aware clamp
  (`eps_clamp = -1e-6` for float32, `-1e-12` for float64); (b)
  `torch.where(mask, value, 0)` instead of `* mask` so masked
  positions can never contaminate the sum via `inf * 0`. Regression
  tests `test_plackett_luce_logprob_handles_repeated_indices_in_float32`
  and `test_isco_detailed_balance_on_tiny_qubo` ensure this stays
  fixed. Lessons L48-L50 in `tasks/lessons.md`.

## [0.6.0] - 2026-04-20

### Added

- **`qqa.polish.apply_polish_if_improves`**: single entry point for the
  greedy 1-flip QUBO polish post-processing. `qqa.anneal`,
  `qqa.simulated_annealing`, `qqa.population_annealing` and both
  PI-GNN trainers now route through this helper so every backend has
  the same "monotone free improvement" contract without five copies of
  the same `if polish and Q_mat is not None: …` block.
- **Shared test fixtures** at `tests/conftest.py`: `APP`, `PAGE_DIR`
  path constants, a `make_problem_config(kind, size, **extra)` factory
  and a `set_slider` helper. Test modules now import these directly,
  eliminating twelve copies of the same ``problem_config`` literal in
  `test_gui_apptest.py`.
- **`app/_common.retheme_plotly(fig)`**: replaces the ``_retheme`` clone
  previously defined once per Streamlit page. Import it alongside
  `plotly_layout` so every chart stays in step with the active theme.
- **`app/_common.as_numpy(x)`** (public alias of the former `_as_np`):
  imported by `_solution_viz.py` so the two modules share a single
  tensor-to-numpy conversion path.

### Changed

- **Benchmark suite refreshed**: the project version now tracks the
  "qqa4co-bench" HF dataset (coloring / mis-rrg / ea3d /
  balanced-partition / MaxCut G-set families), wired through the
  `qqa.bench` public API and `qqa bench run|plot|list|setup` CLI.
- `SpinRelaxation.perturb_` now inherits from `BinaryRelaxation` —
  both relaxations share the same latent cube `[0, 1]` and therefore
  the same noise + ``clamp_`` schedule. Removes a silent copy-paste
  drift risk.
- `qqa.bench` collapsed `_load_bench_discs` and `_load_plot_benchmarks`
  onto a shared `_load_scripts_module(name)` helper so the two
  ``sys.path`` / ``importlib`` call sites no longer drift.
- ``tests/`` directory is now on the pytest ``pythonpath`` so test
  modules can ``from conftest import …`` the shared helpers.

### Removed

- **`qqa.sa._qubo_glauber_sweep` deprecated alias** dropped — it
  forwarded to `_qubo_seq_glauber_sweep` and was only referenced by an
  in-tree diagnostic script (updated). The buggy parallel-update
  semantics it warned about have been gone since 0.4.0.

## [0.5.3] - 2026-04-20

### Added

- **Backend-aware Visualize layout**: the Streamlit Visualize page now
  shows PQQA-only tabs for PQQA runs (family tree, PCA embedding,
  diversity, parallel coordinates) and PA-only tabs for PA runs
  (ESS, free-energy trajectory, equilibration diagnostic,
  Thermodynamics, Lineage vs energy, Ancestry Sankey). Empty
  "No snapshots recorded" placeholders are gone.
- **Up-front PA capability probe** in the Solve page: problems that
  PA cannot sample (categorical / structured binary, e.g. TSP, QAP,
  Coloring, NQueens) now trigger a clear warning banner and disable
  the Run button, instead of surfacing a cryptic ``einsum`` /
  ``NotImplementedError`` mid-run.
- **Three PA-specific visualisation tabs**: Thermodynamics (Q vs β,
  internal energy, specific heat), Lineage vs energy, Ancestry
  Sankey.

### Changed

- `qqa.simulated_annealing` / `qqa.population_annealing` now accept
  `polish=True/False` and expose a `polished_sol` field, matching the
  contract `qqa.anneal` has always had. The 1-flip polish is default-on
  across all backends so the "best_obj" score card reflects the same
  post-processing everywhere.
- `_validate_chain_problem` (used by both SA and PA) now rejects
  structured `BinaryRelaxation` (non-flat `shape_fn`, e.g. TSP)
  with an actionable error steering users to `qqa.anneal`.

## [0.5.2] - 2026-04-20

### Added

- **`qqa.bench` public Python API** (`run`, `plot`, `list_suites`,
  `resolve_suite`) mirroring the `qqa bench` CLI so notebooks can
  dispatch a benchmark without subprocess boilerplate.
- **Polished benchmark report figure** (`scripts/plot_benchmarks.py`)
  and the corresponding `qqa bench plot` CLI flow.

### Changed

- HF Hub dataset renamed to `qqa4co-bench` (was `discs-benchmarks`);
  `scripts/setup_discs_data.sh` and all docs follow suit.

## [0.5.1] - 2026-04-19

### Added

- **`qqa.population_annealing`**: Population Annealing backend with
  parallel chain sampling, importance resampling between inverse
  temperatures, full free-energy / log-Z estimates and an optional
  genealogy / ancestry record. `PAResult` dataclass and
  `qqa solve --backend pa` CLI expose the new path.
- **MaxCut G-set benchmark family** via
  `scripts/fetch_gset_data.py` + `scripts/maxcut_gset_g70.py`.

## [0.5.0] - 2026-04-19

### Added

- Streamlit Compare page now offers a **PQQA vs SA shootout** mode that
  runs both backends on the same problem instance and reports the
  per-backend best objective, runtime and a "SA time to PQQA best"
  speed-up factor side-by-side, including a convergence plot.

### Changed

- Internal refactor: `qqa.utils` now exposes
  `require_cuda_if_requested(device)` and
  `safe_score_summary(problem, sol, fallback_obj)` helpers. The QQA,
  SA and PI-GNN/CPRA trainers now route their CUDA-availability check
  and `problem.score_summary` fallback through these shared helpers,
  removing duplicated inline `try/except` blocks while preserving the
  exact user-facing error messages and result dictionaries.
- Marked the legacy graph-evaluation helpers in `qqa.utils`
  (`approximate_mis`, `mis_stats`, `max_cut_stats`, `_gen_combinations`)
  as superseded by `problem.score_summary`. They are kept for backward
  compatibility but are no longer used internally.

### Documentation

- Repo-wide audit of the QQA / CPRA paper citations. Three places had
  silently swapped the QQA paper (Ichikawa & Arai, ICLR 2025) with the
  CPRA paper (Ichikawa & Iwashita, TMLR 2025) — fixed in
  `src/qqa/__init__.py` docstring, `examples/16_cra_pignn.ipynb`
  and `examples/17_cpra_pignn.ipynb`. Adopted the TMLR-published
  title for CPRA ("Continuous Parallel Relaxation for Finding Diverse
  Solutions in Combinatorial Optimization Problems"); the older
  arXiv-preprint title ("Continuous Tensor Relaxation …") is no longer
  used.
- Added a Codecov coverage badge to `README.md` and a placeholder for
  the Zenodo DOI badge (uncommented and DOI-substituted as soon as the
  first release is minted).
- Fixed `CITATION.cff` `preferred-citation` block: title now correctly
  matches the URL (both point at the QQA ICLR 2025 paper); arXiv:2409.02135
  added as an explicit identifier so citation tooling (Zenodo, ORCID,
  OpenAlex) resolves to the same artefact.

### Infrastructure

- `publish.yml` Trusted Publishing wired up end-to-end on PyPI:
  GitHub Actions environment `pypi` is now connected to the registered
  Trusted Publisher, so future tagged releases upload automatically
  without manual `twine` invocations.
- Broadened PyPI classifiers in `pyproject.toml`
  (`Environment :: Console`, `Environment :: GPU :: NVIDIA CUDA`,
  `Intended Audience :: Education / Developers`, OS-specific tags,
  `Topic :: Mathematics / Physics`, `Typing :: Typed`) for better PyPI
  discoverability.

## [0.4.0] - 2026-04-19

### Added

- **`qqa.simulated_annealing`**: GPU-parallel Simulated Annealing
  baseline with two execution paths:
  - QUBO fast path (Glauber-like parallel update, single matmul per
    sweep) for any problem exposing `Q_mat`.
  - Generic single-spin sequential Metropolis fallback for non-QUBO
    problems.
  - New `SAResult` dataclass mirroring `AnnealResult` for
    interchangeable downstream tooling.
- **CLI**: `qqa solve --backend sa` with `--sa-num-sweeps`,
  `--sa-beta-start`, `--sa-beta-end`, `--sa-schedule`.
- **`qqa.utils.enable_tf32`** helper to opt into TF32 matmul / cuDNN
  on Ampere+ GPUs.
- **`anneal(..., mixed_precision="bf16")`** opt-in for bfloat16
  autocast on the QQA forward pass (CUDA only; falls back to fp32
  silently elsewhere).
- **`train_cra_pi_gnn` / `train_cpra_pi_gnn`**: new
  `early_stop_disc_patience` argument that terminates training when
  the best discrete objective stops improving.
- **CPRA `multi_problem` batching**: when every replica problem has a
  same-shape `Q_mat`, the trainer stacks them into one tensor and
  computes all replica costs in a single batched `einsum`, replacing
  the previous Python-level per-replica loop.
- **`docs/explanation/algorithm.md`**: SA section documenting the
  parallel-Glauber fast path and when to reach for SA vs QQA / CRA /
  CPRA.
- **`examples/18_solver_benchmark.ipynb`**: head-to-head
  benchmark notebook comparing all four solver families on a common
  MIS instance with controlled compute budget.

### Changed

- `HistoryRecorder` now buffers per-epoch metrics as GPU scalars and
  performs a single bulk `cpu()` transfer in `on_train_end`,
  eliminating per-epoch host-device synchronisation. Public
  `result.history` shape is unchanged.
- `qqa.anneal` and the PI-GNN trainers now use
  `optimizer.zero_grad(set_to_none=True)` (PyTorch 2.x best practice).
- `SpinRelaxation.project` no longer allocates two `ones_like(x)`
  intermediates per call; uses scalar-broadcast `torch.where`.
- `CategoricalRelaxation.penalty` no longer triggers a redundant
  `forward`: the relaxation now exposes `penalty_from_forward` so
  `anneal` reuses the already-normalised tensor.

### Performance

- ~15 % wall-clock reduction on CPU for `qqa.anneal`-driven workloads
  (HistoryRecorder + `set_to_none` + `SpinRelaxation` together).
- CPRA `multi_problem` runs are 2–4× faster on GPU at `R = 16` thanks
  to the batched `einsum` path.

### Notes

- No public API removed. `qqa.anneal`, `qqa.pignn.train_*` and the
  `AnnealResult` dataclass are unchanged. New keyword arguments
  (`mixed_precision`, `early_stop_disc_patience`) are opt-in and
  default to the prior behaviour.

## [0.3.0] - 2026-04-18

### Added

- **Spin problem family** in `qqa.problems`:
  - `Ising1D`, `EdwardsAnderson`, `SherringtonKirkpatrick`
  - `BinaryPerceptron` (teacher-student), `HopfieldMemory`
  - New `SpinRelaxation` that maps `[0,1]` → `±1` with differentiable forward.
- **Visualization** (`qqa.visualization`):
  - Dual backend (`"matplotlib"` default, `"plotly"` optional).
  - `plot_best_trajectory`, `plot_schedule`, `plot_run_comparison`,
    `plot_parallel_coordinates`, `plot_solution_heatmap`.
- **CLI** (`qqa` entry point): `qqa version`, `qqa solve`, `qqa bench`,
  `qqa gui`.
- **Streamlit GUI** (`qqa gui` / `uv run streamlit run app/streamlit_app.py`):
  problem definition → live annealing → visualization → comparison.
- **Example notebooks**: MIS, coloring, MaxCut, 3D Edwards–Anderson, SK,
  binary perceptron, Hopfield memory, parallel benchmark.
- **Docs site** via MkDocs + Material with auto API reference.
- **Tooling**: GitHub Actions CI, `pre-commit`, `CONTRIBUTING.md`,
  `CITATION.cff`.

### Changed

- `qqa.problems` is now a subpackage (`qubo.py`, `categorical.py`, `spin.py`).
  Public symbols (`MaximumIndependentSet`, `Coloring`, ...) are preserved via
  re-export, so existing code keeps working.

### Deprecated

- `qqa.legacy.*` wrappers still work and emit `DeprecationWarning`; use
  `qqa.anneal` instead.

## [0.2.0]

- Initial unified `qqa.anneal` API, package reorganization under `src/qqa`,
  `uv`/`pyproject.toml` based install, smoke tests and demo scripts.

## [0.1.0]

- Original research release accompanying the ICLR 2025 paper.
