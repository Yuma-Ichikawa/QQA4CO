# Semantic audit implementation

Version 0.11.0 closes the 18 correctness, runtime, and product-boundary
findings from the September 2026 semantic audit. This page maps each finding
to its implemented contract and regression evidence. It is a correctness
matrix, not a claim that a heuristic dominates every exact solver.

## Finding closure matrix

| Finding | Implemented contract | Primary implementation | Regression evidence |
| --- | --- | --- | --- |
| F01 | Every candidate records original or presolved coordinates; exact incumbents remain in original coordinates and populations are restored independently. | `qqa.result.CandidateRecord`, `qqa.api.solve` | `test_exact_incumbent_has_an_explicit_original_coordinate_contract` |
| F02 | Exact-only routes initialise QQA allocation to zero and report the stages that actually executed. | `qqa.api.solve` | `test_exact_only_plan_does_not_claim_a_skipped_qqa_stage` |
| F03 | Independent verification checks shape, finite values, bounds, integrality, structured domains, constraints, and original objective. | `ModelIR.domain_violations`, `ModelIR.verify_solution`, `ModelIRProblem.score_summary` | `test_feasibility_checks_variable_domains`; property and exhaustive ModelIR tests |
| F04 | Raw, repaired, and selected candidates stay linked by IDs; selection is feasibility-first and monotone; a changed candidate cannot inherit an unrelated certificate or gap. | `SolveResult.candidates`, `qqa.api.solve` | `test_repair_is_retained_but_cannot_replace_a_better_verified_candidate` |
| F05 | A single monotonic deadline covers planning, presolve, search, repair, and certification; expired stages are skipped explicitly. | `qqa.runtime.SolveContext`, solver `time_limit` contracts | deadline tests in `test_audit_contracts.py` and `test_algebraic_benchmarks.py` |
| F06 | Pickle-free checkpoints retain latent state, optimizer, RNG, schedule, adaptive control, incumbent, population, augmented-Lagrangian state, and archives. Resume and extend semantics are distinct. | `qqa.runtime.checkpoint`, `qqa.annealing` | checkpoint parity, corruption, and resume tests in `test_next_generation_runtime.py` |
| F07 | Search merit and verified primal events are distinct, and event time is observed rather than interpolated. | `qqa.runtime.events` | `test_event_recorder_uses_observed_times_and_labels_search_merit` |
| F08 | Unknown objective, bound, and verification remain unavailable; they are never rendered as zero or verified. | `qqa.visuals.cockpit`, `SolveResult` | `test_cockpit_keeps_unknown_values_unknown` |
| F09 | Replica learning-rate and preconditioner scaling is applied to the effective optimizer update after Adam normalization. | `qqa.annealing._apply_optimizer_step_scale_` | `test_adam_role_scale_changes_the_actual_update` |
| F10 | Convexification is adjusted for objective normalization and problem dimension. | `qqa.runtime.population.estimate_convexification_beta` | `test_convexification_uses_objective_scale_and_dimension` and curvature oracle tests |
| F11 | Replica exchange is explicitly reported as `heuristic_role_exchange`; no detailed-balance or sampling guarantee is claimed for Adam QQA. | `qqa.runtime.population`, `qqa.annealing` diagnostics | exchange semantics and checkpoint tests |
| F12 | Augmented-Lagrangian state, archive ranking, restart control, and feasibility-first candidate keys remain device-resident until bounded reporting points. | `qqa.mixed.augmented_lagrangian`, `ModelIRProblem.incumbent_keys` | CUDA parity and device-residency tests |
| F13 | Built-in linear, quadratic, higher-order, cardinality, and clause factors are structurally conditioned during presolve instead of wrapped with full-dimensional reconstruction. | `qqa.model.presolve` | `test_structural_presolve_preserves_sparse_builtin_factors`; exhaustive equivalence tests |
| F14 | Dynamic search merit is separated from static original-objective incumbent ranking and the historical feasible archive. | `qqa.local.archive`, `qqa.annealing` | archive alignment and feasibility-first selection tests |
| F15 | Capability, plan, and actual execution are separate records; `require_qqa_primal=True` fails if no QQA primal stage executes. | `qqa.model.capabilities`, `qqa.portfolio`, `qqa.api.solve` | exact-only and required-QQA stage tests |
| F16 | Optimizer restart state uses in-place indexed assignment for selected replica rows. | `qqa.annealing._reset_optimizer_rows` | optimizer restart-state tests |
| F17 | The obsolete Polyfill dependency is absent; documentation builds strictly and browser smoke tests cover the application. | `mkdocs.yml`, documentation CI | strict MkDocs and GUI/browser smoke tests |
| F18 | Per-coordinate bounds project directly; representation, QQA search, repair, lower-bound, exact encoding, and proof-safe capabilities are reported independently. | `qqa.model.bounds`, `qqa.model.capabilities`, Model Doctor | `test_model_ir_problem_accepts_per_coordinate_bounds`; capability/planner tests |

## Verification layers

The repository enforces these contracts at several independent boundaries:

- exhaustive and property-based semantic tests for small models;
- CPU tests on every supported CPython version;
- opt-in CUDA correctness, parity, compilation, and precision tests;
- native SCIP/HiGHS/CP-SAT adapter tests when their extras are installed;
- wheel, source-distribution, CLI, GUI, strict documentation, and secret scans;
- cold-start public MIPLIB/QPLIB campaigns with original-coordinate solution
  validation, full failure accounting, and separate QQA intervention counts.

The machine-readable factor capability registry is authoritative. A factor
being representable does not imply that pure QQA can differentiate it, that a
GPU-fused implementation exists, that repair is available, or that an exact
backend can prove the model. `qqa.inspect`, `qqa.doctor`, and `qqa.plan` consume
the same registry so those distinctions remain visible before execution.

## Re-run the core audit gates

```bash
python -m pip install --upgrade "qqa[all]"
pytest -q tests/test_audit_contracts.py tests/test_semantic_p0.py
pytest -q tests/test_next_generation_runtime.py tests/test_algebraic_benchmarks.py
ruff check .
mypy --ignore-missing-imports --check-untyped-defs src
mkdocs build --strict
```

CUDA-only tests skip clearly on CPU hosts. Run them on a CUDA host with:

```bash
pytest -q tests/test_cuda_runtime.py tests/test_gpu_primitives.py tests/test_semantics.py
```
