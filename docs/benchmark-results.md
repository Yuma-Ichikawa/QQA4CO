# Public MIPLIB/QPLIB results

This page reports the complete 2026-08-26 campaign without selecting only
favourable instances. Machine-readable compact results, full incumbent
trajectories, snapshot metadata, implementation revision, and SHA-256
manifests are committed under
[`benchmark-results/2026-08-26`](https://github.com/Yuma-Ichikawa/QQA4CO/tree/main/benchmark-results/2026-08-26).

## Protocol

- The primary comparison is `sg-cqqa` against the matched
  `scip-aggressive` ablation. Both use one SCIP/LP thread, seed 0, the same
  public instance, reference snapshot, and a 30-second total solver budget.
- MIPLIB contains all 240 benchmark-set instances. QPLIB contains all 453
  public instances. MIPLIB SG-CQQA used the generic `cuda` device class;
  QPLIB used `cpu`. No machine, scheduler, host, or private path metadata is
  retained.
- Final-quality W/T/L first compares feasibility and then directional
  reference error (or the original objective when no reference is available).
  Primal-integral W/T/L uses a common 30-second horizon; lower is better.
- QPLIB's sparse algebraic import is common preparation outside each paired
  solver clock. Solver-model construction, plugin setup, QQA, completion, and
  SCIP are inside the budget. Each native run is process-isolated.
- The QQA-only ablation sets `fast_candidates=0` and permits 20% heuristic
  overhead. The primary hybrid uses two fast candidates, a continuous-
  completion escalation gate, and a 10% overhead cap.

## Primary full-suite comparison

| Library | Paired instances | Feasible baseline → SG-CQQA | Final W/T/L | Integral W/T/L | Median primal error baseline → SG-CQQA | Median integral baseline → SG-CQQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIPLIB | 240/240 | 177 → 177 | 6 / 194 / 40 | 12 / 116 / 112 | 0.07450 → 0.08511 | 25.573 → 26.875 |
| QPLIB | 437/453 | 348 → 351 | 43 / 361 / 33 | 145 / 133 / 159 | 0.06129 → 0.07407 | 14.669 → 14.144 |

On QPLIB's 437 successfully paired instances, the primary hybrid has more
final-quality wins than losses (43 versus 33), finds three additional feasible
solutions, and lowers the median primal integral. It does not dominate every
anytime comparison: integral wins/losses are 145/159, and its median final
reference error is higher.

On MIPLIB, the 30-second primary hybrid is weaker than aggressive SCIP: six
final wins versus 40 losses, with equal feasible counts. This negative result
is retained in full. The primary MIPLIB candidate path generated 631
candidates; 47.70% completed feasibly, 44.37% were accepted, and 6.97%
improved the original incumbent. The QQA subset generated 175 candidates with
76.57% completion, 70.29% acceptance, and 2.86% incumbent improvement.

For QPLIB, the primary path generated 489 candidates with 39.47% feasible
completion, 38.85% acceptance, and 14.72% incumbent improvement. Its 63 QQA
candidates had 46.03% completion, 44.44% acceptance, and 3.17% incumbent
improvement.

## QQA-only ablation

| Library | Paired instances | Paired feasible baseline → SG-CQQA | Final W/T/L | Integral W/T/L | QQA acceptance | QQA incumbent improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIPLIB | 240/240 | 177 → 177 | 4 / 191 / 45 | 6 / 115 / 119 | 43.51% | 7.42% |
| QPLIB | 437/453 | 350 → 350 | 23 / 361 / 53 | 97 / 135 / 205 | 36.09% | 10.06% |

QQA-only loses more often than the gated primary hybrid on both libraries.
The result supports using QQA conditionally inside SCIP, with cheap surrogate
candidates and completion evidence, rather than invoking it as an unconditional
replacement for native primal heuristics.

## Additional diagnostics

The five-second CPU MIPLIB screening compares plain SCIP, aggressive SCIP,
and the primary hybrid on all 240 instances. Against aggressive SCIP,
SG-CQQA records 8/226/6 final W/T/L but 13/166/61 primal-integral W/T/L;
feasible counts are 139 versus 138. The configured minimum QQA reserve prevents
QQA calls at this short horizon, so this is a fast-surrogate screening result,
not evidence for the learned QQA path.

A two-instance QPLIB diagnostic uses three seeds on one binary `QBL` and one
mixed-binary/continuous `QML` instance. Across six pairs, SG-CQQA records
4/0/2 for both final quality and primal integral. This small, previously
explored diagnostic is published for seed sensitivity only and is not used as
an estimate of full-library performance.

## Failures and limits

All requested runs were attempted. MIPLIB completed 480/480 primary and
480/480 QQA-only runs without failure. In the QPLIB primary campaign,
874/906 runs completed and 32 anonymous failures covered the same 16 instances
for both solvers: 28 bounded worker timeouts and four native-process failures.
The QQA-only campaign completed 875/906 runs with 31 failures: 28 worker
timeouts and three native-process failures. Consequently 437/453 QPLIB
instances have direct paired measurements in each campaign.

These failures are not counted as wins or losses. They are preserved by
basename, solver, seed, and error class in the artifacts. Native isolation
keeps a failed nonlinear or exceptionally large model from terminating the
remaining campaign; it does not turn an unmeasured run into a result.

The benchmark establishes a QPLIB final-quality advantage for this fixed
30-second primary campaign, but not universal superiority. In particular,
MIPLIB and the QQA-only ablations remain negative, one seed is insufficient
for a broad stochastic claim, and longer budgets may change the ranking.

## Snapshot identity

- MIPLIB 2017 benchmark-v2 archive: `c756eefd544d83b31809306b45d3549a1a5b9378e6aa78b68738b1a3b6a418fa`
- MIPLIB solution snapshot v36: `9236602294c1a5aca5248b6f7d03689a7533ea3cbf3f3d92cff962e752e20af84`
- QPLIB public archive: `b3596e1264ed57c5f6a44e822679f5c9138e1985fe74bc8341c3becbc666b9fd`
- QPLIB solution snapshot: `6e024cd786b3049e572a96fbe89ddd0d73ff4d6bcac0c0427e9086d378dc0c43`

The snapshots were retrieved from the official public sources on 2026-08-25
(UTC). See the [MIPLIB download page](https://miplib.zib.de/download.html),
[MIPLIB benchmark set](https://miplib.zib.de/set_benchmark.html),
[QPLIB instances](https://qplib.zib.de/instances.html), and
[QPLIB format documentation](https://qplib.zib.de/doc.html).
