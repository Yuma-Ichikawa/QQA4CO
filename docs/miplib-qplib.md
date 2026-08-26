# MIPLIB and QPLIB

QQA4CO provides a separate sparse algebraic path for public MIPLIB and QPLIB
instances. It does not route these models through the bounded callable
`MixedProblem` schema: infinite bounds, large sparse rows, quadratic
constraints, and original variable identities are retained directly.

Install the two optional readers/solvers:

```bash
pip install "qqa[scip,qplib]"
```

## Fetch and inspect an instance

The downloader uses the official public hosts, records the download timestamp,
source URL, byte count, and SHA-256 digest, and rejects unsafe ZIP members.

```bash
# One small instance
qqa benchmark fetch miplib --instance pk1 --output benchmarks/miplib
qqa benchmark fetch qplib --instance 31 --output benchmarks/qplib

# Complete public snapshots (large downloads)
qqa benchmark fetch miplib --output benchmarks/miplib
qqa benchmark fetch qplib --output benchmarks/qplib

qqa benchmark inspect benchmarks/miplib/pk1.mps.gz
qqa benchmark inspect benchmarks/qplib/QPLIB_0031.qplib
```

Result metadata contains only portable source names, public URLs, versions,
timestamps, and content hashes. It deliberately does not record hostnames,
absolute filesystem paths, private endpoints, or machine-specific server
configuration.

The official sources are the [MIPLIB download
page](https://miplib.zib.de/download.html) and the [QPLIB
site](https://qplib.zib.de/). Pin the generated `snapshot.json` alongside a
published result because reference solutions can be updated independently of
QQA4CO.

## Run SCIP or SG-CQQA

```bash
# Exact/global SCIP path
qqa benchmark run benchmarks/miplib/pk1.mps.gz \
  --solver scip --time-limit 60 --threads 1 --output pk1-scip.json

# SCIP-guided conditional QQA primal heuristic
qqa benchmark run benchmarks/miplib/pk1.mps.gz \
  --solver sg-cqqa --time-limit 60 --threads 1 \
  --core-size 64 --sol-size 32 --epochs 120 --max-calls 4 \
  --completion-time 1 --seed 0 --output pk1-sgcqqa.json

# QPLIB uses the same command
qqa benchmark run benchmarks/qplib/QPLIB_0031.qplib \
  --solver sg-cqqa --time-limit 60 --output qplib-31.json

# Several files: emits per-instance results plus overall/PROBTYPE summaries
qqa benchmark run benchmarks/qplib/QPLIB_*.qplib \
  --format qplib --solver sg-cqqa --time-limit 60 --output qplib-suite.json

# Paired comparison at equal budgets, including the native-heuristic ablation
qqa benchmark compare benchmarks/qplib/QPLIB_0031.qplib \
  --format qplib \
  --solvers scip scip-aggressive sg-cqqa \
  --baseline-solver scip-aggressive --seeds 0 1 2 \
  --time-limit 60 --threads 1 --reference-file qplib.solu \
  --output qplib-comparison.json --quiet

# Full, resumable campaigns checkpoint after every solver/instance/seed run
qqa benchmark compare benchmarks/miplib/instances/*.mps.gz \
  --format miplib --time-limit 60 --threads 1 --seeds 0 \
  --baseline-solver scip-aggressive \
  --reference-file benchmarks/miplib/miplib2017-v36.solu \
  --continue-on-error --output miplib-campaign.json --quiet

# Repeat the identical command with --resume after an interruption
qqa benchmark compare benchmarks/miplib/instances/*.mps.gz \
  --format miplib --time-limit 60 --threads 1 --seeds 0 \
  --baseline-solver scip-aggressive \
  --reference-file benchmarks/miplib/miplib2017-v36.solu \
  --continue-on-error --resume --output miplib-campaign.json --quiet

# Or run disjoint instance shards independently, then validate and merge them
qqa benchmark merge miplib-shard-*.json --output miplib-campaign.json
```

`--time-limit` is one total wall-clock budget. Input conversion and plugin
setup are deducted before SCIP starts; every QQA call and continuous
completion then runs inside SCIP's remaining solve time. This makes
`--solver scip` and `--solver sg-cqqa` comparable at a matched budget.
For a paired `compare` campaign, sparse algebraic import is common preparation
outside each solver's identical deadline. MIP input is parsed once and reused.
Each QPLIB solver run instead reparses the same public file in a disposable
worker before starting its solver clock. This prevents nonlinear native state
or allocator fragmentation from crossing instance/solver boundaries while
keeping the measured phases symmetric. Solver-model setup, plugin setup,
completion, QQA, and SCIP remain inside the matched deadline.

SG-CQQA is a primal heuristic, not a replacement for SCIP's proof machinery:

Models without binary or integer variables bypass QQA and use the same
aggressive-SCIP configuration as the direct ablation baseline. This keeps
continuous QPLIB comparisons matched while avoiding an empty plugin and
completion-model setup cost. The result records `qqa_applicable: false` and
`qqa_plugin_active: false`.

1. SCIP presolves the model and solves a root or node LP.
2. RENS is used before an incumbent exists; RINS-style agreement is used
   afterwards. For nonlinear models, the RINS centre is the best feasible
   point measured in the original algebraic model rather than an objective
   auxiliary maintained by SCIP.
3. A normalised score built from fractionality, incumbent disagreement,
   pseudocosts, and reduced costs selects a small uncertain integer core.
4. General integers are restricted to node-local `floor`/`ceil` or incumbent
   neighbourhoods. Wide global integer intervals are never represented by one
   high-frequency periodic penalty.
5. The original linear/quadratic objective and either original linear rows or
   selected active LP rows form a normalised core surrogate. Selected rows use
   independent PHR multipliers and penalty growth during QQA, with separate
   feasibility/objective archives. Cheap floor/ceil coordinate moves are
   tried first.
6. If the fast path does not improve the original incumbent and the configured
   time reserve remains, float64 QQA proposes a diverse population of integer
   assignments. In the primary hybrid, escalation also requires at least one
   fast candidate to have a feasible continuous completion; this prevents an
   expensive QQA call in a locally uncompletable neighbourhood. Setting
   `--fast-candidates 0` intentionally bypasses that gate for QQA-only
   ablations. The default requires more than 20 seconds remaining; change this
   explicitly with `--min-qqa-time` for long-run experiments.
7. An independent sub-SCIP fixes each assignment and completes all remaining
   continuous variables. If fixing the rounded core complement is infeasible,
   a bounded second-stage LNS repair keeps only the proposed core fixed and
   releases the complement. If that remains infeasible, a final broad LNS
   fixes only the highest-scored quarter of the core. All stages share the
   same completion time/node allowance and plugin overhead cap. Full solutions
   return through `trySol()`.
8. SCIP retains cuts, dual bounds, branch-and-bound, and certification.

The plugin runs only after useful LP-node timings, leaves a minimum time reserve
for SCIP, and caps calls, candidates, nodes, completion time, and total plugin
overhead. By default fast completion and QQA together may consume at most 10%
of SCIP's allotted time. If the first QQA call does not improve the original
incumbent, later QQA calls in that run are suppressed while fast LNS and SCIP
continue. This safeguard can be disabled explicitly for an ablation with
`--continue-qqa-without-improvement`. See the official [PySCIPOpt heuristic
tutorial](https://pyscipopt.readthedocs.io/en/latest/tutorials/heuristic.html)
and [model API](https://pyscipopt.readthedocs.io/en/latest/api/model.html).

## Python API

```python
from qqa.benchmarking import run_miplib, run_qplib
from qqa.hybrid import QQAHeuristicConfig
from qqa.io import load_mps, load_qplib

mip = load_mps("benchmarks/miplib/pk1.mps.gz")
qp = load_qplib("benchmarks/qplib/QPLIB_0031.qplib")
print(mip.summary())
print(qp.problem_type, qp.evaluate(qp.lower_bounds).maximum_infeasibility)

config = QQAHeuristicConfig(
    core_size=64,
    sol_size=32,
    epochs=120,
    max_calls=4,
    max_candidates=8,
    seed=0,
    threads=1,
)
result = run_miplib(
    "benchmarks/miplib/pk1.mps.gz",
    solver="sg-cqqa",
    time_limit=60,
    qqa_config=config,
)
result.to_dict()
```

`AlgebraicModel` stores linear and quadratic coefficients as SciPy CSR
matrices. `evaluate()` reports objective, constraint values, maximum row
violation, bound violation, integrality violation, and their maximum. QPLIB's
lower-triangle quadratic convention is converted to the symmetric Hessian of
`0.5 * x.T @ Q @ x` and cross-checked against `pyqplib` at both the supplied
initial point and a deterministic nonzero point.

## QPLIB routing and non-convexity

The three-character QPLIB `PROBTYPE` is retained in every result. Its second
character identifies continuous (`C`), binary (`B`), mixed-binary (`M`),
integer (`I`), or general mixed-integer (`G`) variables. The third character
describes the constraint class. Definitions and the official maximum
infeasibility metric follow the [QPLIB format
documentation](https://qplib.zib.de/doc.html).

For convex and linear continuous completions, SCIP solves the resulting
subproblem directly. The package also exposes sparse difference-of-convex
decomposition, eigenvalue shifting, and tangent concave linearisation in
`qqa.hybrid.nonconvex`; these are building blocks for non-convex QPLIB local
models. A feasible heuristic result does not imply global optimality. Report
`status`, `dual_bound`, and `gap`, and claim proof only when SCIP reports an
appropriate certified status.

## Metrics and reproducibility

`BenchmarkResult` includes:

- total runtime and SCIP solving time;
- time to first finite incumbent and the incumbent trajectory;
- original-space objective and maximum infeasibility;
- dual bound and SCIP-compatible relative gap;
- primal integral when a reference objective is supplied;
- node count, overall hybrid and QQA-only completion/acceptance rates, and QQA
  call timings;
- source basename, SHA-256, parser version, and reference snapshot name.

For nonlinear QPLIB models, every incumbent event is re-evaluated against the
original sparse objective and constraints. The tracker keeps the best feasible
original-space point monotonically even if SCIP's internal epigraph/hypograph
auxiliary contains slack. Reference error and primal integral are directional:
an incumbent better than the supplied reference has zero error, not a positive
absolute error.

Multi-file runs add overall and `PROBTYPE`-grouped feasible rates, median
runtime/time-to-first-feasible/gap/infeasibility, and aggregate QQA
completion/acceptance rates.

Use the same archive, reference file, total time, thread count, seed set, and
hardware class for comparisons. MIPLIB solution records and QPLIB solution
records can be passed with `--reference-file`; they are read as reference
values, never as a substitute for checking the returned point.
For a single `benchmark run`, runtime, time to first feasible, and primal
integral start before input parsing. In paired `compare`, one common algebraic
import is excluded from every solver equally; all three clocks then start
before solver-model construction. Primal integral always uses the configured
time limit as its common horizon. If model setup exhausts that limit, SCIP is
not started and the run is reported as `setup-time-limit`.
The thread option constrains both SCIP parallel workers and LP-solver threads;
SG-CQQA also applies it to Torch. For externally reproducible CPU runs, cap
BLAS/OpenMP threads in the execution environment as well.

`benchmark compare` automates the matched instance/seed runs. `scip` is the
library default, `scip-aggressive` enables SCIP's aggressive native heuristic
setting, and `sg-cqqa` uses that same setting plus the conditional plugin.
Therefore `scip-aggressive` is the direct ablation baseline. The output reports
paired final-primal-quality and anytime primal-integral win/tie/loss counts;
publish the complete JSON rather than only a favourable aggregate.

Long campaigns write the complete JSON atomically after every run when
`checkpoint_file` is supplied through Python, or automatically to `--output`
through the CLI. `--resume` first verifies the complete portable comparison
configuration, then skips finished tuples. `--continue-on-error` stores only
the source basename, format, solver, seed, and exception class; exception text
is deliberately omitted because it can contain a machine path. Use
`--retry-failures` with `--resume` after correcting an optional dependency or
solver issue.

QPLIB execution uses a disposable native-solver process for both the Python
API and CLI. A native crash or bounded worker timeout therefore cannot corrupt
the caller or an existing checkpoint. Campaign mode records that run as a
path-free failure and continues; normal process exit also releases large
nonlinear models before the next run. This isolation works from regular
scripts, notebooks, and interactive Python without requiring an
`if __name__ == "__main__"` guard.

To publish results without machine details, create deterministic compact JSON,
gzip-compressed full trajectories, and a hash manifest:

```python
from qqa.benchmarking import publish_benchmark_campaigns

publish_benchmark_campaigns(
    {
        "miplib": "miplib-campaign.json",
        "qplib": "qplib-campaign.json",
    },
    {
        "miplib": "benchmarks/miplib/snapshot.json",
        "qplib": "benchmarks/qplib/snapshot.json",
    },
    "public-results",
    implementation_revision="0123456789abcdef0123456789abcdef01234567",
)
```

Publication rejects absolute POSIX/Windows paths, loopback/private/link-local
addresses, local/internal host suffixes, and environment-specific metadata
keys before writing an artifact.

## Third-party reproduction checklist

The workflow requires no repository-specific directory layout:

1. Create a fresh Python 3.10+ environment and install
   `pip install "qqa[scip,qplib]"` (or a wheel built from the repository).
2. Fetch instances with `qqa benchmark fetch`, or download them from the
   official public MIPLIB/QPLIB hosts.
3. Keep the generated `snapshot.json` and public `.solu` snapshot with the
   result.
4. Run `benchmark compare` with explicit time, threads, seeds, solver list,
   baseline, and reference file.
5. Verify source hashes, package versions, feasibility, original objective,
   and full trajectories in the JSON before drawing a performance conclusion.

The emitted model, run, and comparison metadata use source basenames and
public content hashes only. Absolute paths, usernames, hostnames, private URLs,
and machine-local environment settings are intentionally excluded.
