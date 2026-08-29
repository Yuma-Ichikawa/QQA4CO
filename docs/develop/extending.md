# Extending QQA4CO

This page is the **single entry point** for everyone who wants to teach
QQA4CO a new problem, a new relaxation, a new schedule, a new callback,
or a whole new solver backend.

Everything below is fully covered by the public Python API — you do
**not** have to fork the package or modify any installed file. If you
*do* want to upstream your extension, the bottom of each section lists
the exact files you need to touch in a PR.

---

## TL;DR — five extension points

| You want to add … | Implement … | Read |
|---|---|---|
| A new combinatorial problem | a subclass of [`qqa.COProblem`](../api.md#qqa.problems.base) | §[Custom problem](#a-new-problem) |
| A new continuous lift of a discrete variable | a class that satisfies the [`qqa.Relaxation`](../api.md#qqa.relaxation) `Protocol` | §[Custom relaxation](#a-new-relaxation) |
| A new annealing schedule | any `Callable[[int, int], float]` | §[Custom schedule](#a-new-schedule) |
| A new training-time hook | a subclass of [`qqa.Callback`](../api.md#qqa.callbacks) | §[Custom callback](#a-new-callback) |
| A new solver backend (next to `qqa.anneal` and `qqa.pignn`) | a function returning [`qqa.AnnealResult`](../api.md#qqa.annealing.AnnealResult) | §[Custom backend](#a-new-solver-backend) |

QQA4CO's design philosophy is that **every extension point is a small,
pure-Python contract**, not a registration hook or a metaclass dance.
All five of the contracts above are listed in `src/qqa/relaxation.py`,
`src/qqa/problems/base.py`, `src/qqa/schedule.py`, `src/qqa/callbacks.py`
and `src/qqa/annealing.py` respectively, in well under 100 lines of code
each.

---

## A new problem

Subclass `qqa.COProblem` (or one of its more specific bases like
`qqa.QUBOProblem` / `qqa.SpinProblem`), attach a `relaxation`, and
implement `loss_fn`. That is enough for `qqa.anneal()` to drive it.

```python
import torch
from qqa import COProblem, BinaryRelaxation, anneal, fix_seed


class MaxOnes(COProblem):
    """Trivial example: maximise the number of 1-bits in an N-bit string."""

    def __init__(self, num_nodes: int, device: str = "cpu") -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.relaxation = BinaryRelaxation()

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (B, N); we minimise -sum(x), i.e. maximise sum(x).
        return -x.sum(dim=-1)


fix_seed(0)
result = anneal(MaxOnes(num_nodes=64), sol_size=128, num_epochs=500)
print(int(-result.best_obj))  # 64
```

Required attributes / methods:

* `self.num_nodes: int` (or `num_node` for categorical, or `num_spins`
  for spin) — the relaxation reads this to size the latent tensor.
* `self.relaxation: qqa.Relaxation` — pick `BinaryRelaxation()`,
  `SpinRelaxation()`, `CategoricalRelaxation()`, or your own.
* `loss_fn(self, x) -> torch.Tensor` — vectorised over the leading batch
  axis. Return shape `(B,)` for single-instance problems, `(B, I)` for
  batched-instance problems.

Strongly recommended:

* `score_summary(self, x_disc) -> dict` — returns a human-readable
  breakdown so the CLI and the Streamlit GUI can print *"MIS size: 22"*
  rather than *"loss: -22"*. The dict shape is documented in
  `qqa.problems.base.COProblem.score_summary`.

### Want to upstream it?

Open a PR that touches:

1. The right file under `src/qqa/problems/` (e.g. `qubo.py` for binary
   QUBO, `spin.py` for spin glasses, `extras.py` for everything else).
2. `src/qqa/problems/__init__.py` — add to the imports and `__all__`.
3. `src/qqa/__init__.py` — add to the imports and `__all__` so users
   can do `from qqa import YourProblem`.
4. `tests/test_extra_problems.py` (or the matching file) — at minimum a
   smoke test that constructs a tiny instance, anneals 50 epochs, and
   asserts the result is feasible.
5. `docs/problems.md` — add a row to the catalogue table.
6. *(optional)* `src/qqa/cli.py` — add a `--problem` choice and a
   matching branch in `_build_problem`. Skip this if your problem is
   esoteric; CLI users can always use `--problem-file`.
7. *(optional)* `scripts/verify_all_problems.py` — register a strong
   baseline so the verification sweep covers your problem.

---

## A new relaxation

Implement the `qqa.Relaxation` `Protocol`. The contract is exactly seven
methods (`init`, `forward`, `project`, `penalty`, `diversity`,
`perturb_`, `num_variables`) and is documented in
`src/qqa/relaxation.py`.

```python
import torch
from qqa import Relaxation


class TrinaryRelaxation:
    """Three-valued variables x ∈ {-1, 0, +1}."""

    def init(self, sol_size, problem, device):
        return torch.rand((sol_size, problem.num_nodes), device=device, requires_grad=True)

    def forward(self, x):
        # Map to (-1, +1) via tanh so loss_fn sees a smooth proxy.
        return torch.tanh(2.0 * (x - 0.5))

    def project(self, x):
        s = self.forward(x)
        return torch.where(s.abs() < 0.33, torch.zeros_like(s), torch.sign(s))

    def penalty(self, x, curve_rate):
        s = self.forward(x)
        # Penalise the |s| ≈ 0.5 region (continuous middle ground).
        return (1 - (3 * s.abs() - 1).clamp(min=0)).pow(curve_rate).sum(dim=-1)

    def diversity(self, x):
        return x.std(dim=0).sum()

    def perturb_(self, x, learning_rate, temp):
        if temp <= 0:
            return
        with torch.no_grad():
            x.add_(torch.randn_like(x) * (2 * learning_rate * temp) ** 0.5)
            x.clamp_(0.0, 1.0)

    def num_variables(self, problem):
        return problem.num_nodes
```

The `Protocol` is a structural type — Python checks the methods at use
time, not at class definition time, so you do **not** have to inherit
from anything. Pass an instance to your problem as `self.relaxation =
TrinaryRelaxation()` and `qqa.anneal()` will drive it.

### Want to upstream it?

1. Add the class to `src/qqa/relaxation.py` (or a sibling module).
2. Re-export it from `src/qqa/__init__.py`.
3. Add a unit test in `tests/test_problems.py` covering `init`,
   `project`, `penalty`, and `diversity` shape contracts.

---

## A new schedule

A schedule is *just* a callable `(epoch, num_epochs) -> float`. Pass any
function or `dataclass` with `__call__` to `anneal(schedule=...)`.

```python
import math
import qqa


def cosine_bg(epoch: int, num_epochs: int) -> float:
    """min_bg --(cosine)--> max_bg over T epochs."""
    if num_epochs <= 1:
        return 0.1
    t = epoch / (num_epochs - 1)
    return -2.0 + 2.1 * 0.5 * (1 - math.cos(math.pi * t))


qqa.anneal(problem, num_epochs=2000, schedule=cosine_bg)
```

For a stateful schedule (e.g. one that adapts to the current diversity)
prefer a [Callback](#a-new-callback) that mutates `state.bg`, since the
schedule itself is called once per epoch with no state.

---

## A new callback

Subclass `qqa.Callback` and override any of `on_train_begin`,
`on_epoch_end`, `on_train_end`. The single argument
[`CallbackState`](../api.md#qqa.callbacks.CallbackState) carries
`epoch`, `num_epochs`, `bg`, `x`, `losses`, `penalties`, `diversity`,
`best_obj`, `hyperparams` (a mutable dict), `problem`, `relaxation`,
and a free-form `extras` dict.

```python
from qqa import Callback


class EarlyStopOnBest(Callback):
    """Stop annealing once we hit a known best objective."""

    def __init__(self, target: float) -> None:
        self.target = target
        self.stopped = False

    def on_epoch_end(self, state):
        if not self.stopped and float(state.best_obj) <= self.target:
            self.stopped = True
            state.hyperparams["_stop_requested"] = True
```

Built-in callbacks worth reading as further examples:

* `qqa.HistoryRecorder` — accumulates per-epoch metrics (always added by
  default unless you pass `record_history=False`).
* `qqa.AutoDivTuner` — adapts `div_param` online to hit a target
  diversity ratio.
* `qqa.PopulationTracker` — snapshots the parallel population for
  post-hoc PCA / heatmap visualisation.
* `qqa.TrajectoryTracker` — tracks an auxiliary problem's objective
  during a penalised QUBO solve.

The annealing loop respects `state.hyperparams["div_param"]` and reads
`state.bg` for the current penalty weight, so callbacks have a real
control surface.

### Caveat — early stopping

The current `qqa.anneal()` loop does not honour an arbitrary
"stop-now" signal yet. If you need hard early stopping today, wrap
`anneal()` in your own loop and call it with `num_epochs=1` per outer
step, or open an issue / PR proposing the API.

---

## A new solver backend

The `qqa.pignn` subpackage is the canonical example of a third backend
sitting next to `qqa.anneal()`. The recipe is short:

1. **Take a `qqa.COProblem`** as the first positional argument.
2. **Run your training loop** (with whatever optimiser, GNN, schedule
   you like).
3. **Return a `qqa.AnnealResult`**, populating at least `best_sol`,
   `best_obj`, `runtime`, and (for graph problems) `score`.

```python
from time import time
import torch
from qqa import AnnealResult


def my_backend(problem, *, num_epochs: int = 1000, device: str = "cpu") -> AnnealResult:
    t0 = time()
    x = torch.rand(problem.num_nodes, device=device, requires_grad=True)
    optim = torch.optim.Adam([x], lr=1e-2)
    for _ in range(num_epochs):
        optim.zero_grad()
        loss = problem.loss_fn(x.unsqueeze(0)).sum()
        loss.backward()
        optim.step()
        x.data.clamp_(0.0, 1.0)
    bits = (x.detach() >= 0.5).float()
    best_obj = float(problem.loss_fn(bits.unsqueeze(0)).item())
    score = problem.score_summary(bits)
    return AnnealResult(best_sol=bits, best_obj=best_obj, runtime=time() - t0, score=score)
```

By returning `AnnealResult`, downstream tooling — the Streamlit GUI,
`qqa.visualization`, the CLI's `--output` pickle, the example notebooks
— all keep working with your backend.

### Heavy dependencies?

Mirror the `qqa.pignn` pattern:

* Put your code in a sub-package, never imported from
  `src/qqa/__init__.py` at top level.
* Add an `_import.py` with a `require_xxx()` helper that raises an
  `ImportError` whose message tells the user how to install the extra.
* Add a new `[project.optional-dependencies]` group in
  `pyproject.toml`.
* Skip the test file with `pytest.importorskip("your_dep")` so CI keeps
  passing without the heavy dep.

### Want to upstream it?

1. New sub-package `src/qqa/<your_backend>/` with `__init__.py`,
   `_import.py`, and the trainer module.
2. Optional dependency in `pyproject.toml`.
3. CLI hook in `src/qqa/cli.py` (add a `--backend <name>` choice).
4. Tests in `tests/test_<your_backend>.py`.
5. Documentation: `docs/reference/backends.md` (comparison table) and
   `docs/api.md` (mkdocstrings stanza).

---

## Where to look in the source

If you have not opened `src/qqa/` yet, the shortest tour is:

| File | Lines | What you get from reading it |
|---|---|---|
| `src/qqa/annealing.py` | ~300 | The whole annealer in one function — every extension point routes through here |
| `src/qqa/problems/base.py` | ~80 | The `COProblem` / `QUBOProblem` contracts |
| `src/qqa/relaxation.py` | ~220 | All four bundled relaxations + the `Protocol` |
| `src/qqa/callbacks.py` | ~170 | The full callback API + four worked examples |
| `src/qqa/schedule.py` | ~40 | The schedule contract is *literally* a callable |
| `src/qqa/pignn/trainer.py` | ~700 | A complete reference for "how do I write a new backend?" |

That is roughly 1500 lines for everything you need to understand to
extend the package end-to-end.

---

## Still stuck?

* Read [`docs/develop/internals.md`](internals.md) for a guided tour of
  the rest of the source tree.
* Skim [`tasks/lessons.md`](https://github.com/Yuma-Ichikawa/QQA4CO/blob/main/tasks/lessons.md)
  in the repository — every entry is a concrete pitfall the maintainers
  hit so you do not have to.
* Open an issue with the *"extension question"* template at
  <https://github.com/Yuma-Ichikawa/QQA4CO/issues>.
