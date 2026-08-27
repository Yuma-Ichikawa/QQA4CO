# Migration from 0.2.x

QQA 0.3 is backwards compatible with 0.2 at the Python level: the classes
you already use (`MaximumIndependentSet`, `MaxCut`, `Coloring`, …) are still
imported from the same `qqa.*` top-level namespace and the `qqa.anneal()`
signature has not changed.

## What changed

- **`qqa.problems` is now a subpackage.** Internally the classes live in
  `qqa/problems/{qubo,categorical,spin}.py`, but `from qqa import
  MaximumIndependentSet` keeps working.
- **New problem classes** are available: `Ising1D`, `EdwardsAnderson`,
  `SherringtonKirkpatrick`, `BinaryPerceptron`, `HopfieldMemory`, plus the
  supporting `SpinRelaxation`.
- **`qqa.visualization`** now accepts a `backend=` argument and exposes
  `plot_best_trajectory`, `plot_schedule`, `plot_run_comparison`,
  `plot_parallel_coordinates`, `plot_solution_heatmap` in addition to
  `plot_history`.
- **CLI** (`qqa version / solve / bench / gui`) and **Streamlit GUI** are
  new in 0.3.
- **`HistoryRecorder`** now records `best_obj` and `loss_min` per epoch in
  addition to the existing fields.

## Deprecations

`qqa.legacy.*` (wrappers for the old `batch_annealing_*` functions) keeps
working and emits `DeprecationWarning`. Port to `qqa.anneal()` when you get
a chance. These wrappers are scheduled for removal in QQA 1.0.0; the stable
`qqa.solve()` / `qqa.anneal()` APIs and result contracts replace them.

## Checklist

- [ ] Replace `batch_annealing(...)` calls with `qqa.anneal(problem, ...)`.
- [ ] If you used a custom plotting helper, consider switching to
      `qqa.visualization.plot_history(result, backend="plotly")`.
- [ ] If you want spin problems, use the new `qqa.SpinRelaxation` via one
      of the `Ising1D` / `EdwardsAnderson` / `SherringtonKirkpatrick` /
      `BinaryPerceptron` / `HopfieldMemory` classes.
