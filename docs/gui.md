# GUI

The Streamlit dashboard exposes QQA as an interactive tool with four pages.

```bash
pip install "qqa[gui]"
qqa gui
# or equivalently:
streamlit run app/streamlit_app.py
```

## Pages

### Home

Pick a problem family, size, seed, and problem-specific parameters. A
preview is rendered as a graph (for combinatorial problems), a coupling
heatmap (for spin problems), or a pattern matrix (for the binary
perceptron).

### Solve

Tune QQA hyper-parameters (`sol_size`, `learning_rate`, `temp`, `min_bg`,
`max_bg`, `curve_rate`, `div_param`, `num_epochs`) and hit **Run QQA** to
launch the anneal. A `StreamlitCallback` streams the progress bar, current
metrics, and a live loss/best plot.

### Visualize

Four tabs — dynamics, best trajectory, schedule, solution heatmap — all
backed by the :mod:`qqa.visualization` helpers in their Plotly flavour.

### Compare

Run a small grid over `min_bg × max_bg × div_param` on the current problem
and inspect the outcome with a parallel-coordinates plot plus an overlaid
best-objective trajectory.

## Programmatic access

`StreamlitCallback` lives inside the Solve page (it depends on
Streamlit's runtime and session state). If you want to reuse it in your
own Streamlit app, copy the class out of `app/pages/1_Solve.py` or
import it from there:

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "solve_page", Path("app/pages/1_Solve.py")
)
solve_page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solve_page)
cb = solve_page.StreamlitCallback(...)
```

For non-Streamlit use cases, just subclass :class:`qqa.callbacks.Callback`
directly — the base class is framework-agnostic.
