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

You can reuse the `StreamlitCallback` from the Solve page in your own
Streamlit apps:

```python
from app.pages._common import StreamlitCallback  # adjust to your layout
```

Or instantiate your own subclass of :class:`qqa.callbacks.Callback`.
