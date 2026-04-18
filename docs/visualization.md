# Visualization

`qqa.visualization` offers consistent plotting helpers that accept a
`backend="matplotlib" | "plotly"` argument. Plotly requires the optional
extra `pip install qqa[plotly]`; missing plotly automatically falls back to
matplotlib with a warning.

## Catalog

```python
from qqa import visualization as viz

viz.plot_history(result)                 # loss / penalty / diversity
viz.plot_best_trajectory(result)         # best objective over epochs
viz.plot_schedule(schedule, num_epochs)  # bg annealing schedule
viz.plot_run_comparison(results, labels=[...])
viz.plot_parallel_coordinates(df, objective="best_obj", backend="plotly")
viz.plot_solution_heatmap(result, problem)
```

## Example: compare multiple runs

```python
import qqa
from qqa import visualization as viz

problem = qqa.SherringtonKirkpatrick(N=80, seed=0)
runs = [
    qqa.anneal(problem, num_epochs=600, min_bg=-3, verbose=False),
    qqa.anneal(problem, num_epochs=600, min_bg=-1, verbose=False),
]
viz.plot_run_comparison(runs, labels=["min_bg=-3", "min_bg=-1"])
```

## Integration with dashboards

All Plotly figures are raw `plotly.graph_objects.Figure` instances, so
they drop straight into `st.plotly_chart(fig)`.

## Gallery

Figures below are regenerated deterministically by
`scripts/make_gallery.py`. The full set (eight problems × four plot kinds
plus the annealing schedule) lives under `data/fig/gallery/` and is
referenced from the top-level README. The MkDocs site loads the PNGs
directly from GitHub's raw endpoint so the canonical copies stay in
`data/` without duplication inside `docs/`.

### Schedule

![Default LinearBGSchedule](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/schedule_default.png)

### Per-problem dynamics

| Problem | History | Solution | Population |
| --- | --- | --- | --- |
| MIS         | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/history_mis.png)         | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/solution_mis.png)        | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/population_mis.png)        |
| Max-Cut     | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/history_maxcut.png)      | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/solution_maxcut.png)     | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/population_maxcut.png)     |
| Coloring    | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/history_coloring.png)    | —                                                                                                          | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/population_coloring.png)   |
| Ising 1D    | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/history_ising1d.png)     | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/solution_ising1d.png)    | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/population_ising1d.png)    |
| EA 3D       | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/history_ea3d.png)        | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/solution_ea3d.png)       | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/population_ea3d.png)       |
| SK          | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/history_sk.png)          | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/solution_sk.png)         | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/population_sk.png)         |
| Perceptron  | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/history_perceptron.png)  | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/solution_perceptron.png) | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/population_perceptron.png) |
| Hopfield    | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/history_hopfield.png)    | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/solution_hopfield.png)   | ![](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/population_hopfield.png)   |

Regenerate with `uv run python scripts/make_gallery.py`.
