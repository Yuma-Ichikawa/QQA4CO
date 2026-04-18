"""Visualisation helpers for :class:`~qqa.annealing.AnnealResult`.

Every plotting function accepts a ``backend`` argument:

* ``"matplotlib"`` (default) — static figures, no optional deps.
* ``"plotly"`` — interactive figures, requires ``pip install qqa[plotly]``.

If Plotly is not installed and ``backend="plotly"`` is requested, the
functions automatically fall back to matplotlib with a warning.

Plot catalog:

* :func:`plot_history` — loss / penalty / diversity dynamics.
* :func:`plot_best_trajectory` — best objective value over epochs.
* :func:`plot_schedule` — the annealing schedule ``bg(epoch)``.
* :func:`plot_run_comparison` — overlay multiple runs.
* :func:`plot_parallel_coordinates` — hyper-parameter sweep view.
* :func:`plot_solution_heatmap` — spins / bits of the best solution.
* :func:`plot_population_evolution` — parallel-population loss heat-map.
* :func:`plot_population_embedding` — PCA trajectory of the population.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from qqa.annealing import AnnealResult
    from qqa.schedule import LinearBGSchedule


def _have_plotly() -> bool:
    try:
        importlib.import_module("plotly.graph_objects")
        return True
    except Exception:
        return False


def _resolve_backend(backend: str) -> str:
    """Return the backend to actually use, warning on fallback."""
    if backend == "plotly" and not _have_plotly():
        warnings.warn(
            "Plotly backend requested but 'plotly' is not installed. "
            "Install with `pip install qqa[plotly]`. Falling back to matplotlib.",
            stacklevel=3,
        )
        return "matplotlib"
    if backend not in ("matplotlib", "plotly"):
        raise ValueError(f"Unknown backend {backend!r}. Use 'matplotlib' or 'plotly'.")
    return backend


# ---------------------------------------------------------------------------
# plot_history
# ---------------------------------------------------------------------------


def plot_history(
    result: AnnealResult,
    title: str = "QQA dynamics",
    backend: str = "matplotlib",
    show: bool = True,
):
    """Plot mean loss, mean penalty and diversity across epochs.

    Returns the backend-native figure object (``(fig, axes)`` for matplotlib,
    ``go.Figure`` for plotly).
    """
    h = result.history
    if not h:
        raise ValueError("No history recorded. Pass record_history=True to anneal().")
    backend = _resolve_backend(backend)
    if backend == "matplotlib":
        return _plot_history_mpl(h, title, show)
    return _plot_history_plotly(h, title, show)


def _plot_history_mpl(h: dict, title: str, show: bool):
    import matplotlib.pyplot as plt

    epochs = np.arange(len(h["loss_mean"]))
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    for ax in axs:
        ax.grid(ls="--", alpha=0.6)

    mean_l = np.asarray(h["loss_mean"])
    std_l = np.asarray(h["loss_std"])
    axs[0].plot(epochs, mean_l, label="Mean Loss", color="darkblue", lw=2)
    axs[0].fill_between(epochs, mean_l - std_l, mean_l + std_l, alpha=0.2, color="cornflowerblue")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    mean_p = np.asarray(h["penalty_mean"])
    std_p = np.asarray(h["penalty_std"])
    axs[1].plot(epochs, mean_p, label="Mean Penalty", color="firebrick", lw=2)
    axs[1].fill_between(epochs, mean_p - std_p, mean_p + std_p, alpha=0.2, color="salmon")
    axs[1].set_title("Penalty")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Penalty")
    axs[1].legend()

    div = np.asarray(h["diversity"])
    axs[2].plot(epochs, div, label="Diversity", color="green", lw=2)
    axs[2].set_title("Diversity")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Diversity")
    axs[2].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axs


def _plot_history_plotly(h: dict, title: str, show: bool):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    epochs = np.arange(len(h["loss_mean"]))
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Loss", "Penalty", "Diversity"))

    mean_l = np.asarray(h["loss_mean"])
    std_l = np.asarray(h["loss_std"])
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([epochs, epochs[::-1]]),
            y=np.concatenate([mean_l + std_l, (mean_l - std_l)[::-1]]),
            fill="toself",
            fillcolor="rgba(100, 149, 237, 0.25)",
            line={"color": "rgba(0,0,0,0)"},
            name="Loss ±1σ",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=mean_l, mode="lines", name="Mean Loss", line={"color": "darkblue"}),
        row=1,
        col=1,
    )

    mean_p = np.asarray(h["penalty_mean"])
    std_p = np.asarray(h["penalty_std"])
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([epochs, epochs[::-1]]),
            y=np.concatenate([mean_p + std_p, (mean_p - std_p)[::-1]]),
            fill="toself",
            fillcolor="rgba(250, 128, 114, 0.25)",
            line={"color": "rgba(0,0,0,0)"},
            name="Penalty ±1σ",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=mean_p, mode="lines", name="Mean Penalty", line={"color": "firebrick"}
        ),
        row=1,
        col=2,
    )

    div = np.asarray(h["diversity"])
    fig.add_trace(
        go.Scatter(x=epochs, y=div, mode="lines", name="Diversity", line={"color": "green"}),
        row=1,
        col=3,
    )

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 18}},
        template="plotly_white",
        height=420,
        width=1200,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Epoch")
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# plot_best_trajectory
# ---------------------------------------------------------------------------


def plot_best_trajectory(
    result: AnnealResult,
    title: str = "Best objective per epoch",
    backend: str = "matplotlib",
    show: bool = True,
):
    """Plot ``best_obj`` vs epoch (monotonically non-increasing)."""
    h = result.history
    if not h or "best_obj" not in h or len(h["best_obj"]) == 0:
        raise ValueError(
            "No best-objective history recorded. Pass record_history=True to anneal()."
        )
    backend = _resolve_backend(backend)
    best = np.asarray(h["best_obj"], dtype=float)
    if best.ndim > 1:
        best = best.mean(axis=-1)
    epochs = np.arange(len(best))

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
        ax.plot(epochs, best, color="#333", lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Best objective")
        ax.set_title(title)
        ax.grid(ls="--", alpha=0.5)
        if show:
            plt.show()
        return fig, ax

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=best, mode="lines", name="Best objective"))
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Epoch",
        yaxis_title="Best objective",
        template="plotly_white",
        height=420,
    )
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# plot_schedule
# ---------------------------------------------------------------------------


def plot_schedule(
    schedule: LinearBGSchedule,
    num_epochs: int,
    title: str = "Annealing schedule",
    backend: str = "matplotlib",
    show: bool = True,
):
    """Visualise the ``bg`` annealing schedule over ``num_epochs``."""
    backend = _resolve_backend(backend)
    epochs = np.arange(num_epochs)
    bg = np.asarray([float(schedule(int(e), num_epochs)) for e in epochs])

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        ax.plot(epochs, bg, color="purple", lw=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("bg")
        ax.set_title(title)
        ax.grid(ls="--", alpha=0.5)
        if show:
            plt.show()
        return fig, ax

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=bg, mode="lines", line={"color": "purple"}))
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Epoch",
        yaxis_title="bg",
        template="plotly_white",
        height=400,
    )
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# plot_run_comparison
# ---------------------------------------------------------------------------


def plot_run_comparison(
    results: list[AnnealResult],
    labels: list[str] | None = None,
    title: str = "Run comparison",
    backend: str = "matplotlib",
    show: bool = True,
):
    """Overlay ``best_obj`` trajectories from multiple runs."""
    if labels is None:
        labels = [f"run {i}" for i in range(len(results))]
    if len(labels) != len(results):
        raise ValueError("labels must have the same length as results")

    series: list[tuple[str, np.ndarray]] = []
    for lab, r in zip(labels, results, strict=True):
        if not r.history or "best_obj" not in r.history:
            continue
        best = np.asarray(r.history["best_obj"], dtype=float)
        if best.ndim > 1:
            best = best.mean(axis=-1)
        series.append((lab, best))
    if not series:
        raise ValueError("No runs had a recorded best_obj history.")

    backend = _resolve_backend(backend)
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
        for lab, best in series:
            ax.plot(np.arange(len(best)), best, lw=1.8, label=lab)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Best objective")
        ax.set_title(title)
        ax.grid(ls="--", alpha=0.5)
        ax.legend()
        if show:
            plt.show()
        return fig, ax

    import plotly.graph_objects as go

    fig = go.Figure()
    for lab, best in series:
        fig.add_trace(go.Scatter(x=np.arange(len(best)), y=best, mode="lines", name=lab))
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Epoch",
        yaxis_title="Best objective",
        template="plotly_white",
        height=460,
    )
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# plot_parallel_coordinates
# ---------------------------------------------------------------------------


def plot_parallel_coordinates(
    sweep_df: Any,
    objective: str = "best_obj",
    title: str = "Hyperparameter sweep",
    backend: str = "plotly",
    show: bool = True,
):
    """Parallel-coordinates plot of a hyper-parameter sweep.

    ``sweep_df`` is expected to be a ``pandas.DataFrame`` (or any object with a
    ``to_dict(orient="list")`` method) whose columns are the hyper-parameters
    plus one ``objective`` column.

    The Plotly backend produces a coloured interactive figure (recommended).
    Matplotlib falls back to a simple scatter-matrix-like rendering.
    """
    if hasattr(sweep_df, "to_dict"):
        data = sweep_df.to_dict(orient="list")
    elif isinstance(sweep_df, dict):
        data = dict(sweep_df)
    else:
        raise TypeError("sweep_df must be a pandas DataFrame or a dict of equal-length lists.")

    if objective not in data:
        raise KeyError(f"objective column {objective!r} not found in sweep_df.")

    backend = _resolve_backend(backend)
    if backend == "plotly":
        import plotly.graph_objects as go

        obj_vals = np.asarray(data[objective], dtype=float)
        dims = []
        for k, v in data.items():
            arr = np.asarray(v, dtype=float)
            dims.append({"label": k, "values": arr})
        fig = go.Figure(
            data=go.Parcoords(
                line={"color": obj_vals, "colorscale": "Viridis", "showscale": True},
                dimensions=dims,
            )
        )
        fig.update_layout(
            title={"text": title, "x": 0.5},
            template="plotly_white",
            height=500,
        )
        if show:
            fig.show()
        return fig

    import matplotlib.pyplot as plt

    keys = [k for k in data if k != objective]
    obj_vals = np.asarray(data[objective], dtype=float)
    fig, axs = plt.subplots(1, len(keys), figsize=(3.5 * max(1, len(keys)), 4.5), facecolor="white")
    if len(keys) == 1:
        axs = [axs]
    for ax, k in zip(axs, keys, strict=True):
        ax.scatter(np.asarray(data[k], dtype=float), obj_vals, c=obj_vals, cmap="viridis")
        ax.set_xlabel(k)
        ax.set_ylabel(objective)
        ax.grid(ls="--", alpha=0.5)
    fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axs


# ---------------------------------------------------------------------------
# plot_solution_heatmap
# ---------------------------------------------------------------------------


def plot_solution_heatmap(
    result: AnnealResult,
    problem: Any = None,
    title: str = "Best solution",
    backend: str = "matplotlib",
    show: bool = True,
):
    """Render the best discrete solution as a 1D/2D heatmap.

    For lattice spin problems (``EdwardsAnderson`` with ``dim == 2``) the
    solution is reshaped to ``(L, L)`` automatically.
    """
    sol = result.best_sol
    if hasattr(sol, "detach"):
        sol = sol.detach().cpu().numpy()
    sol = np.asarray(sol)
    if sol.ndim == 2:
        # For a single-instance problem best_sol is (sol_size, N): pick the
        # first (most diverse) row for display.
        if problem is not None and getattr(problem, "num_instance", None) is not None:
            arr = sol
        else:
            arr = sol[0][None, :]
    elif sol.ndim == 1:
        arr = sol[None, :]
    else:
        raise ValueError(f"Unsupported solution shape: {sol.shape}")

    # Try to reshape lattice problems to 2D for clarity.
    if (
        problem is not None
        and hasattr(problem, "dim")
        and getattr(problem, "dim", None) == 2
        and hasattr(problem, "L")
    ):
        L = int(problem.L)
        if arr.shape[-1] == L * L:
            arr = arr[0].reshape(L, L)
            arr = arr[None, :, :]  # add leading axis for consistent downstream

    backend = _resolve_backend(backend)
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        data = arr if arr.ndim == 2 else arr.squeeze()
        if data.ndim == 1:
            data = data[None, :]
        im = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    import plotly.graph_objects as go

    data = arr if arr.ndim == 2 else arr.squeeze()
    if getattr(data, "ndim", 2) == 1:
        data = data[None, :]
    fig = go.Figure(data=go.Heatmap(z=data, colorscale="RdBu", zmin=-1, zmax=1, zmid=0))
    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        height=400,
    )
    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# plot_population_evolution — parallel search visualiser
# ---------------------------------------------------------------------------


def plot_population_evolution(
    tracker: Any,
    title: str = "Parallel population loss",
    backend: str = "plotly",
    show: bool = True,
):
    """Render the parallel population's loss landscape across epochs.

    ``tracker`` must be a :class:`~qqa.callbacks.PopulationTracker` instance
    that captured snapshots during the run. Each column of the resulting
    heat-map is one snapshot epoch, each row is one replica in the
    ``sol_size`` population, and colour encodes loss. Rows are sorted once,
    by final-epoch loss, to keep the panel readable.

    The best trajectory is overlaid as a thin white curve.
    """
    if not tracker.loss:
        raise ValueError("PopulationTracker recorded no snapshots.")
    loss = np.stack(tracker.loss, axis=1)  # (sol_size, T)
    epochs = np.asarray(tracker.epochs)
    order = np.argsort(loss[:, -1])
    loss_sorted = loss[order]
    best_traj = loss.min(axis=0)

    backend = _resolve_backend(backend)
    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=loss_sorted,
                x=epochs,
                colorscale="Viridis",
                colorbar={"title": "loss"},
                zsmooth="best",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=np.full_like(epochs, 0, dtype=float),  # drawn on top band
                mode="lines",
                line={"color": "rgba(255,255,255,0.0)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=best_traj,
                mode="lines",
                name="best-of-batch",
                yaxis="y2",
                line={"color": "#f8fafc", "width": 2.5},
            )
        )
        fig.update_layout(
            title={"text": title, "x": 0.5},
            template="plotly_dark",
            height=460,
            xaxis_title="Epoch",
            yaxis_title="Replica (sorted by final loss)",
            yaxis2={
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "title": "best loss",
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        if show:
            fig.show()
        return fig

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    im = ax.imshow(
        loss_sorted,
        aspect="auto",
        cmap="viridis",
        extent=[epochs[0], epochs[-1], 0, loss_sorted.shape[0]],
    )
    ax2 = ax.twinx()
    ax2.plot(epochs, best_traj, color="black", lw=1.5, label="best-of-batch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Replica (sorted)")
    ax2.set_ylabel("best loss")
    fig.colorbar(im, ax=ax, label="loss")
    ax.set_title(title)
    if show:
        plt.show()
    return fig, (ax, ax2)


# ---------------------------------------------------------------------------
# plot_population_embedding — PCA trajectory of the replicas
# ---------------------------------------------------------------------------


def plot_population_embedding(
    tracker: Any,
    title: str = "Population PCA trajectory",
    backend: str = "plotly",
    show: bool = True,
):
    """2D PCA of the parallel population's continuous variables over time.

    Projects every snapshot of ``tracker.x`` (shape ``(sol_size, N, ...)``)
    onto the 2 principal components computed from the concatenation of all
    snapshots, then draws the resulting trajectory as a scatter coloured by
    epoch with replica paths drawn as light-grey lines.

    Requires :class:`~qqa.callbacks.PopulationTracker` to have been run with
    ``record_x=True``.
    """
    if not tracker.x:
        raise ValueError("PopulationTracker has no x-snapshots; instantiate it with record_x=True.")
    # Flatten each snapshot to (sol_size, D)
    snaps = [np.asarray(x).reshape(x.shape[0], -1) for x in tracker.x]
    sol_size = snaps[0].shape[0]
    X = np.concatenate(snaps, axis=0)  # (T*sol_size, D)
    X = X - X.mean(axis=0, keepdims=True)
    # PCA via SVD (cap dim for speed).
    if X.shape[1] > 256:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[1], size=256, replace=False)
        Xp = X[:, idx]
    else:
        Xp = X
    _u, _s, vh = np.linalg.svd(Xp, full_matrices=False)
    components = vh[:2]  # (2, d')
    Y = Xp @ components.T  # (T*sol_size, 2)
    Y = Y.reshape(len(snaps), sol_size, 2)  # (T, sol_size, 2)
    epochs = np.asarray(tracker.epochs)

    backend = _resolve_backend(backend)
    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()
        # Faint per-replica paths
        for r in range(sol_size):
            fig.add_trace(
                go.Scatter(
                    x=Y[:, r, 0],
                    y=Y[:, r, 1],
                    mode="lines",
                    line={"color": "rgba(148, 163, 184, 0.15)", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        # Scatter points coloured by epoch
        t_grid = np.broadcast_to(epochs[:, None], (len(epochs), sol_size))
        fig.add_trace(
            go.Scatter(
                x=Y[..., 0].ravel(),
                y=Y[..., 1].ravel(),
                mode="markers",
                marker={
                    "size": 5,
                    "color": t_grid.ravel(),
                    "colorscale": "Plasma",
                    "colorbar": {"title": "Epoch"},
                    "opacity": 0.85,
                },
                showlegend=False,
            )
        )
        fig.update_layout(
            title={"text": title, "x": 0.5},
            template="plotly_dark",
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        if show:
            fig.show()
        return fig

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    for r in range(sol_size):
        ax.plot(Y[:, r, 0], Y[:, r, 1], color="grey", alpha=0.15, lw=1)
    t_grid = np.broadcast_to(epochs[:, None], (len(epochs), sol_size))
    sc = ax.scatter(Y[..., 0].ravel(), Y[..., 1].ravel(), c=t_grid.ravel(), cmap="plasma", s=10)
    fig.colorbar(sc, ax=ax, label="Epoch")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    if show:
        plt.show()
    return fig, ax
