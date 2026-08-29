"""Backend-neutral data extraction for advanced result diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def solution_rows(result: Any, problem: Any | None = None) -> list[dict[str, Any]]:
    """Flatten a solution into labelled, normalised plotting rows."""
    solution = _as_numpy(result.best_sol).reshape(-1)
    space = getattr(problem, "space", None)
    if space is None:
        lo = float(np.nanmin(solution)) if solution.size else 0.0
        hi = float(np.nanmax(solution)) if solution.size else 1.0
        span = hi - lo if hi > lo else 1.0
        return [
            {
                "label": f"x[{index}]",
                "group": "x",
                "kind": "variable",
                "value": float(value),
                "lower": lo,
                "upper": hi,
                "normalised": float((value - lo) / span),
            }
            for index, value in enumerate(solution)
        ]

    rows: list[dict[str, Any]] = []
    for metadata in space.describe():
        start = int(metadata["start"])
        stop = int(metadata["stop"])
        lower = float(metadata["lower"])
        upper = float(metadata["upper"])
        span = upper - lower
        for local_index, value in enumerate(solution[start:stop]):
            label = str(metadata["name"])
            if int(metadata["size"]) > 1:
                label = f"{label}[{local_index}]"
            rows.append(
                {
                    "label": label,
                    "group": metadata["name"],
                    "kind": metadata["kind"],
                    "value": float(value),
                    "lower": lower,
                    "upper": upper,
                    "normalised": float(np.clip((value - lower) / span, 0.0, 1.0)),
                }
            )
    return rows


def constraint_rows(result: Any, problem: Any | None = None) -> list[dict[str, Any]]:
    """Extract constraint diagnostics from a result score."""
    score = getattr(result, "score", {}) or {}
    if not score and problem is not None and hasattr(problem, "score_summary"):
        score = problem.score_summary(result.best_sol)
    constraints = score.get("extra", {}).get("constraints", {})
    return [{"name": name, **details} for name, details in constraints.items()]


def trajectory(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return an epoch axis and best-known scalar objective trajectory."""
    history = getattr(result, "history", {}) or {}
    values = history.get("best_obj") or history.get("loss_min") or []
    objective = np.asarray(values, dtype=float)
    if objective.ndim > 1:
        objective = objective.min(axis=tuple(range(1, objective.ndim)))
    return np.arange(objective.size), objective


def serialisable_summary(result: Any, problem: Any | None = None) -> dict[str, Any]:
    """Build a compact JSON-safe result payload."""

    def convert(value: Any) -> Any:
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    rows = constraint_rows(result, problem)
    report = getattr(result, "violations", None)
    report_status = getattr(getattr(report, "status", None), "value", None)
    declared = getattr(result, "score", {}).get("feasible")
    feasibility_status = report_status or (
        "feasible" if declared is True else "infeasible" if declared is False else "unknown"
    )
    guarantee = getattr(getattr(result, "guarantee_level", None), "value", "unknown")
    solve_status = getattr(getattr(result, "status", None), "value", "unknown")
    return {
        "problem": getattr(
            problem, "name", type(problem).__name__ if problem is not None else None
        ),
        "best_obj": convert(result.best_obj),
        "runtime_seconds": float(result.runtime),
        "feasible": feasibility_status == "feasible",
        "feasibility_status": feasibility_status,
        "solve_status": solve_status,
        "guarantee_level": guarantee,
        "score": convert(getattr(result, "score", {})),
        "solution": convert(result.best_sol),
        "variables": solution_rows(result, problem),
        "constraints": rows,
        "history_points": len(getattr(result, "history", {}).get("loss_mean", [])),
    }
