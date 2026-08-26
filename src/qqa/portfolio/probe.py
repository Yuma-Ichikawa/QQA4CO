"""Small empirical probes for explicit, opt-in solver routing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from qqa.config import SolverConfig
from qqa.portfolio.planner import SolverPlan, build_plan


@dataclass(frozen=True, slots=True)
class ProbeRecord:
    backend: str
    objective: float
    feasible: bool
    runtime: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeResult:
    selected_backend: str
    records: tuple[ProbeRecord, ...]
    plan: SolverPlan


def probe_portfolio(
    model: Any,
    config: SolverConfig,
    *,
    backends: tuple[str, ...] = ("qqa", "sa", "pa", "isco"),
    epochs: int = 50,
    replicas: int = 16,
) -> ProbeResult:
    """Compare tiny runs by feasibility, objective, then wall time."""
    from qqa.api import solve  # noqa: PLC0415

    records = []
    for backend in backends:
        started = perf_counter()
        try:
            result = solve(
                model,
                config=SolverConfig.from_mapping(
                    {
                        **config.to_dict(),
                        "backend": backend,
                        "epochs": epochs,
                        "replicas": replicas,
                        "exact_backend": "none",
                        "require_certificate": False,
                        "budget": None,
                    }
                ),
            )
            records.append(ProbeRecord(backend, result.best_obj, result.feasible, result.runtime))
        except (ImportError, NotImplementedError, RuntimeError, ValueError) as exc:
            records.append(
                ProbeRecord(
                    backend,
                    float("inf"),
                    False,
                    perf_counter() - started,
                    type(exc).__name__,
                )
            )
    usable = [record for record in records if record.error is None]
    if not usable:
        raise RuntimeError("Every mini-probe backend failed.")
    selected = min(usable, key=lambda row: (not row.feasible, row.objective, row.runtime))
    selected_config = SolverConfig.from_mapping({**config.to_dict(), "backend": selected.backend})
    return ProbeResult(selected.backend, tuple(records), build_plan(model, selected_config))


__all__ = ["ProbeRecord", "ProbeResult", "probe_portfolio"]
