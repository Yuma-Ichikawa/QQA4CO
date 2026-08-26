"""Small dependency-light command helpers shared by CLI frontends."""

from __future__ import annotations

import json


def command_version() -> int:
    import qqa

    print(qqa.__version__)
    return 0


def resolve_device(device: str) -> str:
    from qqa.utils import resolve_device as resolve

    return resolve(device)


def print_score(score: dict) -> None:
    """Print a compact human summary; complete nested data belongs in JSON."""
    label = score.get("label", "objective")
    value = score.get("value")
    unit = score.get("unit", "")
    rendered_value = f"{value:.8g}" if isinstance(value, (int, float)) else str(value)
    suffix = f" {unit}" if unit else ""
    feasible = str(bool(score.get("feasible", False))).lower()
    print(f"score      : {label}={rendered_value}{suffix}; feasible={feasible}")

    extra = score.get("extra", {})
    variables = extra.get("variables", {}) if isinstance(extra, dict) else {}
    if isinstance(variables, dict) and variables:
        rendered_variables = json.dumps(variables, ensure_ascii=False)
        if len(rendered_variables) > 500:
            rendered_variables = rendered_variables[:497] + "..."
        print(f"solution   : {rendered_variables}")
    constraints = extra.get("constraints", {}) if isinstance(extra, dict) else {}
    if isinstance(constraints, dict) and constraints:
        violations = [
            (name, float(row.get("violation", 0.0)), bool(row.get("feasible", False)))
            for name, row in constraints.items()
            if isinstance(row, dict)
        ]
        failed = [name for name, _, feasible_row in violations if not feasible_row]
        maximum = max((violation for _, violation, _ in violations), default=0.0)
        print(f"constraints: {len(failed)}/{len(violations)} failed; max_violation={maximum:.6g}")


__all__ = ["command_version", "print_score", "resolve_device"]
