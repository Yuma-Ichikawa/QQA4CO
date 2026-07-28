"""Ready-to-run optimisation models based on realistic planning tasks.

The builders in this module are deliberately small enough for tutorials and
CI, while retaining the variable coupling, engineering constraints, and
competing objectives that make real models interesting.
"""

from qqa.applications.microgrid import (
    build_microgrid_dispatch,
    build_microgrid_pareto,
)
from qqa.applications.process import build_process_blackbox

APPLICATIONS = (
    "microgrid-dispatch",
    "microgrid-pareto",
    "process-blackbox",
)


def build_application(name: str):
    """Build one of the packaged application models by stable CLI name."""
    builders = {
        "microgrid-dispatch": build_microgrid_dispatch,
        "microgrid-pareto": build_microgrid_pareto,
        "process-blackbox": build_process_blackbox,
    }
    try:
        builder = builders[name]
    except KeyError as exc:
        raise ValueError(f"Unknown application {name!r}; choose from {APPLICATIONS}.") from exc
    return builder()


__all__ = [
    "APPLICATIONS",
    "build_application",
    "build_microgrid_dispatch",
    "build_microgrid_pareto",
    "build_process_blackbox",
]
