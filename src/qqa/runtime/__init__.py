"""Event-driven runtime contracts shared by solvers, services, and UIs."""

from qqa.runtime.checkpoint import Checkpoint, fingerprint_problem, load_checkpoint, save_checkpoint
from qqa.runtime.context import SolveContext
from qqa.runtime.events import EventKind, EventRecorder, SolveEvent
from qqa.runtime.package import PackageManifest, export_result_package, verify_result_package
from qqa.runtime.population import ReplicaPortfolio, ReplicaRole, WarmStateBundle
from qqa.runtime.security import validate_portable_payload

__all__ = [
    "Checkpoint",
    "EventKind",
    "EventRecorder",
    "PackageManifest",
    "ReplicaPortfolio",
    "ReplicaRole",
    "SolveEvent",
    "SolveContext",
    "WarmStateBundle",
    "export_result_package",
    "fingerprint_problem",
    "load_checkpoint",
    "save_checkpoint",
    "verify_result_package",
    "validate_portable_payload",
]
