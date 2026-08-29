"""Environment diagnostics and Streamlit launcher commands."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_RUNTIME_PROBE = """
import importlib.util
import json

import torch

try:
    from pyscipopt import Model, quicksum
    from pyscipopt.recipes.nonlinear import set_nonlinear_objective
except (ImportError, OSError):
    scip = False
else:
    scip = callable(Model) and callable(quicksum) and callable(set_nonlinear_objective)

cuda = bool(torch.cuda.is_available())
mps_backend = getattr(torch.backends, "mps", None)
mps = bool(mps_backend is not None and mps_backend.is_available())
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": cuda,
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if cuda else None,
    "recommended_device": "cuda" if cuda else "mps" if mps else "cpu",
    "optional": {
        "scip": scip,
        "pignn": importlib.util.find_spec("torch_geometric") is not None,
        "streamlit": importlib.util.find_spec("streamlit") is not None,
        "plotly": importlib.util.find_spec("plotly") is not None,
        "pandas": importlib.util.find_spec("pandas") is not None,
    },
    "probe_status": "complete",
}))
"""


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _fallback_runtime_payload(status: str) -> dict:
    return {
        "torch": _installed_version("torch"),
        "cuda_available": None,
        "cuda_version": None,
        "gpu": None,
        "recommended_device": "cpu",
        "optional": {
            "scip": importlib.util.find_spec("pyscipopt") is not None,
            "pignn": importlib.util.find_spec("torch_geometric") is not None,
            "streamlit": importlib.util.find_spec("streamlit") is not None,
            "plotly": importlib.util.find_spec("plotly") is not None,
            "pandas": importlib.util.find_spec("pandas") is not None,
        },
        "probe_status": status,
    }


def _probe_runtime(timeout: float) -> dict:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _RUNTIME_PROBE],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return _fallback_runtime_payload("timed_out")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return _fallback_runtime_payload("failed")


def command_doctor(args) -> int:
    if getattr(args, "model", None):
        import qqa

        report = qqa.doctor(args.model, replicas=args.replicas)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(report.explain())
        return 0 if report.ready else 2

    timeout = float(getattr(args, "probe_timeout", 30.0))
    if not timeout > 0.0:
        raise ValueError("probe_timeout must be positive.")
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **_probe_runtime(timeout),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"python     : {payload['python']}")
        print(f"torch      : {payload['torch']}")
        print(f"device     : {payload['recommended_device']}")
        print(f"gpu        : {payload['gpu'] or 'not available'}")
        print(f"probe      : {payload['probe_status']}")
        for name, available in payload["optional"].items():
            print(f"{name:<11}: {'ready' if available else 'not installed'}")
    return 0


def resolve_streamlit_app() -> Path | None:
    """Locate the packaged dashboard or the source-tree fallback."""
    try:
        from importlib.resources import files

        candidate = Path(str(files("qqa").joinpath("_app", "streamlit_app.py")))
        if candidate.exists():
            return candidate
    except (ModuleNotFoundError, FileNotFoundError, OSError):
        pass

    candidate = Path(__file__).resolve().parents[3] / "app" / "streamlit_app.py"
    return candidate if candidate.exists() else None


def command_gui(args) -> int:
    if shutil.which("streamlit") is None:
        print(
            "[qqa gui] 'streamlit' is not on PATH. Install the GUI extras with "
            "'pip install qqa[gui]'.",
            file=sys.stderr,
        )
        return 2

    app = resolve_streamlit_app()
    if app is None:
        print(
            "[qqa gui] Streamlit app not found. Re-install qqa[gui] or run "
            "from a QQA4CO source checkout.",
            file=sys.stderr,
        )
        return 2

    command = [
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if args.headless:
        command.extend(["--server.headless", "true"])
    environment = os.environ.copy()
    environment.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    return subprocess.call(command, env=environment)


__all__ = ["command_doctor", "command_gui", "resolve_streamlit_app"]
