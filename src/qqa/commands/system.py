"""Environment diagnostics and Streamlit launcher commands."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from qqa.commands.runtime import resolve_device


def command_doctor(args) -> int:
    import torch

    from qqa.hybrid import scip_available

    optional = {
        "scip": scip_available(),
        "pignn": importlib.util.find_spec("torch_geometric") is not None,
        "streamlit": importlib.util.find_spec("streamlit") is not None,
        "plotly": importlib.util.find_spec("plotly") is not None,
        "pandas": importlib.util.find_spec("pandas") is not None,
    }
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "recommended_device": resolve_device("auto"),
        "optional": optional,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"python     : {payload['python']}")
        print(f"torch      : {payload['torch']}")
        print(f"device     : {payload['recommended_device']}")
        print(f"gpu        : {payload['gpu'] or 'not available'}")
        for name, available in optional.items():
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
