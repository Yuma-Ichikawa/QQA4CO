"""Headless smoke test for the Streamlit dashboard.

Starts ``streamlit run app/streamlit_app.py`` in a subprocess, polls the
HTTP endpoint briefly, and asserts the server eventually responds.

The test is automatically skipped if either ``streamlit`` or ``requests``
is not installed (so core CI stays lean).
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
requests = pytest.importorskip("requests")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_streamlit_app_boots():
    if shutil.which("streamlit") is None:
        pytest.skip("streamlit executable not on PATH")

    repo_root = Path(__file__).resolve().parents[1]
    app = repo_root / "app" / "streamlit_app.py"
    assert app.exists(), app

    port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 40
        ok = False
        while time.time() < deadline:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=1.5)
                if r.status_code == 200:
                    ok = True
                    break
            except Exception:
                time.sleep(0.5)
        assert ok, "Streamlit server did not become healthy in time"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
