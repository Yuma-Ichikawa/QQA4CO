"""Tiny health-check script for the public Streamlit deployment.

Exits 0 iff ``<url>/_stcore/health`` returns HTTP 200 with the body ``"ok"``.
This is the same endpoint that Streamlit Cloud uses internally, so it stays
green even when the app is configured as Private (which would otherwise
redirect the root URL to ``/-/auth/app``).

Usage::

    uv run python scripts/check_streamlit_deploy.py
    uv run python scripts/check_streamlit_deploy.py --url https://my.app
"""

from __future__ import annotations

import argparse
import sys
import urllib.request

DEFAULT_URL = "https://parallelquasiquantum4co.streamlit.app"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL (no trailing slash)")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    url = f"{args.url.rstrip('/')}/_stcore/health"
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace").strip()
            if resp.status == 200 and body == "ok":
                print(f"[health] OK  {url} -> {body!r}")
                return 0
            print(f"[health] FAIL status={resp.status} body={body!r}", file=sys.stderr)
            return 1
    except Exception as exc:  # pragma: no cover - network-dependent
        print(f"[health] FAIL {url} -> {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
