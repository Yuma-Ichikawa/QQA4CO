"""Private process entry point for one optional native exact backend."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        return 2
    request, response, ready = map(Path, arguments)
    try:
        from qqa.hybrid.exact import _run_backend_payload, _safe_error

        with request.open("rb") as stream:
            payload, backend, kwargs = pickle.load(stream)
        ready.touch(exist_ok=False)
        envelope = _run_backend_payload(payload, backend, kwargs)
    except Exception as exc:  # noqa: BLE001 - final native-process boundary
        try:
            envelope = _safe_error(exc)
        except NameError:
            message = str(exc)
            if "/" in message or "\\" in message:
                message = "backend initialisation or execution failed"
            envelope = ("error", type(exc).__name__, message)
    try:
        with response.open("wb") as stream:
            pickle.dump(envelope, stream, protocol=5)
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
