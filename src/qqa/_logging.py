"""Structured logging for ``qqa``.

Library users who want to capture, redirect, or silence QQA's output
should configure the standard ``logging`` module:

    import logging
    logging.getLogger("qqa").setLevel(logging.INFO)

The package itself does **not** install handlers (per the standard
"libraries should not configure logging" rule), so ``import qqa`` has
zero side effects on the user's logging configuration.

The existing ``verbose=True / False`` flag on :func:`qqa.anneal` and
the PyG trainers is unchanged — that flag controls human-readable
``print()`` output for terminal users. Use ``logging`` if you want
machine-parseable logs in addition to (or instead of) the printed
output.
"""

from __future__ import annotations

import logging

__all__ = ["get_logger"]

_ROOT = "qqa"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the canonical ``qqa`` logger (or a named child).

    Examples::

        from qqa._logging import get_logger
        log = get_logger(__name__)         # qqa.<module> child logger
        log.info("starting solve %s", problem)

    Library code inside this package uses ``get_logger(__name__)`` to
    inherit the central ``qqa`` log level. Calls remain no-ops unless
    the user attaches a handler — :func:`logging.basicConfig` works.
    """
    if name is None or name == _ROOT or not name.startswith(_ROOT + "."):
        return logging.getLogger(_ROOT) if name is None else logging.getLogger(name)
    return logging.getLogger(name)


# Attach a NullHandler so consumers without `logging.basicConfig()` do
# not see "No handlers could be found for logger qqa" warnings on
# Python 3.10+ (the warning is gone, but the convention is still good
# hygiene; see https://docs.python.org/3/howto/logging.html#library-config).
logging.getLogger(_ROOT).addHandler(logging.NullHandler())
