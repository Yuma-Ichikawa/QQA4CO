"""Lazy / actionable import guard for :mod:`qqa.pignn`.

PyTorch Geometric is an *optional* dependency. Importing
:mod:`qqa.pignn.model` or :mod:`qqa.pignn.trainer` without it should
raise a single, clear error that tells the user how to install it —
**not** a bare ``ModuleNotFoundError`` whose stack trace forces the user
to dig through PyG internals.
"""

from __future__ import annotations


def require_pyg() -> None:
    """Raise an actionable ImportError if torch-geometric is missing.

    This is intentionally cheap (``import torch_geometric`` only) so it
    can be called at the top of every public function in
    :mod:`qqa.pignn` without measurable overhead on warm imports.
    """
    try:
        import torch_geometric  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via pytest skip
        raise ImportError(
            "qqa.pignn requires torch-geometric, which is not installed. "
            "Install it with one of:\n"
            '  pip install "qqa[pignn]"\n'
            "  pip install torch-geometric\n"
            "See https://pytorch-geometric.readthedocs.io for platform-"
            "specific wheels."
        ) from exc
