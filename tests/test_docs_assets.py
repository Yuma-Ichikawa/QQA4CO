"""Lightweight guard that mirrors the `mkdocs build --strict` CI step.

The failing CI run `24658520429` tripped `mkdocs --strict` because
`docs/how-to/benchmark.md` once referenced a local image path that lived
outside the MkDocs doc root. Gallery assets now live under `docs/assets/`.
We fixed the link, but the CI pipeline is slow and forgiving mistakes
is easy. This test encodes the same invariant as a pytest assertion so
the mistake is caught locally *before* pushing.

Invariants
----------
1. Every `![alt](path)` and `<img src="path">` reference in any
   markdown file under `docs/` either points to an absolute URL
   (http/https) or resolves to a path inside `docs/`.
2. No markdown file under `docs/` links to `../../` paths outside the
   doc root — which is exactly the class of issue that broke CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"

# ![alt text](target)  OR  <img src="target">
_IMAGE_MD = re.compile(r"!\[[^\]]*\]\(([^)\s#]+)")
_IMAGE_HTML = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']")


def _all_image_refs() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for md in DOCS_ROOT.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        for target in _IMAGE_MD.findall(text):
            out.append((md, target))
        for target in _IMAGE_HTML.findall(text):
            out.append((md, target))
    return out


@pytest.mark.parametrize(("md", "target"), _all_image_refs())
def test_doc_image_target_is_valid(md: Path, target: str) -> None:
    if target.startswith(("http://", "https://", "data:", "mailto:")):
        return  # absolute URL — MkDocs does not need to resolve it
    if target.startswith("/"):
        # Project-absolute — MkDocs resolves from the site root.
        resolved = DOCS_ROOT / target.lstrip("/")
    else:
        resolved = (md.parent / target).resolve()

    # Must stay inside docs/ (this is exactly what broke 24658520429).
    try:
        resolved.relative_to(DOCS_ROOT)
    except ValueError:
        pytest.fail(
            f"{md.relative_to(REPO_ROOT)}: image link '{target}' escapes docs/ "
            f"(resolved to {resolved}). `mkdocs build --strict` will reject this. "
            "Move the asset into docs/ or link to a raw GitHub URL instead."
        )

    assert resolved.is_file(), (
        f"{md.relative_to(REPO_ROOT)}: image '{target}' resolves to "
        f"{resolved}, which does not exist."
    )
