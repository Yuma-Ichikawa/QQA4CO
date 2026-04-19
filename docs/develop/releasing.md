# Releasing a new version

This page is the canonical checklist for cutting a new QQA4CO release.
Follow it top-to-bottom; never skip a step. Every previous release that
went sideways skipped exactly one item on this list.

## Versioning

QQA4CO follows [SemVer](https://semver.org/) (`MAJOR.MINOR.PATCH`).
Until 1.0 we will:

* bump **PATCH** for bug fixes and additive doc changes,
* bump **MINOR** for new public API,
* bump **MAJOR** only when removing or renaming a public symbol.

`__version__` is **single-sourced** from `pyproject.toml` via
`importlib.metadata.version("qqa")`. **Do not** edit
`src/qqa/__init__.py` to bump versions — only `pyproject.toml`.

## Pre-release checklist

1. **Working tree is clean** — no uncommitted changes (`git status`).
2. **All CI passes locally**:
   ```bash
   make lint test docs
   ```
3. **`CHANGELOG.md` has an "Unreleased" section** with every public
   change since the last tag. Move it under a new dated heading
   `## [X.Y.Z] - YYYY-MM-DD`.
4. **Bump `version` in `pyproject.toml`** to the new value.
5. **Update `CITATION.cff`** — bump `version:` and `date-released:` so
   Zenodo and the GitHub citation widget pick it up.
6. **Re-run the verification sweep** if you touched any solver:
   ```bash
   uv run python scripts/verify_all_problems.py
   ```
   and commit the regenerated `docs/verification.md`.

## Cutting the release

```bash
# 1. One commit with the version bump + CHANGELOG move + CITATION bump.
git add pyproject.toml CHANGELOG.md CITATION.cff docs/verification.md
git commit -m "release: vX.Y.Z"

# 2. Tag and push.
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin main vX.Y.Z

# 3. Build the wheel and sdist.
rm -rf dist/
uv build

# 4. Upload to PyPI (Trusted Publishing via the GitHub Actions release
#    workflow is the preferred path — see below). For a manual upload:
uv run twine upload dist/*
```

## Post-release checklist

1. **Verify** `pip install qqa==X.Y.Z` works from a fresh venv on
   another machine.
2. **GitHub Release** — create one against the new tag, paste the
   matching `CHANGELOG.md` section as the body.
3. **Zenodo** — confirm the new DOI was minted (CITATION.cff is what
   triggers it).
4. **Docs** — confirm the GitHub Pages workflow pushed the new
   `mkdocs build` to <https://yuma-ichikawa.github.io/QQA4CO/>.
5. **Streamlit Cloud** — bump the pinned `qqa` version in the deploy's
   `requirements.txt` if needed (see `deploy/STREAMLIT_DEPLOY.md`).
6. **Open a `CHANGELOG.md` "Unreleased" stub** for the next cycle.

## Trusted Publishing (recommended once configured)

Once Trusted Publishing is set up on PyPI for the `qqa` project, a
push of the `vX.Y.Z` tag should trigger a GitHub Actions workflow that
builds and uploads the wheel + sdist with no API token in the repo.
Track the workflow run from the GitHub *Actions* tab.

If the workflow fails:

* `workflows/release.yml` logs always show the exact failure step.
* `dist/` build failures usually mean a stale `dist/` directory shipped
  in the wheel — purge it locally and re-tag.
* Trusted Publishing rejection means the PyPI configuration drifted —
  re-configure under "Publishing" on the PyPI project page.

## Rolling back a bad release

PyPI does **not** allow re-uploading the same version. If you ship a
broken wheel:

1. Yank it on PyPI: `uv run twine yank qqa --version X.Y.Z` (do not
   delete — yanking lets `pip install qqa==X.Y.Z` keep working for
   pinned users while preventing fresh installs).
2. Bump to `X.Y.(Z+1)` with the fix and follow the full release
   checklist again.
3. Document the yanked version in `CHANGELOG.md` so users searching
   the changelog understand why a version is "missing".

Never amend a tag that has been pushed; create a new one.
