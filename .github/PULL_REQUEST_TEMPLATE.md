<!--
Thanks for contributing to QQA4CO! Please fill in the sections below
so reviewers can land your change quickly.
-->

## Summary

<!-- 1-3 sentences: what does this PR change and why? -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing behaviour to change)
- [ ] Documentation only
- [ ] Build / CI / tooling

## Checklist

- [ ] I ran `uv run ruff check src tests scripts app` and it is clean.
- [ ] I ran `uv run ruff format src tests scripts app` (or `pre-commit run --all-files`).
- [ ] I ran `uv run pytest -q` and all tests pass locally.
- [ ] If I touched solver behaviour, I added or updated a regression
      test under `tests/`.
- [ ] If I changed a user-visible API, I updated `docs/` and the
      relevant Quickstart / How-to entries.
- [ ] I updated `CHANGELOG.md` under `## [Unreleased]`.
- [ ] If I added a new dependency, I justified it in the PR description.

## Related issue / discussion

<!-- Closes #..., refs #..., or "n/a" -->

## Reproduction or benchmark

<!--
For perf or numerical changes, paste before/after numbers (best_obj,
runtime, GPU memory). For UI changes, attach a screenshot or a short
recording of the dashboard.
-->
