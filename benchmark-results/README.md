# Public benchmark artifacts

This directory contains compact, portable summaries produced from official
MIPLIB and QPLIB snapshots. Full per-instance records, incumbent trajectories,
and run manifests are generated artifacts published by CI, a release, or an
external registry rather than committed source files.

The `2026-08-26` summary covers the primary SG-CQQA comparison, a QQA-only
ablation, and the conservative screening campaign. See
[`docs/benchmark-results.md`](../docs/benchmark-results.md) for the conditions,
aggregate results, and limitations.

Artifacts intentionally omit filesystem paths, usernames, hostnames, private
addresses, scheduler details, and machine identifiers. The generic `cpu` or
`cuda` device class and solver thread count remain because they are required
to interpret a benchmark.
