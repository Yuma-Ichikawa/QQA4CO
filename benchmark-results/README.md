# Public benchmark artifacts

This directory contains portable benchmark outputs produced from official
MIPLIB and QPLIB snapshots. Each campaign directory contains:

- `manifest.json`: source snapshot summary, implementation revision, campaign
  configuration, aggregate metrics, and artifact SHA-256 hashes;
- `*-results.json`: readable results without full incumbent trajectories;
- `*-campaign.json.gz`: deterministic gzip containing the complete campaign,
  including trajectories and anonymous failure records.

The `2026-08-26` snapshot contains the full primary SG-CQQA comparison, a
QQA-only ablation, a short CPU screening campaign, and a fixed two-instance,
three-seed QPLIB diagnostic. See
[`docs/benchmark-results.md`](../docs/benchmark-results.md) for the conditions,
aggregate results, and limitations.

Artifacts intentionally omit filesystem paths, usernames, hostnames, private
addresses, scheduler details, and machine identifiers. The generic `cpu` or
`cuda` device class and solver thread count remain because they are required
to interpret a benchmark.
