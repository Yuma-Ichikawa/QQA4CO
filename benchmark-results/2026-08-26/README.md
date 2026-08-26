# 2026-08-26 public benchmark campaign

This directory keeps only the compact, machine-readable campaign summary.
Full per-instance results and incumbent trajectories are generated artifacts;
publish them as release or CI artifacts rather than source files.

The complete protocol, failures, snapshot hashes, and interpretation are in
[`docs/benchmark-results.md`](../../docs/benchmark-results.md). Reproduce a
campaign with the public benchmark CLI:

```bash
qqa benchmark fetch miplib --output benchmarks/miplib
qqa benchmark fetch qplib --output benchmarks/qplib
qqa benchmark compare benchmarks/miplib/instances/*.mps.gz \
  --format miplib --seeds 0 1 2 --time-limit 30 \
  --continue-on-error --output results/miplib.json
qqa benchmark compare benchmarks/qplib/instances/*.qplib \
  --format qplib --seeds 0 1 2 --time-limit 30 \
  --continue-on-error --output results/qplib.json
```

Use multiple seeds and matched wall-clock budgets for new claims. Never infer
universal solver dominance from the single-seed historical campaign.
