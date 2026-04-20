# Benchmark your method against QQA

This page is the **zero-to-report** guide for third parties who want to
compare a new solver against the benchmark suite shipped with QQA4CO.
No edits to the repo or knowledge of its internals are required.

Every benchmark instance lives on the Hugging Face Hub:

> **[`huggingface.co/datasets/Yuma-Ichikawsa/qqa4co-bench`](https://huggingface.co/datasets/Yuma-Ichikawsa/qqa4co-bench)**

The dataset bundles the [DISCS](https://arxiv.org/abs/2311.04730) CO
benchmarks (NeurIPS 2023), the **MaxCut G-set** superset (71 graphs,
Helmberg & Rendl 2000 via Yinyu Ye's mirror), and four extra families
described in the [PQQA paper](https://openreview.net/forum?id=9EfBeXaXf0):
Graph Coloring, MIS on `d`-regular random graphs, 3D Edwards-Anderson
spin glass and Balanced k-way partition.

---

## 1. One-command setup

```bash
# download every family from the Hub into ./data/
make bench-all-setup
```

That will fetch ~6.7 GB of DISCS subsets + the PQQA extras. Use the
finer-grained flags if you only need one family:

```bash
./scripts/setup_benchmarks.sh --only coloring,ea3d
```

## 2. One-line run

Three equivalent ways to run the benchmark:

=== "CLI"

    ```bash
    # Whole suite, default PQQA hyperparameters, CPU.
    qqa bench-run --suite all --output mine.json

    # Scoped to MIS on SATLIB only, 3 instances.
    qqa bench-run --suite mis-satlib --instances 3 --output mine.json
    ```

=== "Python"

    ```python
    from qqa import bench

    payload = bench.run(
        "all",                      # or a specific suite id
        backend="qqa",
        instances=None,             # None = every instance on disk
        output="mine.json",
    )
    ```

=== "make"

    ```bash
    make bench-all SUITE=mis-satlib INSTANCES=3 OUTPUT=mine.json
    ```

All outputs land under `./bench_results/` (git-ignored by default), so
you can keep multiple runs side-by-side.

## 3. Visualise the result

```bash
qqa bench-plot bench_results/mine.json --output report.png
```

For an A/B comparison against a baseline:

```bash
qqa bench-plot bench_results/mine.json bench_results/sa.json \
    --labels "my method" "SA baseline" \
    --title "My method vs SA" \
    --output report.png
```

The rendered image is a single 2x2 panel — radar (per-family ratio),
per-subset horizontal bars, feasibility bars, and a per-instance
violin+strip plot. `--theme dark` produces a dark-mode variant suitable
for talk slides.

![example report](https://raw.githubusercontent.com/Yuma-Ichikawa/QQA4CO/main/data/fig/gallery/bench_report_example.png)

## 4. Suite identifiers

```bash
qqa bench-list             # tree view
qqa bench-list --as-suites # every resolvable --suite id, one per line
```

`--suite` accepts:

| Pattern                          | Example                    |
|----------------------------------|----------------------------|
| `all`                            | every subset on disk       |
| `<family>`                       | `coloring`, `ea3d`         |
| `<family>-<graph-type>`          | `mis-rrg`, `ea3d-gaussian` |
| `<family>-<graph-type>-<subset>` | `ea3d-gaussian-L6`         |

Family names themselves can contain a hyphen (`mis-rrg`,
`balanced-partition`); resolution is longest-prefix-first.

## 5. Reporting approximation ratios honestly

For each solver we report a single "higher-is-better" number. The
convention differs per family so that the ratio is always in
`(0, 1]` at the published optimum:

| Family                 | Objective      | Ratio formula                 |
|------------------------|----------------|-------------------------------|
| `mis`, `mis-rrg`       | maximisation   | `solver / best_known`         |
| `maxcut`, `maxclique`  | maximisation   | `solver / best_known`         |
| `gset` (MaxCut)        | maximisation   | `solver / best_known`         |
| `normcut`              | minimisation   | `best_known / solver`         |
| `coloring`             | conflict count | `feasible` flag only          |
| `ea3d`                 | energy (≤ 0)   | `solver / best_known`         |
| `balanced-partition`   | edge cut       | feasibility only (no ref)     |

`qqa.bench.plot` weighs each subset's mean ratio by the number of
instances it contains, so a family with multiple subsets stays
well-represented in the family-level radar.

## 6. Custom solver?

`qqa.bench.run` only routes through the three built-in backends
(`qqa` / `sa` / `pa`). To benchmark your own solver:

1. Load the exact instances the runner uses:

    ```python
    from qqa import datasets
    ds = datasets.discs_mis(graph_type="satlib", subset="uf")
    for problem, best_known in zip(ds.problems, ds.best_known):
        x_star = my_solver(problem)       # your code
        ...
    ```

2. Dump your results in the same JSON schema as `scripts/bench_discs.py`
   writes (see [`scripts/bench_discs.py`](../reference/cli.md) for the
   exact shape). Once written, `qqa bench-plot` will happily render
   your results next to the bundled baselines.
