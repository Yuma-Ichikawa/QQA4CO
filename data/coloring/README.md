# Graph Coloring Benchmark

Graph Coloring instances used by `qqa.datasets.coloring` and
`qqa bench-run --suite coloring-*`.

Following the PQQA paper (arXiv:2409.02135v2, §5.5 / Table 6) we
evaluate the classical COLOR benchmark (Trick 2002). Coverage is
organised in three families:

| graph_type | generator / source                                           | chromatic number (`num_colors`)                   |
|------------|--------------------------------------------------------------|---------------------------------------------------|
| `myciel`   | `networkx.mycielski_graph(k)` (Mycielski 1955)               | `k` (procedural)                                  |
| `queen`    | queen-attack graph on a k×k chessboard                       | tabulated, k∈{5..13} (Chvátal; DeLaVina)         |
| `dimacs`   | DIMACS `.col` fetched from `mat.tepper.cmu.edu/COLOR/instances` | tabulated (Trick 2002 and follow-ups)          |

`best_known` is always `0` (the minimum number of edge conflicts a
proper coloring must reach). `num_colors` stores the target chromatic
number so `qqa.problems.Coloring` is instantiated with a feasible
palette.

## arXiv-2409.02135v2 Table 6 coverage

The paper tabulates 12 specific instances. All 12 are reachable through
the public loader:

| Paper row   | Colors | Local file path                           | `num_nodes` | `num_edges` |
|-------------|--------|-------------------------------------------|-------------|-------------|
| anna        | 11     | `coloring/dimacs/0001.gpickle`            | 138         | 493         |
| jean        | 10     | `coloring/dimacs/0002.gpickle`            | 80          | 254         |
| queen8_12   | 12     | `coloring/dimacs/0003.gpickle`            | 96          | 1368        |
| myciel5     |  6     | `coloring/myciel/0003.gpickle`            | 23          | 71          |
| myciel6     |  7     | `coloring/myciel/0004.gpickle`            | 47          | 236         |
| queen5_5    |  5     | `coloring/queen/0001.gpickle`             | 25          | 160         |
| queen6_6    |  7     | `coloring/queen/0002.gpickle`             | 36          | 290         |
| queen7_7    |  7     | `coloring/queen/0003.gpickle`             | 49          | 476         |
| queen8_8    |  9     | `coloring/queen/0004.gpickle`             | 64          | 728         |
| queen9_9    | 10     | `coloring/queen/0005.gpickle`             | 81          | 1056        |
| queen11_11  | 11     | `coloring/queen/0007.gpickle`             | 121         | 1980        |
| queen13_13  | 13     | `coloring/queen/0009.gpickle`             | 169         | 3328        |

This mapping is enforced by
`QQA4CO_plugin/tests/test_hf_bench_coverage.py::test_coloring_paper_table6_coverage`,
which fails the test suite if any of the 12 instances disappears.

## Layout

```
data/coloring/
├── README.md           (git-tracked)
├── .gitignore          (git-tracked, ignores the rest)
├── myciel/
│   ├── manifest.jsonl
│   ├── 0001.gpickle    # myciel3 .. myciel7
│   └── ...
├── queen/
│   ├── manifest.jsonl
│   ├── 0001.gpickle    # queen5_5 .. queen13_13
│   └── ...
└── dimacs/
    ├── manifest.jsonl
    ├── 0001.gpickle    # anna
    ├── 0002.gpickle    # jean
    └── 0003.gpickle    # queen8_12
```

Everything below `.gitignore` is re-generated locally by:

```bash
python scripts/generate_coloring_instances.py       # fetches DIMACS once, caches in data/coloring/_dimacs_cache
python scripts/generate_coloring_instances.py --skip-dimacs   # offline rebuild
```

The DIMACS family is downloaded once (≈10 KB per `.col`) from
Trick's canonical mirror; the cached ascii sources remain in
`data/coloring/_dimacs_cache/` for reproducibility.

Upstream distribution: `Yuma-Ichikawa/qqa4co-bench` on the Hugging Face
Hub (same repo as DISCS; see `data/discs/README.md` for credentials /
fallbacks).
