# Graph Coloring Benchmark

Procedurally generated Graph Coloring instances used by
`qqa.datasets.coloring` and `scripts/bench_discs.py --suite coloring-*`.

Following the PQQA paper (arXiv:2409.02135v2) we evaluate the classical
COLOR benchmark families:

| graph_type | generator                    | chromatic number |
|------------|-------------------------------|------------------|
| `myciel`   | `networkx.mycielski_graph(k)` | known (k+1)      |
| `queen`    | queen-attack graph on k×k     | known (see code) |

`best_known` is always `0` (the number of edge conflicts a proper coloring
must reach). `num_colors` stores the target chromatic number so
`qqa.problems.Coloring` is instantiated with a feasible palette.

## Layout

```
data/coloring/
├── README.md         (git-tracked)
├── .gitignore        (git-tracked, ignores the rest)
├── myciel/
│   ├── manifest.jsonl
│   ├── myciel3.gpickle
│   └── ...
└── queen/
    ├── manifest.jsonl
    ├── queen5_5.gpickle
    └── ...
```

Everything below `.gitignore` is re-generated locally by:

```bash
python scripts/generate_coloring_instances.py
```

Upstream distribution: `yuma-ichikawa/discs-co-bench` on the Hugging Face
Hub (same repo as DISCS; see `data/discs/README.md` for credentials /
fallbacks).
