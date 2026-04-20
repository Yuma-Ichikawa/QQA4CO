# Maximum Independent Set on d-regular Random Graphs

MIS instances on d-regular random graphs, matching the experiment
setup of PQQA (arXiv:2409.02135v2, Table 2 / Figure 3). Generated
locally via `networkx.random_regular_graph` for full reproducibility.

`best_known` is the theoretical MIS density upper bound
(Barbier, Krzakala, Zdeborová 2013) multiplied by `n`, serving as a
published reference for approximation-ratio comparisons.

## Layout

```
data/mis-rrg/
├── README.md              (git-tracked)
├── .gitignore             (git-tracked)
├── d20_n10000/
│   ├── manifest.jsonl     (5 instances, best_known, graph metadata)
│   ├── 0001.gpickle       (networkx.Graph; n = 10 000, d = 20)
│   └── ...
└── d100_n10000/           (same structure, d = 100)
```

`num_nodes`, `degree`, `seed`, and `best_known` are stored in
`manifest.jsonl`; individual graphs are pickled via
`pickle.HIGHEST_PROTOCOL`.

## Regenerate locally

```bash
python scripts/generate_rrg_instances.py
```

## Why only `n = 10_000` and not `n = 10^6`?

The PQQA paper also evaluates `n = 10^6`; at that scale a single
`.gpickle` balloons to ~1 GB and graph construction alone dominates any
solver benchmark. We ship the `n = 10^4` variant here (covers both
`d = 20` and `d = 100`) and keep the million-node generator disabled so
the Hugging Face Hub mirror stays lean. Reinstate it by passing
`--nodes 1000000` to `scripts/generate_rrg_instances.py` if needed.

Upstream distribution: `Yuma-Ichikawa/qqa4co-bench` on the Hugging Face
Hub.
