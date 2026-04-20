# 3D Edwards-Anderson Spin Glass Benchmark

User-requested hard benchmark to complement the PQQA reproduction: 3D
Edwards-Anderson spin glass on the cubic lattice
`L × L × L` (nearest-neighbour couplings, periodic BCs).

Two coupling distributions are shipped:

| distribution | `J_ij`         | notes                                    |
|--------------|----------------|------------------------------------------|
| `gaussian`   | `N(0, 1)`      | continuous spin glass; standard EA model |
| `bimodal`    | `{-1, +1}`     | ±J spin glass                            |

## Layout

```
data/ea3d/
├── README.md                (git-tracked)
├── .gitignore               (git-tracked)
├── gaussian/
│   ├── L4/
│   │   ├── manifest.jsonl
│   │   ├── 0001.npz          (coo_matrix of couplings)
│   │   └── ...
│   ├── L6/
│   └── L8/
└── bimodal/                  (same sub-structure)
```

Each instance is saved as an `.npz` with `row`, `col`, `data`, `num_spins`
arrays so it can be reloaded into a `qqa.problems.EdwardsAnderson`
problem with one call.

`best_known` is the brute-force ground-state energy for tiny lattices
(`N ≤ 20`) and `NaN` otherwise; QQA/SA approximation ratios are reported
against this published ground-state reference when available.

## Regenerate locally

```bash
python scripts/generate_ea3d_instances.py
```

Upstream distribution: `yuma-ichikawa/discs-co-bench` on the Hugging Face
Hub.
