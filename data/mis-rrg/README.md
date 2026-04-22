# Maximum Independent Set on d-regular Random Graphs

MIS instances on d-regular random graphs (RRGs), matching the
experiment setup of PQQA (Ichikawa and Arai 2024, arXiv:2409.02135v2,
Table 2 / Figure 3). Graphs are generated locally with
`networkx.random_regular_graph` for full reproducibility, then pickled
with `pickle.HIGHEST_PROTOCOL`.

`best_known` is the replica-symmetric (RS) asymptotic MIS density from
Barbier, Krzakala and Zdeborova (2013) multiplied by `n`. It is used as
a published reference for approximation-ratio (ApR) comparisons
`|IS| / best_known`.

## Coverage matrix (arXiv-2409.02135v2 Table 2)

| Degree d | Nodes n   | Seeds | best_known | Bundled | On HF Hub |
| ---: | ---: | ---: | ---: | :---: | :---: |
| 20  | 10 000    | 5 |   2 498 | yes | yes |
| 20  | 100 000   | 5 |  24 980 | yes | yes |
| 20  | 1 000 000 | 5 | 249 800 | yes | yes |
| 100 | 10 000    | 5 |     669 | yes | yes |
| 100 | 100 000   | 5 |   6 690 | yes | yes |
| 100 | 1 000 000 | 5 |  66 900 | yes | yes |

The underlying Barbier RS densities used by the generator are
d=3:0.457, d=5:0.381, d=10:0.307, d=20:0.2498, d=50:0.1738,
d=100:0.0669.

Historical note: earlier snapshots of the generator set d=100 to 0.1360
by mistake (a d=20-like value), inflating `best_known` by ~2x on
d100_n10000 and silently breaking ApR calculations. The bundled
manifests here have been rebuilt with the correct value.

## Layout

```
data/mis-rrg/
  README.md
  .gitignore           # binary .gpickle tracked on HF, not in git
  d20_n10000/
    manifest.jsonl     # 5 instances, best_known, num_edges, seed
    0001.gpickle       # networkx.Graph; n = 10 000, d = 20
    ...
  d20_n100000/         # same structure, n = 10^5
  d20_n1000000/        # same structure, n = 10^6 (--include-huge)
  d100_n10000/
  d100_n100000/
  d100_n1000000/
```

Per-instance metadata example (from d100_n100000/manifest.jsonl):

```
id: mis-rrg-d100-n100000-s0001
file: 0001.gpickle
problem: mis
graph_type: rrg
subset: d100_n100000
num_nodes: 100000
num_edges: 5000000
best_known: 6690.0
best_known_source: Barbier 2013 RS density rho_d=0.0669 x N
generator: rrg
d: 100, n: 100000, seed: 1
```

## Regenerate locally

```bash
python scripts/generate_rrg_instances.py
python scripts/generate_rrg_instances.py --include-huge
```

Run via the qqa CLI:

```bash
qqa bench-run --suite mis-rrg-d20_n10000   --instances 1
qqa bench-run --suite mis-rrg-d100_n100000 --instances 1
qqa bench-run --suite mis-rrg-d20_n1000000 --instances 1
```

## Upstream distribution

All subsets (including n = 10^6) mirror to
`Yuma-Ichikawa/qqa4co-bench` on the Hugging Face Hub, so external
users can reproduce the paper without regenerating graphs locally.

## Citations

```bibtex
@article{ichikawa2024pqqa,
  title   = {Continuous Tensor Relaxation for Combinatorial
             Optimization via Gradient-Based Annealing},
  author  = {Ichikawa, Yuma and Arai, Yamato},
  journal = {arXiv preprint arXiv:2409.02135},
  year    = {2024}
}

@article{barbier2013hard,
  title   = {The Hard-Core Model on Random Graphs Revisited},
  author  = {Barbier, Jean and Krzakala, Florent and
             Zdeborova, Lenka and Zhang, Pan},
  journal = {Journal of Physics: Conference Series},
  volume  = {473},
  pages   = {012021},
  year    = {2013}
}
```
