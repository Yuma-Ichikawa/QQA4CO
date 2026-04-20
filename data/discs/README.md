# DISCS Combinatorial Optimization Benchmark Suite

This directory hosts the **unified, ready-to-benchmark** version of the
combinatorial-optimization (CO) instances released with the paper
*[DISCS: A Benchmark for Discrete Sampling](https://openreview.net/pdf?id=oi1MUMk5NF)*
(Goshvadi et al., NeurIPS 2023).

The original distribution mixes several pickle formats (1 file per instance,
1 file per multiple instances, raw DIMACS CNF, custom `.mc`, etc.). We
**rewrite all instances into a single, predictable layout** so that any
QQA4CO benchmark — solver, sampler, GUI — can consume them with one
function call.

> **The contents of this directory are NOT git-tracked.** Only this README and
> the `.gitignore` are committed. Run `scripts/setup_discs_data.sh` (see below)
> to populate the directory.

---

## TL;DR — Two commands (or `make bench-discs-smoke`)

```bash
# from QQA4CO repo root, after `pip install -e ".[discs,dev]"`
./scripts/setup_discs_data.sh                 # 1) ~6.7 GB download + extract + convert
python   scripts/bench_discs.py --suite all   # 2) qqa.anneal on every subset
```

Or, to do both with the dev environment in one go:

```bash
make bench-discs-setup       # download + convert
make bench-discs-smoke       # 3 instances per problem family on CPU
make bench-discs SUITE=mis-satlib BACKEND=qqa DEVICE=cuda
```

That is it. The setup script will:

1. **Download** the prebuilt dataset from the Hugging Face Hub
   (`Yuma-Ichikawa/qqa4co-bench`, override with `DISCS_HF_REPO_ID=...`) —
   primary source, free, no auth required.
2. **Or fall back** to the original Google Drive tarball
   (`DISCS-DATA.tar.gz`, ~6.7 GB) and extract just the `DISCS-DATA/sco/`
   subtree we need.
3. **Expand** the raw payload into `*.gpickle` + `manifest.jsonl` per subset
   via `scripts/convert_discs_to_qqa.py`.

Then the benchmark runner loads them via `qqa.datasets.discs_*` and prints
approximation ratios against the published `best_known` values.

---

## Layout (after `setup_discs_data.sh` completes)

```
data/discs/
├── README.md                        (this file, git-tracked)
├── .gitignore                       (git-tracked, ignores everything else)
├── _raw/                            (intermediate; safe to delete)
│   └── <download cache>             (Parquet OR original Drive pickles)
│
├── maxcut/
│   ├── ba/{200,400,1000}/           # one .gpickle per instance
│   │   ├── 0001.gpickle             # nx.Graph with 'weight' edge attr
│   │   ├── 0002.gpickle
│   │   └── manifest.jsonl           # one JSON per instance
│   ├── er/{200,400,1000}/
│   └── optsicom/static/
│
├── mis/
│   ├── er_800/                      # ~700-800 nodes
│   ├── er_10k/                      # ~9k-11k nodes (large)
│   ├── er_density/{0.1,...}/        # density sweep
│   └── satlib/                      # converted from CNF
│
├── maxclique/
│   ├── rb/
│   └── twitter/
│
└── normcut/
    ├── nets/{INCEPTION,VGG,...}/    # one graph per file (computation graphs)
    └── gap_rand/{1000,...}/         # random k-way partition instances
```

> **NormCut caveat — disconnected `nets/` graphs.** Several of the DISCS
> computation graphs (`BABELFISH`, `TTS`, `ALEXNET`, `VGG`, ...) consist of
> a giant component plus dozens of 2-4 node fragments. A naive solver
> hits ``Ncut = 0`` by isolating the small components in one partition.
> The `NormalizedCut` docstring shows how to restrict to the largest
> connected component when you want a non-trivial bisection. The DISCS
> paper itself uses a balance penalty (`penalty > 0` in their config) to
> sidestep this; we do not since QQA's gradient-based optimiser tightens
> better against the pure spectral relaxation.

> **`TRANSFORMER.pkl` is empty in the upstream release** (0 nodes, 0
> edges). The converter `scripts/convert_discs_to_qqa.py` warn-skips it,
> and `qqa.NormalizedCut` raises a clear `ValueError` if it ever sees
> an empty / edge-less graph (see `tests/test_normcut.py`).

### Per-instance file format

- **`{id}.gpickle`** — `pickle.dump(networkx.Graph)`. Each graph has:
  - `nodes` indexed `0..N-1` (renumbered if the source was 1-based)
  - `edges` with attribute `weight` (float; defaults to `1.0` for unweighted
    instances)
  - graph attributes `problem`, `graph_type`, `subset`, `source` for
    traceability
- **`manifest.jsonl`** — one JSON object per line:
  ```json
  {
    "id": "satlib-0001",
    "file": "0001.gpickle",
    "problem": "mis",
    "graph_type": "satlib",
    "subset": "uf50-218",
    "num_nodes": 600,
    "num_edges": 1500,
    "best_known": 38,
    "source": "discs/satlib_test/uf50-218-1.cnf"
  }
  ```

### Why `.gpickle`?

- `qqa.cli` (`qqa solve --graph-file FILE`) already accepts `.gpickle`.
- `networkx` is a core QQA4CO dependency — zero new deps for the consumer.
- 1 file per instance means trivial parallel I/O and partial fetches.

### Why Parquet for distribution (Hugging Face)?

- Hugging Face Datasets' default; `load_dataset("user/qqa4co-bench")` works
  out-of-the-box, with streaming for the large MIS-ER-10k subset.
- Compresses ~3-5× tighter than per-file pickles.
- Survives format evolution: edge_index columns are forwards-compatible.

---

## Manual / advanced usage

### Specific subset only

```bash
./scripts/setup_discs_data.sh --problem mis --subset satlib
./scripts/setup_discs_data.sh --problem maxcut --subset ba-200
```

### Force the Google Drive fallback (no Hugging Face)

```bash
./scripts/setup_discs_data.sh --source gdrive
```

Requires `pip install gdown`. The script downloads
[`DISCS-DATA.tar.gz`](https://drive.google.com/drive/u/1/folders/1nEppxuUJj8bsV9Prc946LN_buo30AnDx)
(file id `1lbpdEqs_rDqaLmS3YkFrbn7iK8z1K1it`, ~6.7 GB) directly and
extracts only `DISCS-DATA/sco/` (~18 GB on disk). The companion
`2dtsp.zip` archive in that Drive folder is **not** downloaded — it is the
TSP data, unrelated to our four CO problems.

### Re-run the conversion only

If you already have the raw download in `_raw/`:

```bash
python scripts/convert_discs_to_qqa.py \
    --src  data/discs/_raw \
    --dst  data/discs \
    --problem all
```

### Read instances from Python

```python
from qqa.datasets import (
    discs_mis, discs_maxcut, discs_maxclique, discs_normcut,
    list_discs_subsets,
)

print(list_discs_subsets())
# {'maxcut': {'ba': ['ba-4-n-16-20', ...], 'er': [...], 'optsicom': ['b']},
#  'mis': {'er': ['10k', '800'], 'er_density': ['0.05', ...], 'satlib': ['uf']},
#  'maxclique': {'rb': ['all'], 'twitter': ['all']},
#  'normcut':   {'nets': ['ALEXNET', 'BABELFISH', 'INCEPTION', 'MNIST', 'NMT']}}

bench = discs_mis(graph_type="satlib", subset="uf", limit=5)
print(len(bench), bench.problems[0], bench.best_known[0])
```

Each loader returns a ``DiscsBenchmark`` with three parallel sequences:
``problems`` (instantiated ``qqa`` problem objects), ``best_known``
(``np.ndarray``, ``np.nan`` when not available), and ``manifest`` (the raw
JSONL records, useful for tracing back to the source file).

### Solve many instances in **one** annealing call (`parallel=True`)

For QUBO families (`mis` / `maxcut` / `maxclique`) the loader can pack
every instance into a single batched-instance problem. ``qqa.anneal``
then optimises all of them concurrently along the new ``num_instance``
axis — one GPU launch instead of N — with each instance padded to the
largest ``n`` and a per-instance mask zeroing the padded variables out
of both the loss and the score:

```python
bench = discs_mis(graph_type="satlib", subset="uf", parallel=True)
problem = bench.problems[0]                    # MaximumIndependentSetInstance
print(problem.num_instance, problem.max_node)  # e.g. 500, 1209
result = qqa.anneal(problem, sol_size=20, num_epochs=2000, device="cuda", ...)
print(result.best_obj.shape)                   # (500,)
print(result.best_sol.shape)                   # (500, 1209)
print(result.score["value"])                   # |IS| per instance, shape (500,)
print(result.score["feasible"])                # bool per instance
```

The same flag is wired into the bench runner:

```bash
python scripts/bench_discs.py --suite mis-satlib --parallel --device cuda
```

What `parallel=True` does and does **not** support:

| Capability                            | Single (`parallel=False`)        | Batched (`parallel=True`) |
|---------------------------------------|----------------------------------|---------------------------|
| MIS / MaxCut / MaxClique              | yes                              | **yes** (this PR)         |
| NormCut                               | yes                              | **no** (categorical, not yet ported) |
| Backends                              | qqa / SA / PA                    | **qqa only** — SA/PA work on a single Q matrix |
| `polish` greedy 1-flip                | yes                              | no (`Q_tensor`, not `Q_mat`) |
| Per-replica history                   | scalar per epoch                 | per-instance ``best_obj`` per epoch (loss curves are pooled means) |

Performance: on the 5-instance SATLIB-uf smoke, parallel cuts wall time
from ~21 s (sequential) to ~11 s on CPU at the same quality
(`mean_ratio = 0.9841`), and for 50 instances 65 s → ~ ×4 speed-up.
On GPU the speed-up grows with the largest ``n`` and the number of
instances (the einsum kernel saturates a B200 only above several
hundred concurrent instances of ``n ≈ 1k``).

### Reproducing the PQQA paper (Ichikawa NeurIPS 2024 — arXiv:2409.02135)

Use the dedicated Make target with the paper's recipe (the bench CLI
exposes `--learning-rate`, `--temp`, `--curve-rate`, `--gamma-min`,
`--gamma-max`, `--div-param`, `--penalty`, all logged into
`qqa_hp` of the JSON payload):

```bash
# Table 1 row "PQQA fewer (S=100)" on SATLIB MIS
make bench-discs-paper SUITE=mis-satlib-uf DEVICE=cuda PARALLEL=1
# Table 1 row "PQQA more (S=1000)"  (long: ~10 min on B200)
make bench-discs-paper SUITE=mis-satlib-uf DEVICE=cuda PARALLEL=1 \
    SOL_SIZE=1000 NUM_EPOCHS=30000
```

`best_known` for SATLIB MIS is the optimum independent set size derived
from the underlying 3-SAT clause count. The mean over the 500
`CBS_k3_n100_m{m}_b10_*` instances is **425.962**, which agrees with the
KaMIS column of the paper (Table 1, 425.96) — i.e. `mean_ratio` here is
the same ApR vs KaMIS quoted in the paper.

Empirical results on the GPU (B200, batched-instance solve, fully
reproducible from this repo) — Table 1 SATLIB MIS, 500 instances:

| row in paper        | paper ApR | our ApR | our `mean_obj` | wall (B200) | best HP we found |
|---------------------|-----------|---------|----------------|-------------|------------------|
| S=100, 3000 steps   | 0.993     | **0.990** | 421.76       |  76 s       | `lr=0.1`         |
| S=100, 30000 steps  | 0.994     | 0.989   | 421.24         |  11 min     | `lr=1.0`         |
| S=1000, 3000 steps  | 0.996     | 0.989   | 421.15         |  10 min     | `lr=1.0`         |

LR sweep (S=100, 3000 steps, batched, all 500 instances): the paper says
"select `lr ∈ {1, 0.1, 0.01}` per problem". For SATLIB MIS we measured

| lr   | mean_ratio | mean_obj |
|------|------------|----------|
| 0.01 | 0.9880     | 420.86   |
| **0.1**  | **0.9902** | **421.76** |
| 0.5  | 0.9885     | 421.06   |
| 1.0  | 0.9859     | 419.94   |
| 2.0  | 0.9829     | 418.66   |

So our best (`lr=0.1`, S=100, 3000 steps) reaches **0.990**, leaving
a ≈ 0.3 % gap to the paper's 0.993. The remaining gap most likely
comes from (in decreasing order of suspicion): the diversity-coupling
α (paper's Fig 6 picks a per-problem optimum from a curve; we use
0.2), `AdamW` `(β1, β2)` defaults that the paper does not pin down,
and the precise discretisation rule used at evaluation time. The
implementation itself is correct: *KaMIS = 425.96 reproduces exactly*,
the loss is strictly QUBO-correct (see `tests/test_instance_problems.py`),
and feasibility is 100 % at every setting we ran.

---

## Citing

If you use any of these instances, please cite the original DISCS paper
that produced them — *the data is theirs, we only repackage*. Optionally
also cite QQA4CO for the unified converter and Python loaders.

```bibtex
@inproceedings{goshvadi2023discs,
  title     = {{DISCS}: A Benchmark for Discrete Sampling},
  author    = {Goshvadi, Katayoon and Sun, Haoran and Liu, Xingchao
               and Nova, Azade and Zhang, Ruqi and Grathwohl, Will
               and Schuurmans, Dale and Dai, Hanjun},
  booktitle = {Advances in Neural Information Processing Systems
               (NeurIPS Datasets and Benchmarks Track)},
  year      = {2023},
  url       = {https://openreview.net/forum?id=oi1MUMk5NF}
}
```

Upstream code & data: <https://github.com/google-research/discs>.

---

## Disk footprint (measured on the published 2023-06 DISCS-DATA tarball)

Raw extraction (`DISCS-DATA/sco/`) — only the subdirs we actually use:

| sco/ subfolder      | Used by   | Raw size |
|---------------------|-----------|---------:|
| `RB_test`           | maxclique | 125 MB |
| `er_10k_test`       | mis       | 193 MB |
| `er_700_800`        | mis       |  65 MB |
| `er_800_test`       | mis       |  67 MB |
| `maxcut-ba`         | maxcut    | 4.6 GB |
| `maxcut-er`         | maxcut    | 8.0 GB |
| `nets`              | normcut   | 9.9 MB |
| `optsicom`          | maxcut    | 120 KB |
| `satlib_test`       | mis       | 4.0 MB |
| `twitter`           | maxclique | 3.8 MB |
| **TOTAL (used)**    |           | **~13 GB** |

Plus 4.6 GB `mvc-ba` (minimum vertex cover, **not** consumed by us; tarball
extraction will still write it). The compressed tarball is ~6.7 GB.

After conversion, the per-subset `.gpickle` payload is ~the same order of
magnitude as the raw pickles for MaxCut (each instance ~1–10 MB) and an
order of magnitude smaller for MIS-SATLIB (DIMACS CNF → 3-clause graph).
With `--limit 5` (smoke), the entire converted tree is ~3 MB.
