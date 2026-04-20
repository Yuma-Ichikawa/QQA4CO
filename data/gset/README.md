# MaxCut — G-set benchmark

**71 graphs**, best-known cuts tracked, hosted on
[`Yuma-Ichikawa/qqa4co-bench`](https://huggingface.co/datasets/Yuma-Ichikawa/qqa4co-bench)
under `gset/`.

| size               | instances            | typical n, m                                 |
| ------------------ | -------------------- | -------------------------------------------- |
| small              | G1..G21              | n=800, m=800-19 176                          |
| medium             | G22..G54             | n=1 000..3 000                               |
| large              | G55..G67             | n=5 000..7 000                               |
| sparse near-bipart | G70, G72, G77, G81   | n=10 000..20 000                             |

Upstream G68, G69, G71, G73-76, G78-80 are missing from the canonical
Ye mirror and are therefore not included here (the fetch script skips
them automatically).

Layout

```
data/gset/standard/
    G1.gpickle, G2.gpickle, ..., G81.gpickle
    manifest.jsonl
```

Each manifest record carries at least

```json
{"id": "G70", "file": "G70.gpickle", "nodes": 10000, "edges": 9999,
 "best_known": 9591,
 "best_known_source": "Benlic & Hao 2013; Matsuda 2018; Ichikawa NeurIPS 2024",
 "source_url": "https://web.stanford.edu/~yyye/yyye/Gset/G70",
 "problem": "maxcut", "graph_type": "gset", "subset": "standard"}
```

Fetch / regenerate

```bash
# pull the whole family (30 MB) from Hugging Face
./scripts/setup_benchmarks.sh --only gset

# …or re-download from Stanford and re-pickle locally (no HF required)
python scripts/fetch_gset_data.py
```

Run

```bash
qqa bench-run --suite gset --instances 5 --output gset.json
qqa bench-plot bench_results/gset.json --output gset.png
```

Approximation-ratio convention: ``ratio = found_cut / best_known``
(higher = better; 1.0 = matches the published upper bound).
