# Large Erdős–Rényi MIS instances

The full public benchmark payload is intentionally not stored in Git. Fetch
the benchmark collection with:

```bash
pip install -e ".[discs]"
./scripts/setup_benchmarks.sh --source hf
```

The files used by the original bundled snapshot were:

| File | SHA-256 |
| --- | --- |
| `ER_9000_11000_0.02_0.gpickle` | `f0bd48ee42ac9e2c2a38a4e4ec4eb28942d14ad5e5bafc73a5c1e66de49ce355` |
| `ER_9000_11000_0.02_1.gpickle` | `26b24ab10d64c8815c95ba9bb546ae6834e0a891e37d39a38191d23e8d4ce396` |
| `ER_9000_11000_0.02_2.gpickle` | `742d9b6fcc92b264a89bfbd8b63706eb4c2793a59180d6fd327ac99de5254362` |
| `ER_9000_11000_0.02_3.gpickle` | `6abebe6f41101a2daf07b9fb00e54ae44d33b0dd91191386c3719994268cb9b2` |
| `ER_9000_11000_0.02_4.gpickle` | `b3752b026c1cd3c3d619418432e142a7659f56b9dc67026e2cc7631c3d86a1f1` |
| `ER_9000_11000_0.02_5.gpickle` | `c1c45e6fe248c475631097bcc5aeb2308e921e51900d2a421ff62c05df981218` |

Source and license metadata are maintained with the public dataset described
in [the benchmark guide](../../../docs/how-to/benchmark.md). Verify a fetched
copy with `sha256sum data/mis/er-large/*.gpickle`.
