# Advanced opt-in runtime

QQA4CO keeps `qqa.solve(...)` on the portable pure-QQA route unless you
explicitly enable an advanced runtime. None of the features on this page is
required for ordinary CPU, CUDA, or MPS solves.

## Fused sparse CUDA and CUDA Graphs

Install the optional Triton kernels on a supported Linux/CUDA environment:

```bash
python -m pip install --upgrade "qqa[triton]"
```

Select the fused sparse kernel and static-step CUDA Graph replay explicitly:

```python
import qqa

config = qqa.SolverConfig.for_profile(
    "quality",
    device="cuda",
    sparse_kernel="triton",
    cuda_graphs=True,
)
result = qqa.solve(problem, config=config)
```

`sparse_kernel="auto"` uses Triton only when it is importable and the values
are CUDA tensors; otherwise it uses the portable PyTorch operation.
`cuda_graphs=True` requires CUDA and AdamW. It preserves the schedule inputs and
optimizer state while replaying a fixed-shape training step. Keep it disabled
for dynamic-shape custom relaxations.

## One QQA island per GPU

The distributed engine exchanges a bounded, diverse elite set at coarse round
boundaries. NCCL keeps CUDA migrants device-to-device; Gloo provides the same
contract for CPU testing. Save the following as `distributed_solve.py`:

```python
import os

import networkx as nx
import torch
import torch.distributed as dist

import qqa
from qqa.engines import anneal_distributed_island


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    graph = nx.random_regular_graph(d=3, n=500, seed=0)
    problem = qqa.MaximumIndependentSet(graph, penalty=2.0)
    result = anneal_distributed_island(
        problem,
        device=device,
        sol_size=256,
        num_epochs=2000,
        rounds=4,
        migration_size=16,
        seed=local_rank,
    )
    if dist.get_rank() == 0:
        print(result.best_obj)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Launch one process per visible GPU:

```bash
torchrun --standalone --nproc-per-node=4 distributed_solve.py
```

The function requires an already-initialised process group. This keeps cluster
launch policy outside QQA4CO and makes the same code portable across standard
PyTorch launchers.

## Bounded advanced local search

Sparse binary QUBOs expose explicit, auditable local-search methods:

```python
from qqa.local import iterated_local_search, k_flip_search, tabu_search

incumbent = result.best_sol
tabu = tabu_search(sparse_qubo, incumbent, iterations=500, tenure=11)
kflip = k_flip_search(sparse_qubo, tabu.solution, candidate_width=24)
polished = iterated_local_search(sparse_qubo, kflip.solution, seed=0)
print(polished.objective, polished.moves)
```

Structure-specific functions include `two_opt_tour`, `three_opt_tour`,
`maxcut_fm_search`, `mis_swap_search`, `kempe_coloring_search`, and
`walksat_search`. They return a solution, objective, move count, and method
name without modifying the caller's input.

## Safe QUBO presolve

The QUBO presolve helpers never infer a fixing from an unproved heuristic:

```python
from qqa.presolve import (
    detect_qubo_symmetries,
    dominance_fixings,
    exact_probe_persistency,
    submodular_roof_duality,
)

fixings = dominance_fixings(sparse_qubo)
symmetry_groups = detect_qubo_symmetries(sparse_qubo)

# Bounded exact persistency; raises instead of silently exceeding the limit.
proof = exact_probe_persistency(sparse_qubo, max_variables=24)

# Available when every pair coefficient is non-positive.
roof = submodular_roof_duality(submodular_qubo)
```

General `ModelIR` presolve also performs singleton linear bound propagation and
retains a reversible ledger so returned solutions remain in original space.

## Experimental learned helpers

Learned helpers are imported from `qqa.learned` and are never selected by the
default planner:

```python
from qqa.learned import (
    DiscreteDiffusionGenerator,
    OnlineSolverSelector,
    factor_graph_warm_start,
    model_features,
)

warm_start = factor_graph_warm_start(model_ir, steps=100, device="cuda")

generator = DiscreteDiffusionGenerator(sparse_qubo)
candidates = generator.generate(128, warm_start=warm_start, seed=0)

selector = OnlineSolverSelector(["qqa", "diffusion"])
features = model_features(model_ir)
backend = selector.select(features)
selector.update(backend, features, reward=1.0)
```

Treat the selector's reward definition, learned weights, and diffusion budget as
application policy. For reproducible comparisons, record them alongside the
QQA seed and equal wall-clock budget.
