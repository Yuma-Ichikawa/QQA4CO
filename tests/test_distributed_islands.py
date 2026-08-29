from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from qqa.compile import SparseQUBO
from qqa.engines import SparseQUBOProblem, anneal_distributed_island
from qqa.engines.distributed import exchange_elites, select_diverse_migrants
from qqa.local import EliteArchive, EliteEntry


def test_select_diverse_migrants_keeps_best_and_diversity():
    candidates = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]])
    selected = select_diverse_migrants(
        candidates,
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
        count=2,
    )
    assert torch.equal(selected[0], candidates[0])
    assert torch.equal(selected[1], candidates[2])


def test_elite_archive_bitpacked_distance_matches_unpacked():
    packed = EliteArchive(maximum_size=4, minimum_distance=0.5, bitpack=True)
    plain = EliteArchive(maximum_size=4, minimum_distance=0.5, bitpack=False)
    entries = (
        EliteEntry(torch.tensor([0, 0, 0, 0]), 3.0, True, 0.0),
        EliteEntry(torch.tensor([0, 0, 0, 1]), 2.0, True, 0.0),
        EliteEntry(torch.tensor([1, 1, 1, 1]), 4.0, True, 0.0),
    )
    assert [packed.add(entry) for entry in entries] == [plain.add(entry) for entry in entries]
    assert [entry.objective for entry in packed.entries] == [
        entry.objective for entry in plain.entries
    ]


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_single_rank_gloo_exchange(tmp_path):
    if dist.is_initialized():
        pytest.skip("A process group is already active in this test process.")
    rendezvous = tmp_path / "gloo-rendezvous"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        elites = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.float32)
        result = exchange_elites(elites)
        assert result.backend == "gloo"
        assert result.world_size == 1
        assert torch.equal(result.gathered_elites, elites)

        qubo = SparseQUBO(
            torch.tensor([-1.0, -0.5, 0.25]),
            torch.tensor([[0, 1], [1, 2]]),
            torch.tensor([1.5, -0.75]),
        )
        island = anneal_distributed_island(
            SparseQUBOProblem(qubo),
            rounds=2,
            migration_size=2,
            sol_size=4,
            num_epochs=4,
            learning_rate=0.05,
            device="cpu",
            seed=7,
            record_history=False,
            verbose=False,
            polish=False,
        )
        assert island.diagnostics["distributed_rounds"] == 2
        assert island.diagnostics["distributed_completed_rounds"] == 2
        assert island.diagnostics["distributed_exchanges"] == 1
        assert island.final_population.shape == (4, 3)
    finally:
        dist.destroy_process_group()
