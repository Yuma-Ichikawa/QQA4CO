"""Repository layout invariants for portable source checkouts."""

from __future__ import annotations

from pathlib import Path

from qqa.benchmarking import builtin_benchmark_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_redundant_root_directories_do_not_return() -> None:
    forbidden = (
        "benchmark-results",
        "benchmarks",
        "deploy",
        "notebooks",
    )
    assert not [name for name in forbidden if (REPO_ROOT / name).exists()]


def test_documentation_assets_are_not_stored_as_dataset_files() -> None:
    assert not (REPO_ROOT / "data" / "fig").exists()
    assert (REPO_ROOT / "docs" / "assets" / "gallery" / "schedule_default.png").is_file()


def test_builtin_benchmark_manifest_is_the_single_source_of_truth() -> None:
    manifest = builtin_benchmark_manifest("qqa-core")
    assert manifest.name == "qqa-core-portable"
    assert manifest.instances
    audit = builtin_benchmark_manifest("audit-public")
    assert audit.budgets == (1.0, 10.0, 30.0, 300.0)
    assert audit.seeds == (0, 1, 2, 3, 4)
    assert {instance.format for instance in audit.instances} == {
        "miplib-archive",
        "qplib-archive",
    }
