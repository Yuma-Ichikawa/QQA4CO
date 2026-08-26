"""Disposable native-solver worker used by portable benchmark APIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from qqa.benchmarking.algebraic_runner import _isolated_benchmark_worker
from qqa.hybrid import QQAHeuristicConfig


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("benchmark worker requires request and output files")
    request_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    _isolated_benchmark_worker(
        str(output_path),
        str(request["source_path"]),
        str(request["resolved_format"]),
        str(request["solver"]),
        int(request["seed"]),
        QQAHeuristicConfig(**request["qqa_config"]),
        request.get("reference_records"),
        request["run_kwargs"],
        bool(request["common_import"]),
    )


if __name__ == "__main__":
    main()
