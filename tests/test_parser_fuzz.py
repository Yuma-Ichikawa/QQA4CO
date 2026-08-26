from __future__ import annotations

import json
import string
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qqa.io.formats import (
    load_dimacs,
    load_ising_text,
    load_model_ir_json,
    load_opb,
    load_qubo_text,
)

_TEXT_PARSERS = (
    (load_qubo_text, ".qubo"),
    (load_ising_text, ".ising"),
    (load_dimacs, ".cnf"),
    (load_opb, ".opb"),
    (load_model_ir_json, ".json"),
)


@pytest.mark.parametrize(("loader", "suffix"), _TEXT_PARSERS)
@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.text(alphabet=string.printable, max_size=256))
def test_portable_parsers_never_leak_internal_exceptions(loader, suffix: str, payload: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"input{suffix}"
        path.write_text(payload, encoding="utf-8")
        try:
            model = loader(path)
        except (ValueError, TypeError, UnicodeError):
            return
        variable_count = getattr(model, "num_variables", getattr(model, "num_nodes", None))
        assert isinstance(variable_count, int)
        assert 1 <= variable_count <= 1_000_000


@pytest.mark.parametrize(
    ("loader", "suffix", "payload"),
    (
        (load_qubo_text, ".qubo", "999999999999 999999999999 1"),
        (load_ising_text, ".ising", "999999999999 1"),
        (load_dimacs, ".cnf", "p cnf 999999999999 0"),
    ),
)
def test_declared_size_guard_rejects_memory_exhaustion_inputs(
    tmp_path: Path,
    loader,
    suffix: str,
    payload: str,
) -> None:
    path = tmp_path / f"huge{suffix}"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="declare between"):
        loader(path)


def test_json_schema_errors_are_normalized_to_value_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "variables": [{"name": "x", "domain": "binary"}],
                "objective": {"factors": [{"type": "linear", "indices": [0]}]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required"):
        load_model_ir_json(path)
