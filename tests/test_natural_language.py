"""Unified natural-language planning, routing, and execution tests."""

from __future__ import annotations

import json

import pytest

import qqa
from qqa.cli import main


def _spec(*, objectives: int = 1) -> dict:
    objective_rows = [
        {
            "name": "cost",
            "direction": "min",
            "expression": "square(x - 0.25) + square(n - 2)",
            "unit": "",
        }
    ]
    if objectives > 1:
        objective_rows.append(
            {
                "name": "quality",
                "direction": "max",
                "expression": "x + n",
                "unit": "",
            }
        )
    return {
        "name": "natural-test",
        "variables": [
            {"name": "x", "kind": "real", "lower": 0, "upper": 1, "size": 1},
            {"name": "n", "kind": "integer", "lower": 0, "upper": 4, "size": 1},
        ],
        "objectives": objective_rows,
        "constraints": [],
        "notes": "",
    }


class _NaturalClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.system_prompt = None
        self.prompt = None

    def generate_model_json(self, prompt, *, system_prompt=None):
        self.prompt = prompt
        self.system_prompt = system_prompt
        return json.dumps(self.payload)


def test_natural_language_uses_a_separate_system_prompt_and_routes_blackbox():
    client = _NaturalClient(_spec())
    plan = qqa.compile_natural_language(
        "Tune this expensive black-box experiment under a budget.",
        client=client,
    )
    assert plan.selected_solver == "blackbox"
    assert plan.blackbox_intent
    assert "untrusted" in client.system_prompt
    assert "expensive black-box" in client.prompt


def test_multiobjective_routing_is_local_and_deterministic():
    spec = qqa.ModelSpec.from_dict(_spec(objectives=2))
    plan = qqa.plan_spec(spec, solver="auto")
    assert plan.selected_solver == "pareto"
    with pytest.raises(ValueError, match="one objective"):
        qqa.plan_spec(spec, solver="blackbox")


def test_blackbox_adapter_handles_vector_reductions_and_constraints():
    spec = qqa.ModelSpec.from_dict(
        {
            "name": "vector-blackbox",
            "variables": [{"name": "x", "kind": "real", "lower": 0, "upper": 1, "size": 3}],
            "objectives": [
                {
                    "name": "distance",
                    "direction": "min",
                    "expression": "square(sum(x) - 1.5)",
                    "unit": "",
                }
            ],
            "constraints": [
                {
                    "name": "capacity",
                    "expression": "sum(x)",
                    "sense": "<=",
                    "rhs": 2,
                    "weight": 100,
                    "scale": 2,
                    "tolerance": 1e-8,
                }
            ],
            "notes": "",
        }
    )
    problem = qqa.blackbox_from_spec(spec)
    value, violations, point = problem.evaluate_one(problem.space.pack({"x": [0.5, 0.5, 0.5]}))
    assert value == pytest.approx(0.0)
    assert violations == [0.0]
    assert point["x"] == pytest.approx([0.5, 0.5, 0.5])


def test_ask_accepts_reviewed_spec_without_api_and_solves():
    answer = qqa.ask(
        _spec(),
        solver="qqa",
        sol_size=32,
        num_epochs=120,
        seed=2,
    )
    assert answer.solver == "qqa"
    assert answer.result.score["feasible"]
    assert answer.result.score["value"] < 0.01


def test_cli_ask_can_plan_reviewed_json_without_api_key(tmp_path, capsys):
    source = tmp_path / "model.json"
    source.write_text(json.dumps(_spec()), encoding="utf-8")
    assert (
        main(
            [
                "ask",
                "--spec",
                str(source),
                "--solver",
                "qqa",
                "--plan-only",
                "--show-model",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "solver     : qqa" in output
    assert "validated (plan only)" in output
    assert '"natural-test"' in output
