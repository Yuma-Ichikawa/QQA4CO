"""Safe TeX model compilation, API extraction, and CLI tests."""

from __future__ import annotations

import json

import pytest

import qqa
from qqa.cli import main
from qqa.tex.expressions import UnsafeExpressionError


def _spec_dict(expression: str = "square(x - 2) + square(n - 3)") -> dict:
    return {
        "name": "tex-quadratic",
        "variables": [
            {"name": "x", "kind": "real", "lower": -5, "upper": 5, "size": 1},
            {"name": "n", "kind": "integer", "lower": 0, "upper": 6, "size": 1},
        ],
        "objectives": [
            {
                "name": "loss",
                "direction": "min",
                "expression": expression,
                "unit": "",
            }
        ],
        "constraints": [],
        "notes": "test",
    }


class _FakeClient:
    def generate_model_json(self, prompt):
        assert "<tex>" in prompt
        return json.dumps(_spec_dict())


class _RepairingFakeClient:
    def __init__(self):
        self.calls = 0

    def generate_model_json(self, prompt):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"objective": "wrong shape"})
        assert "validation-error" in prompt
        return json.dumps(_spec_dict())


def test_compile_tex_to_safe_mixed_problem_and_solve():
    spec = qqa.compile_tex(r"\min_{x,n} (x-2)^2 + (n-3)^2", client=_FakeClient())
    problem = qqa.problem_from_spec(spec)
    result = problem.solve(sol_size=32, num_epochs=200, verbose=False)
    assert result.score["value"] < 1e-4
    assert result.score["extra"]["variables"]["n"] == 3
    assert result.score["extra"]["variables"]["x"] == pytest.approx(2, abs=0.02)


def test_compile_tex_repairs_schema_invalid_output_once():
    client = _RepairingFakeClient()
    spec = qqa.compile_tex(r"\min x^2", client=client)
    assert spec.name == "tex-quadratic"
    assert client.calls == 2


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "x.__class__",
        "[value for value in x]",
        "open('/tmp/pwned')",
    ],
)
def test_model_spec_rejects_executable_or_unknown_syntax(expression):
    with pytest.raises((UnsafeExpressionError, ValueError)):
        qqa.ModelSpec.from_dict(_spec_dict(expression))


def test_cli_solves_audited_spec_without_api_key(tmp_path, capsys):
    source = tmp_path / "model.json"
    source.write_text(json.dumps(_spec_dict()), encoding="utf-8")
    assert main(["tex", "--spec", str(source), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "validated (dry run)" in output


def test_client_requires_environment_key(monkeypatch):
    monkeypatch.delenv("QQA_LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="QQA_LLM_API_KEY"):
        qqa.OpenAICompatibleClient()


def test_api_secret_is_redacted_from_transport_failures(monkeypatch):
    import urllib.error

    secret = "do-not-leak-this-test-secret"
    client = qqa.OpenAICompatibleClient(api_key=secret, max_retries=0)

    def fail(*args, **kwargs):
        raise urllib.error.URLError(f"Authorization: Bearer {secret}")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(qqa.LLMAPIError) as error:
        client.generate_model_json("test")
    assert secret not in str(error.value)


def test_client_caches_structured_output_fallback(monkeypatch):
    client = qqa.OpenAICompatibleClient(api_key="test-key", max_retries=0)
    calls = []
    response = {
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(_spec_dict()),
                    }
                ]
            }
        ]
    }

    def fake_request(prompt, *, structured, timeout=None, max_retries=None):
        calls.append(structured)
        if structured:
            raise qqa.LLMAPIError("request timed out")
        return response

    monkeypatch.setattr(client, "_request", fake_request)
    assert json.loads(client.generate_model_json("first"))["name"] == "tex-quadratic"
    assert json.loads(client.generate_model_json("second"))["name"] == "tex-quadratic"
    assert calls == [True, False, False]
