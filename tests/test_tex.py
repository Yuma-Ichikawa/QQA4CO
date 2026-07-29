"""Safe TeX model compilation, API extraction, and CLI tests."""

from __future__ import annotations

import json

import pytest

import qqa
from qqa.cli import main
from qqa.tex.expressions import UnsafeExpressionError
from qqa.tex.schema import MAX_CONSTRAINTS, MAX_OBJECTIVES, MAX_TOTAL_DIMENSION


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
    def __init__(self):
        self.system_prompt = None

    def generate_model_json(self, prompt, *, system_prompt=None):
        assert "<tex>" in prompt
        self.system_prompt = system_prompt
        return json.dumps(_spec_dict())


class _RepairingFakeClient:
    def __init__(self):
        self.calls = 0

    def generate_model_json(self, prompt, *, system_prompt=None):
        assert system_prompt == qqa.TEX_SYSTEM_PROMPT
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"objective": "wrong shape"})
        assert "validation-error" in prompt
        return json.dumps(_spec_dict())


def test_compile_tex_to_safe_mixed_problem_and_solve():
    client = _FakeClient()
    spec = qqa.compile_tex(r"\min_{x,n} (x-2)^2 + (n-3)^2", client=client)
    assert "untrusted" in client.system_prompt
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


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.com",
        "https://user:password@example.com",
        "https://example.com?token=secret",
        "https://example.com/#fragment",
    ],
)
def test_client_rejects_unsafe_base_urls(base_url):
    with pytest.raises(ValueError, match="base_url must be an HTTPS URL"):
        qqa.OpenAICompatibleClient(api_key="test-key", base_url=base_url)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("api_key", "test-key\ninjected", "control characters"),
        ("base_url", 123, "base_url must be a string"),
        ("model", "model\rheader", "control characters"),
        ("verify_ssl", 1, "verify_ssl must be a boolean"),
        ("max_retries", True, "max_retries must be an integer"),
    ],
)
def test_client_rejects_ambiguous_or_header_unsafe_configuration(keyword, value, message):
    options = {"api_key": "test-key", keyword: value}
    with pytest.raises((TypeError, ValueError), match=message):
        qqa.OpenAICompatibleClient(**options)


def test_client_wraps_malformed_embedded_json_as_api_error(monkeypatch):
    client = qqa.OpenAICompatibleClient(api_key="test-key", max_retries=0)
    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: {"output_text": "prefix {not-json} suffix"},
    )
    with pytest.raises(qqa.LLMAPIError, match="valid JSON object"):
        client.generate_model_json("model request")


@pytest.mark.parametrize(
    ("api_style", "system_field"),
    [("responses", "instructions"), ("messages", "system")],
)
def test_client_sends_system_prompt_separately_and_does_not_redirect_key(
    monkeypatch, api_style, system_field
):
    requests = []
    response_payload = (
        {"content": [{"type": "text", "text": json.dumps(_spec_dict())}]}
        if api_style == "messages"
        else {"output_text": json.dumps(_spec_dict())}
    )

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            assert limit > 1024
            return json.dumps(response_payload).encode()

    def open_request(request, **kwargs):
        requests.append(request)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    client = qqa.OpenAICompatibleClient(
        api_key="test-key",
        api_style=api_style,
        max_retries=0,
    )
    client.generate_model_json("user model", system_prompt="trusted system")
    payload = json.loads(requests[0].data)
    assert payload[system_field] == "trusted system"
    assert "Authorization" not in requests[0].headers
    assert requests[0].unredirected_hdrs["Authorization"] == "Bearer test-key"


def test_model_spec_rejects_excessive_total_variable_dimension():
    source = _spec_dict()
    source["variables"] = [
        {
            "name": "huge",
            "kind": "real",
            "lower": 0,
            "upper": 1,
            "size": MAX_TOTAL_DIMENSION + 1,
        }
    ]
    source["objectives"][0]["expression"] = "sum(huge)"
    with pytest.raises(ValueError, match="total variable dimension"):
        qqa.ModelSpec.from_dict(source)


def test_model_spec_rejects_excessive_objective_count():
    source = _spec_dict()
    objective = source["objectives"][0]
    source["objectives"] = [
        {**objective, "name": f"objective_{index}"} for index in range(MAX_OBJECTIVES + 1)
    ]
    with pytest.raises(ValueError, match=rf"At most {MAX_OBJECTIVES} objectives"):
        qqa.ModelSpec.from_dict(source)


def test_model_spec_rejects_excessive_constraint_count():
    source = _spec_dict()
    source["constraints"] = [
        {
            "name": f"constraint_{index}",
            "expression": "x",
            "sense": "<=",
            "rhs": 5,
            "weight": 1,
            "scale": 1,
            "tolerance": 0,
        }
        for index in range(MAX_CONSTRAINTS + 1)
    ]
    with pytest.raises(ValueError, match=rf"At most {MAX_CONSTRAINTS} constraints"):
        qqa.ModelSpec.from_dict(source)


def test_model_spec_preflight_rejects_vector_valued_objective():
    source = _spec_dict("x")
    source["variables"][0]["size"] = 2
    with pytest.raises(ValueError, match="one scalar per candidate"):
        qqa.ModelSpec.from_dict(source)


def test_model_spec_preflight_rejects_vector_valued_constraint():
    source = _spec_dict("sum(square(x)) + square(n - 3)")
    source["variables"][0]["size"] = 2
    source["constraints"] = [
        {
            "name": "vector-limit",
            "expression": "x",
            "sense": "<=",
            "rhs": 5,
            "weight": 1,
            "scale": 1,
            "tolerance": 0,
        }
    ]
    with pytest.raises(ValueError, match="Constraint 'vector-limit'.*one scalar per candidate"):
        qqa.ModelSpec.from_dict(source)


@pytest.mark.parametrize("expression", ["log(x)", "1 / x"])
def test_model_spec_preflight_rejects_nonfinite_expression_on_declared_domain(expression):
    source = _spec_dict(expression)
    with pytest.raises(ValueError, match="returned NaN or infinity"):
        qqa.ModelSpec.from_dict(source)


def test_model_spec_preflight_accepts_finite_vector_reduction():
    source = _spec_dict("sum(square(x)) + square(n - 3)")
    source["variables"][0]["size"] = 3
    spec = qqa.ModelSpec.from_dict(source)
    assert spec.variables[0].size == 3


def test_model_spec_rejects_boolean_numbers_and_oversized_expressions():
    source = _spec_dict()
    source["variables"][0]["lower"] = False
    with pytest.raises(ValueError, match="JSON real number"):
        qqa.ModelSpec.from_dict(source)

    source = _spec_dict("x" + " + 0" * 1_000)
    with pytest.raises(ValueError, match="too long"):
        qqa.ModelSpec.from_dict(source)
