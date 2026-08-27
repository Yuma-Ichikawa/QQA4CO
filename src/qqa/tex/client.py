"""Minimal Responses/Messages client with credential-safe failures."""

from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from numbers import Real
from typing import Any, Literal

from qqa.tex.schema import MODEL_JSON_SCHEMA

# QQA intentionally ships no provider-specific endpoint or model. Empty
# constants preserve the public configuration surface while requiring users
# to select their own OpenAI-compatible profile explicitly.
DEFAULT_BASE_URL = ""
DEFAULT_MODEL = ""
MAX_PROMPT_CHARACTERS = 250_000


class LLMAPIError(RuntimeError):
    """An API transport or response-shape failure with secrets redacted."""


def _redact(message: str, secret: str) -> str:
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return re.sub(
        r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?bearer\s+)[^\s\"']+", r"\1[REDACTED]", message
    )


def _validated_text(
    value: str,
    label: str,
    *,
    maximum: int,
    allow_layout_controls: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    if len(value) > maximum:
        raise ValueError(f"{label} is too long (maximum {maximum:,} characters).")
    allowed_controls = {"\t", "\n", "\r"} if allow_layout_controls else set()
    if any(
        (ord(character) < 32 and character not in allowed_controls) or ord(character) == 127
        for character in value
    ):
        raise ValueError(f"{label} must not contain control characters.")
    return value


def _extract_text(response: dict, style: str) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    if style == "messages":
        content = response.get("content", [])
        if not isinstance(content, list):
            raise LLMAPIError("LLM response content must be a list.")
        texts = [
            item.get("text", "")
            for item in content
            if (
                isinstance(item, dict)
                and item.get("type") in (None, "text")
                and isinstance(item.get("text", ""), str)
            )
        ]
    else:
        texts = []
        output = response.get("output", [])
        if not isinstance(output, list):
            raise LLMAPIError("LLM response output must be a list.")
        for item in output:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content", [])
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if (
                    isinstance(content, dict)
                    and content.get("type") in (None, "output_text", "text")
                    and isinstance(content.get("text"), str)
                ):
                    texts.append(content["text"])
    text = "".join(texts).strip()
    if not text:
        raise LLMAPIError("LLM response did not contain text output.")
    return text


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            candidate = stripped[start : end + 1]
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                pass
            else:
                return candidate
        raise LLMAPIError("LLM response did not contain a valid JSON object.") from None


class OpenAICompatibleClient:
    """Call an OpenAI Responses-compatible or Anthropic Messages endpoint."""

    api_key: str
    base_url: str
    model: str
    api_style: Literal["responses", "messages"]
    verify_ssl: bool
    timeout: float
    max_retries: int
    max_output_tokens: int
    max_response_bytes: int
    _structured_available: bool | None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_style: Literal["responses", "messages"] = "responses",
        verify_ssl: bool = True,
        timeout: float = 120.0,
        max_retries: int = 2,
        max_output_tokens: int = 4096,
        max_response_bytes: int = 2_000_000,
    ):
        resolved_api_key = api_key or os.environ.get("QQA_LLM_API_KEY", "")
        if not resolved_api_key:
            raise ValueError(
                "QQA_LLM_API_KEY is not set. Export it in your shell; "
                "do not pass credentials in source code."
            )
        self.api_key = _validated_text(resolved_api_key, "api_key", maximum=16_384)
        resolved_base_url = base_url or os.environ.get("QQA_LLM_BASE_URL") or DEFAULT_BASE_URL
        if not resolved_base_url:
            raise ValueError(
                "QQA_LLM_BASE_URL is not set. Configure the base URL of your "
                "OpenAI-compatible endpoint."
            )
        if not isinstance(resolved_base_url, str):
            raise TypeError("base_url must be a string.")
        self.base_url = resolved_base_url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an HTTPS URL with a host and without credentials, "
                "query parameters, or a fragment."
            )
        resolved_model = model or os.environ.get("QQA_LLM_MODEL") or DEFAULT_MODEL
        if not resolved_model:
            raise ValueError(
                "QQA_LLM_MODEL is not set. Configure a model supported by your endpoint."
            )
        self.model = _validated_text(resolved_model, "model", maximum=512)
        if api_style not in ("responses", "messages"):
            raise ValueError("api_style must be 'responses' or 'messages'.")
        if not isinstance(verify_ssl, bool):
            raise TypeError("verify_ssl must be a boolean.")
        if isinstance(timeout, bool) or not isinstance(timeout, Real) or not 0 < timeout <= 600:
            raise ValueError("timeout must be in (0, 600] seconds.")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 10
        ):
            raise ValueError("max_retries must be an integer in [0, 10].")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 128 <= max_output_tokens <= 65536
        ):
            raise ValueError("max_output_tokens must be an integer in [128, 65536].")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or not 1024 <= max_response_bytes <= 20_000_000
        ):
            raise ValueError("max_response_bytes must be an integer in [1024, 20000000].")
        self.api_style = api_style
        self.verify_ssl = verify_ssl
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.max_response_bytes = max_response_bytes
        self._structured_available: bool | None = None

    def generate_model_json(self, prompt: str, *, system_prompt: str | None = None) -> str:
        """Generate one model JSON document, preferring Structured Outputs."""
        prompt = _validated_text(
            prompt,
            "prompt",
            maximum=MAX_PROMPT_CHARACTERS,
            allow_layout_controls=True,
        )
        if system_prompt is not None:
            system_prompt = _validated_text(
                system_prompt,
                "system_prompt",
                maximum=MAX_PROMPT_CHARACTERS,
                allow_layout_controls=True,
            )
        structured = self.api_style == "responses" and self._structured_available is not False
        request_options: dict[str, Any] = {}
        if system_prompt is not None:
            request_options["system_prompt"] = system_prompt
        try:
            response = self._request(
                prompt,
                structured=structured,
                timeout=min(self.timeout, 30.0) if structured else self.timeout,
                max_retries=0 if structured else self.max_retries,
                **request_options,
            )
            if structured:
                self._structured_available = True
        except LLMAPIError as exc:
            # Some OpenAI-compatible gateways have not implemented
            # Responses ``text.format``. Retry once with prompt-enforced JSON.
            fallback_errors = ("HTTP 400", "HTTP 422", "timed out", "timeout")
            if not structured or not any(item in str(exc) for item in fallback_errors):
                raise
            self._structured_available = False
            response = self._request(prompt, structured=False, **request_options)
        return _extract_json_text(_extract_text(response, self.api_style))

    def _request(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        structured: bool,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> dict:
        timeout = self.timeout if timeout is None else timeout
        max_retries = self.max_retries if max_retries is None else max_retries
        if self.api_style == "responses":
            url = f"{self.base_url}/v1/responses"
            payload: dict = {
                "model": self.model,
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": self.max_output_tokens,
            }
            if system_prompt:
                payload["instructions"] = system_prompt
            if structured:
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "qqa_optimization_model",
                        "description": "A validated optimisation model for QQA.",
                        "strict": True,
                        "schema": MODEL_JSON_SCHEMA,
                    }
                }
            headers = {}
        else:
            url = f"{self.base_url}/v1/messages"
            payload = {
                "model": self.model,
                "max_tokens": self.max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                payload["system"] = system_prompt
            headers = {"anthropic-version": "2023-06-01"}

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **headers,
            },
        )
        # urllib copies ordinary headers to redirected requests. Mark the
        # credential as unredirected so a gateway cannot forward it to another
        # origin through a 30x response.
        request.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        context = ssl.create_default_context()
        if not self.verify_ssl:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=timeout,
                    context=context,
                ) as response:
                    content_type = ""
                    if getattr(response, "headers", None) is not None:
                        content_type = response.headers.get("Content-Type", "")
                    if content_type and not (
                        "application/json" in content_type.lower()
                        or "+json" in content_type.lower()
                    ):
                        raise LLMAPIError(
                            "LLM endpoint returned a non-JSON Content-Type "
                            f"({content_type.split(';', 1)[0]})."
                        )
                    raw = response.read(self.max_response_bytes + 1)
                    if len(raw) > self.max_response_bytes:
                        raise LLMAPIError(f"LLM response exceeded {self.max_response_bytes} bytes.")
                    result = json.loads(raw.decode("utf-8"))
                if not isinstance(result, dict):
                    raise LLMAPIError("LLM endpoint returned a non-object JSON response.")
                return result
            except urllib.error.HTTPError as exc:
                detail = exc.read(4096).decode("utf-8", errors="replace")
                message = _redact(f"LLM API HTTP {exc.code}: {detail}", self.api_key)
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt >= max_retries:
                    raise LLMAPIError(message) from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                message = _redact(f"LLM API request failed: {exc}", self.api_key)
                if attempt >= max_retries:
                    raise LLMAPIError(message) from exc
            time.sleep(min(2**attempt, 8))
        raise AssertionError("unreachable")


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "LLMAPIError",
    "OpenAICompatibleClient",
]
