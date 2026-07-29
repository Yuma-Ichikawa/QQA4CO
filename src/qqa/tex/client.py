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
from typing import Literal

from qqa.tex.schema import MODEL_JSON_SCHEMA

DEFAULT_BASE_URL = "https://api.example.com"
DEFAULT_MODEL = "your-model-id"


class LLMAPIError(RuntimeError):
    """An API transport or response-shape failure with secrets redacted."""


def _redact(message: str, secret: str) -> str:
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return re.sub(
        r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?bearer\s+)[^\s\"']+", r"\1[REDACTED]", message
    )


def _extract_text(response: dict, style: str) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    if style == "messages":
        content = response.get("content", [])
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in (None, "text")
        ]
    else:
        texts = []
        for item in response.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
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
            json.loads(candidate)
            return candidate
        raise LLMAPIError("LLM response did not contain a valid JSON object.") from None


class OpenAICompatibleClient:
    """Call an OpenAI Responses-compatible or Anthropic Messages endpoint."""

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
        self.api_key = (
            api_key
            or os.environ.get("QQA_LLM_API_KEY", "")
            or os.environ.get("QQA_LLM_API_KEY", "")
        )
        if not self.api_key:
            raise ValueError(
                "QQA_LLM_API_KEY (or the legacy QQA_LLM_API_KEY) is not set. "
                "Export it in your shell; do not pass credentials in source code."
            )
        self.base_url = (base_url or os.environ.get("QQA_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
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
        self.model = model or os.environ.get("QQA_LLM_MODEL") or DEFAULT_MODEL
        if api_style not in ("responses", "messages"):
            raise ValueError("api_style must be 'responses' or 'messages'.")
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
            raise ValueError("timeout must be in (0, 600] seconds.")
        if not isinstance(max_retries, int) or not 0 <= max_retries <= 10:
            raise ValueError("max_retries must be an integer in [0, 10].")
        if not isinstance(max_output_tokens, int) or not 128 <= max_output_tokens <= 65536:
            raise ValueError("max_output_tokens must be an integer in [128, 65536].")
        if not isinstance(max_response_bytes, int) or not 1024 <= max_response_bytes <= 20_000_000:
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
        structured = self.api_style == "responses" and self._structured_available is not False
        request_options = {"system_prompt": system_prompt} if system_prompt else {}
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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
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
