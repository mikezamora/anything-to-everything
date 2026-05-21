"""LLM shims. MockLLM is used by tests and the default demo.

AnthropicLLM is the optional real-LLM adapter. The `anthropic` SDK is
imported lazily inside AnthropicLLM.__init__ so importing this module
never requires the SDK to be installed.

OllamaLLM talks to a locally-running Ollama HTTP server using stdlib
urllib only — no extra dependencies.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .errors import LlmBadOutputError, LlmUnavailableError


class MockLLM:
    """Deterministic stub LLM: emits canned DSL specs and a fixed-template
    verbalization."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None):
        self.responses = responses or {}

    def emit_dsl(self, prompt: str) -> dict[str, Any]:
        if prompt not in self.responses:
            raise KeyError(f"MockLLM has no canned response for prompt {prompt!r}")
        return self.responses[prompt]

    def verbalize(self, prompt: str, result: dict[str, Any]) -> str:
        lines = [f"Result for: {prompt}"]
        for o in result.get("observables", []):
            lines.append(f"  - {o['field']}@{o['site']} ({o['op']}) = "
                         f"{o['value']:.4f}")
        if "converged" in result:
            lines.append("converged" if result["converged"]
                         else "(did not converge)")
        return "\n".join(lines)


class AnthropicLLM:
    """Real Anthropic-API-backed LLM. The SDK is imported lazily inside
    __init__; importing this module never requires the SDK to be installed.
    """

    def __init__(self, *, model: str = "claude-opus-4-7-1m",
                 schema_text: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "anthropic SDK not installed. `pip install anthropic`."
            ) from exc
        self.model = model
        self._schema_text = schema_text
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def emit_dsl(self, prompt: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError(
            "AnthropicLLM.emit_dsl: implement using the claude-api skill's "
            "prompt-caching pattern. Out of scope for sub-project G tests."
        )

    def verbalize(self, prompt: str, result: dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError(
            "AnthropicLLM.verbalize: implement using the claude-api skill."
        )


class OllamaLLM:
    """Local-Ollama-backed LLM shim.

    Talks to an Ollama HTTP server (default http://localhost:11434) over
    stdlib urllib — no extra deps. The DSL-emission path uses Ollama's
    ``format: "json"`` constraint so the assistant returns structured JSON
    directly; the verbalization path uses free-form text.

    Connection failures and malformed responses are raised as typed
    :class:`BridgeError` subclasses (``BRIDGE_E_LLM_UNAVAILABLE`` /
    ``BRIDGE_E_LLM_BAD_OUTPUT``) so callers can pattern-match on the
    bridge's structured-error contract.
    """

    def __init__(self, model: str = "gemma4:31b", host: str = "localhost",
                 port: int = 11434, timeout: float = 120.0):
        self.model = model
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

    def _chat(self, messages: list[dict[str, str]], *,
              force_json: bool) -> str:
        """POST to /api/chat and return the assistant message content."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if force_json:
            body["format"] = "json"
        data = json.dumps(body).encode("utf-8")
        req = Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except (URLError, HTTPError, OSError) as exc:
            raise LlmUnavailableError(
                f"Could not reach Ollama at {self.base_url}: {exc}",
                details={"base_url": self.base_url, "model": self.model,
                         "error": str(exc)},
            ) from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
            content = payload["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
            raise LlmBadOutputError(
                f"Ollama returned malformed chat response: {exc}",
                details={"model": self.model, "error": str(exc)},
            ) from exc
        if not isinstance(content, str):
            raise LlmBadOutputError(
                "Ollama returned non-string assistant content",
                details={"model": self.model, "type": type(content).__name__},
            )
        return content

    def emit_dsl(self, prompt: str) -> dict[str, Any]:
        content = self._chat(
            [{"role": "user", "content": prompt}],
            force_json=True,
        )
        try:
            spec = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LlmBadOutputError(
                f"Ollama assistant content was not valid JSON: {exc}",
                details={"model": self.model, "content": content[:512]},
            ) from exc
        if not isinstance(spec, dict):
            raise LlmBadOutputError(
                "Ollama assistant JSON was not an object",
                details={"model": self.model, "type": type(spec).__name__},
            )
        return spec

    def verbalize(self, prompt: str, result: dict[str, Any]) -> str:
        user = (
            f"Original prompt: {prompt}\n\n"
            f"Result: {json.dumps(result, indent=2, default=str)}\n\n"
            "Verbalize this result for a user in one sentence."
        )
        return self._chat(
            [{"role": "user", "content": user}],
            force_json=False,
        )
