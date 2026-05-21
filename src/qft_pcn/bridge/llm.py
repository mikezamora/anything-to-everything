"""LLM shims. MockLLM is used by tests and the default demo.

AnthropicLLM is the optional real-LLM adapter. The `anthropic` SDK is
imported lazily inside AnthropicLLM.__init__ so importing this module
never requires the SDK to be installed.
"""

from __future__ import annotations

from typing import Any


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
