"""Thin wrapper around the local Ollama HTTP API.

The viz never talks to a managed inference API; the LLM lives on the user's
machine. We expose three entry points:

  * `list_models()` -> Ollama's installed models.
  * `generate_dsl(prompt, model, schema, examples)` -> validated DSL dict
    or {error, raw, validation_errors} on validation failure.
  * `verbalize(observations, model, original_prompt)` -> natural-language
    summary, with `<think>...</think>` chain-of-thought blocks stripped
    out so reasoning models (e.g. deepseek-r1) don't leak internals.

`OLLAMA_URL` is overridable via the `OLLAMA_HOST` env var (matches the
Ollama CLI's own convention).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from .dsl import validate as dsl_validate

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def list_models() -> list[dict]:
    """Return the Ollama-installed model list (name + size + modified)."""
    r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
    r.raise_for_status()
    return r.json().get("models", [])


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _system_prompt_for_dsl(schema: dict, examples: list[dict]) -> str:
    ex_block = "\n\n".join(
        f"Example {i + 1}:\n```json\n{json.dumps(ex, indent=2)}\n```"
        for i, ex in enumerate(examples)
    )
    return (
        "You are a QPCN DSL emitter. Given a problem in natural language, "
        "you respond with ONLY a JSON object conforming to this schema:\n\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n\n"
        "Do not include any prose, markdown, or explanation. Return raw JSON. "
        "Use the cutoff field to bound the per-site Fock truncation (default 2). "
        "Available hamiltonian term kinds: mass, kinetic, quartic, yukawa, "
        "density, curvature_coupling. Available observable operators: n, phi, phi2.\n\n"
        f"{ex_block}"
    )


def generate_dsl(prompt: str, *, model: str, schema: dict,
                 examples: list[dict]) -> dict:
    """Ask Ollama to emit a DSL for `prompt`. Validate before returning.

    Returns `{"dsl": <validated dict>}` on success, or
    `{"error": str, "raw": str, "validation_errors": list[str]}` on failure.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "system": _system_prompt_for_dsl(schema, examples),
        "stream": False,
        "format": "json",
    }
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=120.0)
    r.raise_for_status()
    raw = _strip_think(r.json().get("response", ""))

    try:
        candidate = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"error": f"LLM returned non-JSON: {exc}",
                "raw": raw, "validation_errors": []}

    errors = dsl_validate(candidate)
    if errors:
        return {"error": "DSL validation failed",
                "raw": raw, "validation_errors": errors}

    return {"dsl": candidate}


def verbalize(observations: Any, *, model: str,
              original_prompt: str) -> str:
    """Ask Ollama to explain the run's observations in plain language."""
    body = {
        "model": model,
        "prompt": (
            "Original question:\n"
            f"  {original_prompt}\n\n"
            "Final observables from the QPCN run:\n"
            f"  {json.dumps(observations, indent=2)}\n\n"
            "Briefly explain what the observables mean in the context of the "
            "original question. Keep it under 120 words. No preamble."
        ),
        "stream": False,
    }
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=120.0)
    r.raise_for_status()
    return _strip_think(r.json().get("response", ""))
