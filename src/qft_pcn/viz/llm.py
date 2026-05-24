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
from pathlib import Path
from typing import Any

import httpx

from .dsl import validate as dsl_validate

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_EXAMPLES_DIR = Path(__file__).resolve().parent / "llm_examples"


def _load_few_shot() -> list[tuple[str, dict]]:
    """Load the canonical few-shot example DSLs.

    Returns list of (name, parsed-dict). Sorted for deterministic prompt
    ordering."""
    items = []
    for path in sorted(_EXAMPLES_DIR.glob("*.json")):
        items.append((path.stem, json.loads(path.read_text())))
    return items


def build_system_prompt() -> str:
    """Build the §9.2 v1 DSL system prompt with inlined few-shot examples."""
    preamble = (
        "You are a translator from natural-language programming and proof "
        "tasks into a JSON DSL consumed by the QPCN reasoning substrate. "
        "Output ONLY a single JSON object — no prose, no markdown fences.\n"
        "\n"
        "Schema (v1):\n"
        "  - version: \"1\" (required)\n"
        "  - fields: list of {name, cutoff} per quantum species\n"
        "  - sites: integer site count\n"
        "  - boundary: {site_idx_str: {field_name: value}} clamps (optional)\n"
        "  - constraints: list (see below)\n"
        "  - observables: list of {site, field, op}\n"
        "  - search: {method, runtime, steps, chi_max, dt}\n"
        "  - decomposition: {children, execution} (optional, §10.10)\n"
        "\n"
        "Constraint kinds:\n"
        "  - local: {kind, site, term, weight} — predicate over one site\n"
        "  - two_site: {kind, sites, term, weight} — predicate over a pair\n"
        "  - well_typed_subtree: {kind, root, weight} — §10.2 typing rules at root\n"
        "  - example: {kind, input, output, weight} — input/output evaluation\n"
        "  - vocabulary: {kind, primitives, weight} — restrict node_kind labels\n"
        "  - use_lemma: {kind, lemma_id, sites, weight} — §10.8 cached lemma clamp\n"
        "\n"
        "Search runtime:\n"
        "  - mera: hierarchical TEBD; pick for recursive / nested-scope tasks\n"
        "  - mps: flat TEBD; pick for small or non-recursive tasks\n"
        "\n"
        "Observable ops: argmax | n | phi | pi | a | adag | identity\n"
        "(argmax returns the dominant basis label — use it for AST decode.)\n"
    )
    examples = _load_few_shot()
    few_shot = "\n\nExamples:\n\n" + "\n\n".join(
        f"# {name}\n{json.dumps(spec, indent=2)}"
        for name, spec in examples
    )
    return preamble + few_shot


SYSTEM_PROMPT = build_system_prompt()


def list_models() -> list[dict]:
    """Return the Ollama-installed model list (name + size + modified)."""
    r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
    r.raise_for_status()
    return r.json().get("models", [])


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()



def _attempt_generate_dsl(prompt: str, *, model: str, system: str) -> dict:
    """Single round-trip to Ollama; returns a success or failure dict.

    Success: `{"dsl": <validated dict>}`.
    Failure: `{"error", "raw", "validation_errors"}`.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "system": system,
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


def generate_dsl(prompt: str, *, model: str, schema: dict,
                 examples: list[dict], max_retries: int = 0) -> dict:
    """Ask Ollama to emit a DSL for `prompt`. Validate before returning.

    Returns `{"dsl": <validated dict>}` on success, or
    `{"error": str, "raw": str, "validation_errors": list[str]}` on failure.

    If `max_retries > 0` and the first attempt fails validation (or returns
    non-JSON), build a follow-up prompt that quotes the raw output and the
    validation errors, asking the LLM to emit a corrected DSL. Retry up to
    `max_retries` additional times. Return the first success or the final
    failure dict (so callers always see the *last* raw/errors).

    Note: `schema` and `examples` are accepted for API compatibility but the
    §9.2 v1 schema and canonical few-shot examples are baked into SYSTEM_PROMPT.
    """
    result = _attempt_generate_dsl(prompt, model=model, system=SYSTEM_PROMPT)
    if "dsl" in result or max_retries <= 0:
        return result

    for _ in range(max_retries):
        retry_prompt = (
            "Your previous DSL failed validation: "
            f"{result.get('validation_errors') or [result.get('error', '')]}. "
            "The raw output was:\n"
            f"{result.get('raw', '')}\n"
            "Emit a corrected DSL that satisfies the schema."
        )
        result = _attempt_generate_dsl(retry_prompt, model=model,
                                       system=SYSTEM_PROMPT)
        if "dsl" in result:
            return result
    return result


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
