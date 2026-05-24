"""§10.5 LLM-to-DSL emitter — wraps an Ollama-backed LLM as the upstream
producer of theorem-synthesis DSL.

This closes the B1 gap from ``spec_gap_analysis.md`` (§10.5 acceptance
test): a hardcoded NL prompt → real LLM → structured DSL spec the
substrate consumes. The emitter sits ABOVE the existing ``OllamaLLM``
HTTP shim; the LLM emits a JSON document conforming to
:class:`TheoremSpec` (theorem-statement string + example-verification
inputs), and :func:`emit_theorem_spec` validates + retries on
schema/parse failure.

Contract with the LLM:
- prompt: ``role=system`` carries the schema + few-shot example;
  ``role=user`` carries the NL request.
- response: Ollama's ``format: "json"`` constraint forces a JSON
  object. The emitter validates required keys (``theorem``,
  ``examples``) and that ``theorem`` is parseable by
  :func:`src.qft_pcn.logic.ast.parse`.

Anti-shortcut (§1.1 / memory:anti-shortcut-directive): no canned
fallback when the LLM returns garbage — we raise
:class:`LlmBadOutputError` so the demo loud-fails and the caller can
file an EXTENSIONS entry. The LLM is the load-bearing producer; mocking
it out would defeat the purpose of B1.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .errors import LlmBadOutputError
from .llm import OllamaLLM


# ---------------------------------------------------------------------------
# Schema for the LLM's JSON response.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TheoremSpec:
    """Structured theorem-synthesis spec the LLM produces.

    Attributes:
        theorem: surface-syntax theorem string parseable by
            ``src.qft_pcn.logic.ast.parse``. Example:
            ``"forall x:Nat. (x + 0) == x"``.
        examples: list of ``{"input": str, "expected": str}`` pairs used
            for example-verification (§15.11 step). Each entry is a
            concrete instantiation of the universally-quantified
            theorem.
        signature: free-form human-readable type signature
            (``"length :: List a -> Nat"`` in the spec's literal §15
            target; here adapted to the substrate-supported analog).
        explanation: one-line natural-language description (§15.12
            verbalization seed).
    """
    theorem: str
    examples: tuple[dict[str, str], ...]
    signature: str
    explanation: str


# System prompt — embeds the substrate-supported grammar + a worked
# example so the LLM emits a parseable string on the first try.
_SYSTEM_PROMPT = """\
You are a theorem-spec emitter for a tensor-network proof assistant.

Given a natural-language request for a property of a function on
naturals or lists, you respond with a single JSON object of the form:

{
  "theorem":   "<surface-syntax theorem string>",
  "examples":  [{"input": "...", "expected": "..."}, ...],
  "signature": "<type signature, e.g. 'length :: List a -> Nat'>",
  "explanation": "<one-line natural-language description>"
}

The "theorem" field MUST be parseable by the host's surface-syntax
parser. The supported surface is:

    e  ::= x | n | true | false | \\x:T. e | e e
         | if e then e else e | e op e
         | Zero | Succ e | Nil | Cons e e
         | forall x:T. e | e == e
    T  ::= Int | Bool | Nat | List T | T -> T
    op ::= + | - | * | < | ==

Use Peano naturals: write 0 as "Zero", 1 as "Succ Zero", 3 as
"Succ (Succ (Succ Zero))". The "(x + 0) == x" form is preferred over
"add x Zero == x" because + is a built-in binary operator.

Forall introduces a universally-quantified variable. The substrate
supports forall over Nat with arithmetic and propositional equality.

EXAMPLE INPUT: "Prove that adding zero to any natural number gives
back the same number."

EXAMPLE OUTPUT:
{
  "theorem":   "forall x:Nat. (x + Zero) == x",
  "examples":  [
    {"input": "x = Zero",                       "expected": "Zero"},
    {"input": "x = Succ Zero",                  "expected": "Succ Zero"},
    {"input": "x = Succ (Succ (Succ Zero))",    "expected": "Succ (Succ (Succ Zero))"}
  ],
  "signature": "add_zero_right :: forall x:Nat. (x + 0) == x",
  "explanation": "Right-identity of zero under addition on Peano naturals."
}

Respond with the JSON object only.
"""


class LLMToDslEmitter:
    """Wraps an LLM (default: Ollama gemma4:31b) as the upstream producer
    of theorem-synthesis DSL.

    The LLM is the load-bearing source of the structured spec; this
    class only validates + retries on malformed JSON. There is no
    canned fallback (anti-shortcut §1.1).
    """

    def __init__(
        self,
        llm: OllamaLLM | None = None,
        *,
        model: str = "gemma4:31b",
        host: str | None = None,
        port: int = 11434,
        timeout: float = 180.0,
        max_retries: int = 2,
    ):
        if llm is None:
            # Default host: env QFT_PCN_OLLAMA_HOST -> WSL default ->
            # 'localhost'. On WSL the Windows Ollama instance sits at
            # the host-network gateway; the env override lets the demo
            # find it without hardcoding a brittle IP.
            if host is None:
                host = os.environ.get("QFT_PCN_OLLAMA_HOST", "localhost")
            self.llm = OllamaLLM(model=model, host=host, port=port,
                                 timeout=timeout)
        else:
            self.llm = llm
        self.max_retries = max(1, int(max_retries))

    def emit_theorem_spec(self, nl_prompt: str) -> TheoremSpec:
        """Drive the LLM to emit a parseable :class:`TheoremSpec`.

        Retries up to ``max_retries`` times on JSON-parse failure; the
        last failure is re-raised. No silent fallbacks.
        """
        # Lazy import to avoid the surface-parser dependency at module
        # import time (the demo may run inside contexts where the AST
        # surface is not available).
        from src.qft_pcn.logic.ast import parse as parse_ast

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": nl_prompt},
        ]
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            content = self.llm._chat(messages, force_json=True)
            try:
                obj = json.loads(content)
            except json.JSONDecodeError as exc:
                last_exc = LlmBadOutputError(
                    f"LLM emitted non-JSON content "
                    f"(attempt {attempt + 1}/{self.max_retries}): {exc}",
                    details={"content_head": content[:512]},
                )
                continue
            if not isinstance(obj, dict):
                last_exc = LlmBadOutputError(
                    "LLM emitted non-object JSON",
                    details={"type": type(obj).__name__},
                )
                continue
            missing = [k for k in ("theorem", "examples", "signature",
                                    "explanation") if k not in obj]
            if missing:
                last_exc = LlmBadOutputError(
                    f"LLM emission missing keys: {missing}",
                    details={"got_keys": sorted(obj.keys())},
                )
                continue
            theorem = obj["theorem"]
            if not isinstance(theorem, str):
                last_exc = LlmBadOutputError(
                    "LLM 'theorem' field is not a string",
                    details={"type": type(theorem).__name__},
                )
                continue
            # The host's surface parser is the authoritative validator;
            # an unparseable theorem string is a hard failure.
            try:
                parse_ast(theorem)
            except Exception as exc:
                last_exc = LlmBadOutputError(
                    f"LLM 'theorem' string did not parse: {exc}",
                    details={"theorem": theorem},
                )
                continue
            examples = obj["examples"]
            if not isinstance(examples, list):
                last_exc = LlmBadOutputError(
                    "LLM 'examples' field is not a list",
                    details={"type": type(examples).__name__},
                )
                continue
            ex_tuple: list[dict[str, str]] = []
            for e in examples:
                if not isinstance(e, dict) or "input" not in e \
                        or "expected" not in e:
                    last_exc = LlmBadOutputError(
                        "LLM example missing 'input'/'expected' keys",
                        details={"example": e},
                    )
                    break
                ex_tuple.append({"input":    str(e["input"]),
                                 "expected": str(e["expected"])})
            else:
                return TheoremSpec(
                    theorem=theorem,
                    examples=tuple(ex_tuple),
                    signature=str(obj["signature"]),
                    explanation=str(obj["explanation"]),
                )
        # All retries exhausted.
        assert last_exc is not None
        raise last_exc


__all__ = ["LLMToDslEmitter", "TheoremSpec"]
