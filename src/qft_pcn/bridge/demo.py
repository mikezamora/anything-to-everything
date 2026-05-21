"""End-to-end demo: mock LLM emits DSL -> runtime runs -> verbalize.

Run via `python -m src.qft_pcn.bridge.demo`.
"""

from __future__ import annotations

import argparse
import json
import sys

from .llm import MockLLM, AnthropicLLM
from .runtime import run_problem


CANNED_PROMPT = "find a configuration where site 0 holds occupation 2 on field x"

CANNED_DSL = {
    "fields": [{"name": "x", "cutoff": 4}],
    "sites": 2,
    "constraints": [],
    "boundary": {"0": {"x": 2}},
    "observables": [{"site": 0, "field": "x", "op": "n"}],
    "search": {"method": "imag_time", "steps": 20, "chi_max": 4},
}


def render_transcript(prompt: str, dsl: dict, result: dict,
                      verbalization: str, *, out=None) -> None:
    if out is None:
        out = sys.stdout
    out.write("=" * 60 + "\n")
    out.write("PROMPT\n")
    out.write("=" * 60 + "\n")
    out.write(prompt + "\n\n")

    out.write("=" * 60 + "\n")
    out.write("DSL\n")
    out.write("=" * 60 + "\n")
    out.write(json.dumps(dsl, indent=2) + "\n\n")

    out.write("=" * 60 + "\n")
    out.write("OBSERVABLES\n")
    out.write("=" * 60 + "\n")
    for o in result["observables"]:
        out.write(f"  site={o['site']:>3} field={o['field']:<8} "
                  f"op={o['op']:<8} value={o['value']:.6f} "
                  f"(imag={o['imag_part']:.2e})\n")
    out.write("\n")

    out.write("=" * 60 + "\n")
    out.write("ENERGY\n")
    out.write("=" * 60 + "\n")
    out.write(f"  total = {result['energy']:.6f}\n")
    out.write(f"  per-term = {result['energy_per_term']}\n")
    out.write(f"  converged = {result['converged']}\n\n")

    out.write("=" * 60 + "\n")
    out.write("VERBALIZATION\n")
    out.write("=" * 60 + "\n")
    out.write(verbalization + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", choices=["mock", "anthropic"], default="mock")
    args = p.parse_args(argv)

    if args.llm == "mock":
        llm = MockLLM(responses={CANNED_PROMPT: CANNED_DSL})
    else:                                              # pragma: no cover
        llm = AnthropicLLM()

    prompt = CANNED_PROMPT
    dsl = llm.emit_dsl(prompt)
    result = run_problem(dsl).to_dict()
    verbalization = llm.verbalize(prompt, result)
    render_transcript(prompt, dsl, result, verbalization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
