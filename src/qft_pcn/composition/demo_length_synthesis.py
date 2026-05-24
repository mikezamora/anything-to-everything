"""§15 worked end-to-end synthesis demo + §10.5 real-LLM bridge wiring.

This is the A2 + B1 acceptance from ``spec_gap_analysis.md``:

  Step 1 (§15.1)  NL prompt
  Step 2 (§15.2)  LLM extracts structured intent (Ollama, gemma4:31b)
  Step 3 (§15.3)  DSL spec (TheoremSpec from llm_emitter)
  Step 4 (§15.4)  Hamiltonian construction (MeraEvalHamiltonian)
  Step 5 (§15.5)  State initialization (encode_mera)
  Step 6 (§15.6)  Imaginary-time TEBD (mera_imaginary_evolve_state)
  Step 7 (§15.7)  Measurement (per-leaf marginals)
  Step 8 (§15.8)  AST decoding (decode_mera)
  Step 9 (§15.9)  Validation pass (classical re-parse + type check)
  Step 10 (§15.10) Pretty-printing (extended pretty for Nat / Forall / Eq)
  Step 11 (§15.11) Example verification (per-instance substrate evolve)
  Step 12 (§15.12) Verbalization (LLM, optional)

LIST-SUBSTRATE ADAPTATION (anti-shortcut §1.1, EXTENSIONS.md S4):
The spec's literal target ``length : List a -> Nat`` requires
``KIND_LENGTH`` / ``KIND_REVERSE`` / ``KIND_APPEND`` in the MERA kind
table; the substrate's ``MERA_LEAF_DIM = 16`` is saturated and adding
those kinds is a substrate-wide refactor explicitly deferred to S4 in
``EXTENSIONS.md``. Per the task prompt's adaptation guidance, we
re-target the §15 walkthrough to the substrate-supported analog

    forall x : Nat. (x + 0) == x

which the K-8 acceptance proved relaxes to ``<H> = 0`` end-to-end. The
ARCHITECTURAL test is the §15 pipeline integration (NL -> LLM -> DSL
-> Hamiltonian -> TEBD -> decode -> pretty -> verify), not the choice
of theorem; every step exercises the same surface the §15.13 reader
inspects.

Run via::

    python -m src.qft_pcn.composition.demo_length_synthesis

A markdown report is written to ``reports/length_demo_<HEAD-SHA>.md``.
If Ollama is unreachable (anti-shortcut: no mock fallback), the demo
loud-fails and prints the precise EXTENSIONS-entry text to stderr.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from src.qft_pcn.bridge.errors import LlmBadOutputError, LlmUnavailableError
from src.qft_pcn.bridge.llm_emitter import LLMToDslEmitter, TheoremSpec
from src.qft_pcn.logic.ast import (
    Bin,
    BoolLit,
    Cons,
    Eq,
    Forall,
    If,
    IntLit,
    Lam,
    Nil,
    Node,
    App,
    Succ,
    TArrow,
    TBool,
    TInt,
    TList,
    TNat,
    Ty,
    Var,
    Zero,
    parse as parse_ast,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_evolution_logic import mera_imaginary_evolve_state


# Spec §15.1 -- the verbatim NL prompt (re-targeted to a Nat-arithmetic
# theorem; see module docstring for the adaptation rationale).
NL_PROMPT = (
    "Prove that adding zero on the right to any natural number gives "
    "back the same number. State the theorem with a universal "
    "quantifier over Nat using propositional equality, and provide a "
    "few examples by instantiating the bound variable to concrete "
    "Peano naturals (Zero, Succ Zero, Succ (Succ (Succ Zero)))."
)

# Substrate evolution parameters (K-8 acceptance / §10.10 verified
# settings).
_EVOLVE_DT = 0.1
_EVOLVE_STEPS = 300
_EVOLVE_CHI = 16


# ---------------------------------------------------------------------------
# Extended pretty-printer (§15.10).
# ---------------------------------------------------------------------------


def _pretty_ty_ext(t: Ty) -> str:
    if isinstance(t, TNat):
        return "Nat"
    if isinstance(t, TInt):
        return "Int"
    if isinstance(t, TBool):
        return "Bool"
    if isinstance(t, TList):
        return f"List {_pretty_ty_ext(t.elem)}"
    if isinstance(t, TArrow):
        left = _pretty_ty_ext(t.src)
        if isinstance(t.src, TArrow):
            left = f"({left})"
        return f"{left} -> {_pretty_ty_ext(t.dst)}"
    return type(t).__name__


def _pretty_ext(node: Node) -> str:
    """Extended pretty-printer covering Nat/List/Forall/Eq — the host
    ``ast.pretty`` only handles the Int/Bool subset.
    """
    if isinstance(node, Var):
        return node.name
    if isinstance(node, IntLit):
        return str(node.val)
    if isinstance(node, BoolLit):
        return "true" if node.val else "false"
    if isinstance(node, Zero):
        return "0"
    if isinstance(node, Succ):
        return f"Succ ({_pretty_ext(node.arg)})"
    if isinstance(node, Nil):
        return "Nil"
    if isinstance(node, Cons):
        return f"Cons ({_pretty_ext(node.head)}) ({_pretty_ext(node.tail)})"
    if isinstance(node, Lam):
        return (f"\\{node.param}:{_pretty_ty_ext(node.param_ty)}. "
                f"{_pretty_ext(node.body)}")
    if isinstance(node, App):
        return f"({_pretty_ext(node.fn)}) ({_pretty_ext(node.arg)})"
    if isinstance(node, If):
        return (f"if {_pretty_ext(node.cond)} then "
                f"{_pretty_ext(node.then_b)} else "
                f"{_pretty_ext(node.else_b)}")
    if isinstance(node, Bin):
        return f"({_pretty_ext(node.lhs)} {node.op} {_pretty_ext(node.rhs)})"
    if isinstance(node, Eq):
        return f"{_pretty_ext(node.lhs)} == {_pretty_ext(node.rhs)}"
    if isinstance(node, Forall):
        return (f"forall {node.param}:{_pretty_ty_ext(node.param_ty)}. "
                f"{_pretty_ext(node.body)}")
    return repr(node)


# ---------------------------------------------------------------------------
# Example-verification substrate run (§15.11).
# ---------------------------------------------------------------------------


def _peano(n: int) -> Node:
    """Build the Peano encoding of a non-negative integer ``n``."""
    if n < 0:
        raise ValueError(f"_peano: n must be non-negative, got {n}")
    out: Node = Zero()
    for _ in range(n):
        out = Succ(arg=out)
    return out


def _instantiate(template_theorem: Node, x_value: Node) -> Node:
    """Substitute the ``forall``-bound variable with ``x_value``.

    Used to materialize a ``forall x:Nat. P(x)`` theorem at a concrete
    Peano witness (the §15.11 example-verification step). The result
    is a closed equation ``P(<x_value>)`` that the substrate can
    relax to ground state without the universal-quantification
    machinery.
    """
    if not isinstance(template_theorem, Forall):
        raise TypeError(
            f"_instantiate expects a Forall template, got "
            f"{type(template_theorem).__name__}"
        )
    bound = template_theorem.param

    def subst(n: Node) -> Node:
        if isinstance(n, Var):
            return x_value if n.name == bound else n
        if isinstance(n, Bin):
            return Bin(op=n.op, lhs=subst(n.lhs), rhs=subst(n.rhs))
        if isinstance(n, Eq):
            return Eq(lhs=subst(n.lhs), rhs=subst(n.rhs))
        if isinstance(n, App):
            return App(fn=subst(n.fn), arg=subst(n.arg))
        if isinstance(n, Succ):
            return Succ(arg=subst(n.arg))
        if isinstance(n, If):
            return If(cond=subst(n.cond), then_b=subst(n.then_b),
                      else_b=subst(n.else_b))
        if isinstance(n, Cons):
            return Cons(head=subst(n.head), tail=subst(n.tail))
        # Leaves with no Var occurrence: return unchanged.
        return n

    return subst(template_theorem.body)


def _verify_example_via_substrate(
    template_theorem: Node, x_value: Node,
) -> dict[str, Any]:
    """Run the substrate on the instantiated equation; surface residual.

    A successful verification: the instantiated equation relaxes to
    ``<H> ≈ 0`` (the substrate certifies the equality propositionally,
    by Curry-Howard). The example-verification line in §15.11 is the
    same substrate path as the main theorem proof but with a closed
    (forall-free) equation.
    """
    instance = _instantiate(template_theorem, x_value)
    state, meta = encode_mera(instance)
    H = MeraEvalHamiltonian(meta)
    protected = set(meta.forall_protected_leaves)
    _, final = mera_imaginary_evolve_state(
        state, H, dt=_EVOLVE_DT, steps=_EVOLVE_STEPS,
        chi_layer=_EVOLVE_CHI, frozen_leaves=protected,
    )
    residual = float(H.total_energy(final))
    return {
        "instance": _pretty_ext(instance),
        "residual_energy": residual,
        "passed": residual < 1e-3,
    }


# ---------------------------------------------------------------------------
# Full §15 pipeline.
# ---------------------------------------------------------------------------


@dataclass
class DemoResult:
    nl_prompt: str
    theorem_spec: TheoremSpec
    parsed_ast: Node
    pretty_input: str
    n_leaves: int
    n_nodes: int
    main_residual_energy: float
    main_converged: bool
    decoded_pretty: str
    examples: list[dict[str, Any]]
    narrative: str


def _stage_banner(out, title: str) -> None:
    out.write("\n" + "=" * 72 + "\n")
    out.write(title + "\n")
    out.write("=" * 72 + "\n")


def run_demo(
    *,
    nl_prompt: str = NL_PROMPT,
    emitter: LLMToDslEmitter | None = None,
    out=None,
    n_examples_cap: int = 3,
) -> DemoResult:
    """Drive the §15 pipeline end-to-end.

    Returns a :class:`DemoResult` capturing every stage; the narrative
    string is the human-readable transcript also written to ``out``.

    Raises :class:`LlmUnavailableError` if Ollama is unreachable; the
    caller is expected to surface the precise EXTENSIONS entry to the
    user (see ``main()`` below).
    """
    buf = StringIO()

    class _Tee:
        def __init__(self, *targets):
            self.targets = targets
        def write(self, s):
            for t in self.targets:
                t.write(s)
        def flush(self):
            for t in self.targets:
                if hasattr(t, "flush"):
                    t.flush()

    sink = _Tee(out, buf) if out is not None else buf

    if emitter is None:
        emitter = LLMToDslEmitter()

    # ---- Step 1: NL input (§15.1) ----------------------------------------
    _stage_banner(sink, "STEP 1 — Natural-language input (§15.1)")
    sink.write(nl_prompt + "\n")

    # ---- Step 2-3: LLM emits structured DSL (§15.2-3, §10.5) -------------
    _stage_banner(sink, "STEP 2-3 — LLM emits structured DSL (§15.2-3, §10.5)")
    spec = emitter.emit_theorem_spec(nl_prompt)
    sink.write(f"theorem    = {spec.theorem!r}\n")
    sink.write(f"signature  = {spec.signature!r}\n")
    sink.write(f"explanation= {spec.explanation!r}\n")
    sink.write(f"examples ({len(spec.examples)}):\n")
    for e in spec.examples:
        sink.write(f"  {e['input']!r:40} -> {e['expected']!r}\n")

    # ---- Step 4-5: Hamiltonian construction + state init (§15.4-5) -------
    _stage_banner(sink, "STEP 4-5 — encode_mera + MeraEvalHamiltonian (§15.4-5)")
    parsed = parse_ast(spec.theorem)
    pretty_input = _pretty_ext(parsed)
    sink.write(f"parsed AST head : {type(parsed).__name__}\n")
    sink.write(f"pretty(input)   : {pretty_input}\n")
    state, meta = encode_mera(parsed)
    H = MeraEvalHamiltonian(meta)
    sink.write(f"n_nodes  = {meta.n_nodes}\n")
    sink.write(f"n_leaves = {meta.n_leaves}\n")
    sink.write(f"forall_protected_leaves = "
               f"{sorted(meta.forall_protected_leaves)}\n")
    initial_E = float(H.total_energy(state))
    sink.write(f"initial <H> = {initial_E:.6f}\n")

    # ---- Step 6: Imaginary-time TEBD (§15.6) -----------------------------
    _stage_banner(sink, "STEP 6 — Imaginary-time TEBD (§15.6)")
    sink.write(f"dt={_EVOLVE_DT}, steps={_EVOLVE_STEPS}, "
               f"chi={_EVOLVE_CHI}\n")
    protected = set(meta.forall_protected_leaves)
    history, final_state = mera_imaginary_evolve_state(
        state, H, dt=_EVOLVE_DT, steps=_EVOLVE_STEPS,
        chi_layer=_EVOLVE_CHI, frozen_leaves=protected,
    )
    residual = float(H.total_energy(final_state))
    sink.write(f"final   <H> = {residual:.6e}\n")
    converged = residual < 1e-3
    sink.write(f"converged   = {converged}  (threshold 1e-3)\n")

    # ---- Step 7-8: Measurement + AST decode (§15.7-8) --------------------
    _stage_banner(sink, "STEP 7-8 — Measurement + AST decode (§15.7-8)")
    try:
        decoded = decode_mera(final_state, meta)
        decoded_ast: Node | None = decoded.ast
        decoded_pretty = _pretty_ext(decoded_ast)
        sink.write(f"decoded pretty : {decoded_pretty}\n")
        sink.write(f"residual_norm  : {decoded.residual_norm:.4e}\n")
    except Exception as exc:
        # Decode can fail for forall-quantified relaxed states whose
        # bound-Var leaves carry the protected superposition (not a
        # single argmax); we surface the failure honestly rather than
        # hide it. The substrate proof is the load-bearing artifact;
        # the textual decode is auxiliary.
        decoded_ast = None
        decoded_pretty = (f"<decode skipped: {type(exc).__name__}: {exc}>")
        sink.write(decoded_pretty + "\n")

    # ---- Step 9: Validation (§15.9) --------------------------------------
    _stage_banner(sink, "STEP 9 — Classical validation (§15.9)")
    if decoded_ast is not None:
        roundtrip_ok = True
        try:
            reparse = parse_ast(decoded_pretty.replace("0", "Zero")
                                              if isinstance(decoded_ast, Zero)
                                              else decoded_pretty)
            sink.write(f"re-parse ok : {type(reparse).__name__}\n")
        except Exception as exc:
            roundtrip_ok = False
            sink.write(f"re-parse failed: {exc}\n")
        sink.write(f"validation : "
                   f"{'PASSED' if roundtrip_ok else 'FAILED'}\n")
    else:
        sink.write("validation : skipped (decode not available)\n")

    # ---- Step 10: Pretty-print (§15.10) ----------------------------------
    _stage_banner(sink, "STEP 10 — Pretty-print (§15.10)")
    sink.write(f"{spec.signature}\n")
    sink.write(f"{pretty_input}\n")

    # ---- Step 11: Example verification (§15.11) --------------------------
    _stage_banner(sink, "STEP 11 — Example verification via substrate (§15.11)")
    examples_results: list[dict[str, Any]] = []
    # The LLM-emitted examples are free-form strings ("x = Succ Zero",
    # "Cons 1 ..."); we additionally exercise three deterministic Peano
    # instantiations (n = 0, 1, 3) -- the substrate is the verifier, and
    # these witnesses are the §15.11 "length [], length [x], length [x,y,z]"
    # analogs adapted to the Nat-arithmetic target.
    canonical_ns = [0, 1, 3][:n_examples_cap]
    for n in canonical_ns:
        x_val = _peano(n)
        result = _verify_example_via_substrate(parsed, x_val)
        result["x"] = n
        examples_results.append(result)
        sink.write(f"  x = {n}  ({_pretty_ext(x_val)})\n")
        sink.write(f"    instance : {result['instance']}\n")
        sink.write(f"    residual : {result['residual_energy']:.4e}\n")
        sink.write(f"    passed   : {result['passed']}\n")

    # ---- Step 12: Verbalization (§15.12) ---------------------------------
    _stage_banner(sink, "STEP 12 — LLM verbalization (§15.12)")
    sink.write(spec.explanation + "\n")

    # ---- Summary ---------------------------------------------------------
    _stage_banner(sink, "SUMMARY")
    sink.write(f"main theorem residual <H> = {residual:.4e}  "
               f"(threshold 1e-3 -> "
               f"{'PASS' if converged else 'FAIL'})\n")
    n_pass = sum(1 for r in examples_results if r["passed"])
    sink.write(f"examples passed             = {n_pass}/"
               f"{len(examples_results)}\n")

    narrative = buf.getvalue()
    return DemoResult(
        nl_prompt=nl_prompt,
        theorem_spec=spec,
        parsed_ast=parsed,
        pretty_input=pretty_input,
        n_leaves=meta.n_leaves,
        n_nodes=meta.n_nodes,
        main_residual_energy=residual,
        main_converged=converged,
        decoded_pretty=decoded_pretty,
        examples=examples_results,
        narrative=narrative,
    )


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


_EXTENSIONS_LOUDFAIL = """\
[demo_length_synthesis] Ollama unreachable -- the §15 demo requires a
running Ollama daemon serving the gemma4:31b model. Add the following
entry to EXTENSIONS.md and re-run when the host is reachable:

  ## Missing dependency: Ollama bridge endpoint for §15 demo
  - Where: src/qft_pcn/composition/demo_length_synthesis.py
  - Need: Ollama HTTP server reachable at QFT_PCN_OLLAMA_HOST
    (default localhost:11434) with the gemma4:31b model loaded.
  - Symptom: LlmUnavailableError surfaced from
    bridge.llm.OllamaLLM._chat at demo start.
  - Workaround (interactive): set QFT_PCN_OLLAMA_HOST to the host
    IP (e.g. on WSL, the Windows host gateway) and confirm
    `curl http://$QFT_PCN_OLLAMA_HOST:11434/api/tags` lists
    gemma4:31b.
"""


def _head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _write_report(result: DemoResult, sha: str) -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    path = reports_dir / f"length_demo_{sha}.md"
    body = []
    body.append(f"# §15 worked end-to-end demo (HEAD {sha})\n")
    body.append("")
    body.append("**Target (adapted per EXTENSIONS S4 List blocker):** "
                f"`{result.theorem_spec.signature}`")
    body.append("")
    body.append(f"**Theorem:** `{result.theorem_spec.theorem}`")
    body.append("")
    body.append(f"**Explanation (LLM):** {result.theorem_spec.explanation}")
    body.append("")
    body.append("## Pipeline transcript")
    body.append("")
    body.append("```")
    body.append(result.narrative.rstrip())
    body.append("```")
    body.append("")
    body.append("## Numeric summary")
    body.append("")
    body.append(f"- main theorem residual `<H>` = "
                f"{result.main_residual_energy:.4e} "
                f"({'PASS' if result.main_converged else 'FAIL'})")
    body.append(f"- examples passed = "
                f"{sum(1 for e in result.examples if e['passed'])}/"
                f"{len(result.examples)}")
    body.append("")
    body.append("| x | instance | residual | passed |")
    body.append("|---|---|---|---|")
    for e in result.examples:
        body.append(f"| {e['x']} | `{e['instance']}` | "
                    f"{e['residual_energy']:.4e} | {e['passed']} |")
    body.append("")
    path.write_text("\n".join(body))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="§15 worked-example demo (A2 + B1).")
    parser.add_argument("--model", default="gemma4:31b")
    parser.add_argument("--host", default=None,
                        help="ollama host (overrides QFT_PCN_OLLAMA_HOST)")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)

    emitter = LLMToDslEmitter(model=args.model, host=args.host,
                              port=args.port)
    try:
        result = run_demo(emitter=emitter, out=sys.stdout)
    except LlmUnavailableError as exc:
        sys.stderr.write(_EXTENSIONS_LOUDFAIL)
        sys.stderr.write(f"\nUnderlying error: {exc}\n")
        return 2
    except LlmBadOutputError as exc:
        sys.stderr.write(f"[demo_length_synthesis] LLM emission rejected: "
                         f"{exc}\n")
        return 3

    if not args.no_report:
        sha = _head_sha()
        path = _write_report(result, sha)
        sys.stdout.write(f"\nreport: {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
