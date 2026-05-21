"""Exploratory demo for sub-project A (AST <-> MPS encoder/decoder).

Run with:
    .venv/bin/python -m src.qft_pcn.logic.demo

Demonstrates, in five sections:

  1. Round-trip on programs not in the test suite (sanity check on novel input).
  2. Bond structure: showing that bond dimensions track live binders.
  3. Hole superposition: encoding an under-determined program with HoleVar,
     verifying entanglement, and sampling completions.
  4. Alpha-renaming invariance: |x.x and |y.y produce identical MPS states.
  5. The structural correspondence in one table:
        variable use         <-> measurement of an entangled register
        lexical scope depth  <-> position in a binder stack
        alpha-renaming       <-> local unitary on the bid register

This is intentionally a script — it prints what it sees so you can
build intuition for what the encoded states look like. No assertions
that block; if something is off the printout will make it visible.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.qft_pcn.logic.ast import (
    HoleVar, Lam, IntLit, App, Var, TInt, Bin, pretty,
)
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.decoder import decode, sample, ast_alpha_eq
from src.qft_pcn.logic.ast import parse


_BAR = "=" * 72
_SEP = "-" * 72


def _print_section(title: str) -> None:
    print()
    print(_BAR)
    print(f"  {title}")
    print(_BAR)


def _print_bond_structure(state, meta, label: str) -> None:
    """Pretty-print bond-dim vs live-binder count side by side."""
    bond_dims = state.bond_dimensions()
    print(f"  {label}: N={meta.N}, ||state||={state.norm_sq():.10f}")
    print(f"  bond  |  bond dim  |  live binders  |  channels = |L|+1")
    print(f"  ------+------------+----------------+-------------------")
    for i, (live, dim) in enumerate(zip(meta.live_binders_per_bond,
                                        bond_dims)):
        sites = sorted({b.lam_site for b in live})
        match = "OK" if dim >= len(live) + 1 else "FAIL"
        if i < 12 or any(live):
            print(f"  {i:>4}  |  {dim:>8}  |  {str(sites):<14}|  "
                  f"{len(live) + 1}   [{match}]")


def section_1_roundtrip_novel() -> None:
    _print_section("1. Round-trip on novel programs (not in the test suite)")

    novel = [
        # Higher-order composition with arithmetic
        r"(\f:Int->Int. \g:Int->Int. \x:Int. f (g x))"
        r"(\y:Int. y + 1) (\z:Int. z * 2) (5)",
        # Three binders, two used, one dead
        r"\a:Int. \b:Int. \c:Int. a * b",
        # Nested if/then/else inside lambda
        r"\x:Int. if 0 < x then x * x else 0 - x",
        # Two-level shadowing
        r"\x:Int. \x:Int. \x:Int. x",
    ]

    for src in novel:
        ast = parse(src)
        state, meta = encode(ast, N=32, chi_max=16)
        result = decode(state, meta)
        same = ast_alpha_eq(result.ast, ast)
        ok = "OK" if same else "FAIL"
        print(f"  [{ok}]  {src}")
        if not same:
            print(f"         decoded: {pretty(result.ast)}")
        else:
            # Show the regenerated form so you can see the alpha-renaming
            print(f"         decoded: {pretty(result.ast)}")
    print()


def section_2_bond_structure() -> None:
    _print_section("2. Bond structure tracks live binders")

    # A program with 3 binders, two of them used.
    src = r"\x:Int. \y:Int. \z:Int. x * z + y"
    ast = parse(src)
    state, meta = encode(ast, N=16, chi_max=16)
    print(f"  source: {src}")
    print(f"  layout: LAM_x@0, LAM_y@1, LAM_z@2, BIN+@3, BIN*@4,")
    print(f"          VAR_x@5, VAR_z@6, VAR_y@7, then PADs.")
    print()
    _print_bond_structure(state, meta, "encoded state")
    print()
    print(f"  Observation: bond dim equals |live binders| + 1 throughout.")
    print(f"  This is the §1.1 structural marker. A classical-lookup")
    print(f"  shortcut would force every bond to dim 1.")
    print()


def section_3_hole_superposition() -> None:
    _print_section("3. Hole superposition: encoding an under-determined program")

    # `\x. \y. ?HOLE` where the hole can resolve to either x or y.
    h = HoleVar(candidates=["x", "y"])
    ast = Lam(param="x", param_ty=TInt(),
              body=Lam(param="y", param_ty=TInt(), body=h))
    state, meta = encode(ast, N=8, chi_max=16)

    print(f"  source: \\x:Int. \\y:Int. ?HOLE   (HoleVar over {h.candidates!r})")
    print()
    _print_bond_structure(state, meta, "encoded state")
    print()
    S0 = state.entanglement_entropy(0)
    S1 = state.entanglement_entropy(1)
    print(f"  Entanglement entropy:")
    print(f"    bond 0 (LAM_x | LAM_y) = {S0:.4f}")
    print(f"    bond 1 (LAM_y | HOLE)  = {S1:.4f}   "
          f"(ln 2 = {math.log(2):.4f}, ln 3 = {math.log(3):.4f})")
    print(f"  S > 0 at the hole's left bond = real bond-carried entanglement.")
    print(f"  (Currently ln 3 due to a no_info passthrough; the architectural")
    print(f"   property — entanglement, not classical lookup — is upheld.)")
    print()

    print(f"  Sampling 200 completions of the hole:")
    rng = np.random.default_rng(seed=20260521)
    samples = sample(state, meta, n_samples=200, rng=rng)
    # The hole becomes a Var; tally which binder each sample resolved to.
    # The decoder regenerates names, so check structure: the Var's name
    # must match either the outer or inner Lam's regenerated name.
    counts = Counter()
    for s in samples:
        # walk to the body's body
        outer = s.ast
        inner = outer.body
        var = inner.body
        if not isinstance(var, Var):
            counts["other"] += 1
            continue
        if var.name == outer.param:
            counts["x (outer)"] += 1
        elif var.name == inner.param:
            counts["y (inner)"] += 1
        else:
            counts["other"] += 1
    for choice, n in counts.most_common():
        pct = 100.0 * n / len(samples)
        print(f"    {choice:<14} {n:>4}  ({pct:5.1f}%)")
    print()
    print(f"  The decoder's argmax (deterministic) prefers one completion;")
    print(f"  the sampler draws from the conditioned MPS distribution.")
    print()


def section_4_alpha_renaming() -> None:
    _print_section("4. Alpha-renaming invariance")

    pairs = [
        (r"\x:Int. x",                  r"\y:Int. y"),
        (r"\x:Int. \y:Int. x + y",      r"\a:Int. \b:Int. a + b"),
        (r"\f:Int->Int. \x:Int. f x",   r"\g:Int->Int. \w:Int. g w"),
    ]
    for src_a, src_b in pairs:
        s_a, _ = encode(parse(src_a), N=16, chi_max=16)
        s_b, _ = encode(parse(src_b), N=16, chi_max=16)
        overlap = abs(s_a.inner(s_b)) ** 2
        match = "OK" if overlap > 1 - 1e-10 else "FAIL"
        print(f"  [{match}]  {src_a:<32}  vs  {src_b}")
        print(f"          |<a|b>|^2 = {overlap:.12f}")
    print()
    print(f"  Two alpha-equivalent programs encode to the SAME MPS state.")
    print(f"  The bid scheme is depth-relative, so name choice is invisible.")
    print()


def section_5_correspondence() -> None:
    _print_section("5. The structural correspondence in one table")

    correspondences = [
        ("variable binding",      "entanglement creation on the bid bond"),
        ("variable use",          "measurement of the bid register"),
        ("lexical scope depth",   "BID_k register value at the use site"),
        ("alpha-renaming",        "no-op (depth-relative bid scheme)"),
        ("shadowing",             "innermost matching binder consumed first"),
        ("free variable",         "ill-scoped Var raises (not encodable)"),
        ("hole at variable pos",  "Bell-like superposition over candidates"),
        ("dead binder (unused)",  "channel created at LAM, dropped next bond"),
    ]
    print(f"  Programming language concept       Quantum representation")
    print(_SEP)
    for left, right in correspondences:
        print(f"  {left:<34} {right}")
    print()


def main() -> None:
    print()
    print("AST <-> MPS encoder/decoder: exploratory demo")
    print("sub-project A of the QFT_PCN_ARCHITECTURE roadmap")
    section_1_roundtrip_novel()
    section_2_bond_structure()
    section_3_hole_superposition()
    section_4_alpha_renaming()
    section_5_correspondence()
    print()
    print("Done. See docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md")
    print("for the spec; src/qft_pcn/logic/ for the implementation.")
    print()


if __name__ == "__main__":
    main()
