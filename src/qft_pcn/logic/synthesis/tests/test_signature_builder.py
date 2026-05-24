"""Tests for the free-form signature ingestion module.

Covers EXTENSIONS.md entry "free-form signature ingestion for QPCN
synthesis": ``parse_signature_string`` + ``signature_to_sketch`` +
QPCN baseline integration via ``payload["signature"]: str``.
"""
from __future__ import annotations

from src.qft_pcn.logic.ast import (
    HoleVar,
    Lam,
    Node,
    TArrow,
    TList,
    TNat,
    TypeHole,
)
from src.qft_pcn.logic.synthesis.signature_builder import (
    SignatureSpec,
    parse_signature_string,
    signature_to_sketch,
)


def test_parse_simple_signature():
    spec = parse_signature_string("length : List Nat -> Nat")
    assert isinstance(spec, SignatureSpec)
    assert spec.name == "length"
    assert isinstance(spec.ty, TArrow)
    assert isinstance(spec.ty.src, TList)
    assert isinstance(spec.ty.src.elem, TNat)
    assert isinstance(spec.ty.dst, TNat)
    # One top-level arrow + 1 body hole.
    assert spec.hole_count == 2


def test_signature_to_sketch_lambda_with_hole_body():
    # `id : a -> a` should yield `\_sig_x1:?. HoleVar()`.
    spec = parse_signature_string("id : a -> a")
    sketch = signature_to_sketch(spec)
    assert isinstance(sketch, Lam)
    assert sketch.param == "_sig_x1"
    # The polymorphic 'a' became a TypeHole.
    assert isinstance(sketch.param_ty, TypeHole)
    # The body is a structural HoleVar.
    assert isinstance(sketch.body, HoleVar)
    # Default HoleVar is the "any in-scope binder" var-hole form
    # (empty candidates).
    assert sketch.body.candidates == ()


def test_signature_to_sketch_handles_polymorphic():
    # `map : (a->b) -> List a -> List b` -> three-argument? No -- two
    # arguments (the function + the list) with two nested lambdas.
    spec = parse_signature_string("map : (a -> b) -> List a -> List b")
    assert spec.hole_count == 3  # 2 arrows + body hole
    sketch = signature_to_sketch(spec)
    # Outer: \_sig_x1:(a->b). \_sig_x2:List a. HoleVar()
    assert isinstance(sketch, Lam)
    assert sketch.param == "_sig_x1"
    # First arg is itself an arrow type a -> b (polymorphic).
    assert isinstance(sketch.param_ty, TArrow)
    assert isinstance(sketch.param_ty.src, TypeHole)
    assert isinstance(sketch.param_ty.dst, TypeHole)
    # Second lambda over List a.
    inner = sketch.body
    assert isinstance(inner, Lam)
    assert inner.param == "_sig_x2"
    assert isinstance(inner.param_ty, TList)
    assert isinstance(inner.param_ty.elem, TypeHole)
    # Body is the structural hole.
    assert isinstance(inner.body, HoleVar)


def test_qpcn_baseline_accepts_string_signature():
    """The QPCN adapter must accept a payload carrying a signature string
    and NOT short-circuit with "no builder_name" — it should drive the
    synthesis pipeline via signature_builder. The substrate may still
    fail to converge (synthesis is a hard problem), but the adapter
    MUST NOT report the absent-builder error; instead the diagnostics
    must carry the parsed signature.
    """
    from experiments.baselines import QPCNBaseline
    from experiments.schema import ProblemSpec

    problem = ProblemSpec(
        domain="synthesis",
        problem_id="signature-test",
        statement="id : Int -> Int",
        payload={
            "signature": "id : Int -> Int",
            # Keep the substrate run tiny -- we are checking the wiring,
            # not the synthesis outcome (free-form signatures have no
            # IO examples to discriminate on).
            "synth_knobs": {
                "N": 8,
                "chi_max": 4,
                "n_samples": 4,
                "anneal_steps": 8,
            },
        },
        tags=("test",),
    )
    att = QPCNBaseline(seed=0).solve(problem)
    # Either the substrate solved it OR it failed -- but the failure
    # must NOT be the "no builder_name" honest no-attempt; the
    # signature path was taken.
    err = att.error or ""
    assert "no builder_name" not in err
    # Diagnostics must reflect the signature route.
    assert att.diagnostics.get("signature") == "id : Int -> Int" or \
        att.diagnostics.get("signature_name") == "id" or \
        "signature" in att.diagnostics
